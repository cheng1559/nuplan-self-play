"""Train SingleFrameNet with PPO in non-reactive closed loop, evaluating every epoch.

    uv run sim-train --train data/nuplan/train_mini --eval val14=data/nuplan/val14 \
        --eval test14=data/nuplan/test14 --output outputs/train/nr-mini

Each iteration simulates ``--worlds`` training scenes for a full episode (ego is the
policy; everyone else replays the log), then runs PPO on the collected samples. After
every pass over the training scenes ("epoch"), the deterministic policy (Beta mode) is
scored on each evaluation set with the nuPlan-style metrics. Logs go to
``metrics.jsonl`` and TensorBoard in the output directory; checkpoints to
``checkpoints/``.
"""

import argparse
import dataclasses
import json
import math
import time
import warnings
from pathlib import Path

import torch

from cs5478_project.data import SceneDataLoader, SceneDataset
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.evaluation.evaluator import Metrics
from cs5478_project.models.checkpoint import (
    load_single_frame,
    random_single_frame,
    save_single_frame,
)
from cs5478_project.models.config import SingleFrameConfig
from cs5478_project.training.config import PPOConfig
from cs5478_project.training.distributed import Distributed, setup
from cs5478_project.training.mixing import WorldMode, mixed_controls, sample_modes
from cs5478_project.training.ppo import PPO, flatten, merge
from cs5478_project.training.randomization import randomize
from cs5478_project.training.rollout import PPOEnv, collect, ego_controls, evaluate

METRIC_NAMES = [name for name in Metrics._fields if name not in ("world", "slot")]


def scene_paths(directory) -> list[Path]:
    paths = sorted(Path(directory).glob("*.pt"))
    if not paths:
        raise ValueError(f"No scenes in {directory}")
    return paths


def summarize(rows: dict) -> dict:
    """Means of per-scene metric tensors, plus failure rates."""
    summary = {name: float(value.float().mean()) for name, value in rows.items()}
    summary["collision_rate"] = float((rows["no_ego_at_fault_collisions"] < 1).float().mean())
    summary["offroad_rate"] = float((rows["drivable_area_compliance"] < 1).float().mean())
    return summary


def metric_rows(metrics: Metrics) -> dict:
    return {name: getattr(metrics, name) for name in METRIC_NAMES}


class Trainer:
    def __init__(self, args, config: PPOConfig, distributed: Distributed = Distributed()):
        self.args, self.config = args, config
        self.distributed = distributed
        self.device = torch.device(args.device)
        if distributed.enabled and self.device.type == "cuda":
            self.device = torch.device("cuda", distributed.local_rank)
        # Same initial weights everywhere (broadcast below); rank-specific episodes.
        torch.manual_seed(config.seed)
        self.loaded = (
            load_single_frame(args.init, device=self.device)
            if args.init
            else random_single_frame(
                model_config(getattr(args, "model", [])), seed=config.seed, device=self.device
            )
        )
        self.model = self.loaded.model
        distributed.broadcast_parameters(self.model)
        # A continued run (--start-epoch) shuffles differently from the first epochs.
        self.start_epoch = getattr(args, "start_epoch", 1)
        torch.manual_seed(config.seed + distributed.rank + 1000 * (self.start_epoch - 1))
        if config.domain_randomization and not self.loaded.config.reward_conditions:
            raise ValueError("Domain randomization needs a policy with reward_conditions")
        # Each rank trains on its own equal shard of the scenes.
        paths = scene_paths(args.train)
        shard = len(paths) // distributed.world_size
        self.train_paths = paths[distributed.rank :: distributed.world_size][:shard]
        # Only rank 0 evaluates.
        evaluations = args.eval if distributed.main else []
        self.eval_sets = {name: scene_paths(path) for name, path in evaluations}
        self.iterations_per_epoch = len(self.train_paths) // config.worlds
        if self.iterations_per_epoch < 1:
            raise ValueError(f"Need at least {config.worlds} training scenes")
        total = -(-self.iterations_per_epoch * config.epochs // config.rollouts_per_update)
        self.ppo = PPO(self.model, config, total, distributed)
        self.evaluated_at = None
        # Continue the update count and the learning-rate schedule of a stopped run.
        self.iteration = (self.start_epoch - 1) * self.iterations_per_epoch
        self.iteration //= config.rollouts_per_update
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scheduler stepped before the optimizer
            for _ in range(self.iteration):
                self.ppo.scheduler.step()
        # Rollouts waiting for the next update (it may span an epoch boundary).
        self.buffer, self.buffer_stats = [], []
        self.store = torch.device("cpu") if config.store_on_host else None
        self.env = PPOEnv(self.loaded, compile_step=config.compile, amp=config.amp)
        # One evaluation environment per set: an environment keeps its batch shape across
        # resets, and each set runs as its own batch size.
        self.eval_envs = {
            name: PPOEnv(
                self.loaded,
                compile_step=config.compile,
                amp=config.amp,
                stochastic=False,
                deterministic=config.eval_action,
            )
            for name in self.eval_sets
        }
        self.evaluator = DrivingEvaluator(compile_step=config.compile)
        # Loaders keep their workers between epochs and evaluations.
        self.train_loader = self.loader(
            self.train_paths,
            shuffle=True,
            batch_size=config.worlds,
            prefetch=config.prefetch,
            idm_fields=self.mixed,
        )
        self.train_batches = self.train_loader.epochs(config.epochs - self.start_epoch + 1)
        self.eval_loaders = {}
        for name, paths in self.eval_sets.items():
            # Pad with repeats so every batch has the same shape; repeats are dropped.
            size = min(config.eval_worlds, len(paths))
            slots = -(-len(paths) // size) * size
            padded = (paths * -(-slots // len(paths)))[:slots]
            self.eval_loaders[name] = self.loader(
                padded, shuffle=False, batch_size=size, idm_fields=config.eval_reactive
            )
        self.output = Path(args.output)
        self.log_file = self.writer = None
        if not distributed.main:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        run = {"ppo": config.to_dict(), "args": vars(args), "ranks": distributed.world_size}
        (self.output / "config.json").write_text(json.dumps(run, indent=2, default=str))
        self.log_file = (self.output / "metrics.jsonl").open("a")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.output / "tensorboard")
        except ImportError:  # TensorBoard is optional
            self.writer = None

    def log(self, kind: str, step: int, values: dict, **extra):
        if self.log_file is None:
            return
        record = {"kind": kind, "step": step, "time": time.time(), **extra, **values}
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()
        if self.writer is not None:
            prefix = kind if "split" not in extra else f"{kind}/{extra['split']}"
            for name, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar(f"{prefix}/{name}", value, step)

    @property
    def mixed(self) -> bool:
        config = self.config
        return config.reactive_worlds > 0 or config.self_play_worlds > 0

    def loader(self, paths, *, shuffle, batch_size, prefetch=False, idm_fields=False):
        cache = self.config.cache_scenes
        return SceneDataLoader(
            # Exported scenes were validated on save.
            SceneDataset(paths, validate=False, cache=cache),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0 if cache else self.config.loader_workers,
            device=self.device,
            # Workers load compact scenes; padding happens on the device (no host copies).
            collate_on_device=self.device.type == "cuda",
            persistent_workers=True,
            prefetch=prefetch,
            # IDM and rule agents need rails and segment signals; non-reactive
            # episodes skip them.
            idm_fields=idm_fields,
            # A rare export over a capacity is swapped for another scene.
            replace_oversize=True,
        )

    def evaluate(self, epoch: int):
        modes = [("", None)]
        if self.config.eval_reactive:
            modes.append(("-reactive", WorldMode.REACTIVE))
        for name, paths in self.eval_sets.items():
            for suffix, mode in modes:
                start = time.perf_counter()
                rows = []
                for batch in self.eval_loaders[name]:
                    scene = batch.tensors
                    controls = None
                    if mode is not None:
                        modes_ = torch.full_like(scene.ego_index, mode)
                        controls = mixed_controls(scene, modes_, self_play_agents=0)
                    metrics = evaluate(self.eval_envs[name], self.evaluator, scene, controls)
                    rows.append(metric_rows(metrics))
                merged = {
                    key: torch.cat([row[key] for row in rows])[: len(paths)] for key in rows[0]
                }
                summary = summarize(merged)
                summary["seconds"] = time.perf_counter() - start
                split = name + suffix
                self.log(
                    "eval", epoch, summary, split=split, epoch=epoch, update=self.iteration,
                    scenes=len(paths),
                )  # fmt: skip
                print(f"[eval {split} epoch {epoch}] score {summary['score']:.3f} "
                      f"collisions {summary['collision_rate']:.3f} "
                      f"offroad {summary['offroad_rate']:.3f} "
                      f"progress {summary['ego_progress_along_expert_route']:.3f} "
                      f"({summary['seconds']:.0f}s)", flush=True)  # fmt: skip

    def train_epoch(self, epoch: int):
        config = self.config
        clock = time.perf_counter()
        for _ in range(self.iterations_per_epoch):
            batch = next(self.train_batches)
            loaded = time.perf_counter()
            scene, controls, mode_stats = batch.tensors, None, {}
            if self.mixed:
                weights = (config.log_worlds, config.reactive_worlds, config.self_play_worlds)
                modes = sample_modes(config.worlds, weights, self.device)
                controls = mixed_controls(scene, modes, config.self_play_agents)
                mode_stats = {
                    f"{mode.name.lower()}_worlds": float((modes == mode).float().mean())
                    for mode in WorldMode
                }
            blind = weights = None
            if config.domain_randomization:
                if controls is None:
                    controls = ego_controls(scene)
                scene, blind, weights = randomize(scene, controls, config.blind_probability)
            rollout, metrics = collect(self.env, self.evaluator, scene, controls, blind, weights)
            returns = (rollout.reward.sum(-1) * rollout.valid).sum(0)
            self.buffer.append(flatten(rollout, config.gamma, config.gae_lambda, self.store))
            # Ego metrics as before; self-play vehicles separately.
            ego = metrics.slot == scene.ego_index[metrics.world]
            rows = metric_rows(metrics)
            npc_stats = {}
            if bool((~ego).any()):
                npc = summarize({key: value[~ego] for key, value in rows.items()})
                npc_stats = {f"self_play_{key}": npc[key] for key in ("score", "collision_rate")}
                npc_stats["self_play_agents"] = float((~ego).sum()) / config.worlds
            self.buffer_stats.append(
                {
                    **summarize({key: value[ego] for key, value in rows.items()}),
                    **npc_stats,
                    **mode_stats,
                    "episode_return": float(returns.mean()),
                    "episode_length": float(rollout.valid.float().sum(0).mean()),
                    "load_seconds": loaded - clock,
                    "rollout_seconds": time.perf_counter() - loaded,
                }
            )
            del rollout
            clock = time.perf_counter()
            if len(self.buffer) == config.rollouts_per_update:
                self.update(epoch)
                clock = time.perf_counter()

    def update(self, epoch: int):
        start = time.perf_counter()
        stats = self.ppo.update(merge(self.buffer))
        stats["update_seconds"] = time.perf_counter() - start
        collected = self.buffer_stats
        for name in collected[0]:
            mean = sum(row[name] for row in collected) / len(collected)
            stats[name] = mean * len(collected) if name.endswith("_seconds") else mean
        stats["scenes"] = len(collected) * self.config.worlds * self.distributed.world_size
        seconds = stats["load_seconds"] + stats["rollout_seconds"] + stats["update_seconds"]
        stats["samples_per_second"] = stats["samples"] / seconds
        self.buffer, self.buffer_stats = [], []
        self.iteration += 1
        self.log("train", self.iteration, stats, epoch=epoch)
        if self.distributed.main:
            print(f"[train epoch {epoch} update {self.iteration}] {stats['scenes']} scenes, "
                  f"{stats['samples']} samples | return {stats['episode_return']:.3f} "
                  f"score {stats['score']:.3f} coll {stats['collision_rate']:.2f} "
                  f"off {stats['offroad_rate']:.2f} ent {stats['entropy']:.2f} "
                  f"kl {stats['approx_kl']:.4f} ev {stats['explained_variance']:.2f} "
                  f"| load {stats['load_seconds']:.0f}s rollout {stats['rollout_seconds']:.0f}s "
                  f"update {stats['update_seconds']:.0f}s "
                  f"{stats['samples_per_second']:.0f} samples/s", flush=True)  # fmt: skip
        every = self.config.eval_updates
        if every > 0 and self.iteration % every == 0:
            self.checkpoint_and_evaluate(epoch, f"update_{self.iteration:05d}.pt")

    def checkpoint_and_evaluate(self, epoch: int, name: str):
        if self.evaluated_at == self.iteration:
            return  # the last epoch ended on an update that was just evaluated
        self.evaluated_at = self.iteration
        if self.distributed.main:
            save_single_frame(self.output / "checkpoints" / name, self.model, update=self.iteration)
            self.evaluate(epoch)
        self.distributed.barrier()

    def run(self):
        main = self.distributed.main
        if main and self.start_epoch == 1:
            self.evaluate(0)
        self.distributed.barrier()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.train_epoch(epoch)
            last = epoch == self.config.epochs
            if self.config.eval_updates > 0 and not last:
                continue
            if epoch % self.config.eval_interval and not last:
                continue
            self.checkpoint_and_evaluate(epoch, f"epoch_{epoch:03d}.pt")
        if self.log_file is not None:
            self.log_file.close()
        if self.writer is not None:
            self.writer.close()


def model_config(overrides) -> SingleFrameConfig:
    """SingleFrameConfig with KEY=VALUE overrides, parsed by each field's default type."""
    config = SingleFrameConfig()
    values = {}
    for item in overrides:
        key, _, text = item.partition("=")
        if key not in {f.name for f in dataclasses.fields(SingleFrameConfig)}:
            raise ValueError(f"Unknown SingleFrameConfig field {key!r}")
        default = getattr(config, key)
        if isinstance(default, bool):
            if text.lower() not in ("true", "false"):
                raise ValueError(f"{key} takes true or false")
            values[key] = text.lower() == "true"
        else:
            values[key] = type(default)(text)
    return dataclasses.replace(config, **values)


def _prepare_loader_workers(workers: int) -> None:
    """Worker processes send each scene's many tensors through shared memory. The
    default strategy holds one file descriptor per tensor, which exceeds common limits
    (1024) for a batch of 2048 scenes; share through files and raise the soft limit."""
    if workers <= 0:
        return
    import resource

    torch.multiprocessing.set_sharing_strategy("file_system")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True, help="Directory of training scenes.")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        type=lambda value: tuple(value.split("=", 1)),
        metavar="NAME=DIR",
        help="Evaluation set; repeat for several.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path, help="Start from this SingleFrameNet checkpoint.")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Continue a stopped run at this epoch (with --init from the checkpoint before "
        "it): the update count and learning-rate schedule resume; optimizer moments do not.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="SingleFrameConfig field for a fresh network (repeatable), e.g. embed_dim=256.",
    )
    parser.add_argument("--device", default="cuda")
    for field in dataclasses.fields(PPOConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type in (bool, "bool"):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            kind = {"int": int, "float": float}.get(str(field.type), field.type)
            parser.add_argument(flag, type=kind, default=field.default)
    args = parser.parse_args(argv)
    config = PPOConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PPOConfig)})
    _prepare_loader_workers(0 if config.cache_scenes else config.loader_workers)
    if config.cpu_threads > 0:
        torch.set_num_threads(config.cpu_threads)
    distributed = setup(args.device)
    Trainer(args, config, distributed).run()
    if distributed.enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
