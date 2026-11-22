"""Score a SingleFrameNet checkpoint on evaluation sets with the training evaluator.

Non-reactive (NR: other actors replay their logs) by default; ``--reactive`` adds the
reactive setting (R: IDM vehicles, rule-based pedestrians). One JSON line per set,
action, and setting; ``--output`` also writes them to a file.

    uv run python scripts/evaluate_checkpoint.py CHECKPOINT \
        --eval val14=data/nuplan/val14 --eval test14=data/nuplan/test14 --reactive \
        --output results.jsonl
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from cs5478_project.data import SceneDataLoader, SceneDataset
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.models.checkpoint import load_single_frame
from cs5478_project.training.mixing import WorldMode, mixed_controls
from cs5478_project.training.randomization import conditions_from
from cs5478_project.training.rollout import PPOEnv, evaluate
from cs5478_project.training.train import metric_rows, scene_paths, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        type=lambda value: tuple(value.split("=", 1)),
        metavar="NAME=DIR",
    )
    parser.add_argument("--action", nargs="+", default=["mode"], choices=("mean", "mode"))
    parser.add_argument("--reactive", action="store_true", help="Also score the R setting.")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument(
        "--goal-horizon",
        type=float,
        help="Seconds ahead along the route for the evaluation goal (the checkpoint's "
        "goal_horizon_s by default).",
    )
    parser.add_argument(
        "--reward-weights",
        nargs="+",
        default=[],
        metavar="NAME=VALUE",
        help="Reward weights told to a reward-conditioned policy instead of the defaults "
        "e.g. collision=-3 goal=1.",
    )
    parser.add_argument("--output", type=Path, help="Append the JSON lines to this file.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    loaded = load_single_frame(args.checkpoint, device=device)
    if args.goal_horizon is not None:
        config = dataclasses.replace(loaded.config, goal_horizon_s=args.goal_horizon)
        loaded = dataclasses.replace(loaded, config=config)
    overrides = {
        name: float(value) for name, value in (w.split("=", 1) for w in args.reward_weights)
    }
    conditions = conditions_from(overrides, device) if overrides else None
    evaluator = DrivingEvaluator(compile_step=True)
    for name, directory in args.eval:
        paths = scene_paths(directory)
        size = min(args.worlds, len(paths))
        slots = -(-len(paths) // size) * size
        padded = (paths * -(-slots // len(paths)))[:slots]
        loader = SceneDataLoader(
            SceneDataset(padded, validate=False),
            batch_size=size,
            device=device,
            collate_on_device=device.type == "cuda",
            idm_fields=args.reactive,
        )
        settings = ["NR", "R"] if args.reactive else ["NR"]
        for action in args.action:
            env = PPOEnv(loaded, compile_step=True, stochastic=False, deterministic=action)
            for setting in settings:
                start = time.perf_counter()
                rows = []
                for batch in loader:
                    scene, controls = batch.tensors, None
                    if setting == "R":
                        modes = torch.full_like(scene.ego_index, WorldMode.REACTIVE)
                        controls = mixed_controls(scene, modes, self_play_agents=0)
                    metrics = evaluate(env, evaluator, scene, controls, conditions)
                    rows.append(metric_rows(metrics))
                merged = {
                    key: torch.cat([row[key] for row in rows])[: len(paths)] for key in rows[0]
                }
                summary = summarize(merged)
                summary["seconds"] = time.perf_counter() - start
                record = {
                    "checkpoint": str(args.checkpoint),
                    "set": name,
                    "setting": setting,
                    "action": action,
                    "goal_horizon_s": loaded.config.goal_horizon_s,
                    "reward_weights": overrides,
                    "scenes": len(paths),
                    **summary,
                }
                line = json.dumps(record)
                print(line, flush=True)
                if args.output is not None:
                    with args.output.open("a") as file:
                        file.write(line + "\n")


if __name__ == "__main__":
    main()
