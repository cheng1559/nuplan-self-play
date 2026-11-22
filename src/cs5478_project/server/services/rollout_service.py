"""Catalog normalized inputs and execute complete, isolated simulation batches."""

import base64
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

from cs5478_project.checkpoints import (
    RANDOM_INIT_PREFIX,
    discover_single_frame_checkpoints,
)
from cs5478_project.defaults import EPISODE_FRAMES
from cs5478_project.server.schemas import (
    Actor,
    ActorState,
    AddedActor,
    AgentControl,
    BatchRollout,
    CaseDetail,
    ControlPreset,
    Frame,
    LogPose,
    ObservationTrack,
    ObservedChunk,
    ObservedFrame,
    PolicyTrack,
    RolloutSettings,
    RolloutSource,
    RolloutSourceList,
    RolloutWorld,
    WorldMetrics,
    WorldSummary,
)
from cs5478_project.server.services.static_objects import static_actor_metadata
from cs5478_project.visualization.map import display_map_from_data
from cs5478_project.visualization.replay import ACTOR_TYPES, RecordedCase

LEARNED_KINDS = ("single_frame",)


class RolloutBusyError(RuntimeError):
    """Another request currently owns this service's simulation execution slot."""


class RolloutUnavailableError(RuntimeError):
    """The requested execution device is unavailable."""


class UnknownRollout(KeyError):
    """The rollout is not the latest SingleFrameNet run kept for observation queries."""


class UnknownRolloutSource(KeyError):
    """A request referenced an input outside the registered catalog."""


METRIC_FIELDS = tuple(WorldMetrics.model_fields)
# Indexed by ControlType value; code 3 is unused.
CONTROL_NAMES = ("log", "idm", "pedestrian", "unused", "single_frame", "absent")


@dataclass(frozen=True)
class _Source:
    summary: RolloutSource
    path: Path | None = None


class RolloutService:
    """Keep source identities on CPU and allow one live batch per process.

    Completed responses are returned directly, without storing results or
    writing artifacts. Each request reconstructs scenes and a fresh environment.
    """

    def __init__(
        self,
        artifact_roots: list[Path] | None = None,
        *,
        include_demo: bool = False,
        settings: RolloutSettings | None = None,
    ) -> None:
        self.settings = settings or RolloutSettings()
        self._sources: dict[str, _Source] = {}
        self._lock = threading.Lock()
        self._policy_cache = {}
        # Latest SingleFrameNet rollout, kept to rebuild any query's observation on demand.
        self._observation_context: dict | None = None
        # Latest rollout of any kind, kept to serve its worlds one at a time.
        self._latest: dict | None = None
        # The last batch's scenes (by path) and its device batch, reused by a rerun.
        self._scene_cache: dict = {}
        self._batch_cache: tuple | None = None
        if include_demo:
            for index, (scenario_id, frames) in enumerate(
                (("synthetic-demo", EPISODE_FRAMES), ("synthetic-secondary", EPISODE_FRAMES))
            ):
                identifier = self._identifier(f"synthetic:{index}")
                self._sources[identifier] = _Source(
                    summary=RolloutSource(
                        id=identifier,
                        scenario_id=scenario_id,
                        source_kind="demo",
                        num_frames=frames,
                        dt=0.1,
                        duration_s=(frames - 1) * 0.1,
                        actor_count=6,
                        dynamic_actor_count=4,
                        static_object_count=2,
                        num_actor_slots=4,
                        has_map=False,
                    )
                )
        roots = artifact_roots
        if roots is None:
            roots = [
                path for path in (Path("outputs/replay"), Path("data/scenes")) if path.exists()
            ]
        seen: set[Path] = set()
        for root_index, root in enumerate(roots):
            root = Path(root).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Artifact root does not exist: {root}")
            files = [root] if root.is_file() else sorted(root.rglob("*.pt"))
            for path in files:
                if not path.is_file() or path.suffix.lower() != ".pt" or path in seen:
                    continue
                seen.add(path)
                relative = path.name if root.is_file() else path.relative_to(root).as_posix()
                collection = root.parent.name if root.is_file() else root.name
                self._register(path, self._identifier(f"root-{root_index}:{relative}"), collection)

    @staticmethod
    def _identifier(key: str) -> str:
        return base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")

    def _register(self, path: Path, identifier: str, collection: str = "") -> None:
        """List a scene from its summary; the scene itself is loaded when a rollout uses it."""
        from cs5478_project.data.dataset import read_scene_summary

        scene = read_scene_summary(path)
        scenario_type = None
        metadata = path.with_name(path.stem + ".metadata.json")
        if metadata.is_file():
            try:
                scenario_type = json.loads(metadata.read_text())["source"]["scenario_type"]
            except (ValueError, KeyError, TypeError):
                scenario_type = None
        self._sources[identifier] = _Source(
            summary=RolloutSource(
                id=identifier,
                scenario_id=scene.scenario_id,
                source_kind="scene",
                num_frames=scene.num_frames,
                dt=scene.dt,
                duration_s=(scene.num_frames - 1) * scene.dt,
                actor_count=scene.dynamic_actors + scene.static_objects,
                dynamic_actor_count=scene.dynamic_actors,
                static_object_count=scene.static_objects,
                num_actor_slots=scene.num_actors,
                has_map=scene.has_map,
                collection=collection,
                scenario_type=scenario_type,
            ),
            path=path,
        )

    def list_sources(self) -> RolloutSourceList:
        return RolloutSourceList(
            sources=[source.summary for source in self._sources.values()],
            settings=self.settings,
            policy_checkpoints=discover_single_frame_checkpoints(),
        )

    def run(
        self,
        source_ids: list[str],
        *,
        preset: ControlPreset = "log",
        agent_controls: list[AgentControl] | None = None,
        checkpoint: str | None = None,
        checkpoint_config: str | None = None,
        sampling_method: str = "argmax",
        lazy: bool = False,
        goal_horizon_s: float | None = None,
        added_actors: list[AddedActor] | None = None,
    ) -> BatchRollout:
        """Execute a fresh batch to completion before returning any output frames.

        ``lazy`` returns per-world summaries and ego scores only (frames come from
        ``world``) and skips scenes over the capacities instead of failing."""
        if not self._lock.acquire(blocking=False):
            raise RolloutBusyError("A batch rollout is already running; retry after it completes")
        try:
            return self._run(
                source_ids,
                preset=preset,
                agent_controls=agent_controls or [],
                checkpoint=checkpoint,
                checkpoint_config=checkpoint_config,
                sampling_method=sampling_method,
                lazy=lazy,
                goal_horizon_s=goal_horizon_s,
                added_actors=added_actors or [],
            )
        finally:
            self._lock.release()

    def _run(
        self,
        source_ids: list[str],
        *,
        preset: ControlPreset,
        agent_controls: list[AgentControl],
        checkpoint: str | None = None,
        checkpoint_config: str | None = None,
        sampling_method: str = "argmax",
        lazy: bool = False,
        goal_horizon_s: float | None = None,
        added_actors: list[AddedActor] = (),
    ) -> BatchRollout:

        from cs5478_project.data import (
            SceneDataLoader,
            SceneDataset,
            load_scene,
            make_synthetic_scene,
        )
        from cs5478_project.pipelines.rollout import rollout_log_batch, rollout_rule_based_batch
        from cs5478_project.server.services.additions import add_actors
        from cs5478_project.server.services.controls import (
            resolve_blind,
            resolve_controls,
            resolve_goals,
        )

        started = time.perf_counter()
        if not source_ids or len(source_ids) > self.settings.max_batch_size:
            raise ValueError(f"Select between 1 and {self.settings.max_batch_size} rollout sources")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_ids must not contain duplicates")
        try:
            sources = [self._sources[identifier] for identifier in source_ids]
        except KeyError as error:
            raise UnknownRolloutSource(str(error.args[0])) from error

        def read(source):
            if source.path is None:
                return make_synthetic_scene(
                    scenario_id=source.summary.scenario_id,
                    num_frames=source.summary.num_frames,
                    num_actors=source.summary.actor_count,
                    dt=source.summary.dt,
                )
            # The previous batch's scenes are reused unless their files changed.
            return self._scene_cache.get(_file_key(source.path)) or load_scene(source.path)

        with ThreadPoolExecutor(8) as pool:
            scenes = list(pool.map(read, sources))
        self._scene_cache = {
            _file_key(source.path): scene
            for source, scene in zip(sources, scenes, strict=True)
            if source.path is not None
        }
        # Scene edits: the cache above keeps the files' scenes, the batch gets the edited ones.
        edited = [
            add_actors(scene, [item for item in added_actors if item.source_id == source_id])
            for scene, source_id in zip(scenes, source_ids, strict=True)
        ]
        scenes, added = [item[0] for item in edited], [item[1] for item in edited]
        skipped = []
        if lazy:
            fits = [
                scene.num_actors <= self.settings.max_actors
                and scene.num_frames <= self.settings.max_frames
                for scene in scenes
            ]
            if not any(fits):
                raise ValueError("Every selected scene exceeds the visualizer's capacities")
            skipped = [
                scene.scenario_id for scene, fit in zip(scenes, fits, strict=True) if not fit
            ]
            scenes, sources, source_ids, added = (
                [item for item, fit in zip(items, fits, strict=True) if fit]
                for items in (scenes, sources, source_ids, added)
            )
            agent_controls = [item for item in agent_controls if item.source_id in source_ids]
        for scene in scenes:
            if scene.num_actors > self.settings.max_actors:
                raise ValueError(
                    f"Scene {scene.scenario_id!r} needs {scene.num_actors} actor slots; "
                    f"max_actors={self.settings.max_actors}"
                )
            if scene.num_frames > self.settings.max_frames:
                raise ValueError(
                    f"Scene {scene.scenario_id!r} needs {scene.num_frames} frames; "
                    f"max_frames={self.settings.max_frames}"
                )
        control_types, modes = resolve_controls(
            scenes, source_ids, preset, agent_controls, self.settings.max_actors
        )
        blind = resolve_blind(scenes, source_ids, agent_controls, self.settings.max_actors)
        pinned, pinned_goals = resolve_goals(
            scenes, source_ids, agent_controls, self.settings.max_actors
        )
        reactive = any(mode != "log" for selected in modes for mode in selected)
        families = {mode for selected in modes for mode in selected if mode in LEARNED_KINDS}
        if len(families) > 1:
            raise ValueError("A batch runs one learned model")
        learned = next(iter(families), None)
        device = self._resolve_device()
        batch_key = (
            tuple(_file_key(item.path) if item.path else item.summary.id for item in sources),
            device,
            tuple(item.model_dump_json() for item in added_actors),
        )
        if self._batch_cache is not None and self._batch_cache[0] == batch_key:
            loaded = self._batch_cache[1]
        else:
            self._batch_cache = None  # free the old batch before building the new one
            loader = SceneDataLoader(
                SceneDataset(scenes),
                batch_size=len(scenes),
                max_actors=self.settings.max_actors,
                max_frames=self.settings.max_frames,
                max_static_objects=self.settings.max_static_objects,
                max_map_segments=self.settings.max_map_segments,
                device=device,
                drop_last=False,
                collate_on_device=device == "cuda",  # much faster than a CPU collate
            )
            loaded = next(iter(loader))
            self._batch_cache = (batch_key, loaded)
        if preset == "ego_single_frame_reactive":
            control_types = _reactive_background(
                loaded.tensors, modes, source_ids, scenes, agent_controls
            )
        rollout_options = {
            "num_steps": max(scene.num_frames for scene in scenes) - 1,
            "compile_step": self.settings.compiled,
        }
        if lazy:
            from cs5478_project.evaluation import DrivingEvaluator

            rollout_options["evaluator"] = DrivingEvaluator(compile_step=self.settings.compiled)
        if learned is not None:
            from cs5478_project.pipelines.rollout import rollout_learned_batch

            loaded_policy = self._load_policy(learned, checkpoint, checkpoint_config, device)
            checkpoint = loaded_policy.checkpoint
            if learned == "single_frame" and goal_horizon_s is not None:
                import dataclasses

                # A copy: the cached policy keeps its own setting.
                config = dataclasses.replace(loaded_policy.config, goal_horizon_s=goal_horizon_s)
                loaded_policy = dataclasses.replace(loaded_policy, config=config)
            trace = rollout_learned_batch(
                loaded.tensors,
                loaded_policy=loaded_policy,
                **rollout_options,
                control_types=control_types.to(device=device),
                blind=blind.to(device=device),
                sampling_method=sampling_method,
                goal_overrides=(pinned.to(device=device), pinned_goals.to(device=device))
                if bool(pinned.any())
                else None,
            )
        elif reactive:
            trace = rollout_rule_based_batch(
                loaded.tensors,
                **rollout_options,
                control_types=control_types.to(device=device),
                blind=blind.to(device=device),
            )
        else:
            trace = rollout_log_batch(loaded.tensors, **rollout_options)
        run_id = str(uuid4())
        self._latest = {
            "run_id": run_id,
            "scenes": scenes,
            "sources": sources,
            "trace": trace,
            "modes": modes,
            "blind": blind,
            # Fixed goals apply only to actors a learned policy drives.
            "pinned": pinned if learned else pinned & False,
            "pinned_goals": pinned_goals,
            "added": added,
            "config": loaded_policy.config if learned == "single_frame" else None,
        }
        self._observation_context = (
            {
                "run_id": run_id,
                "scene": loaded.tensors,
                "trace": trace,
                "control_types": control_types.to(device=device),
                "blind": blind.to(device=device),
                "config": loaded_policy.config,
            }
            if learned == "single_frame"
            else None
        )
        worlds = [] if lazy else [self._world(world) for world in range(len(scenes))]
        summaries = _summaries(scenes, sources, trace.metrics)
        return BatchRollout(
            id=run_id,
            preset=preset,
            policy=learned or ("rule_based" if reactive else "log"),
            pedestrian_oracle=any(mode == "pedestrian" for selected in modes for mode in selected),
            control_ego=any(
                selected[scene.ego_index] != "log"
                for selected, scene in zip(modes, scenes, strict=True)
            ),
            device=device,
            compiled=self.settings.compiled,
            batch_size=len(scenes),
            elapsed_s=time.perf_counter() - started,
            world_transitions=sum(scene.num_frames - 1 for scene in scenes),
            worlds=worlds,
            checkpoint=checkpoint if learned else None,
            summaries=summaries,
            skipped=skipped,
        )

    def world(self, run_id: str, world: int) -> RolloutWorld:
        """Frames, actors, and map of one world of the latest rollout."""
        context = self._latest
        if context is None or context["run_id"] != run_id:
            raise UnknownRollout(run_id)
        if not 0 <= world < len(context["scenes"]):
            raise ValueError(f"World {world} is not in rollout {run_id}")
        return self._world(world)

    def policy(self, run_id: str, world: int, actor_id: str) -> PolicyTrack:
        """Action distributions, values, and (ego) rewards of one SingleFrameNet actor."""
        import torch

        from cs5478_project.models.distributions import decode_unit
        from cs5478_project.training.config import PPOConfig

        context = self._latest
        if context is None or context["run_id"] != run_id:
            raise UnknownRollout(run_id)
        trace, config = context["trace"], context["config"]
        if trace.policy is None or config is None:
            raise ValueError("This rollout has no SingleFrameNet decisions")
        if not 0 <= world < len(context["scenes"]):
            raise ValueError(f"World {world} is not in rollout {run_id}")
        scene = context["scenes"][world]
        try:
            slot = scene.actor_ids.tolist().index(int(actor_id.removeprefix("actor-")))
        except ValueError as error:
            raise ValueError(f"Unknown dynamic actor: {actor_id}") from error
        policy = trace.policy
        match = ((policy["query_world"] == world) & (policy["query_slot"] == slot)).nonzero()
        if not len(match):
            raise ValueError(f"{actor_id} is not controlled by SingleFrameNet")
        query = int(match[0, 0])
        frames = scene.num_frames
        alpha, beta, unit = (policy[name][:frames, query] for name in ("alpha", "beta", "unit"))
        value = policy["value"][:frames, query]
        jerk, steering_rate = decode_unit(unit, config)
        action = torch.stack((jerk, steering_rate), -1)

        def pairs(tensor):
            return [None if bool(row.isnan().any()) else tuple(row.tolist()) for row in tensor]

        def numbers(tensor):
            return [None if math.isnan(item) else item for item in tensor.tolist()]

        rewards = None
        if trace.rewards is not None and slot == int(scene.ego_index):
            rewards = {
                name: numbers(value[:frames, world]) for name, value in trace.rewards.items()
            }
        return PolicyTrack(
            actor_id=actor_id,
            jerk_range=(config.min_jerk, config.max_jerk),
            steering_rate_range=(-config.max_steering_rate, config.max_steering_rate),
            gamma=PPOConfig().gamma,
            alpha=pairs(alpha),
            beta=pairs(beta),
            action=[
                None if row is None else pair
                for row, pair in zip(pairs(unit), pairs(action), strict=True)
            ],
            value=numbers(value),
            rewards=rewards,
        )

    def _world(self, world: int) -> RolloutWorld:
        import numpy as np

        context = self._latest
        run_id, trace = context["run_id"], context["trace"]
        modes, blind = context["modes"], context["blind"]
        scene, source = context["scenes"][world], context["sources"][world]
        scene_map = (
            display_map_from_data(scene.map, scene.scenario_id) if scene.map is not None else None
        )
        case = RecordedCase(
            scenario_id=scene.scenario_id,
            dt=scene.dt,
            actor_type=scene.actor_type.numpy(),
            actor_mask=scene.actor_mask.numpy(),
            actor_ids=scene.actor_ids.numpy(),
            static={name: value.numpy() for name, value in scene.static._asdict().items()},
            valid=trace.actor_mask[: scene.num_frames, world, : scene.num_actors].numpy(),
            size=scene.size.numpy(),
            reference_to_center=scene.reference_to_center.numpy(),
            state={
                name: value[: scene.num_frames, world, : scene.num_actors].numpy()
                for name, value in zip(trace.state._fields, trace.state, strict=True)
            },
        )
        case.validate()
        actors = [
            Actor(
                id=case.actor_id(slot),
                slot=int(slot),
                type=ACTOR_TYPES[case.actor_type[slot]],
                length=float(case.size[slot, 0]),
                width=float(case.size[slot, 1]),
                reference_to_center=float(case.reference_to_center[slot]),
                is_ego=int(slot) == scene.ego_index,
                blind=bool(blind[world, slot]),
                control_type=modes[world][slot],
                added_serial=context["added"][world].get(int(case.actor_ids[slot])),
                goal_override=tuple(context["pinned_goals"][world, slot].tolist())
                if bool(context["pinned"][world, slot]) and modes[world][slot] in LEARNED_KINDS
                else None,
                route_start_s=float(scene.routes.start_time_s[slot])
                if scene.routes is not None
                else 0.0,
                route_horizon_s=float(scene.routes.horizon_s[slot])
                if scene.routes is not None
                else 0.0,
                route=scene.routes.points[slot, scene.routes.mask[slot]].tolist()
                if scene.routes is not None
                else [],
                route_source=("none", "map", "gt")[int(scene.routes.source[slot])]
                if scene.routes is not None
                else "none",
                has_rail=bool(scene.rails.rail_valid[slot]) if scene.rails is not None else False,
                reactive_role=("absent", "idm", "open_loop", "ego")[int(scene.rails.role[slot])]
                if scene.rails is not None
                else None,
            )
            for slot in case.slots
        ]
        actors.extend(
            item.model_copy(update={"added_serial": context["added"][world].get(int(item.id[6:]))})
            for item in static_actor_metadata(case.static)
        )
        detail = CaseDetail(
            id=f"{run_id}:{world}",
            scenario_id=case.scenario_id,
            source_kind=source.summary.source_kind,
            num_frames=case.num_frames,
            dt=case.dt,
            duration_s=(case.num_frames - 1) * case.dt,
            actor_count=len(actors),
            dynamic_actor_count=int(scene.actor_mask.sum()),
            static_object_count=int(scene.static.mask.sum()),
            actors=actors,
            bounds=case.bounds(),
            has_map=scene_map is not None,
        )
        centers = case.centers()
        logged_heading = scene.reference_state.heading.numpy()
        logged_centers = scene.reference_state.position.numpy() + (
            np.stack((np.cos(logged_heading), np.sin(logged_heading)), axis=-1)
            * scene.reference_to_center.numpy()[None, :, None]
        )
        frames = []
        for tick in range(case.num_frames):
            states = []
            for slot in case.slots:
                if not bool(trace.actor_mask[tick, world, slot]):
                    continue
                velocity = case.state["velocity"][tick, slot]
                states.append(
                    ActorState(
                        id=case.actor_id(slot),
                        position=case.state["position"][tick, slot].tolist(),
                        center=centers[tick, slot].tolist(),
                        heading=float(case.state["heading"][tick, slot]),
                        velocity=velocity.tolist(),
                        speed=math.hypot(*velocity),
                        log_pose=LogPose(
                            center=logged_centers[tick, slot].tolist(),
                            heading=float(logged_heading[tick, slot]),
                        )
                        if modes[world][slot] != "log" and bool(scene.reference_valid[tick, slot])
                        else None,
                        goals=trace.goals[tick, world, slot].tolist()
                        if trace.goals is not None and bool(trace.goal_mask[tick, world, slot])
                        else [],
                        **{
                            name: float(case.state[name][tick, slot])
                            for name in (
                                "acceleration",
                                "acceleration_command",
                                "steering",
                                "yaw_rate",
                            )
                        },
                    )
                )
            step = int(trace.step[tick, world])
            frames.append(
                Frame(
                    index=step,
                    time_s=step * case.dt,
                    done=bool(trace.done[tick, world]),
                    actors=states,
                    ego_events=[
                        name for name, flags in (trace.events or {}).items() if flags[tick, world]
                    ],
                )
            )
        return RolloutWorld(
            source_id=source.summary.id, detail=detail, frames=frames, map=scene_map
        )

    def observations(self, run_id: str, world: int, actor_id: str) -> ObservationTrack:
        """Rebuild what a SingleFrameNet actor observed at every frame of the latest rollout."""
        import torch

        from cs5478_project.datatypes.control import ControlType
        from cs5478_project.datatypes.map import MapElementType
        from cs5478_project.datatypes.state import WorldState
        from cs5478_project.env.observations import build_observation

        context = self._observation_context
        if context is None or context["run_id"] != run_id:
            raise UnknownRollout(run_id)
        scene, trace = context["scene"], context["trace"]
        if not 0 <= world < scene.dt.shape[0]:
            raise ValueError(f"World {world} is not in rollout {run_id}")
        ids = scene.actor_ids[world].tolist()
        try:
            slot = ids.index(int(actor_id.removeprefix("actor-")))
        except ValueError as error:
            raise ValueError(f"Unknown dynamic actor: {actor_id}") from error
        if int(context["control_types"][world, slot]) != ControlType.SINGLE_FRAME:
            raise ValueError(f"{actor_id} is not controlled by SingleFrameNet")
        device = scene.dt.device
        # Work on this world alone: the batch may hold hundreds of worlds.
        one = _one_world(scene, world)
        controls = context["control_types"][world : world + 1]
        blind = context["blind"][world : world + 1]
        states = WorldState(*(value[:, world : world + 1].to(device) for value in trace.state))
        steps, masks, goals = (
            value[:, world : world + 1].to(device)
            for value in (trace.step, trace.actor_mask, trace.goals)
        )
        query_world = torch.tensor([0], device=device)
        query_slot = torch.tensor([slot], device=device)
        static_ids = scene.static.ids[world].tolist()
        names = [f"actor-{i}" for i in ids] + [f"actor-{i}" for i in static_ids]
        chunk_slot: dict[int, int] = {}
        chunks: list[ObservedChunk] = []
        frames: list[ObservedFrame] = []
        with torch.no_grad():
            for frame in range(trace.step.shape[0]):
                _, selection = build_observation(
                    WorldState(*(value[frame] for value in states)),
                    one,
                    steps[frame],
                    masks[frame],
                    controls,
                    goals[frame],
                    query_world,
                    query_slot,
                    context["config"],
                    blind=blind,
                    return_selection=True,
                )
                kept = selection.agent_index[0][selection.agent_mask[0]].tolist()
                lanes = []
                for index in selection.lane_index[0][selection.lane_mask[0]].tolist():
                    if index not in chunk_slot:
                        chunk_slot[index] = len(chunks)
                        chunks.append(
                            ObservedChunk(
                                points=scene.map.chunk_points[world, index].tolist(),
                                type=MapElementType(
                                    int(scene.map.chunk_type[world, index])
                                ).name.lower(),
                            )
                        )
                    lanes.append(chunk_slot[index])
                frames.append(ObservedFrame(agents=[names[i] for i in kept], lanes=lanes))
        return ObservationTrack(
            actor_id=actor_id,
            agent_capacity=selection.agent_index.shape[1],
            lane_capacity=selection.lane_index.shape[1],
            blind=bool(context["blind"][world, slot]),
            chunks=chunks,
            frames=frames,
        )

    def _load_policy(self, kind, checkpoint, checkpoint_config, device):
        """Load (and cache one) model of the requested learned family."""
        if kind != "single_frame":
            raise ValueError(f"Unknown learned policy kind: {kind}")
        from cs5478_project.models.checkpoint import load_single_frame, random_single_frame

        if checkpoint is None or checkpoint.startswith(RANDOM_INIT_PREFIX):
            suffix = (checkpoint or RANDOM_INIT_PREFIX)[len(RANDOM_INIT_PREFIX) :]
            try:
                seed = int(suffix or 0)
            except ValueError as error:
                raise ValueError(f"Invalid random-init seed: {suffix!r}") from error
            key = ("single_frame", "random", seed, device)
            load = partial(random_single_frame, seed=seed, device=device)
        else:
            path = Path(checkpoint).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Checkpoint does not exist: {path}")
            key = ("single_frame", str(path), path.stat().st_mtime_ns, device)
            load = partial(load_single_frame, path, device=device)
        if key not in self._policy_cache:
            loaded = load()
            self._policy_cache.clear()
            self._policy_cache[key] = loaded
        return self._policy_cache[key]

    def _resolve_device(self) -> str:
        import torch

        device = self.settings.device
        available = torch.cuda.is_available()
        if device == "cuda" and not available:
            raise RolloutUnavailableError(
                "CUDA was requested but no CUDA device is available; restart with --device cpu"
            )
        if device == "auto":
            return "cuda" if available else "cpu"
        return device


def _summaries(scenes, sources, metrics) -> list[WorldSummary]:
    """One row per world, with its ego's scores when an evaluator ran."""
    scores = {}
    if metrics is not None:
        values = {name: getattr(metrics, name).float().cpu().tolist() for name in METRIC_FIELDS}
        for index, world in enumerate(metrics.world.cpu().tolist()):
            scores[world] = WorldMetrics(**{name: values[name][index] for name in values})
    return [
        WorldSummary(
            index=world,
            source_id=source.summary.id,
            scenario_id=scene.scenario_id,
            scenario_type=source.summary.scenario_type,
            collection=source.summary.collection,
            num_frames=scene.num_frames,
            dynamic_actor_count=int(scene.actor_mask.sum()),
            static_object_count=int(scene.static.mask.sum()),
            metrics=scores.get(world),
        )
        for world, (scene, source) in enumerate(zip(scenes, sources, strict=True))
    ]


def _reactive_background(tensors, modes, source_ids, scenes, overrides):
    """Training's reactive (R) evaluation: the ego is SingleFrameNet, the 64 railed vehicles
    nearest to it follow IDM, pedestrians are rule-based, the rest replay their logs.
    Per-actor overrides still apply; ``modes`` is updated in place."""
    import torch

    from cs5478_project.training.mixing import WorldMode, mixed_controls

    codes = mixed_controls(
        tensors, torch.full_like(tensors.ego_index, WorldMode.REACTIVE), self_play_agents=0
    ).cpu()
    overridden = {(item.source_id, item.actor_id) for item in overrides}
    for world, (scene, source_id) in enumerate(zip(scenes, source_ids, strict=True)):
        for slot in range(scene.num_actors):
            if (source_id, f"actor-{int(scene.actor_ids[slot])}") not in overridden:
                modes[world][slot] = CONTROL_NAMES[int(codes[world, slot])]
    result = torch.zeros_like(codes)
    for world, selected in enumerate(modes):
        result[world, : len(selected)] = torch.tensor(
            [CONTROL_NAMES.index(mode) for mode in selected]
        )
    return result


def _one_world(value, world: int):
    """A batch-first tensor tree (SceneBatch and its parts) cut to one world."""
    if value is None:
        return None
    if hasattr(value, "_fields"):
        return type(value)(*(_one_world(item, world) for item in value))
    return value[world : world + 1]


def _file_key(path: Path) -> tuple[Path, int]:
    """A scene file and its modification time: an edited file is read again."""
    return path, path.stat().st_mtime_ns
