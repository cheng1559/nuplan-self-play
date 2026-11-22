"""Register local artifacts and assemble bounded CPU replay responses."""

import base64
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cs5478_project.server.schemas import (
    Actor,
    ActorState,
    CaseDetail,
    CaseSummary,
    Frame,
    FrameChunk,
    SourceKind,
)
from cs5478_project.server.services.static_objects import static_actor_metadata
from cs5478_project.visualization.map import SceneMap, display_map_from_data, load_scene_map
from cs5478_project.visualization.replay import (
    ACTOR_TYPES,
    RecordedCase,
    load_replay_json,
    load_scene_record,
)


@dataclass(frozen=True)
class _Entry:
    path: Path
    world: int
    summary: CaseSummary
    map_path: Path | None


class CaseStore:
    """Index configured roots once and cache at most four decoded source files."""

    def __init__(self, artifact_roots: list[Path] | None = None, *, include_demo=False):
        self._entries: dict[str, _Entry] = {}
        if include_demo:
            demo = Path(__file__).parents[2] / "visualization" / "fixtures" / "demo.replay.json"
            self._register(demo, "demo", "demo")
        roots = artifact_roots
        if roots is None:
            roots = [
                path for path in (Path("outputs/replay"), Path("data/scenes")) if path.exists()
            ]
        seen = set()
        for root_index, root in enumerate(roots):
            root = Path(root).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Artifact root does not exist: {root}")
            files = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in files:
                if not path.is_file() or path.suffix.lower() not in (".json", ".pt"):
                    continue
                if path.name.lower().endswith(".map.json"):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                relative = path.name if root.is_file() else path.relative_to(root).as_posix()
                source = "scene" if path.suffix.lower() == ".pt" else "replay"
                self._register(path, source, f"root-{root_index}:{relative}")

    @lru_cache(maxsize=4)
    def _load(self, path: Path) -> list[RecordedCase]:
        if path.suffix.lower() == ".pt":
            return [load_scene_record(path)]
        return load_replay_json(path)

    @lru_cache(maxsize=4)
    def _load_map(self, path: Path) -> SceneMap | None:
        if path.suffix.lower() == ".pt":
            from cs5478_project.data import load_scene

            scene = load_scene(path)
            return (
                display_map_from_data(scene.map, scene.scenario_id)
                if scene.map is not None
                else None
            )
        return load_scene_map(path)

    def _register(self, path: Path, source: SourceKind, key: str) -> None:
        if path.suffix.lower() == ".pt":
            self._register_scene(path, key)
            return
        cases = self._load(path)
        if not cases:
            return
        map_path = path if path.suffix.lower() == ".pt" else path.with_suffix(".map.json")
        scene_map = self._load_map(map_path) if map_path.exists() else None
        if scene_map is not None and scene_map.scenario_id not in {
            case.scenario_id for case in cases
        }:
            raise ValueError(
                f"Scene map {map_path} has scenario_id {scene_map.scenario_id!r}, "
                f"which does not match any scenario in {path}"
            )
        for world, case in enumerate(cases):
            identifier = base64.urlsafe_b64encode(f"{key}:{world}".encode()).decode().rstrip("=")
            summary = CaseSummary(
                id=identifier,
                scenario_id=case.scenario_id,
                source_kind=source,
                num_frames=case.num_frames,
                dt=case.dt,
                duration_s=(case.num_frames - 1) * case.dt,
                actor_count=case.total_actors,
                dynamic_actor_count=int(case.actor_mask.sum()),
                static_object_count=case.total_actors - int(case.actor_mask.sum()),
            )
            self._entries[identifier] = _Entry(
                path=path,
                world=world,
                summary=summary,
                map_path=(
                    map_path
                    if scene_map is not None and scene_map.scenario_id == case.scenario_id
                    else None
                ),
            )

    def _register_scene(self, path: Path, key: str) -> None:
        """List a scene file from its summary; it is decoded when first opened."""
        from cs5478_project.data.dataset import read_scene_summary

        scene = read_scene_summary(path)
        identifier = base64.urlsafe_b64encode(f"{key}:0".encode()).decode().rstrip("=")
        self._entries[identifier] = _Entry(
            path=path,
            world=0,
            summary=CaseSummary(
                id=identifier,
                scenario_id=scene.scenario_id,
                source_kind="scene",
                num_frames=scene.num_frames,
                dt=scene.dt,
                duration_s=(scene.num_frames - 1) * scene.dt,
                actor_count=scene.dynamic_actors + scene.static_objects,
                dynamic_actor_count=scene.dynamic_actors,
                static_object_count=scene.static_objects,
            ),
            map_path=path if scene.has_map else None,
        )

    def list_cases(self) -> list[CaseSummary]:
        return [entry.summary for entry in self._entries.values()]

    def _get(self, case_id: str) -> tuple[_Entry, RecordedCase]:
        entry = self._entries[case_id]
        return entry, self._load(entry.path)[entry.world]

    def detail(self, case_id: str) -> CaseDetail:
        entry, case = self._get(case_id)
        actors = [
            Actor(
                id=case.actor_id(slot),
                slot=int(slot),
                type=ACTOR_TYPES[case.actor_type[slot]],
                length=float(case.size[slot, 0]),
                width=float(case.size[slot, 1]),
                reference_to_center=float(case.reference_to_center[slot]),
            )
            for slot in case.slots
        ]
        actors.extend(static_actor_metadata(case.static))
        return CaseDetail(
            **entry.summary.model_dump(),
            actors=actors,
            bounds=case.bounds(),
            has_map=entry.map_path is not None,
        )

    def scene_map(self, case_id: str) -> SceneMap:
        entry = self._entries[case_id]
        if entry.map_path is None:
            raise FileNotFoundError("No map is available for this case")
        scene_map = self._load_map(entry.map_path)
        if scene_map is None:
            raise FileNotFoundError("No map is available for this case")
        return scene_map

    def frames(self, case_id: str, start: int, count: int) -> FrameChunk:
        _, case = self._get(case_id)
        if start >= case.num_frames:
            raise IndexError(f"start must be below the scenario frame count ({case.num_frames})")
        centers = case.centers(start, start + count)
        frames = []
        for index in range(start, min(start + count, case.num_frames)):
            actors = []
            for slot in case.active_slots(index):
                velocity = case.state["velocity"][index, slot]
                actors.append(
                    ActorState(
                        id=case.actor_id(slot),
                        position=case.state["position"][index, slot].tolist(),
                        center=centers[index - start, slot].tolist(),
                        heading=float(case.state["heading"][index, slot]),
                        velocity=velocity.tolist(),
                        speed=math.hypot(*velocity),
                        **{
                            name: float(case.state[name][index, slot])
                            for name in (
                                "acceleration",
                                "acceleration_command",
                                "steering",
                                "yaw_rate",
                            )
                        },
                    )
                )
            frames.append(
                Frame(
                    index=index,
                    time_s=index * case.dt,
                    done=index == case.num_frames - 1,
                    actors=actors,
                )
            )
        return FrameChunk(case_id=case_id, start=start, total_frames=case.num_frames, frames=frames)
