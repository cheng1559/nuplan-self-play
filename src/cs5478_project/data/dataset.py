"""Scene datasets and a versioned tensor-and-primitive scenario file format."""

import io
import zlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import Dataset

from cs5478_project.datatypes import MapData, RouteData, Scene, StaticObjects, WorldState
from cs5478_project.datatypes.rails import RailData
from cs5478_project.defaults import MAX_DYNAMIC_ACTORS, MAX_STATIC_OBJECTS

SCENE_FORMAT_VERSION = 9


def save_scene(scene: Scene, path: str | Path, *, compress: bool = False) -> None:
    """Save bounded 256/512 dynamic/static scenes with a canonical RDP 2048 map.

    ``compress`` deflates the file (zlib level 1, about 5x smaller for nuPlan scenes);
    load_scene reads either form.
    """
    from cs5478_project.data.map_chunks import with_chunks
    from cs5478_project.data.map_simplification import simplify_map
    from cs5478_project.data.preparation import prepare_scene

    scene = prepare_scene(
        scene, max_dynamic_actors=MAX_DYNAMIC_ACTORS, max_static_objects=MAX_STATIC_OBJECTS
    )
    if scene.map is not None:
        # Chunks are cut before simplification so they keep the source geometry.
        if scene.map.chunk_points is None:
            scene = replace(scene, map=with_chunks(scene.map))
        scene = replace(scene, map=simplify_map(scene.map))
    scene.validate()
    scene_map = scene.map
    payload = {
        "format_version": SCENE_FORMAT_VERSION,
        "scenario_id": scene.scenario_id,
        "dt": scene.dt,
        "ego_index": scene.ego_index,
        "routes": scene.routes._asdict() if scene.routes is not None else None,
        "actor_type": scene.actor_type,
        "actor_ids": scene.actor_ids,
        "static": scene.static._asdict(),
        "rails": scene.rails._asdict() if scene.rails is not None else None,
        "reference_size": scene.reference_size,
        "actor_mask": scene.actor_mask,
        "size": scene.size,
        "wheelbase": scene.wheelbase,
        "reference_to_center": scene.reference_to_center,
        "initial_state": scene.initial_state._asdict(),
        "reference_state": scene.reference_state._asdict(),
        "reference_valid": scene.reference_valid,
        "map": (
            {
                "map_name": scene_map.map_name,
                "world_from_scene": scene_map.world_from_scene,
                "bounds": scene_map.bounds,
                "segments": scene_map.segments,
                "segment_type": scene_map.segment_type,
                "polyline_index": scene_map.polyline_index,
                "polyline_ids": scene_map.polyline_ids,
                "segment_signal": scene_map.segment_signal,
                "chunk_points": scene_map.chunk_points,
                "chunk_type": scene_map.chunk_type,
                "chunk_signal": scene_map.chunk_signal,
                "drivable_distance": scene_map.drivable_distance,
                "drivable_origin": scene_map.drivable_origin,
                "lane_info": scene_map.lane_info,
            }
            if scene_map is not None
            else None
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        path.write_bytes(zlib.compress(buffer.getbuffer(), 1))
    else:
        torch.save(payload, path)


def _read_payload(path: Path, *, mmap: bool = False):
    """torch.save's zip archive starts with "PK"; anything else is a deflated one."""
    with path.open("rb") as file:
        plain = file.read(2) == b"PK"
    if plain:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    data = zlib.decompress(path.read_bytes())
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def load_scene(path: str | Path, *, validate: bool = True) -> Scene:
    """Load the safe tensor format on CPU and validate its replay contract.

    ``validate=False`` skips the contract check for trusted files (save_scene validates
    before writing); it is several times faster on large scenes.
    """
    payload = _read_payload(Path(path))
    if not isinstance(payload, dict) or payload.get("format_version") != SCENE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported scene format in {path}; re-export with format {SCENE_FORMAT_VERSION}"
        )
    try:
        fields = {name: value for name, value in payload.items() if name != "format_version"}
        rails = fields.pop("rails")
        fields["rails"] = RailData(**rails) if rails is not None else None
        fields["static"] = StaticObjects(**fields["static"])
        routes = fields.pop("routes")
        fields["routes"] = RouteData(**routes) if routes is not None else None
        scene_map = fields.pop("map")
        fields["map"] = MapData(**scene_map) if scene_map is not None else None
        fields["initial_state"] = WorldState(**fields["initial_state"])
        fields["reference_state"] = WorldState(**fields["reference_state"])
        scene = Scene(**fields)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed scene payload in {path}: {error}") from error
    if validate:
        scene.validate()
    return scene


class SceneSummary(NamedTuple):
    scenario_id: str
    dt: float
    num_frames: int
    num_actors: int  # slots, including padding
    dynamic_actors: int
    static_objects: int
    has_map: bool


def read_scene_summary(path: str | Path) -> SceneSummary:
    """Listing facts of a scene file without decoding it: an uncompressed file is
    memory-mapped, so only the few small tensors read here are paged in (a compressed one
    is inflated in full). Call load_scene to use the scene."""
    payload = _read_payload(Path(path), mmap=True)
    if not isinstance(payload, dict) or payload.get("format_version") != SCENE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported scene format in {path}; re-export with format {SCENE_FORMAT_VERSION}"
        )
    mask = payload["actor_mask"]
    return SceneSummary(
        scenario_id=payload["scenario_id"],
        dt=float(payload["dt"]),
        num_frames=int(payload["reference_state"]["heading"].shape[0]),
        num_actors=int(mask.shape[0]),
        dynamic_actors=int(mask.sum()),
        static_objects=int(payload["static"]["mask"].sum()),
        has_map=payload["map"] is not None,
    )


class SceneDataset(Dataset[Scene]):
    """Read prebuilt CPU scenes or normalized scene files by index.

    Paths are loaded lazily. This reader deliberately has no nuPlan dependency;
    the eventual source adapter must export the same normalized scene contract.
    ``validate=False`` trusts scene files written by save_scene. ``cache=True`` loads
    every scene into memory once, with ``cache_threads`` threads, so each later batch
    skips file reads (checked once, at load time, when ``validate``).
    """

    def __init__(
        self,
        scenes: Sequence[Scene | str | Path],
        *,
        validate: bool = True,
        cache: bool = False,
        cache_threads: int = 32,
    ) -> None:
        self._scenes = tuple(scenes)
        self._validate = validate
        self._cached: tuple[Scene, ...] | None = None
        if not self._scenes:
            raise ValueError("SceneDataset requires at least one scene")
        if any(not isinstance(scene, (Scene, str, Path)) for scene in self._scenes):
            raise TypeError("Dataset items must be Scene objects or scene file paths")
        if cache:
            from concurrent.futures import ThreadPoolExecutor

            def read(item):
                if isinstance(item, Scene):
                    if validate:
                        item.validate()
                    return item
                return load_scene(item, validate=validate)

            with ThreadPoolExecutor(cache_threads) as pool:
                self._cached = tuple(pool.map(read, self._scenes))

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, index: int) -> Scene:
        if self._cached is not None:
            return self._cached[index]
        item = self._scenes[index]
        if isinstance(item, Scene):
            item.validate()
            return item
        return load_scene(item, validate=self._validate)
