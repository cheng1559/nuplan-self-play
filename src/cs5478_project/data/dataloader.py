"""Host-side fixed-capacity collation and batch transfer."""

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cs5478_project.data.dataset import SceneDataset
from cs5478_project.datatypes import MapBatch, Scene, SceneBatch, WorldState
from cs5478_project.datatypes.map import (
    CHUNK_POINTS,
    MAX_DRIVABLE_CELLS,
    MAX_MAP_CHUNKS,
    SEGMENT_SIGNALS,
)
from cs5478_project.datatypes.rails import RailData, empty_rails
from cs5478_project.datatypes.route import empty_routes
from cs5478_project.datatypes.static import empty_static
from cs5478_project.defaults import EPISODE_FRAMES, MAX_DYNAMIC_ACTORS, MAX_STATIC_OBJECTS


@dataclass(frozen=True)
class LoadedBatch:
    """Keep CPU string metadata separate from simulation tensor inputs."""

    tensors: SceneBatch
    scenario_ids: tuple[str, ...]

    def pin_memory(self) -> "LoadedBatch":
        """Let PyTorch's DataLoader pin every tensor in this custom container."""
        return LoadedBatch(_map_batch(self.tensors, Tensor.pin_memory), self.scenario_ids)


class SceneDataLoader:
    """Produce identical [B, N, F] capacities for every yielded batch.

    Capacity overflow is rejected instead of dropping actors or frames. With
    drop_last=False, a non-divisible dataset is rejected before iteration.
    Completed-world replacement is handled by the environment's reset mask.
    """

    def __init__(
        self,
        dataset: Dataset[Scene],
        *,
        batch_size: int,
        max_actors: int = MAX_DYNAMIC_ACTORS,
        max_frames: int = EPISODE_FRAMES,
        max_static_objects: int = MAX_STATIC_OBJECTS,
        max_map_segments: int = 2048,
        max_map_chunks: int = MAX_MAP_CHUNKS,
        device: str | torch.device = "cpu",
        shuffle: bool = False,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_on_device: bool = False,
        persistent_workers: bool = False,
        prefetch: bool = False,
        idm_fields: bool = True,
        replace_oversize: bool = False,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size),
            ("max_actors", max_actors),
            ("max_frames", max_frames),
            ("max_static_objects", max_static_objects),
            ("max_map_segments", max_map_segments),
            ("max_map_chunks", max_map_chunks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if not drop_last and len(dataset) % batch_size:
            raise ValueError("drop_last=False requires a dataset divisible by batch_size")
        if len(dataset) < batch_size:
            raise ValueError("Dataset must contain at least one complete batch")
        self.device = torch.device(device)
        capacities = dict(
            max_actors=max_actors,
            max_frames=max_frames,
            max_static_objects=max_static_objects,
            max_map_segments=max_map_segments,
            max_map_chunks=max_map_chunks,
        )
        # SceneDataset already validates (or deliberately trusts) its scenes.
        validate = not isinstance(dataset, SceneDataset)
        self._collate = partial(
            _collate_scenes, validate=validate, idm_fields=idm_fields, **capacities
        )
        self._write = partial(_write, **capacities)
        # On-device collation: workers (or the caller) pack sub-batches of scenes into a
        # few flat tensors per field, and the padded batch is filled on the device, so
        # host memory holds no padded batches and worker results cross processes cheaply.
        self._collate_on_device = collate_on_device
        self._group = batch_size // _sub_batch_size(batch_size) if collate_on_device else 1
        # On CUDA, prepare the next batch on a side stream while the caller uses this one.
        self._prefetch = prefetch and self.device.type == "cuda"
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size // self._group,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory and not collate_on_device,
            collate_fn=(
                partial(
                    _stage_scenes,
                    validate=validate,
                    idm_fields=idm_fields,
                    replace_oversize=replace_oversize,
                    **capacities,
                )
                if collate_on_device
                else self._collate
            ),
            persistent_workers=persistent_workers and num_workers > 0,
        )

    def __len__(self) -> int:
        return len(self._loader) // self._group

    def __iter__(self) -> Iterator[LoadedBatch]:
        return self.epochs(1)

    def epochs(self, count: int) -> Iterator[LoadedBatch]:
        """Batches of ``count`` passes over the dataset; prefetching continues across
        pass boundaries."""
        batches = (batch for _ in range(count) for batch in self._batches())
        if self._prefetch:
            return _prefetched(batches, self.device)
        return batches

    def _batches(self) -> Iterator[LoadedBatch]:
        parts = []
        for loaded in self._loader:
            if self._collate_on_device:
                parts.append(loaded)
                if len(parts) == self._group:  # a partial last group is dropped
                    yield self._write(parts, device=self.device)
                    parts = []
                continue
            tensors = _map_batch(
                loaded.tensors, lambda value: value.to(self.device, non_blocking=True)
            )
            yield LoadedBatch(tensors, loaded.scenario_ids)


def _tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _tensors(item)


def _prefetched(batches: Iterator[LoadedBatch], device: torch.device) -> Iterator[LoadedBatch]:
    """Build batches in a background thread on a side CUDA stream, one ahead of the
    consumer: at most two batches (the one in use and the next) are held at a time."""
    stream = torch.cuda.Stream(device)
    ready: queue.Queue = queue.Queue()
    slots = threading.Semaphore(2)
    stop = threading.Event()
    finished = object()

    def produce():
        try:
            iterator = iter(batches)
            while not stop.is_set():
                if not slots.acquire(timeout=0.1):
                    continue
                with torch.cuda.stream(stream):
                    batch = next(iterator, finished)
                if batch is finished:
                    ready.put(finished)
                    return
                stream.synchronize()
                ready.put(batch)
        except BaseException as error:  # surfaced in the consumer
            ready.put(error)

    thread = threading.Thread(target=produce, name="batch-prefetch", daemon=True)
    thread.start()
    first = True
    try:
        while True:
            if not first:
                slots.release()  # the previous batch is done; build one more
            first = False
            item = ready.get()
            if item is finished:
                return
            if isinstance(item, BaseException):
                raise item
            consumer = torch.cuda.current_stream(device)
            for tensor in _tensors(item.tensors):
                tensor.record_stream(consumer)
            yield item
    finally:
        stop.set()
        thread.join()


def _map_batch(batch: SceneBatch, operation) -> SceneBatch:
    return SceneBatch(
        *(
            type(value)(*(operation(field) for field in value))
            if isinstance(value, tuple)
            else operation(value)
            for value in batch
        )
    )


def _sub_batch_size(batch_size: int, limit: int = 256) -> int:
    """The largest divisor of batch_size up to limit: scenes staged per worker task."""
    return max(size for size in range(1, min(batch_size, limit) + 1) if batch_size % size == 0)


class StagedScenes(NamedTuple):
    """Scenes packed on the host, one flat tensor per field, ready to copy (see _write).

    ``fields`` maps a field path (e.g. ``("map", "segments")``) to the concatenated
    pieces, their shapes [P, D], their worlds [P], and whether the field is a grid.
    """

    scenario_ids: tuple[str, ...]
    dt: Tensor
    num_frames: Tensor
    ego_index: Tensor
    fields: dict


def _check_capacity(scene, max_actors, max_frames, max_map_segments, max_static_objects,
                    max_map_chunks):  # fmt: skip
    if scene.num_actors > max_actors:
        raise ValueError(
            f"Scene {scene.scenario_id!r} has {scene.num_actors} actor slots; "
            f"max_actors={max_actors}"
        )
    if scene.num_frames > max_frames:
        raise ValueError(
            f"Scene {scene.scenario_id!r} has {scene.num_frames} frames; max_frames={max_frames}"
        )
    if len(scene.static.mask) > max_static_objects:
        raise ValueError(
            f"Scene {scene.scenario_id!r} exceeds max_static_objects={max_static_objects}"
        )
    if scene.map is None:
        return
    if scene.map.segments.shape[0] > max_map_segments:
        raise ValueError(f"Scene {scene.scenario_id!r} exceeds max_map_segments={max_map_segments}")
    if scene.map.chunk_points is not None and scene.map.chunk_points.shape[0] > max_map_chunks:
        raise ValueError(f"Scene {scene.scenario_id!r} exceeds max_map_chunks={max_map_chunks}")
    grid = scene.map.drivable_distance
    if grid is not None and max(grid.shape) > MAX_DRIVABLE_CELLS:
        raise ValueError(
            f"Scene {scene.scenario_id!r} drivable grid exceeds {MAX_DRIVABLE_CELLS} cells"
        )


def _stage_scenes(
    scenes: list[Scene],
    *,
    max_actors: int,
    max_frames: int,
    max_map_segments: int,
    max_static_objects: int,
    max_map_chunks: int = MAX_MAP_CHUNKS,
    validate: bool = True,
    idm_fields: bool = True,
    replace_oversize: bool = False,
) -> StagedScenes:
    """Check scenes against the capacities and pack each field's pieces into one tensor.

    With ``replace_oversize``, a scene over a capacity is replaced by another scene of
    the same sub-batch (training tolerates the rare oversize export) instead of raising.

    Without ``idm_fields``, rail edges, stop lines, and starting rails, and segment
    signals (used only by IDM and rule agents) are left out: the batch keeps their empty
    defaults, which saves about a third of the transfer.
    """
    from cs5478_project.data.preparation import prepare_scene

    scenes = [prepare_scene(scene) if scene.static is None else scene for scene in scenes]
    capacities = (max_actors, max_frames, max_map_segments, max_static_objects, max_map_chunks)
    fits = []
    for scene in scenes:
        if validate:
            scene.validate()
        try:
            _check_capacity(scene, *capacities)
            fits.append(True)
        except ValueError:
            if not replace_oversize:
                raise
            fits.append(False)
    if not all(fits):
        if not any(fits):
            raise ValueError("Every scene of a sub-batch exceeds a capacity")
        usable = [scene for scene, ok in zip(scenes, fits, strict=True) if ok]
        scenes = [
            scene if ok else usable[index % len(usable)]
            for index, (scene, ok) in enumerate(zip(scenes, fits, strict=True))
        ]
    pieces: dict[tuple, list] = {}

    def add(key, world, source, transfer=None, grid=False):
        # ``transfer`` is a smaller dtype that holds every value, used for the copy.
        entry = pieces.setdefault(key, [[], [], transfer, grid])
        entry[0].append(world)
        entry[1].append(source)

    def ones(count):
        return torch.ones(count, dtype=torch.bool, device="cpu")

    for world, scene in enumerate(scenes):
        sizes = (
            scene.reference_size
            if scene.reference_size is not None
            else scene.size[None].expand(scene.num_frames, -1, -1)
        )
        add(("reference_size",), world, sizes)
        if scene.rails is not None:
            for name in RailData._fields:
                if idm_fields or not name.startswith(("edge_", "stop_", "rail_")):
                    add(("rails", name), world, getattr(scene.rails, name))
        for name in ("actor_type", "actor_mask", "size", "wheelbase", "reference_to_center"):
            add((name,), world, getattr(scene, name))
        for name, source in zip(WorldState._fields, scene.initial_state, strict=True):
            add(("initial_state", name), world, source)
        for name, source in zip(WorldState._fields, scene.reference_state, strict=True):
            add(("reference_state", name), world, source)
        add(("reference_valid",), world, scene.reference_valid)
        add(("actor_ids",), world, scene.actor_ids)
        for name, source in zip(type(scene.static)._fields, scene.static, strict=True):
            add(("static", name), world, source)
        if scene.routes is not None:
            for name, source in zip(type(scene.routes)._fields, scene.routes, strict=True):
                add(("routes", name), world, source)
        scene_map = scene.map
        if scene_map is None:
            continue
        add(("map", "segments"), world, scene_map.segments)
        add(("map", "segment_type"), world, scene_map.segment_type, torch.int16)
        add(("map", "segment_mask"), world, ones(scene_map.segments.shape[0]))
        if idm_fields and scene_map.segment_signal is not None:
            add(("map", "segment_signal"), world, scene_map.segment_signal, torch.int16)
        if scene_map.chunk_points is not None:
            add(("map", "chunk_points"), world, scene_map.chunk_points)
            add(("map", "chunk_type"), world, scene_map.chunk_type, torch.int16)
            add(("map", "chunk_mask"), world, ones(scene_map.chunk_points.shape[0]))
            if scene_map.chunk_signal is not None:
                add(("map", "chunk_signal"), world, scene_map.chunk_signal, torch.int16)
        grid = scene_map.drivable_distance
        if grid is not None:
            add(("map", "drivable_distance"), world, grid, grid=True)
            if scene_map.lane_info is not None:
                add(("map", "lane_info"), world, scene_map.lane_info, grid=True)
            origin = torch.tensor(scene_map.drivable_origin, dtype=torch.float32, device="cpu")
            add(("map", "drivable_origin"), world, origin)
            add(("map", "drivable_size"), world, torch.tensor(grid.shape, device="cpu"))
    fields = {}
    for key, (worlds, sources, transfer, grid) in pieces.items():
        dtype = transfer or sources[0].dtype
        flat = torch.cat([x.reshape(-1).to("cpu", dtype) for x in sources])
        shapes = torch.tensor([list(x.shape) for x in sources], dtype=torch.int64, device="cpu")
        shapes = shapes.reshape(len(sources), sources[0].ndim)
        fields[key] = (flat, shapes, torch.tensor(worlds, dtype=torch.int64, device="cpu"), grid)
    return StagedScenes(
        tuple(scene.scenario_id for scene in scenes),
        torch.tensor([scene.dt for scene in scenes], dtype=torch.float32, device="cpu"),
        torch.tensor([scene.num_frames for scene in scenes], dtype=torch.int64, device="cpu"),
        torch.tensor([scene.ego_index for scene in scenes], dtype=torch.int64, device="cpu"),
        fields,
    )


def _collate_scenes(
    scenes: list[Scene],
    *,
    max_actors: int,
    max_frames: int,
    max_map_segments: int,
    max_static_objects: int,
    max_map_chunks: int = MAX_MAP_CHUNKS,
    validate: bool = True,
    device: torch.device | None = None,
    idm_fields: bool = True,
) -> LoadedBatch:
    """Pad scenes into one batch, allocated on ``device`` (CPU by default)."""
    capacities = dict(
        max_actors=max_actors,
        max_frames=max_frames,
        max_map_segments=max_map_segments,
        max_static_objects=max_static_objects,
        max_map_chunks=max_map_chunks,
    )
    staged = _stage_scenes(scenes, validate=validate, idm_fields=idm_fields, **capacities)
    return _write([staged], device=device, **capacities)


def _write(
    parts: list[StagedScenes],
    *,
    max_actors: int,
    max_frames: int,
    max_map_segments: int,
    max_static_objects: int,
    max_map_chunks: int = MAX_MAP_CHUNKS,
    device: torch.device | None = None,
) -> LoadedBatch:
    """Allocate a padded batch on ``device`` and fill it from staged parts, in order.

    Each part's field goes to the device in one copy and is scattered there with one
    indexed write; grids, which are large, take one strided device copy per scene.
    """
    with torch.device(device or "cpu"):
        return _fill(
            parts, max_actors, max_frames, max_map_segments, max_static_objects, max_map_chunks
        )


def _fill(parts, max_actors, max_frames, max_map_segments, max_static_objects, max_map_chunks):
    device = torch.empty(0).device
    b, n, f = sum(len(part.scenario_ids) for part in parts), max_actors, max_frames
    initial_state = _empty_state((b, n))
    reference_state = _empty_state((b, f, n))

    def joined(name):
        return torch.cat([getattr(part, name) for part in parts]).to(device)

    batch = SceneBatch(
        dt=joined("dt"),
        num_frames=joined("num_frames"),
        actor_type=torch.zeros((b, n), dtype=torch.int64),
        actor_mask=torch.zeros((b, n), dtype=torch.bool),
        size=torch.ones((b, n, 2), dtype=torch.float32),
        wheelbase=torch.ones((b, n), dtype=torch.float32),
        reference_to_center=torch.zeros((b, n), dtype=torch.float32),
        initial_state=initial_state,
        reference_state=reference_state,
        reference_valid=torch.zeros((b, f, n), dtype=torch.bool),
        reference_size=torch.ones((b, f, n, 2), dtype=torch.float32),
        ego_index=joined("ego_index"),
        routes=empty_routes((b, n)),
        rails=empty_rails((b,), f, n),
        actor_ids=torch.full((b, n), -1, dtype=torch.int64),
        static=empty_static((b, max_static_objects), f),
        map=MapBatch(
            segments=torch.zeros((b, max_map_segments, 2, 2), dtype=torch.float32),
            segment_type=torch.zeros((b, max_map_segments), dtype=torch.int64),
            segment_mask=torch.zeros((b, max_map_segments), dtype=torch.bool),
            segment_signal=torch.full(
                (b, max_map_segments, SEGMENT_SIGNALS), -1, dtype=torch.int64
            ),
            chunk_points=torch.zeros((b, max_map_chunks, CHUNK_POINTS, 2), dtype=torch.float32),
            chunk_type=torch.zeros((b, max_map_chunks), dtype=torch.int64),
            chunk_mask=torch.zeros((b, max_map_chunks), dtype=torch.bool),
            chunk_signal=torch.full((b, max_map_chunks, SEGMENT_SIGNALS), -1, dtype=torch.int64),
            drivable_distance=torch.zeros(
                (b, MAX_DRIVABLE_CELLS, MAX_DRIVABLE_CELLS), dtype=torch.float16
            ),
            drivable_origin=torch.zeros((b, 2), dtype=torch.float32),
            drivable_size=torch.zeros((b, 2), dtype=torch.int64),
            lane_info=torch.zeros((b, MAX_DRIVABLE_CELLS, MAX_DRIVABLE_CELLS), dtype=torch.int16),
        ),
    )
    offset = 0
    for part in parts:
        for key, (flat, shapes, worlds, grid) in part.fields.items():
            destination = batch
            for name in key:
                destination = getattr(destination, name)
            if shapes.shape[1] != destination.ndim - 1:
                raise ValueError(
                    f"Cannot place {'.'.join(key)} pieces of rank {shapes.shape[1]} "
                    f"into {tuple(destination.shape)}"
                )
            if flat.numel() == 0:
                continue
            flat = flat.to(device).to(destination.dtype)
            if grid:
                start = 0
                for world, shape in zip(worlds.tolist(), shapes.tolist(), strict=True):
                    count = shape[0] * shape[1]
                    region = destination[offset + world, : shape[0], : shape[1]]
                    region.copy_(flat[start : start + count].view(shape))
                    start += count
            else:
                _scatter(destination, worlds + offset, shapes, flat)
        offset += len(part.scenario_ids)
    # Frames past each scene's end repeat its last frame.
    world = torch.arange(b)[:, None]
    last = torch.minimum(torch.arange(f)[None], batch.num_frames[:, None] - 1)
    for destination in (*reference_state, batch.reference_size):
        destination.copy_(destination[world, last])
    ids = tuple(scenario for part in parts for scenario in part.scenario_ids)
    return LoadedBatch(batch, ids)


def _scatter(destination: Tensor, worlds: Tensor, shapes: Tensor, flat: Tensor):
    """Write concatenated pieces (``flat``, on the device) of the given shapes [P, D] to
    the leading corner of each destination[world] with one indexed write."""
    device = destination.device
    counts = shapes.prod(1)
    total = int(counts.sum())
    shapes, counts = shapes.to(device), counts.to(device)
    piece = torch.repeat_interleave(
        torch.arange(len(shapes), device=device), counts, output_size=total
    )
    local = torch.arange(total, device=device) - (counts.cumsum(0) - counts)[piece]
    stride = destination.stride()
    index = worlds.to(device)[piece] * stride[0]
    for dim in range(destination.ndim - 2, -1, -1):
        size = shapes[piece, dim]
        index += (local % size) * stride[dim + 1]
        local = local // size
    destination.view(-1)[index] = flat


def _empty_state(leading_shape: tuple[int, ...]) -> WorldState:
    return WorldState(
        *(
            torch.zeros(
                leading_shape + ((2,) if name in ("position", "velocity") else ()),
                dtype=torch.float32,
            )
            for name in WorldState._fields
        )
    )
