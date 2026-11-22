"""Scene-owned map persistence, batching, and environment reset behavior."""

from dataclasses import replace

import pytest
import torch

from cs5478_project.data import (
    SceneDataLoader,
    SceneDataset,
    load_scene,
    make_synthetic_scene,
    save_scene,
)
from cs5478_project.data.map_chunks import polyline_chunks
from cs5478_project.datatypes import MapData, MapElementType, validate_scene_batch
from cs5478_project.datatypes.map import CHUNK_POINTS
from cs5478_project.env import ReplayEnv


def load_batch(scenes, *, segments=5):
    return next(
        iter(
            SceneDataLoader(
                SceneDataset(scenes),
                batch_size=len(scenes),
                max_actors=4,
                max_frames=5,
                max_map_segments=segments,
            )
        )
    ).tensors


def test_standalone_scene_roundtrip_includes_geometry_and_metadata(tmp_path, map_data):
    original = replace(make_synthetic_scene(num_frames=3), map=map_data)
    path = tmp_path / "standalone.pt"
    save_scene(original, path)
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 9
    assert set(tmp_path.iterdir()) == {path}
    restored = load_scene(path)
    assert restored.map.map_name == map_data.map_name
    assert restored.map.world_from_scene == map_data.world_from_scene
    assert restored.map.bounds == map_data.bounds
    assert restored.map.polyline_ids == map_data.polyline_ids
    for name in ("segments", "segment_type", "polyline_index"):
        assert torch.equal(getattr(restored.map, name), getattr(map_data, name))
    # Saving cuts chunks from the unsimplified polylines.
    assert restored.map.chunk_points.shape == (2, CHUNK_POINTS, 2)
    assert restored.map.chunk_type.tolist() == [
        MapElementType.LANE_CENTERLINE,
        MapElementType.ROAD_BOUNDARY,
    ]


def test_compressed_scene_reads_back_like_a_plain_one(tmp_path, map_data):
    from cs5478_project.data.dataset import read_scene_summary

    original = replace(make_synthetic_scene(num_frames=3), map=map_data)
    plain, packed = tmp_path / "plain.pt", tmp_path / "packed.pt"
    save_scene(original, plain)
    save_scene(original, packed, compress=True)
    assert packed.read_bytes()[:2] != b"PK"
    a, b = load_scene(plain), load_scene(packed)
    assert torch.equal(a.map.chunk_points, b.map.chunk_points)
    assert torch.equal(a.reference_state.position, b.reference_state.position)
    assert read_scene_summary(plain) == read_scene_summary(packed)


def test_polyline_chunks_cover_each_polyline_with_bounded_even_pieces():
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 25.0)]  # 55 m with a corner
    data = MapData(
        map_name="chunks",
        world_from_scene=(0.0, 0.0, 0.0),
        bounds=(-1.0, -1.0, 31.0, 26.0),
        segments=torch.tensor([[points[0], points[1]], [points[1], points[2]]]),
        segment_type=torch.tensor([MapElementType.LANE_BOUNDARY] * 2),
        polyline_index=torch.tensor([0, 0]),
        polyline_ids=("boundary",),
    )
    chunk_points, chunk_type, pieces = polyline_chunks(data)
    assert chunk_points.shape == (3, CHUNK_POINTS, 2)  # ceil(55 / 20) equal pieces
    assert chunk_type.tolist() == [MapElementType.LANE_BOUNDARY] * 3
    assert torch.allclose(chunk_points[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(chunk_points[-1, -1], torch.tensor([30.0, 25.0]))
    # Pieces join end to end and keep the source corner exactly.
    assert torch.equal(chunk_points[1:, 0], chunk_points[:-1, -1])
    assert any(((piece == (30.0, 0.0)).all(-1)).any() for piece in pieces)
    steps = (chunk_points[:, 1:] - chunk_points[:, :-1]).norm(dim=-1)
    assert float(steps.max()) <= 55 / 3 / (CHUNK_POINTS - 1) + 1e-5


def test_map_batch_keeps_geometry_and_masks_unmapped_worlds(map_data):
    mapped = replace(make_synthetic_scene(num_frames=3), map=map_data)
    batch = load_batch([mapped, make_synthetic_scene(num_frames=2)])
    validate_scene_batch(batch)
    assert batch.map.segments.shape == (2, 5, 2, 2)
    assert batch.map.segment_mask.sum(dim=1).tolist() == [3, 0]
    assert torch.equal(batch.map.segments[0, :3], map_data.segments)
    assert torch.equal(batch.map.segment_type[0, :3], map_data.segment_type)
    assert torch.count_nonzero(batch.map.segments[:, 3:]) == 0
    assert torch.isfinite(batch.map.segments).all()


def test_map_capacity_overflow_is_rejected(map_data):
    scene = replace(make_synthetic_scene(num_frames=2), map=map_data)
    with pytest.raises(ValueError, match="max_map_segments"):
        load_batch([scene], segments=2)


def test_partial_reset_replaces_only_selected_world_maps(map_data):
    scene = replace(make_synthetic_scene(num_frames=3), map=map_data)
    previous = load_batch([scene, scene])
    shifted = replace(scene, map=replace(map_data, segments=map_data.segments + 10))
    incoming = load_batch([shifted, make_synthetic_scene(num_frames=2)])
    env = ReplayEnv()
    env.reset(previous)
    env.step()
    env.reset(incoming, mask=torch.tensor([False, True]))
    assert torch.equal(env.scene.map.segments[0], previous.map.segments[0])
    assert torch.equal(env.scene.map.segment_mask[0], previous.map.segment_mask[0])
    assert not env.scene.map.segment_mask[1].any()
    assert env.episode.step.tolist() == [1, 0]
    with pytest.raises(ValueError, match="preserve"):
        env.reset(load_batch([scene, scene], segments=6))


@pytest.mark.parametrize(
    "change, message",
    [
        ("nonfinite", "finite"),
        ("disconnected", "connected"),
        ("unknown_type", "unknown"),
        ("mixed_type", "consistent"),
        ("unordered", "ordered groups"),
        ("duplicate_ids", "unique"),
    ],
)
def test_invalid_embedded_map_is_rejected(map_data, change, message):
    if change == "nonfinite":
        map_data.segments[0, 0, 0] = float("nan")
    elif change == "disconnected":
        map_data.segments[1, 0, 0] += 1
    elif change == "unknown_type":
        map_data.segment_type[0] = 99
    elif change == "mixed_type":
        map_data.segment_type[1] = 3
    elif change == "unordered":
        map_data.polyline_index[0] = 1
    else:
        map_data = replace(map_data, polyline_ids=("duplicate", "duplicate"))
    with pytest.raises(ValueError, match=message):
        replace(make_synthetic_scene(num_frames=2), map=map_data).validate()


def test_map_batch_validation_checks_padding_and_types(map_data):
    batch = load_batch([replace(make_synthetic_scene(num_frames=2), map=map_data)])
    with pytest.raises(ValueError, match="dtype"):
        validate_scene_batch(
            batch._replace(map=batch.map._replace(segment_mask=batch.map.segment_mask.int()))
        )
    batch.map.segments[0, -1, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_scene_batch(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_on_device_collation_matches_host_collation(map_data):
    scenes = [replace(make_synthetic_scene(num_frames=3), map=map_data), make_synthetic_scene()]

    def tensors(**kwargs):
        loader = SceneDataLoader(SceneDataset(scenes), batch_size=2, max_map_segments=4, **kwargs)
        batch = next(iter(loader)).tensors
        flat = []

        def walk(value):
            if isinstance(value, torch.Tensor):
                flat.append(value.cpu())
            else:
                for item in value:
                    walk(item)

        walk(batch)
        return flat

    host = tensors()
    for device in (
        tensors(device="cuda", collate_on_device=True),
        tensors(device="cuda", collate_on_device=True, prefetch=True),
    ):
        assert len(host) == len(device)
        assert all(torch.equal(a, b) for a, b in zip(host, device, strict=True))


def test_cached_dataset_returns_the_loaded_scenes(tmp_path, map_data):
    paths = []
    for index in range(3):
        path = tmp_path / f"scene{index}.pt"
        save_scene(replace(make_synthetic_scene(num_frames=3), map=map_data), path)
        paths.append(path)
    cached = SceneDataset(paths, cache=True, cache_threads=2)
    plain = SceneDataset(paths)
    for index in range(3):
        assert cached[index] is cached[index]  # loaded once
        assert torch.equal(cached[index].map.segments, plain[index].map.segments)


def test_loader_can_leave_idm_fields_empty(map_data):
    scenes = [replace(make_synthetic_scene(num_frames=3), map=map_data), make_synthetic_scene()]

    def batch(**kwargs):
        loader = SceneDataLoader(SceneDataset(scenes), batch_size=2, max_map_segments=4, **kwargs)
        return next(iter(loader)).tensors

    full, lean = batch(), batch(idm_fields=False)
    validate_scene_batch(lean)
    assert not lean.rails.edge_mask.any() and not lean.rails.rail_valid.any()
    assert (lean.map.segment_signal == -1).all()
    # Everything else is the same.
    assert torch.equal(full.map.chunk_signal, lean.map.chunk_signal)
    assert torch.equal(full.rails.signal_red, lean.rails.signal_red)
    assert torch.equal(full.reference_state.position, lean.reference_state.position)


def test_epochs_iterate_several_passes():
    scenes = [make_synthetic_scene() for _ in range(4)]
    loader = SceneDataLoader(SceneDataset(scenes), batch_size=2, max_map_segments=4)
    assert len(list(loader.epochs(3))) == 3 * len(loader)


def test_training_loader_can_swap_oversize_scenes(map_data):
    # The mapped scene has 3 segments; with room for 2 it is replaced by the other scene.
    scenes = [replace(make_synthetic_scene(num_frames=3), map=map_data), make_synthetic_scene()]
    kwargs = dict(batch_size=2, max_map_segments=2, collate_on_device=True)
    with pytest.raises(ValueError, match="max_map_segments"):
        next(iter(SceneDataLoader(SceneDataset(scenes), **kwargs)))
    batch = next(iter(SceneDataLoader(SceneDataset(scenes), replace_oversize=True, **kwargs)))
    assert batch.scenario_ids == (scenes[1].scenario_id,) * 2
