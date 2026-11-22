"""RDP sampling preserves map groups under the canonical segment budget."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cs5478_project.data import load_scene, make_synthetic_scene, save_scene, simplify_map
from cs5478_project.datatypes import MapData, MapElementType


def make_map(*polylines):
    segments, types, indices = [], [], []
    for index, vertices in enumerate(polylines):
        points = torch.as_tensor(np.asarray(vertices), dtype=torch.float32)
        segments.append(torch.stack((points[:-1], points[1:]), dim=1))
        types.append(torch.full((len(points) - 1,), index % len(MapElementType)))
        indices.append(torch.full((len(points) - 1,), index))
    return MapData(
        map_name="curved-map",
        world_from_scene=(1.0, 2.0, 0.3),
        bounds=(-100.0, -100.0, 100.0, 100.0),
        segments=torch.cat(segments),
        segment_type=torch.cat(types),
        polyline_index=torch.cat(indices),
        polyline_ids=tuple(f"line-{index}" for index in range(len(polylines))),
    )


def arc_points(count=201):
    angle = np.linspace(0, np.pi / 2, count)
    return np.column_stack((10 * np.cos(angle), 10 * np.sin(angle)))


def test_budget_spends_vertices_on_curves_instead_of_dense_straights():
    straight = np.column_stack((np.linspace(0, 10, 1001), np.zeros(1001)))
    source = make_map(straight, arc_points())
    result = simplify_map(source, max_segments=18)
    result.validate()
    assert len(result.segments) == 18
    counts = torch.bincount(result.polyline_index).tolist()
    assert counts[0] == 1 and counts[1] > 1
    assert result.polyline_ids == source.polyline_ids
    assert result.map_name == source.map_name
    assert result.bounds == source.bounds
    assert result.world_from_scene == source.world_from_scene
    assert result.segment_type.tolist() == [0] + [1] * counts[1]
    for line in range(2):
        original = source.segments[source.polyline_index == line]
        reduced = result.segments[result.polyline_index == line]
        assert torch.equal(reduced[0, 0], original[0, 0])
        assert torch.equal(reduced[-1, 1], original[-1, 1])


def test_sharp_corner_is_retained_exactly():
    points = np.concatenate(
        (
            np.column_stack((np.linspace(0, 10, 101), np.zeros(101))),
            np.column_stack((np.full(100, 10), np.linspace(0.1, 10, 100))),
        )
    )
    result = simplify_map(make_map(points), max_segments=2)
    torch.testing.assert_close(
        result.segments,
        torch.tensor([[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 10.0]]]),
        rtol=0,
        atol=0,
    )


def test_closed_square_preserves_corners_and_closing_edge():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    points = np.concatenate(
        [
            np.linspace(start, end, 101)[:-1]
            for start, end in zip(corners[:-1], corners[1:], strict=True)
        ]
        + [corners[-1:]]
    )
    result = simplify_map(make_map(points), max_segments=4)
    assert len(result.segments) == 4
    assert torch.equal(result.segments[:, 0], torch.tensor(corners[:-1], dtype=torch.float32))
    assert torch.equal(result.segments[-1, 1], result.segments[0, 0])


def test_closed_curve_cannot_collapse_to_a_line():
    points = np.concatenate((arc_points(), arc_points()[:1]))
    source = make_map(points)
    result = simplify_map(source, max_segments=3)
    assert len(result.segments) == 3
    assert torch.equal(result.segments[0, 0], result.segments[-1, 1])
    with pytest.raises(ValueError, match="at least 3"):
        simplify_map(source, max_segments=2)


def test_insufficient_budget_never_drops_a_polyline():
    source = make_map(arc_points(), arc_points() + 20)
    with pytest.raises(ValueError, match="preserve all polylines"):
        simplify_map(source, max_segments=1)


def test_degenerate_closed_polyline_is_finite_and_preserves_identity():
    result = simplify_map(make_map(np.zeros((101, 2))), max_segments=8)
    result.validate()
    assert len(result.segments) == 3
    assert result.polyline_ids == ("line-0",)
    assert torch.count_nonzero(result.segments) == 0


def test_simplification_is_deterministic_and_keeps_original_vertex_coordinates():
    source = make_map(arc_points())
    before = source.segments.clone()
    first = simplify_map(source, max_segments=12)
    second = simplify_map(source, max_segments=12)
    assert torch.equal(first.segments, second.segments)
    assert torch.equal(source.segments, before)
    original = torch.cat((source.segments[:1, 0], source.segments[:, 1]))
    for point in first.segments.reshape(-1, 2):
        assert (original == point).all(dim=1).any()


def test_larger_budget_reduces_curve_approximation_error():
    source = make_map(arc_points())
    errors = []
    points = torch.cat((source.segments[:1, 0], source.segments[:, 1]))
    for budget in (4, 16):
        result = simplify_map(source, max_segments=budget)
        start, end = result.segments[:, 0], result.segments[:, 1]
        chord = end - start
        offset = points[:, None] - start
        projection = ((offset * chord).sum(-1) / chord.square().sum(-1)).clamp(0, 1)
        distance = (offset - projection[..., None] * chord).norm(dim=-1)
        errors.append(distance.min(dim=1).values.max().item())
    assert errors[1] < errors[0] / 5
    assert errors[1] < 0.03


def test_already_small_map_is_unchanged(map_data):
    assert simplify_map(map_data, max_segments=4) is map_data


@pytest.mark.parametrize("budget", [0, -1, True, 4.0])
def test_invalid_budget_is_rejected(map_data, budget):
    with pytest.raises(ValueError, match="positive integer"):
        simplify_map(map_data, max_segments=budget)


def test_scene_save_preserves_original_map_without_simplifying(tmp_path):
    original_map = make_map(arc_points())
    scene = replace(make_synthetic_scene(num_frames=3), map=original_map)
    path = tmp_path / "curved.pt"
    save_scene(scene, path)
    restored = load_scene(path)
    assert len(restored.map.segments) == 200
    assert torch.equal(restored.map.segments, original_map.segments)
    assert len(scene.map.segments) == 200
    assert torch.equal(restored.reference_state.position, scene.reference_state.position)
    assert restored.map.polyline_ids == scene.map.polyline_ids


def test_scene_save_enforces_rdp_2048_without_mutating_input(tmp_path):
    source = make_map(arc_points(5001))
    scene = replace(make_synthetic_scene(num_frames=3), map=source)
    save_scene(scene, tmp_path / "canonical.pt")
    restored = load_scene(tmp_path / "canonical.pt")
    assert len(restored.map.segments) == 2048
    assert len(source.segments) == 5000
    assert restored.map.polyline_ids == source.polyline_ids
    assert torch.equal(restored.reference_state.position, scene.reference_state.position)
