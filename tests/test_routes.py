"""Actor route preparation, validity, persistence, and batched reset behavior."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cs5478_project.data import (
    SceneDataLoader,
    SceneDataset,
    load_scene,
    make_synthetic_scene,
    save_scene,
)
from cs5478_project.data.routes import build_gt_routes, fit_route
from cs5478_project.datatypes import RouteSource
from cs5478_project.env import ReplayEnv


def test_origin_is_a_valid_stationary_route_with_finite_tail_padding():
    points, mask = fit_route(np.zeros((9, 2)))
    assert points.shape == (100, 2)
    assert mask.sum() == 1 and mask[0]
    assert torch.equal(points, torch.zeros_like(points))


def test_gt_route_preserves_endpoints_and_does_not_smooth_corners():
    points, mask = fit_route(np.array([[0, 0], [10, 0], [10, 10]]))
    assert mask.all()
    assert torch.equal(points[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(points[-1], torch.tensor([10.0, 10.0]))
    assert ((points[:, 1] == 0) | (points[:, 0] == 10)).all()


def test_route_roundtrip(tmp_path):
    scene = make_synthetic_scene(num_frames=7)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    restored = load_scene(path)
    assert restored.ego_index == 0
    for expected, actual in zip(scene.routes, restored.routes, strict=True):
        assert torch.equal(actual, expected)


def test_route_validity_rejects_holes_unknown_sources_and_nonfinite_padding():
    for modification, message in (
        (lambda routes: routes.mask.__setitem__((0, 4), False), "prefix"),
        (lambda routes: routes.source.__setitem__(0, 9), "source"),
        (lambda routes: routes.points.__setitem__((1, 99, 0), float("nan")), "finite"),
    ):
        scene = make_synthetic_scene()
        modification(scene.routes)
        with pytest.raises(ValueError, match=message):
            scene.validate()


def test_route_batch_padding_and_partial_reset_preserve_other_worlds():
    first = make_synthetic_scene(num_frames=4)
    second = make_synthetic_scene(num_frames=4, num_actors=1)

    def batch(scenes):
        return next(
            iter(SceneDataLoader(SceneDataset(scenes), batch_size=2, max_actors=5, max_frames=7))
        ).tensors

    original = batch([first, second])
    assert original.routes.points.shape == (2, 5, 100, 2)
    assert not original.routes.mask[1, 1:].any()
    assert (original.routes.source[1, 1:] == RouteSource.NONE).all()
    replacement = replace(
        first, routes=first.routes._replace(points=first.routes.points + 12), ego_index=1
    )
    incoming = batch([replacement, replacement])
    env = ReplayEnv()
    env.reset(original)
    env.step()
    env.reset(incoming, torch.tensor([False, True]))
    for before, after in zip(original.routes, env.scene.routes, strict=True):
        assert torch.equal(before[0], after[0])
    for expected, actual in zip(incoming.routes, env.scene.routes, strict=True):
        assert torch.equal(expected[1], actual[1])
    assert env.scene.ego_index.tolist() == [0, 1]
    assert env.episode.step.tolist() == [1, 0]


def test_map_route_prefers_connected_lanes_over_a_nearby_unlinked_lane(map_data):
    pytest.importorskip("nuplan")
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    from shapely.geometry import LineString

    from cs5478_project.data.nuplan.routes import build_scene_routes

    def edge(identifier, points):
        return SimpleNamespace(
            id=identifier,
            baseline_path=SimpleNamespace(
                linestring=LineString(points), discrete_path=[StateSE2(x, y, 0) for x, y in points]
            ),
            outgoing_edges=[],
            incoming_edges=[],
        )

    start = edge("start", [(1, 2), (11, 2)])
    linked = edge("linked", [(11.2, 2), (21, 2)])
    unrelated = edge("unrelated", [(11, 2), (21, 2.01)])
    start.outgoing_edges = [linked]
    linked.incoming_edges = [start]
    a = SimpleNamespace(id="a", interior_edges=[start], incoming_edges=[], outgoing_edges=[])
    b = SimpleNamespace(
        id="b", interior_edges=[unrelated, linked], incoming_edges=[a], outgoing_edges=[]
    )
    a.outgoing_edges = [b]

    class FakeMap:
        def get_map_object(self, identifier, layer):
            return {"a": a, "b": b}.get(identifier) if layer == Layer.ROADBLOCK else None

        def get_proximal_map_objects(self, **kwargs):
            return {Layer.ROADBLOCK: [a], Layer.ROADBLOCK_CONNECTOR: []}

    scene = replace(make_synthetic_scene(), map=replace(map_data, world_from_scene=(1.0, 2.0, 0.0)))
    routes, corrected = build_scene_routes(scene, FakeMap(), ["a", "b"])
    assert corrected == ["a", "b"]
    assert routes.source.tolist() == [RouteSource.MAP, RouteSource.GT]
    assert torch.equal(routes.points[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(routes.points[0, -1], torch.tensor([20.0, 0.0]))
    assert torch.count_nonzero(routes.points[0, :, 1]) == 0
    for actual, expected in zip(routes[0][1:], build_gt_routes(scene)[0][1:], strict=True):
        assert torch.equal(actual, expected)


def test_unavailable_map_route_raises_instead_of_using_ego_gt(map_data):
    pytest.importorskip("nuplan")
    from cs5478_project.data.nuplan.routes import build_scene_routes

    scene = replace(make_synthetic_scene(), map=map_data)
    with pytest.raises(ValueError, match="no usable map baselines"):
        build_scene_routes(scene, object(), [])
