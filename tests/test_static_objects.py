"""Compact static storage, deterministic capacity selection, and visibility."""

from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from cs5478_project.data import (
    SceneDataLoader,
    SceneDataset,
    load_scene,
    make_synthetic_scene,
    prepare_scene,
    save_scene,
)
from cs5478_project.datatypes import Scene, WorldState
from cs5478_project.env import ReplayEnv
from cs5478_project.server.app import create_app


def legacy_scene():
    """Build a v4-style input with static jitter and a late observation."""
    base = make_synthetic_scene(num_frames=4, num_actors=1)
    position = torch.cat(
        (
            base.reference_state.position,
            torch.tensor(
                [
                    [[90.0, 90.0]],
                    [[20.0, 4.0]],
                    [[21.0, 4.0]],
                    [[22.0, 4.0]],
                ]
            ),
        ),
        dim=1,
    )
    reference = WorldState(
        *(
            position if name == "position" else torch.cat((value, torch.zeros_like(value)), dim=1)
            for name, value in zip(WorldState._fields, base.reference_state, strict=True)
        )
    )
    return Scene(
        scenario_id="legacy-static",
        dt=0.1,
        actor_type=torch.tensor([0, 2]),
        actor_mask=torch.ones(2, dtype=torch.bool),
        size=torch.tensor([[4.0, 2.0], [1.0, 1.0]]),
        wheelbase=torch.ones(2),
        reference_to_center=torch.zeros(2),
        initial_state=WorldState(*(value[0].clone() for value in reference)),
        reference_state=reference,
        reference_valid=torch.tensor([[True, False], [True, True], [True, False], [True, True]]),
    )


def test_static_geometry_freezes_at_first_observation(tmp_path):
    path = tmp_path / "scene.pt"
    scene = prepare_scene(legacy_scene())
    assert scene.reference_state.position.shape == (4, 1, 2)
    assert scene.actor_ids.tolist() == [0]
    assert scene.static.ids.tolist() == [1]
    assert scene.static.position.tolist() == [[20.0, 4.0]]
    assert scene.static.valid[:, 0].tolist() == [False, True, False, True]
    save_scene(scene, path)
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 9
    assert payload["static"]["position"].shape == (1, 2)
    assert "velocity" not in payload["static"]
    assert load_scene(path).static.position.tolist() == [[20.0, 4.0]]


@pytest.mark.parametrize("compiled", [False, True])
def test_static_masks_and_partial_reset_do_not_allocate_static_world_states(compiled):
    scene = prepare_scene(legacy_scene())

    def batch(value):
        return next(
            iter(
                SceneDataLoader(
                    SceneDataset([value, value]),
                    batch_size=2,
                    max_actors=2,
                    max_static_objects=3,
                    max_frames=5,
                )
            )
        ).tensors

    loaded = batch(scene)
    assert loaded.static.position.shape == (2, 3, 2)
    assert loaded.static.valid.shape == (2, 5, 3)
    assert loaded.reference_state.position.shape == (2, 5, 2, 2)
    env = ReplayEnv(compile_step=compiled)
    result = env.reset(loaded)
    for tick in range(7):
        assert result.static_mask[:, 0].tolist() == [bool(scene.static.valid[min(tick, 3), 0])] * 2
        assert not result.static_mask[:, 1:].any()
        result = env.step()
    shifted = replace(scene, static=scene.static._replace(position=scene.static.position + 50))
    env.reset(batch(shifted), torch.tensor([False, True]))
    assert torch.equal(env.scene.static.position[0], loaded.static.position[0])
    assert env.scene.static.position[1, 0].tolist() == [70.0, 54.0]
    assert env.episode.step.tolist() == [3, 0]


def test_capacity_selection_keeps_future_vehicles_ego_and_nearby_static():
    scene = make_synthetic_scene(num_frames=4, num_actors=9)
    valid = scene.reference_valid.clone()
    late = int((scene.actor_ids == 6).nonzero().item())
    valid[:2, late] = False
    scene = replace(scene, reference_valid=valid)
    selected = prepare_scene(scene, max_dynamic_actors=3, max_static_objects=1)
    assert selected.actor_ids.tolist() == [0, 3, 6]
    assert selected.static.ids.tolist() == [2]
    assert selected.reference_valid[:, 2].tolist() == [False, False, True, True]
    assert torch.equal(selected.routes.points[2], scene.routes.points[late])
    repeat = prepare_scene(selected, max_dynamic_actors=3, max_static_objects=1)
    assert torch.equal(repeat.actor_ids, selected.actor_ids)
    assert torch.equal(repeat.static.position, selected.static.position)


def test_api_sends_static_geometry_once_and_visibility_as_intervals(tmp_path):
    save_scene(legacy_scene(), tmp_path / "scene.pt")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu"))
    sources = client.get("/api/v1/rollout-sources").json()["sources"]
    assert sources[0]["dynamic_actor_count"] == sources[0]["static_object_count"] == 1
    world = client.post("/api/v1/rollouts", json={"source_ids": [sources[0]["id"]]}).json()[
        "worlds"
    ][0]
    metadata = world["detail"]["actors"][1]
    assert metadata["static_position"] == [20.0, 4.0]
    assert metadata["presence_intervals"] == [[1, 2], [3, 4]]
    assert metadata["route"] == [] and metadata["route_source"] == "none"
    assert all(
        [actor["id"] for actor in frame["actors"]] == ["actor-0"] for frame in world["frames"]
    )
    assert world["frames"][-1]["done"]
