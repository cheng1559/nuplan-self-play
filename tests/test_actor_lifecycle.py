"""Temporal actor membership and route horizons independent of episode length."""

from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from cs5478_project.data import (
    SceneDataLoader,
    SceneDataset,
    load_scene,
    make_synthetic_scene,
    save_scene,
)
from cs5478_project.data.routes import build_gt_routes
from cs5478_project.env import ReplayEnv
from cs5478_project.server.app import create_app


def lifecycle_scene():
    scene = make_synthetic_scene(num_frames=6, num_actors=4)
    valid = scene.reference_valid.clone()
    valid[:, 1] = torch.tensor([False, False, True, True, False, True])
    valid[:, 2] = torch.tensor([True, True, False, False, False, False])
    scene = replace(scene, reference_valid=valid, routes=None)
    return replace(scene, routes=build_gt_routes(scene))


@pytest.mark.parametrize("compiled", [False, True])
def test_spawn_despawn_reappearance_and_terminal_freeze_keep_fixed_shapes(compiled):
    scene = lifecycle_scene()
    batch = next(
        iter(SceneDataLoader(SceneDataset([scene]), batch_size=1, max_actors=5, max_frames=8))
    ).tensors
    env = ReplayEnv(compile_step=compiled)
    result = env.reset(batch)
    for tick in range(9):
        index = min(tick, 5)
        assert result.actor_mask.shape == (1, 5)
        assert torch.equal(result.actor_mask[0, :3], scene.reference_valid[index])
        assert not result.actor_mask[0, 3:].any()
        assert result.episode.done.item() == (tick >= 5)
        result = env.step()
    result = env.reset(batch, torch.tensor([True]))
    assert result.actor_mask.tolist() == [[True, False, True, False, False]]
    assert result.episode.step.item() == 0


def test_extended_gt_is_capped_at_30s_from_spawn_and_stops_at_first_gap():
    extended = make_synthetic_scene(num_frames=451, num_actors=4)
    episode = make_synthetic_scene(num_frames=151, num_actors=4)
    valid = torch.ones((451, 3), dtype=torch.bool)
    valid[:50, 1] = False
    valid[190:230, 2] = False
    episode = replace(episode, reference_valid=valid[:151].clone(), routes=None)
    routes = build_gt_routes(
        episode, future_positions=extended.reference_state.position, future_valid=valid
    )
    assert routes.start_time_s.tolist() == [0.0, 5.0, 0.0]
    assert routes.horizon_s.tolist() == pytest.approx([30.0, 30.0, 18.9])
    assert torch.equal(routes.points[1, 0], extended.reference_state.position[50, 1])
    assert torch.equal(routes.points[1, -1], extended.reference_state.position[350, 1])
    assert routes.mask[2].sum() == 100
    assert episode.num_frames == 151


def test_extended_gt_rejects_actor_slot_mismatch():
    scene = make_synthetic_scene(num_frames=4)
    with pytest.raises(ValueError, match="actor slots"):
        build_gt_routes(
            scene,
            future_positions=scene.reference_state.position + 1,
            future_valid=scene.reference_valid,
        )


def test_older_scene_formats_are_rejected(tmp_path):
    path = tmp_path / "old.pt"
    save_scene(make_synthetic_scene(num_frames=4), path)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 6
    torch.save(payload, path)
    with pytest.raises(ValueError, match="re-export"):
        load_scene(path)


def test_live_and_archived_frames_hide_absent_actors_without_dropping_tracks(tmp_path):
    scene = lifecycle_scene()
    save_scene(scene, tmp_path / "dynamic.pt")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu"))
    sources = client.get("/api/v1/rollout-sources").json()["sources"]
    response = client.post("/api/v1/rollouts", json={"source_ids": [sources[0]["id"]]})
    assert response.status_code == 200, response.text
    world = response.json()["worlds"][0]
    assert len(world["detail"]["actors"]) == 4
    for index, frame in enumerate(world["frames"]):
        expected = [
            f"actor-{scene.actor_ids[slot]}"
            for slot in scene.reference_valid[index].nonzero().flatten().tolist()
        ]
        assert [actor["id"] for actor in frame["actors"]] == expected
    assert world["detail"]["actors"][1]["route_start_s"] == pytest.approx(0.2)
    archived = client.get("/api/v1/cases").json()["cases"][0]
    frames = client.get(f"/api/v1/cases/{archived['id']}/frames?start=0&count=6").json()["frames"]
    assert [[actor["id"] for actor in frame["actors"]] for frame in frames] == [
        [actor["id"] for actor in frame["actors"]] for frame in world["frames"]
    ]
