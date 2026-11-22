"""Verify that batch responses come from fresh, completed environment rollouts."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from cs5478_project.data import make_synthetic_scene, save_scene
from cs5478_project.datatypes import WorldState
from cs5478_project.env import ReplayEnv
from cs5478_project.server.app import create_app
from cs5478_project.visualization.map import SceneMap, map_data_from_display


def make_client(tmp_path, scenes, **settings):
    for index, scene in enumerate(scenes):
        save_scene(scene, tmp_path / f"scene-{index}.pt")
    app = create_app(
        artifact_roots=[tmp_path],
        include_demo=False,
        device="cpu",
        max_actors=8,
        max_frames=16,
        **settings,
    )
    return TestClient(app)


def source_ids(client):
    return [source["id"] for source in client.get("/api/v1/rollout-sources").json()["sources"]]


def test_synthetic_catalog_does_not_load_fixture_json_or_torch():
    script = """
import sys
from fastapi.testclient import TestClient
from cs5478_project.server.app import create_app
client = TestClient(create_app(artifact_roots=[]))
assert client.get('/api/v1/rollout-sources').json()['sources'] == []
client = TestClient(create_app(artifact_roots=[], include_demo=True))
result = client.get('/api/v1/rollout-sources').json()
assert [source['scenario_id'] for source in result['sources']] == [
    'synthetic-demo', 'synthetic-secondary'
]
assert [source['num_frames'] for source in result['sources']] == [151, 151]
assert [source['actor_count'] for source in result['sources']] == [6, 6]
assert result['settings'] == dict(
    device='auto', compiled=False, max_actors=256, max_static_objects=512, max_frames=151,
    max_map_segments=2048, max_batch_size=256
)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_actual_environment_outputs_are_captured_and_worlds_finish_independently(
    tmp_path, monkeypatch
):
    scenes = [
        make_synthetic_scene(scenario_id=f"world-{index}", num_frames=frames, num_actors=3)
        for index, frames in enumerate((5, 2, 1))
    ]
    mask = torch.tensor([True, False])
    scenes[0].routes.mask[1] = False
    scenes[0].routes.source[1] = 0
    scenes[0].routes.horizon_s[1] = 0
    scenes[0].routes.start_time_s[1] = 0
    scenes[0] = replace(scenes[0], actor_mask=mask, reference_valid=mask.expand(5, 2).clone())
    client = make_client(tmp_path, scenes)
    calls = []
    original_step = ReplayEnv.step

    def shifted_step(env):
        result = original_step(env)
        calls.append(tuple(result.episode.step.tolist()))
        return result._replace(state=result.state._replace(position=result.state.position + 17))

    monkeypatch.setattr(ReplayEnv, "step", shifted_step)
    ids = source_ids(client)
    response = client.post("/api/v1/rollouts", json={"source_ids": ids})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["policy"] == "log"
    assert result["device"] == "cpu"
    assert result["compiled"] is False
    assert result["batch_size"] == 3
    assert result["world_transitions"] == 5
    assert result["elapsed_s"] > 0
    assert calls == [(1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)]
    assert [len(world["frames"]) for world in result["worlds"]] == [5, 2, 1]
    for index, world in enumerate(result["worlds"]):
        assert world["source_id"] == ids[index]
        assert world["detail"]["id"] == f"{result['id']}:{index}"
        assert world["map"] is None
        assert world["frames"][-1]["done"] is True
        assert all(not frame["done"] for frame in world["frames"][:-1])
        assert world["frames"][-1]["time_s"] == pytest.approx((scenes[index].num_frames - 1) * 0.1)
    first = result["worlds"][0]
    assert [actor["id"] for actor in first["detail"]["actors"]] == ["actor-0", "actor-2"]
    assert [actor["id"] for actor in first["frames"][1]["actors"]] == ["actor-0"]
    assert (
        first["frames"][0]["actors"][0]["position"] == scenes[0].initial_state.position[0].tolist()
    )
    assert first["frames"][1]["actors"][0]["position"] == pytest.approx(
        (scenes[0].reference_state.position[1, 0] + 17).tolist()
    )


def test_rerun_reloads_scene_and_returns_a_new_complete_batch(tmp_path, monkeypatch):
    scene = make_synthetic_scene(scenario_id="fresh", num_frames=3)
    client = make_client(tmp_path, [scene])
    ids = source_ids(client)
    calls = []
    original_step = ReplayEnv.step

    def count_step(env):
        calls.append(1)
        return original_step(env)

    monkeypatch.setattr(ReplayEnv, "step", count_step)
    first = client.post("/api/v1/rollouts", json={"source_ids": ids}).json()
    updated = make_synthetic_scene(scenario_id="fresh", num_frames=4)
    reference = updated.reference_state._replace(position=updated.reference_state.position + 25)
    updated = replace(
        updated,
        reference_state=reference,
        initial_state=WorldState(*(value[0].clone() for value in reference)),
    )
    save_scene(updated, tmp_path / "scene-0.pt")
    second = client.post("/api/v1/rollouts", json={"source_ids": ids}).json()
    assert first["id"] != second["id"]
    assert len(calls) == 5
    assert len(first["worlds"][0]["frames"]) == 3
    assert len(second["worlds"][0]["frames"]) == 4
    assert second["worlds"][0]["frames"][0]["actors"][0]["position"] == [25, 25]
    assert second["worlds"][0]["frames"][-1]["done"] is True


def test_normalized_map_is_returned_inline_and_json_is_not_a_rollout_source(tmp_path):
    scene = make_synthetic_scene(scenario_id="mapped", num_frames=2)
    payload = {
        "format": "scene_map",
        "schema_version": 1,
        "scenario_id": "mapped",
        "map_name": "test-map",
        "coordinate_frame": "scene",
        "world_from_scene": {"x": 123.0, "y": 456.0, "heading": 0.5},
        "bounds": [-10.0, -10.0, 20.0, 20.0],
        "polylines": [
            {"id": "lane-1", "kind": "lane_centerline", "points": [[-5.0, 0.0], [15.0, 0.0]]}
        ],
    }
    scene = replace(scene, map=map_data_from_display(SceneMap.model_validate(payload)))
    save_scene(scene, tmp_path / "scene.pt")
    (tmp_path / "notes.json").write_text(json.dumps({"description": "not a scene"}))
    app = create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu")
    client = TestClient(app)
    listing = client.get("/api/v1/rollout-sources").json()
    assert len(listing["sources"]) == 1
    source = listing["sources"][0]
    assert source["source_kind"] == "scene"
    assert source["num_actor_slots"] == 2
    assert source["has_map"] is True
    result = client.post("/api/v1/rollouts", json={"source_ids": [source["id"]]}).json()
    world = result["worlds"][0]
    assert world["map"] == payload
    assert world["detail"]["has_map"] is True
    assert [frame["index"] for frame in world["frames"]] == [0, 1]
    assert world["frames"][-1]["actors"][0]["center"][0] == pytest.approx(
        scene.reference_state.position[1, 0, 0].item()
        + scene.reference_to_center[0].item()
        * torch.cos(scene.reference_state.heading[1, 0]).item()
    )


def test_live_scene_ignores_sidecars_and_reloads_embedded_map(tmp_path):
    scene = make_synthetic_scene(scenario_id="embedded", num_frames=2)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    (tmp_path / "scene.map.json").write_text("invalid and unrelated map")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu"))
    ids = source_ids(client)
    first = client.post("/api/v1/rollouts", json={"source_ids": ids})
    assert first.status_code == 200
    assert first.json()["worlds"][0]["map"] is None
    payload = SceneMap.model_validate(
        {
            "format": "scene_map",
            "schema_version": 1,
            "scenario_id": scene.scenario_id,
            "map_name": "embedded-map",
            "coordinate_frame": "scene",
            "world_from_scene": {"x": 0.0, "y": 0.0, "heading": 0.0},
            "bounds": [-10.0, -10.0, 10.0, 10.0],
            "polylines": [
                {
                    "id": "boundary",
                    "kind": "road_boundary",
                    "points": [[-5.0, 1.0], [0.0, 2.0], [5.0, 1.0]],
                }
            ],
        }
    )
    save_scene(replace(scene, map=map_data_from_display(payload)), path)
    second = client.post("/api/v1/rollouts", json={"source_ids": ids})
    assert second.status_code == 200
    assert second.json()["worlds"][0]["map"] == payload.model_dump(mode="json")
    assert second.json()["worlds"][0]["detail"]["has_map"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"source_ids": []},
        {"source_ids": ["same", "same"]},
        {"source_ids": [123]},
        {"source_ids": [""]},
        {"source_ids": ["same"], "device": "cuda"},
    ],
)
def test_invalid_batch_requests_are_rejected(payload):
    client = TestClient(create_app(artifact_roots=[], device="cpu"))
    assert client.post("/api/v1/rollouts", json=payload).status_code == 422


def test_unknown_sources_and_batch_size_limit_are_explicit():
    client = TestClient(
        create_app(artifact_roots=[], include_demo=True, device="cpu", max_batch_size=1)
    )
    assert client.post("/api/v1/rollouts", json={"source_ids": ["unregistered"]}).status_code == 404
    result = client.post("/api/v1/rollouts", json={"source_ids": source_ids(client)})
    assert result.status_code == 422
    assert "between 1 and 1" in result.json()["detail"]


@pytest.mark.parametrize(
    "settings, expected", [({"max_actors": 1}, "actor slots"), ({"max_frames": 2}, "frames")]
)
def test_input_capacity_overflow_is_not_silently_truncated(tmp_path, settings, expected):
    save_scene(make_synthetic_scene(num_frames=3), tmp_path / "scene.pt")
    client = TestClient(
        create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu", **settings)
    )
    result = client.post("/api/v1/rollouts", json={"source_ids": source_ids(client)})
    assert result.status_code == 422
    assert expected in result.json()["detail"]


def test_configured_map_capacity_rejects_overflow_before_rollout(tmp_path, map_data):
    scene = replace(make_synthetic_scene(num_frames=2), map=map_data)
    save_scene(scene, tmp_path / "scene.pt")
    client = TestClient(
        create_app(artifact_roots=[tmp_path], include_demo=False, device="cpu", max_map_segments=2)
    )
    assert client.get("/api/v1/rollout-sources").json()["settings"]["max_map_segments"] == 2
    result = client.post("/api/v1/rollouts", json={"source_ids": source_ids(client)})
    assert result.status_code == 422
    assert "max_map_segments=2" in result.json()["detail"]
    client = TestClient(
        create_app(
            artifact_roots=[tmp_path], include_demo=False, device="cpu", max_map_segments=2048
        )
    )
    result = client.post("/api/v1/rollouts", json={"source_ids": source_ids(client)})
    assert result.status_code == 200
    assert result.json()["worlds"][0]["frames"][-1]["done"] is True


def test_unavailable_cuda_has_an_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    client = TestClient(create_app(artifact_roots=[], include_demo=True, device="cuda"))
    result = client.post("/api/v1/rollouts", json={"source_ids": source_ids(client)[:1]})
    assert result.status_code == 503
    assert "--device cpu" in result.json()["detail"]


def test_a_second_request_is_rejected_until_the_first_batch_finishes(tmp_path, monkeypatch):
    import cs5478_project.pipelines.rollout as pipeline

    scene = make_synthetic_scene(num_frames=3)
    client = make_client(tmp_path, [scene])
    ids = source_ids(client)
    original = pipeline.rollout_log_batch
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Test did not release the rollout")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "rollout_log_batch", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/v1/rollouts", json={"source_ids": ids})
        try:
            assert entered.wait(timeout=5)
            assert not first.done()
            second = client.post("/api/v1/rollouts", json={"source_ids": ids})
            assert second.status_code == 409
        finally:
            release.set()
        response = first.result(timeout=10)
    assert response.status_code == 200
    assert response.json()["worlds"][0]["frames"][-1]["done"] is True


def test_failed_execution_releases_the_slot_for_the_next_request(tmp_path, monkeypatch):
    import cs5478_project.pipelines.rollout as pipeline

    client = make_client(tmp_path, [make_synthetic_scene(num_frames=2)])
    ids = source_ids(client)
    original = pipeline.rollout_log_batch

    def failing(*args, **kwargs):
        raise RuntimeError("Deliberate execution failure")

    monkeypatch.setattr(pipeline, "rollout_log_batch", failing)
    with pytest.raises(RuntimeError, match="Deliberate execution failure"):
        client.post("/api/v1/rollouts", json={"source_ids": ids})
    monkeypatch.setattr(pipeline, "rollout_log_batch", original)
    assert client.post("/api/v1/rollouts", json={"source_ids": ids}).status_code == 200
