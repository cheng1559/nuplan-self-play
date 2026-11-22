"""Verify artifact identity, recorded-state responses, and bounded frame access."""

import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from cs5478_project.server.app import create_app
from cs5478_project.server.services.case_store import CaseStore
from cs5478_project.visualization.map import (
    SceneMap,
    display_map_from_data,
    map_data_from_display,
)


def write_replay(path, *, frame_counts=(4, 2)):
    worlds, frames, slots = len(frame_counts), max(frame_counts), 3
    heading = math.pi / 2
    batch = {
        "scenario_ids": [f"scenario-{world}" for world in range(worlds)],
        "num_frames": list(frame_counts),
        "dt": [0.1] * worlds,
        "actor_type": [[0, 1, 2] for _ in range(worlds)],
        "actor_mask": [[True, False, True] for _ in range(worlds)],
        "size": [[[4.8, 2.0], [1.0, 1.0], [1.0, 1.0]] for _ in range(worlds)],
        "reference_to_center": [[1.4, 0.0, 0.0] for _ in range(worlds)],
        "state": {},
    }
    for name in (
        "position",
        "heading",
        "velocity",
        "acceleration",
        "acceleration_command",
        "steering",
        "yaw_rate",
    ):
        values = []
        for tick in range(frames):
            rows = []
            for world in range(worlds):
                index = min(tick, frame_counts[world] - 1)
                if name == "position":
                    row = [[world * 100 + index + slot * 8, 0.0] for slot in range(slots)]
                elif name == "velocity":
                    row = [[1.0, 0.0] for _ in range(slots)]
                else:
                    value = heading if name == "heading" else float(index)
                    row = [value] * slots
                rows.append(row)
            values.append(rows)
        batch["state"][name] = values
    payload = {"format": "replay_debug", "schema_version": 1, "batches": [batch]}
    path.write_text(json.dumps(payload))


def test_replay_cases_keep_world_horizons_and_actor_slots(tmp_path):
    write_replay(tmp_path / "trace.json")
    (tmp_path / "metadata.json").write_text('{"description": "not a replay"}')
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False))
    cases = client.get("/api/v1/cases").json()["cases"]
    assert [case["num_frames"] for case in cases] == [4, 2]
    assert [case["actor_count"] for case in cases] == [2, 2]
    assert [case["source_kind"] for case in cases] == ["replay", "replay"]
    assert cases[0]["duration_s"] == pytest.approx(0.3)
    assert cases[0]["id"] != cases[1]["id"]
    detail = client.get(f"/api/v1/cases/{cases[0]['id']}").json()
    assert [actor["id"] for actor in detail["actors"]] == ["actor-0", "actor-2"]
    assert [actor["slot"] for actor in detail["actors"]] == [0, 2]
    assert [actor["type"] for actor in detail["actors"]] == ["vehicle", "static"]
    assert detail["bounds"][0] < detail["bounds"][2]
    assert detail["bounds"][1] < detail["bounds"][3]


def test_frame_chunks_match_recorded_reference_and_center_convention(tmp_path):
    write_replay(tmp_path / "trace.json")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False))
    cases = client.get("/api/v1/cases").json()["cases"]
    case_id = cases[0]["id"]
    first = client.get(f"/api/v1/cases/{case_id}/frames?start=0&count=2").json()
    second = client.get(f"/api/v1/cases/{case_id}/frames?start=2&count=2").json()
    assert [frame["index"] for frame in first["frames"] + second["frames"]] == [0, 1, 2, 3]
    assert first["total_frames"] == 4
    frame = second["frames"][-1]
    assert frame["time_s"] == pytest.approx(0.3)
    assert frame["done"] is True
    actor = frame["actors"][0]
    assert actor["position"] == [3.0, 0.0]
    assert actor["center"] == pytest.approx([3.0, 1.4])
    assert actor["speed"] == 1.0
    for name in ("acceleration", "acceleration_command", "steering", "yaw_rate"):
        assert actor[name] == 3.0
    short_id = cases[1]["id"]
    final = client.get(f"/api/v1/cases/{short_id}/frames?start=1&count=256").json()
    assert len(final["frames"]) == 1
    assert final["frames"][0]["actors"][0]["position"] == [101.0, 0.0]
    assert final["frames"][0]["done"] is True


def test_api_rejects_unknown_ids_and_invalid_frame_ranges(tmp_path):
    write_replay(tmp_path / "trace.json")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False))
    case_id = client.get("/api/v1/cases").json()["cases"][0]["id"]
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/cases/not-registered").status_code == 404
    assert client.get("/api/v1/cases/not-registered/frames").status_code == 404
    for query in ("start=-1", "count=0", "count=257"):
        assert client.get(f"/api/v1/cases/{case_id}/frames?{query}").status_code == 422
    assert client.get(f"/api/v1/cases/{case_id}/frames?start=4").status_code == 416


def test_normalized_scene_uses_same_state_api(tmp_path):
    from cs5478_project.data import make_synthetic_scene, save_scene

    scene = make_synthetic_scene(scenario_id="normalized", num_frames=3)
    save_scene(scene, tmp_path / "scene.pt")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False))
    summary = client.get("/api/v1/cases").json()["cases"][0]
    assert summary["scenario_id"] == "normalized"
    assert summary["source_kind"] == "scene"
    frames = client.get(f"/api/v1/cases/{summary['id']}/frames").json()["frames"]
    for index, frame in enumerate(frames):
        for slot, actor in enumerate(frame["actors"]):
            assert actor["position"] == scene.reference_state.position[index, slot].tolist()
            assert actor["heading"] == scene.reference_state.heading[index, slot].item()
    assert frames[-1]["done"] is True


def test_demo_is_explicit_and_json_browsing_does_not_import_torch():
    script = """
import sys
from cs5478_project.server.app import create_app
from fastapi.testclient import TestClient
client = TestClient(create_app(artifact_roots=[]))
assert client.get('/api/v1/cases').json()['cases'] == []
client = TestClient(create_app(artifact_roots=[], include_demo=True))
case = client.get('/api/v1/cases').json()['cases'][0]
assert case['source_kind'] == 'demo'
assert case['scenario_id'] == 'synthetic-demo'
response = client.get(f"/api/v1/cases/{case['id']}/frames?count=1")
assert response.status_code == 200
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_malformed_matching_artifact_fails_with_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"format":"replay_debug","schema_version":1,"batches":[{}]}')
    with pytest.raises(ValueError, match="broken.json"):
        CaseStore([tmp_path], include_demo=False)


def test_missing_explicit_root_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Artifact root does not exist"):
        CaseStore([tmp_path / "missing"], include_demo=False)


def write_map(path, *, scenario_id="scenario-0"):
    payload = {
        "format": "scene_map",
        "schema_version": 1,
        "scenario_id": scenario_id,
        "map_name": "test-map",
        "coordinate_frame": "scene",
        "world_from_scene": {"x": 123.0, "y": 456.0, "heading": 0.5},
        "bounds": [-10.0, -10.0, 20.0, 20.0],
        "polylines": [
            {
                "id": "lane-1",
                "kind": "lane_centerline",
                "points": [[-5.0, 0.0], [15.0, 0.0]],
            },
            {
                "id": "boundary-1",
                "kind": "lane_boundary",
                "points": [[-5.0, 2.0], [15.0, 2.0]],
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return payload


def test_static_map_is_served_only_for_matching_world(tmp_path):
    write_replay(tmp_path / "trace.json")
    payload = write_map(tmp_path / "trace.map.json")
    client = TestClient(create_app(artifact_roots=[tmp_path], include_demo=False))
    cases = client.get("/api/v1/cases").json()["cases"]
    assert len(cases) == 2
    first_id, second_id = [case["id"] for case in cases]
    assert client.get(f"/api/v1/cases/{first_id}").json()["has_map"] is True
    assert client.get(f"/api/v1/cases/{second_id}").json()["has_map"] is False
    assert client.get(f"/api/v1/cases/{first_id}/map").json() == payload
    missing = client.get(f"/api/v1/cases/{second_id}/map")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "No map is available for this case"
    assert client.get("/api/v1/cases/not-registered/map").status_code == 404


def test_scene_map_is_embedded_and_sidecars_are_ignored(tmp_path):
    from cs5478_project.data import make_synthetic_scene, save_scene

    scene = make_synthetic_scene(scenario_id="normalized")
    save_scene(scene, tmp_path / "scene.pt")
    store = CaseStore([tmp_path], include_demo=False)
    case_id = store.list_cases()[0].id
    assert store.detail(case_id).has_map is False
    payload = write_map(tmp_path / "scene.map.json", scenario_id="unrelated")
    store = CaseStore([tmp_path], include_demo=False)
    assert store.detail(case_id).has_map is False
    payload["scenario_id"] = scene.scenario_id
    scene = replace(scene, map=map_data_from_display(SceneMap.model_validate(payload)))
    save_scene(scene, tmp_path / "scene.pt")
    (tmp_path / "scene.map.json").unlink()
    store = CaseStore([tmp_path], include_demo=False)
    assert store.detail(case_id).has_map is True
    assert store.scene_map(case_id).scenario_id == "normalized"
    assert store.scene_map(case_id).model_dump(mode="json") == payload


def test_embedded_map_adapter_preserves_polyline_order_and_closing_edges(tmp_path):
    import torch

    payload = write_map(tmp_path / "source.map.json")
    payload["polylines"] = [
        {
            "id": "curved-lane",
            "kind": "lane_connector_centerline",
            "points": [[0.1, 0.2], [1.5, 2.5], [1.5, 2.5], [4.5, 3.5]],
        },
        {
            "id": "closed-road",
            "kind": "road_boundary",
            "points": [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, -5.0]],
        },
        {"id": "stop", "kind": "stop_line", "points": [[0.0, 2.0], [3.0, 2.0]]},
    ]
    display = SceneMap.model_validate(payload)
    data = map_data_from_display(display)
    assert data.polyline_ids == ("curved-lane", "closed-road", "stop")
    assert data.polyline_index.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert data.segment_type.tolist() == [1, 1, 1, 3, 3, 3, 5]
    restored = display_map_from_data(data, display.scenario_id)
    assert restored.map_name == display.map_name
    assert restored.world_from_scene == display.world_from_scene
    assert restored.bounds == display.bounds
    for original, rebuilt in zip(display.polylines, restored.polylines, strict=True):
        assert original.id == rebuilt.id
        assert original.kind == rebuilt.kind
        assert torch.equal(torch.tensor(original.points), torch.tensor(rebuilt.points))
    assert restored.polylines[1].points[-1] == restored.polylines[1].points[0]


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update(bounds=[10.0, -10.0, -10.0, 20.0]),
        lambda payload: payload.update(bounds=[-10.0, -10.0, float("inf"), 20.0]),
        lambda payload: payload["world_from_scene"].update(heading=float("nan")),
        lambda payload: payload["polylines"][0].update(points=[[0.0, 0.0]]),
        lambda payload: payload["polylines"][0].update(points=[[0.0, "0"], [1.0, 0.0]]),
        lambda payload: payload["polylines"][0].update(kind="polygon"),
        lambda payload: payload["polylines"][1].update(id="lane-1"),
        lambda payload: payload.update(coordinate_frame="world"),
    ],
)
def test_malformed_map_fails_with_sidecar_path(tmp_path, change):
    write_replay(tmp_path / "trace.json")
    path = tmp_path / "trace.map.json"
    payload = write_map(path)
    change(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid scene map.*trace.map.json"):
        CaseStore([tmp_path], include_demo=False)
