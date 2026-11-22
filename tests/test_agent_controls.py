"""Exercise per-source controller presets, overrides, validation, and partial reset."""

import pytest
import torch
from fastapi.testclient import TestClient

from cs5478_project.data import SceneDataLoader, SceneDataset, make_synthetic_scene, save_scene
from cs5478_project.datatypes.control import ControlType
from cs5478_project.env import RuleBasedEnv
from cs5478_project.server.app import create_app


@pytest.fixture
def control_inputs(tmp_path):
    scenes = [
        make_synthetic_scene(scenario_id=f"control-{index}", num_frames=8, num_actors=6)
        for index in range(2)
    ]
    for index, scene in enumerate(scenes):
        save_scene(scene, tmp_path / f"{index}.pt")
    client = TestClient(
        create_app(
            artifact_roots=[tmp_path],
            include_demo=False,
            device="cpu",
            max_actors=8,
            max_static_objects=4,
            max_frames=8,
        )
    )
    ids = [source["id"] for source in client.get("/api/v1/rollout-sources").json()["sources"]]
    return client, ids, scenes


def assignments(world):
    return {actor["id"]: actor["control_type"] for actor in world["detail"]["actors"]}


def test_blind_overrides_are_scoped_to_actor_and_source(control_inputs):
    client, ids, _ = control_inputs
    response = client.post(
        "/api/v1/rollouts",
        json={
            "source_ids": ids,
            "preset": "idm_pedestrian",
            "agent_controls": [
                {"source_id": ids[0], "actor_id": "actor-3", "control_type": "idm", "blind": True},
            ],
        },
    )
    assert response.status_code == 200, response.text
    for index, world in enumerate(response.json()["worlds"]):
        for actor in world["detail"]["actors"]:
            assert actor["blind"] is (index == 0 and actor["id"] == "actor-3")
    response = client.post("/api/v1/rollouts", json={"source_ids": ids, "preset": "idm_pedestrian"})
    assert response.status_code == 200, response.text
    assert not any(
        actor["blind"] for world in response.json()["worlds"] for actor in world["detail"]["actors"]
    )


@pytest.mark.parametrize(
    ("preset", "expected", "oracle"),
    [
        ("log", {"actor-0": "log", "actor-1": "log", "actor-3": "log", "actor-4": "log"}, False),
        (
            "ego_idm",
            {"actor-0": "idm", "actor-1": "log", "actor-3": "log", "actor-4": "log"},
            False,
        ),
        (
            "idm_pedestrian",
            {"actor-0": "idm", "actor-1": "pedestrian", "actor-3": "idm", "actor-4": "pedestrian"},
            True,
        ),
    ],
)
def test_control_presets_execute_selected_behaviors(control_inputs, preset, expected, oracle):
    client, ids, scenes = control_inputs
    response = client.post("/api/v1/rollouts", json={"source_ids": ids, "preset": preset})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["preset"] == preset
    assert result["pedestrian_oracle"] is oracle
    for world, scene in zip(result["worlds"], scenes, strict=True):
        assert assignments(world) == expected | {"actor-2": "static", "actor-5": "static"}
        assert world["frames"][-1]["done"]
        for frame in world["frames"]:
            for state in frame["actors"]:
                if expected[state["id"]] == "log":
                    assert state["log_pose"] is None
                    continue
                slot = scene.actor_ids.tolist().index(int(state["id"].removeprefix("actor-")))
                tick = frame["index"]
                heading = scene.reference_state.heading[tick, slot]
                center = scene.reference_state.position[tick, slot] + (
                    torch.stack((heading.cos(), heading.sin())) * scene.reference_to_center[slot]
                )
                assert state["log_pose"]["center"] == pytest.approx(center.tolist(), abs=1e-5)
                assert state["log_pose"]["heading"] == float(heading)
        for slot, actor_id in enumerate(scene.actor_ids.tolist()):
            if expected[f"actor-{actor_id}"] == "log":
                for frame in world["frames"]:
                    state = next(
                        actor for actor in frame["actors"] if actor["id"] == f"actor-{actor_id}"
                    )
                    assert (
                        state["position"]
                        == scene.reference_state.position[frame["index"], slot].tolist()
                    )
                    assert (
                        state["velocity"]
                        == scene.reference_state.velocity[frame["index"], slot].tolist()
                    )
        if preset != "log":
            ego = next(actor for actor in world["frames"][1]["actors"] if actor["id"] == "actor-0")
            assert ego["acceleration_command"] != 0


def test_actor_overrides_use_source_and_stable_actor_id(control_inputs):
    client, ids, scenes = control_inputs
    response = client.post(
        "/api/v1/rollouts",
        json={
            "source_ids": ids,
            "preset": "log",
            "agent_controls": [
                {"source_id": ids[0], "actor_id": "actor-3", "control_type": "idm"},
                {"source_id": ids[1], "actor_id": "actor-1", "control_type": "pedestrian"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    first, second = map(assignments, result["worlds"])
    assert first["actor-3"] == "idm"
    assert second["actor-3"] == "log"
    assert first["actor-1"] == "log"
    assert second["actor-1"] == "pedestrian"
    assert result["policy"] == "rule_based"
    assert result["pedestrian_oracle"] is True
    assert result["control_ego"] is False
    first_frame = result["worlds"][0]["frames"][1]
    state = next(actor for actor in first_frame["actors"] if actor["id"] == "actor-3")
    assert state["acceleration_command"] != 0
    logged = next(actor for actor in first_frame["actors"] if actor["id"] == "actor-0")
    assert logged["velocity"] == scenes[0].reference_state.velocity[1, 0].tolist()


def test_actor_overrides_can_restore_log_within_reactive_preset(control_inputs):
    client, ids, _ = control_inputs
    response = client.post(
        "/api/v1/rollouts",
        json={
            "source_ids": ids,
            "preset": "idm_pedestrian",
            "agent_controls": [
                {"source_id": ids[0], "actor_id": "actor-0", "control_type": "log"},
                {"source_id": ids[0], "actor_id": "actor-1", "control_type": "log"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    first, second = map(assignments, response.json()["worlds"])
    assert first["actor-0"] == first["actor-1"] == "log"
    assert first["actor-3"] == "idm"
    assert first["actor-4"] == "pedestrian"
    assert second["actor-0"] == "idm"
    assert second["actor-1"] == "pedestrian"


def test_idm_is_limited_to_railed_vehicles_and_other_controls_apply_to_any_actor(
    control_inputs,
):
    client, ids, _ = control_inputs
    request = {
        "source_ids": ids,
        "preset": "log",
        "agent_controls": [
            {"source_id": ids[0], "actor_id": "actor-0", "control_type": "pedestrian"},
        ],
    }
    response = client.post("/api/v1/rollouts", json=request)
    assert response.status_code == 200, response.text
    first, second = response.json()["worlds"]
    assert assignments(first)["actor-0"] == "pedestrian"
    states = {actor["id"]: actor for actor in first["frames"][1]["actors"]}
    assert states["actor-0"]["steering"] == 0
    assert assignments(second)["actor-0"] == "log"
    # Pedestrians have no exported rail, so nuPlan's builder never makes them IDM agents.
    request["agent_controls"].append(
        {"source_id": ids[0], "actor_id": "actor-1", "control_type": "idm"}
    )
    response = client.post("/api/v1/rollouts", json=request)
    assert response.status_code == 422 and "no exported rail" in response.text


@pytest.mark.parametrize(
    ("actor_id", "control"),
    [("actor-999", "log"), ("actor-2", "log")],
)
def test_invalid_actor_control_is_rejected(control_inputs, actor_id, control):
    client, ids, _ = control_inputs
    response = client.post(
        "/api/v1/rollouts",
        json={
            "source_ids": ids,
            "agent_controls": [
                {"source_id": ids[0], "actor_id": actor_id, "control_type": control},
            ],
        },
    )
    assert response.status_code == 422


def test_duplicate_unselected_sources_and_oracle_switch_are_rejected(control_inputs):
    client, ids, _ = control_inputs
    override = {"source_id": ids[0], "actor_id": "actor-0", "control_type": "idm"}
    for body in (
        {"source_ids": ids, "agent_controls": [override, override]},
        {"source_ids": ids[1:], "agent_controls": [override]},
        {"source_ids": ids, "pedestrian_oracle": False},
        {"source_ids": ids, "preset": "invalid"},
    ):
        assert client.post("/api/v1/rollouts", json=body).status_code == 422


def test_partial_reset_preserves_unselected_world_control_types():
    scene = make_synthetic_scene(num_frames=8, num_actors=6)
    batch = next(
        iter(
            SceneDataLoader(
                SceneDataset([scene, scene]),
                batch_size=2,
                max_actors=8,
                max_frames=8,
                max_static_objects=4,
                max_map_segments=1,
            )
        )
    ).tensors
    controls = torch.where(batch.actor_mask, batch.actor_type + 1, ControlType.LOG)
    controls[0, 1] = ControlType.LOG
    env = RuleBasedEnv()
    env.reset(batch, control_types=controls)
    env.step()
    env.reset(batch, mask=torch.tensor([True, False]), control_types=torch.zeros_like(controls))
    assert not env.control_types[0].any()
    assert torch.equal(env.control_types[1], controls[1])
    result = env.step()
    assert result.episode.step.tolist() == [1, 2]
    torch.testing.assert_close(result.state.velocity[0, 0], batch.reference_state.velocity[0, 1, 0])
    swapped = controls.clone()
    swapped[:, 0] = ControlType.PEDESTRIAN
    env.reset(batch, control_types=swapped)
    assert all(torch.isfinite(value).all() for value in env.step().state)
    swapped[:, 1] = ControlType.IDM
    with pytest.raises(ValueError, match="exported rail"):
        env.reset(batch, control_types=swapped)
    invalid = controls.clone()
    invalid[0, 1] = 99
    with pytest.raises(ValueError, match="Unknown"):
        env.reset(batch, control_types=invalid)
    invalid = controls.clone()
    invalid[0, -1] = ControlType.IDM
    with pytest.raises(ValueError, match="Padding"):
        env.reset(batch, control_types=invalid)


def test_added_actors_join_the_scene_with_stable_ids(control_inputs):
    client, ids, scenes = control_inputs
    plain = client.post("/api/v1/rollouts", json={"source_ids": ids}).json()["worlds"]
    before = {actor["id"] for actor in plain[0]["detail"]["actors"]}
    largest = max(int(name[6:]) for name in before)
    added = [
        {"source_id": ids[0], "serial": 4, "kind": "vehicle", "position": [30.0, 5.0],
         "heading": 1.5707964, "length": 5.176, "width": 2.0},
        {"source_id": ids[0], "serial": 0, "kind": "static", "position": [-8.0, 2.0],
         "length": 1.0, "width": 0.5},
        {"source_id": ids[0], "serial": 2, "kind": "pedestrian", "position": [3.0, 9.0],
         "length": 0.7, "width": 0.7},
    ]  # fmt: skip
    response = client.post(
        "/api/v1/rollouts",
        json={"source_ids": ids, "preset": "idm_pedestrian", "added_actors": added},
    )
    assert response.status_code == 200, response.text
    worlds = response.json()["worlds"]
    actors = {actor["id"]: actor for actor in worlds[0]["detail"]["actors"]}
    assert set(actors) - before == {f"actor-{largest + 1 + serial}" for serial in (4, 0, 2)}
    vehicle, cone, walker = (actors[f"actor-{largest + 1 + serial}"] for serial in (4, 0, 2))
    assert (vehicle["type"], vehicle["added_serial"], vehicle["control_type"]) == (
        "vehicle",
        4,
        "log",
    )
    assert (cone["type"], cone["added_serial"]) == ("static", 0)
    assert cone["static_position"] == [-8.0, 2.0] and cone["presence_intervals"] == [[0, 8]]
    assert (walker["type"], walker["added_serial"]) == ("pedestrian", 2)
    assert all(actor["added_serial"] is None for name, actor in actors.items() if name in before)
    # They stand still for the whole episode; a vehicle's state is its rear axle.
    for frame in worlds[0]["frames"]:
        states = {state["id"]: state for state in frame["actors"]}
        car = states[vehicle["id"]]
        assert car["center"] == pytest.approx([30.0, 5.0], abs=1e-4) and car["speed"] == 0
        assert car["position"] == pytest.approx([30.0, 5.0 - 1.461], abs=1e-4)
        assert states[walker["id"]]["center"] == pytest.approx([3.0, 9.0])
    # The other world is untouched, and removing an item keeps the others' ids.
    assert {actor["id"] for actor in worlds[1]["detail"]["actors"]} == {
        actor["id"] for actor in plain[1]["detail"]["actors"]
    }
    response = client.post("/api/v1/rollouts", json={"source_ids": ids, "added_actors": added[:1]})
    after = {actor["id"] for actor in response.json()["worlds"][0]["detail"]["actors"]}
    assert after - before == {vehicle["id"]}
    repeated = [added[0], {**added[1], "serial": 4}]
    response = client.post("/api/v1/rollouts", json={"source_ids": ids, "added_actors": repeated})
    assert response.status_code == 422
