"""SingleFrameNet: observation contract, network invariances, Beta helpers, env and API."""

import math

import pytest
import torch
from fastapi.testclient import TestClient
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.agents.goals import sample_goals
from cs5478_project.data import save_scene
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.map import CHUNK_POINTS, MapElementType
from cs5478_project.datatypes.observation import (
    AGENT_FEATURES,
    KINEMATIC_FEATURES,
    LANE_FEATURES,
    Observation,
)
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.env.learned import LearnedEnv
from cs5478_project.env.observations import build_observation
from cs5478_project.models import SingleFrameConfig, SingleFrameNet
from cs5478_project.models.checkpoint import (
    load_single_frame,
    random_single_frame,
    save_single_frame,
)
from cs5478_project.models.distributions import (
    beta_entropy,
    beta_log_prob,
    beta_mode,
    decode_unit,
    sample_beta,
)
from cs5478_project.pipelines.rollout import rollout_learned_batch
from cs5478_project.server.app import create_app

SMALL = SingleFrameConfig(
    embed_dim=16, num_heads=4, max_agents=4, max_lanes=4, last_action_input=False
)


def random_observation(queries=3, agents=5, lanes=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    agent_mask = torch.rand(queries, agents, generator=generator) > 0.4
    agent_mask[:, 0] = True
    lane_mask = torch.rand(queries, lanes, generator=generator) > 0.4
    return Observation(
        torch.randn(queries, agents, AGENT_FEATURES, generator=generator)
        * 10
        * agent_mask[..., None],
        agent_mask,
        torch.randn(queries, 2, generator=generator) * 20,
        torch.randn(queries, KINEMATIC_FEATURES, generator=generator),
        torch.randn(queries, lanes, LANE_FEATURES, generator=generator) * 10 * lane_mask[..., None],
        lane_mask,
    )


def network(layers=1):
    torch.manual_seed(5478)
    config = SingleFrameConfig(
        embed_dim=16, num_heads=4, num_agent_attention_layers=layers, last_action_input=False
    )
    return SingleFrameNet(config).eval()


def assert_output_close(left, right, **kwargs):
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, **kwargs)


def test_outputs_are_two_betas_initialized_near_symmetric_bell():
    output = network()(random_observation())
    assert output.alpha.shape == output.beta.shape == (3, 2)
    assert output.value.shape == (3,)
    assert bool((output.alpha > 1).all() and (output.beta > 1).all())
    # A small final-layer std and bias 0.54 give Beta(2, 2) up to input noise.
    torch.testing.assert_close(output.alpha, torch.full_like(output.alpha, 2.0), atol=0.1, rtol=0)
    torch.testing.assert_close(output.beta, torch.full_like(output.beta, 2.0), atol=0.1, rtol=0)


def test_padding_values_do_not_change_outputs():
    model = network(layers=2)
    observation = random_observation()
    noisy = observation._replace(
        agents=torch.where(observation.agent_mask[..., None], observation.agents, 1e3),
        lanes=torch.where(observation.lane_mask[..., None], observation.lanes, -1e3),
    )
    assert_output_close(model(observation), model(noisy), atol=1e-5, rtol=1e-5)


def test_outputs_are_invariant_to_neighbor_and_lane_order():
    model = network()
    observation = random_observation()
    agents = torch.cat((torch.zeros(1, dtype=torch.long), torch.randperm(4) + 1))
    lanes = torch.randperm(6)
    permuted = observation._replace(
        agents=observation.agents[:, agents],
        agent_mask=observation.agent_mask[:, agents],
        lanes=observation.lanes[:, lanes],
        lane_mask=observation.lane_mask[:, lanes],
    )
    assert_output_close(model(observation), model(permuted), atol=1e-5, rtol=1e-5)


def test_empty_neighbors_and_map_have_finite_outputs_and_gradients():
    model = network().train()
    observation = random_observation()
    observation = observation._replace(
        agent_mask=torch.zeros_like(observation.agent_mask).index_fill(1, torch.tensor([0]), True),
        lane_mask=torch.zeros_like(observation.lane_mask),
    )
    output = model(observation)
    unit = torch.full_like(output.alpha, 0.3)
    loss = -beta_log_prob(output.alpha, output.beta, unit).sum() + output.value.square().sum()
    loss.backward()
    assert all(torch.isfinite(value).all() for value in output)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert model.default_lane.grad is not None and bool(model.default_lane.grad.abs().sum() > 0)


def test_beta_helpers_match_torch_and_stay_finite_at_bounds():
    alpha = torch.tensor([[1.5, 3.0], [2.0, 1.2]])
    beta = torch.tensor([[2.5, 1.1], [2.0, 4.0]])
    unit = torch.tensor([[0.2, 0.9], [0.5, 0.01]])
    reference = torch.distributions.Beta(alpha, beta)
    torch.testing.assert_close(beta_log_prob(alpha, beta, unit), reference.log_prob(unit).sum(-1))
    torch.testing.assert_close(beta_entropy(alpha, beta), reference.entropy().sum(-1))
    edges = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.isfinite(beta_log_prob(alpha, beta, edges)).all()
    torch.testing.assert_close(beta_mode(alpha, beta), (alpha - 1) / (alpha + beta - 2))
    one = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(beta_mode(one, one.flip(0)), one / 4)
    torch.manual_seed(0)
    samples = sample_beta(alpha.expand(1000, -1, -1), beta.expand(1000, -1, -1))
    assert bool(((samples > 0) & (samples < 1)).all())
    torch.testing.assert_close(samples.mean(0), reference.mean, atol=0.03, rtol=0)
    jerk, rate = decode_unit(torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), SMALL)
    torch.testing.assert_close(jerk, torch.tensor([-8.0, 5.0, -1.5]), atol=1e-4, rtol=0)
    torch.testing.assert_close(rate, torch.tensor([-0.8, 0.8, 0.0]), atol=1e-4, rtol=0)


def test_checkpoint_round_trip_is_strict_and_random_init_is_reproducible(tmp_path):
    loaded = random_single_frame(SMALL, seed=3)
    state = torch.get_rng_state()
    again = random_single_frame(SMALL, seed=3)
    assert torch.equal(state, torch.get_rng_state())
    observation = random_observation(agents=4, lanes=4)
    assert_output_close(loaded.model(observation), again.model(observation), atol=0, rtol=0)
    path = tmp_path / "policy.pt"
    save_single_frame(path, loaded.model, update=7)
    restored = load_single_frame(path)
    assert restored.config == SMALL and restored.update == 7
    assert_output_close(loaded.model(observation), restored.model(observation), atol=0, rtol=0)

    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["extra"] = torch.zeros(1)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="extra"):
        load_single_frame(path)
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="Not a SingleFrameNet"):
        load_single_frame(path)
    payload["config"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        SingleFrameConfig.from_dict(payload["config"])


def observe(batch, query_slot, *, blind=None, config=SMALL):
    env = LearnedEnv(random_single_frame(config))
    modes = torch.where(
        batch.actor_mask & (batch.actor_type == ActorType.VEHICLE),
        ControlType.SINGLE_FRAME,
        ControlType.LOG,
    )
    env.reset(batch, control_types=modes)
    goals = sample_goals(env.state, env.scene, env._cache, config)
    return build_observation(
        env.state,
        env.scene,
        env.episode.step,
        env.actor_mask,
        env.control_types,
        goals,
        torch.zeros_like(query_slot),
        query_slot,
        config,
        blind=blind,
    ), goals


def rotated_scene():
    # Ego heads north. A vehicle is 12 m ahead and a static object is 5 m ahead, 2 m right.
    return batch_of(
        traffic_scene(
            ((0.0, 0.0), (0.0, 12.0), (2.0, 5.0)),
            (4.0, 1.0, 0.0),
            (ActorType.VEHICLE, ActorType.VEHICLE, ActorType.STATIC),
            headings=[math.pi / 2, math.pi / 2, 0.0],
        )
    )


def test_observation_uses_query_rear_axle_frame_and_box_centers():
    batch = rotated_scene()
    observation, goals = observe(batch, torch.tensor([0]))
    agents = observation.agents[0]
    assert observation.agent_mask[0].tolist() == [True, True, True, False]
    # Self, then the reserved vehicle, then the static object; vehicles' centers are 1 m ahead.
    torch.testing.assert_close(agents[0, :2], torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(agents[1, :2], torch.tensor([13.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(agents[2, :2], torch.tensor([5.0, -2.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        agents[1, 2:6], torch.tensor([1.0, 0.0, 1.0, 0.0]), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(agents[2, 4:6], torch.tensor([0.0, -1.0]), atol=1e-5, rtol=0)
    assert agents[:3, 16:].tolist() == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    torch.testing.assert_close(agents[0, 8:10], torch.tensor([3.0, 1.0]), atol=1e-5, rtol=0)
    assert not agents[3].any()
    kinematics = observation.kinematics[0]
    torch.testing.assert_close(
        kinematics[:4], torch.tensor([4.0, 0.0, 4.0, 2.0]), atol=1e-5, rtol=0
    )
    expected_goal = goals[0, 0, -1] - torch.tensor([0.0, 0.0])
    torch.testing.assert_close(
        observation.goal[0], torch.stack((expected_goal[1], -expected_goal[0])), atol=1e-4, rtol=0
    )
    assert observation.goal[0, 0] > 0 and abs(float(observation.goal[0, 1])) < 1e-4


def test_blind_queries_keep_only_self_map_and_goal():
    batch = rotated_scene()
    blind = torch.zeros_like(batch.actor_mask)
    blind[0, 0] = True
    seen, _ = observe(batch, torch.tensor([0]))
    hidden, _ = observe(batch, torch.tensor([0]), blind=blind)
    assert hidden.agent_mask[0].tolist() == [True, False, False, False]
    assert not hidden.agents[0, 1:].any()
    torch.testing.assert_close(hidden.agents[0, 0], seen.agents[0, 0])
    torch.testing.assert_close(hidden.goal, seen.goal)
    torch.testing.assert_close(hidden.lanes, seen.lanes)


def test_lanes_are_local_typed_and_report_red_signals():
    batch = rotated_scene()
    t = torch.linspace(0, 1, CHUNK_POINTS)[:, None]
    chunks = torch.stack(
        (
            torch.tensor([0.0, 10.0]) + t * torch.tensor([0.0, 10.0]),
            torch.tensor([50.0, 0.0]) + t * torch.tensor([10.0, 0.0]),
        )
    )
    signal = torch.full((1, 2, batch.map.chunk_signal.shape[-1]), -1)
    signal[0, 1, 0] = 0  # the stop line touches signal 0, which is red at step 0
    scene_map = batch.map._replace(
        chunk_points=chunks[None],
        chunk_type=torch.tensor([[MapElementType.LANE_CENTERLINE, MapElementType.STOP_LINE]]),
        chunk_mask=torch.tensor([[True, True]]),
        chunk_signal=signal,
    )
    rails = batch.rails._replace(signal_red=batch.rails.signal_red.clone())
    rails.signal_red[0, 0, 0] = True
    batch = batch._replace(map=scene_map, rails=rails)
    config = SingleFrameConfig(embed_dim=16, max_agents=4, max_lanes=2)
    observation, _ = observe(batch, torch.tensor([0]), config=config)
    lanes = observation.lanes[0]
    points = 2 * CHUNK_POINTS
    assert observation.lane_mask[0].tolist() == [True, True]
    # The query faces +y, so the first chunk runs straight ahead from 10 m to 20 m.
    expected = torch.stack((10 + 10 * t[:, 0], torch.zeros(CHUNK_POINTS)), -1).flatten()
    torch.testing.assert_close(lanes[0, :points], expected, atol=1e-5, rtol=0)
    assert lanes[0, points + MapElementType.LANE_CENTERLINE] == 1
    assert lanes[1, points + MapElementType.STOP_LINE] == 1
    torch.testing.assert_close(lanes[:, points + 6], batch.rails.speed_limit.expand(2))
    assert lanes[:, points + 7].tolist() == [0.0, 1.0]


def env_scene():
    visible = torch.ones(8, 3, dtype=torch.bool)
    visible[:2, 1] = False
    visible[5:, 1] = False
    return batch_of(
        traffic_scene(
            ((0.0, 0.0), (12.0, 0.0), (5.0, 2.0)),
            (4.0, 1.0, 0.0),
            (ActorType.VEHICLE, ActorType.VEHICLE, ActorType.STATIC),
            frames=8,
            visibility=visible,
        )
    )


def test_selfplay_fullgraph_matches_eager_across_visibility_changes():
    batch = env_scene()
    loaded = random_single_frame(SMALL)
    eager, compiled = LearnedEnv(loaded), LearnedEnv(loaded)
    policy_graphs, step_graphs = [], []

    def backend(graphs):
        def compile_graph(graph, inputs):
            graphs.append(graph)
            return graph.forward

        return compile_graph

    compiled._policy_kernel = torch.compile(
        compiled.policy, fullgraph=True, dynamic=False, backend=backend(policy_graphs)
    )
    # Reset picks the step kernel for the controllers in use; compile what it picks.
    eager_kernel = compiled._kernel
    compiled._kernel = lambda **flags: torch.compile(
        eager_kernel(**flags), fullgraph=True, dynamic=False, backend=backend(step_graphs)
    )
    modes = torch.where(
        batch.actor_mask & (batch.actor_type == ActorType.VEHICLE),
        ControlType.SINGLE_FRAME,
        ControlType.LOG,
    )
    eager.reset(batch, control_types=modes)
    compiled.reset(batch, control_types=modes)
    start = compiled.state.position.clone()
    for _ in range(8):
        expected, actual = eager.step(), compiled.step()
        assert_output_close(expected.state, actual.state, atol=0, rtol=0)
        assert torch.equal(expected.actor_mask, actual.actor_mask)
    assert len(policy_graphs) == len(step_graphs) == 1
    assert eager.values.shape == (2,) and torch.isfinite(eager.values).all()
    # The controlled vehicle moves on its own and does not follow its log.
    assert not torch.allclose(
        compiled.state.position[0, 0], batch.reference_state.position[0, 7, 0]
    )
    assert bool(compiled.state.position[0, 0, 0] > start[0, 0, 0])


def test_policy_decides_every_step_by_default_and_holds_coarser_intervals():
    from dataclasses import replace

    def commands(config):
        env = LearnedEnv(random_single_frame(config))
        env.reset(env_scene())
        assert int(env._policy_gap[0]) == round(config.policy_interval_s / 0.1)
        values = []
        for _ in range(4):
            env.step()
            values.append(float(env.state.acceleration_command[0, 0]))
        return values

    # Default 10 Hz: a fresh target every 0.1 s step.
    every = commands(SMALL)
    assert len(set(every)) == 4
    # policy_interval_s = 0.2 at dt = 0.1: targets change on even steps only.
    held = commands(replace(SMALL, policy_interval_s=0.2))
    assert held[0] == held[1] and held[2] == held[3]


def test_api_runs_single_frame_presets_with_random_init_and_saved_checkpoints(tmp_path):
    scene = traffic_scene(((0.0, 0.0), (12.0, 0.0)), (4.0, 1.0), (0, 0), frames=6)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    app = create_app(
        artifact_roots=[path],
        include_demo=False,
        device="cpu",
        max_actors=4,
        max_static_objects=2,
        max_frames=6,
        max_map_segments=1,
    )
    client = TestClient(app)
    listing = client.get("/api/v1/rollout-sources").json()
    source = listing["sources"][0]["id"]
    random_entry = next(
        item for item in listing["policy_checkpoints"] if item["path"].startswith("random-init")
    )
    payload = {"source_ids": [source], "preset": "ego_single_frame"}
    response = client.post("/api/v1/rollouts", json={**payload, "checkpoint": random_entry["path"]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["policy"] == "single_frame" and data["checkpoint"] == "random-init:0"
    assert data["worlds"][0]["detail"]["actors"][0]["control_type"] == "single_frame"
    frames = data["worlds"][0]["frames"]
    assert len(frames) == 6 and frames[-1]["done"] and frames[0]["actors"][0]["goals"]

    checkpoint = tmp_path / "policy.pt"
    save_single_frame(checkpoint, random_single_frame(SingleFrameConfig(embed_dim=16)).model)
    response = client.post("/api/v1/rollouts", json={**payload, "checkpoint": str(checkpoint)})
    assert response.status_code == 200, response.text
    assert response.json()["checkpoint"] == str(checkpoint.resolve())


def test_rollout_pipeline_records_goals_for_single_frame_actors():
    batch = env_scene()
    trace = rollout_learned_batch(batch, loaded_policy=random_single_frame(SMALL), num_steps=7)
    assert trace.goals.shape == (8, 1, 4, 1, 2)
    assert trace.goal_mask[:, 0, 0].all()


def test_bf16_autocast_keeps_fp32_heads_and_matches_fp32_outputs():
    model = network()
    observation = random_observation()
    reference = model(observation)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = model(observation)
    assert all(value.dtype == torch.float32 for value in mixed)
    for expected, actual in zip(reference, mixed, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(observation)
    loss = -beta_log_prob(output.alpha, output.beta, torch.full_like(output.alpha, 0.4)).sum()
    (loss + output.value.square().sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_api_returns_the_observation_selection_of_a_single_frame_actor(tmp_path):
    from cs5478_project.data import make_synthetic_scene

    scene = make_synthetic_scene(num_frames=6, num_actors=6)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    app = create_app(
        artifact_roots=[path],
        include_demo=False,
        device="cpu",
        max_actors=8,
        max_static_objects=4,
        max_frames=6,
        max_map_segments=1,
    )
    client = TestClient(app)
    source = client.get("/api/v1/rollout-sources").json()["sources"][0]["id"]
    run = client.post(
        "/api/v1/rollouts",
        json={"source_ids": [source], "preset": "single_frame_selfplay"},
    ).json()
    actors = run["worlds"][0]["detail"]["actors"]
    ego = next(actor for actor in actors if actor["is_ego"])
    url = f"/api/v1/rollouts/{run['id']}/observations"
    track = client.get(url, params={"world": 0, "actor_id": ego["id"]}).json()
    # One request covers every frame; chunk geometry is listed once.
    assert len(track["frames"]) == len(run["worlds"][0]["frames"])
    view = track["frames"][3]
    assert view["agents"][0] == ego["id"]
    # Capacity is min(max_agents, dynamic + static slots) = min(96, 8 + 4).
    assert track["agent_capacity"] == 12
    known = {actor["id"] for actor in actors}
    assert set(view["agents"]) <= known and len(view["agents"]) > 1
    assert len(view["lanes"]) <= track["lane_capacity"]
    assert all(
        0 <= index < len(track["chunks"]) for frame in track["frames"] for index in frame["lanes"]
    )
    pedestrian = next(actor for actor in actors if actor["control_type"] == "pedestrian")
    params = {"world": 0, "actor_id": pedestrian["id"]}
    assert client.get(url, params=params).status_code == 422
    params = {"world": 0, "actor_id": ego["id"]}
    assert client.get("/api/v1/rollouts/stale/observations", params=params).status_code == 404


def test_lazy_batch_scores_each_world_and_serves_worlds_on_demand(tmp_path):
    root = tmp_path / "val14"
    root.mkdir()
    for index, gap in enumerate((12.0, 20.0)):
        save_scene(
            traffic_scene(((0.0, 0.0), (gap, 0.0)), (4.0, 1.0), (0, 0), frames=6),
            root / f"scene{index}.pt",
        )
    crowded = traffic_scene(tuple((6.0 * i, 0.0) for i in range(5)), (1.0,) * 5, (0,) * 5)
    save_scene(crowded, root / "crowded.pt")
    client = TestClient(
        create_app(
            artifact_roots=[root],
            device="cpu",
            max_actors=4,
            max_static_objects=2,
            max_frames=6,
            max_map_segments=1,
        )
    )
    sources = client.get("/api/v1/rollout-sources").json()["sources"]
    assert {source["collection"] for source in sources} == {"val14"}
    ids = [source["id"] for source in sources]
    for preset in ("ego_single_frame", "ego_single_frame_reactive", "log"):
        response = client.post(
            "/api/v1/rollouts", json={"source_ids": ids, "preset": preset, "lazy": True}
        )
        assert response.status_code == 200, response.text
        run = response.json()
        # The five-actor scene exceeds max_actors=4: skipped, not an error.
        assert run["worlds"] == [] and run["skipped"] == [crowded.scenario_id]
        assert [row["index"] for row in run["summaries"]] == [0, 1]
        for row in run["summaries"]:
            assert 0.0 <= row["metrics"]["score"] <= 1.0
            assert row["dynamic_actor_count"] == 2
        world = client.get(f"/api/v1/rollouts/{run['id']}/worlds/1").json()
        assert world["source_id"] == run["summaries"][1]["source_id"]
        assert len(world["frames"]) == 6 and world["frames"][-1]["done"]
        ego = next(actor["id"] for actor in world["detail"]["actors"] if actor["is_ego"])
        url = f"/api/v1/rollouts/{run['id']}/policy"
        policy = client.get(url, params={"world": 1, "actor_id": ego})
        if preset == "log":
            assert policy.status_code == 422
            continue
        track = policy.json()
        # One decision per transition; the last frame has none.
        assert len(track["alpha"]) == 6 and track["alpha"][-1] is None
        assert all(a > 0 for pair in track["alpha"][:-1] for a in pair)
        low, high = track["jerk_range"]
        assert all(low <= action[0] <= high for action in track["action"][:-1])
        assert len(track["rewards"]["reward"]) == 6 and track["value"][0] is not None
    # Log replay of the expert itself makes full progress.
    assert all(row["metrics"]["ego_progress_along_expert_route"] == 1.0 for row in run["summaries"])
    assert client.get(f"/api/v1/rollouts/{run['id']}/worlds/2").status_code == 422
    assert client.get("/api/v1/rollouts/missing/worlds/0").status_code == 404


def test_goal_overrides_fix_the_goal_of_chosen_actors_only():
    batch = env_scene()
    pinned = torch.zeros_like(batch.actor_mask)
    pinned[0, 0] = True
    points = torch.zeros((*pinned.shape, 2))
    points[0, 0] = torch.tensor([30.0, -4.0])
    free = rollout_learned_batch(batch, loaded_policy=random_single_frame(SMALL), num_steps=7)
    fixed = rollout_learned_batch(
        batch,
        loaded_policy=random_single_frame(SMALL),
        num_steps=7,
        goal_overrides=(pinned, points),
    )
    assert fixed.goal_mask[:, 0, 0].all()
    assert (fixed.goals[:, 0, 0] == torch.tensor([30.0, -4.0])).all()
    assert not (free.goals[:, 0, 0] == torch.tensor([30.0, -4.0])).all()
    torch.testing.assert_close(fixed.goals[0, 0, 1:], free.goals[0, 0, 1:])


def test_api_applies_a_fixed_goal_and_reports_it(tmp_path):
    scene = traffic_scene(((0.0, 0.0), (12.0, 0.0)), (4.0, 1.0), (0, 0), frames=6)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    client = TestClient(
        create_app(
            artifact_roots=[path],
            include_demo=False,
            device="cpu",
            max_actors=4,
            max_static_objects=2,
            max_frames=6,
            max_map_segments=1,
        )
    )
    source = client.get("/api/v1/rollout-sources").json()["sources"][0]["id"]
    payload = {"source_ids": [source], "preset": "ego_single_frame", "checkpoint": "random-init:0"}
    world = client.post("/api/v1/rollouts", json=payload).json()["worlds"][0]
    ego = next(actor for actor in world["detail"]["actors"] if actor["is_ego"])
    assert ego["goal_override"] is None
    override = {
        "source_id": source,
        "actor_id": ego["id"],
        "control_type": "single_frame",
        "goal": [25.0, 3.0],
    }
    response = client.post("/api/v1/rollouts", json={**payload, "agent_controls": [override]})
    assert response.status_code == 200, response.text
    world = response.json()["worlds"][0]
    ego = next(actor for actor in world["detail"]["actors"] if actor["is_ego"])
    assert ego["goal_override"] == [25.0, 3.0]
    for frame in world["frames"]:
        state = next(item for item in frame["actors"] if item["id"] == ego["id"])
        assert state["goals"] == [[25.0, 3.0]]
    # A fixed goal on an actor the policy does not drive is ignored.
    other = next(a for a in world["detail"]["actors"] if not a["is_ego"] and a["type"] != "static")
    ignored = {**override, "actor_id": other["id"], "control_type": "log"}
    world = client.post("/api/v1/rollouts", json={**payload, "agent_controls": [ignored]}).json()[
        "worlds"
    ][0]
    assert all(actor["goal_override"] is None for actor in world["detail"]["actors"])
