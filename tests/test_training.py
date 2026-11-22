"""PPO pieces: advantage estimation and one rollout plus update on small scenes."""

import json

import pytest
import torch
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.datatypes.observation import Observation
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.models.checkpoint import random_single_frame
from cs5478_project.models.config import SingleFrameConfig
from cs5478_project.training.config import PPOConfig
from cs5478_project.training.ppo import PPO, advantages
from cs5478_project.training.rollout import (
    PPOEnv,
    Rollout,
    _start,
    _step,
    collect,
    ego_controls,
    evaluate,
)

SMALL = SingleFrameConfig(embed_dim=16, num_heads=2, max_agents=4, max_lanes=4)


def test_gae_stops_at_done_and_bootstraps_at_the_time_limit():
    reward = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    value = torch.tensor([[0.5, 0.2], [0.5, 0.2], [0.5, 0.2]])
    done = torch.tensor([[False, False], [False, True], [False, False]])
    empty = torch.zeros(3, 2)
    rollout = Rollout(
        None,
        empty,
        empty,
        value[..., None],
        reward[..., None],
        done,
        done.logical_not(),
        torch.tensor([[2.0], [9.0]]),
    )
    gamma, lam = 0.9, 0.8
    advantage, target = advantages(rollout, gamma, lam)
    # Query 0 never ends: plain GAE with the last value as bootstrap.
    deltas = [1 + gamma * 0.5 - 0.5, 1 + gamma * 0.5 - 0.5, 1 + gamma * 2.0 - 0.5]
    expected = [0.0] * 3
    running = 0.0
    for t in reversed(range(3)):
        running = deltas[t] + gamma * lam * running
        expected[t] = running
    torch.testing.assert_close(advantage[:, 0], torch.tensor(expected))
    # Query 1 ends at t=1: nothing after it reaches t=0 or t=1.
    torch.testing.assert_close(advantage[1, 1], torch.tensor(-0.2))
    torch.testing.assert_close(
        advantage[0, 1], torch.tensor(gamma * 0.2 - 0.2 + gamma * lam * -0.2)
    )
    torch.testing.assert_close(target[..., 0], advantage + value)


def test_rollout_and_update_on_small_scenes():
    torch.manual_seed(0)
    scenes = [
        traffic_scene(((0.0, 0.0), (20.0, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12)
        for _ in range(2)
    ]
    batch = batch_of(*scenes)
    loaded = random_single_frame(SMALL)
    env, evaluator = PPOEnv(loaded, amp=False), DrivingEvaluator()
    rollout, metrics = collect(env, evaluator, batch)
    assert isinstance(rollout.observation, Observation)
    assert rollout.valid.shape == (11, 2) and rollout.reward.shape == (11, 2, 1)
    assert rollout.observation.lanes.shape[:2] == (11, 2)
    assert bool(rollout.valid.all()) and torch.isfinite(rollout.log_prob).all()
    assert metrics.score.shape == (2,)
    config = PPOConfig(worlds=2, minibatch_size=8, microbatch_size=4, compile=False, amp=False)
    before = [p.detach().clone() for p in loaded.model.parameters()]
    stats = PPO(loaded.model, config, total_updates=1).update(rollout)
    # Four epochs of two minibatches of 8 out of 22 samples.
    assert stats["samples"] == 22 and stats["minibatches"] == 4 * 2 and stats["epochs"] == 4
    assert all(isinstance(v, (int, float)) for v in stats.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(before, loaded.model.parameters(), strict=True)
    )
    # Deterministic evaluation reuses the same environment code path.
    deterministic = PPOEnv(loaded, amp=False, stochastic=False)
    first = evaluate(deterministic, DrivingEvaluator(), batch)
    second = evaluate(deterministic, DrivingEvaluator(), batch)
    torch.testing.assert_close(first.score, second.score)


def test_training_goals_stay_fixed_and_evaluation_goals_move_along_the_route():
    scene = traffic_scene(((0.0, 0.0),), (5.0,), (ActorType.VEHICLE,), frames=12)
    batch = batch_of(scene)
    loaded = random_single_frame(SMALL)
    training, evaluation = PPOEnv(loaded, amp=False), PPOEnv(loaded, amp=False, stochastic=False)
    evaluator = DrivingEvaluator()
    collect(training, evaluator, batch)
    # The goal reward targets the fixed goal the policy observes.
    torch.testing.assert_close(evaluator._goal, training.goals[0, :1])
    first = training.goals.clone()
    training.step()
    torch.testing.assert_close(training.goals, first)
    evaluation.reset(batch, control_types=ego_controls(batch))
    before = evaluation.goals.clone()
    evaluation.step()
    assert not torch.equal(evaluation.goals, before)


def test_microbatches_and_merged_rollouts_match_one_large_minibatch():
    scenes = [
        traffic_scene(((0.0, 0.0), (20.0, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12)
        for _ in range(2)
    ]
    batch = batch_of(*scenes)
    torch.manual_seed(0)
    loaded = random_single_frame(SMALL)
    rollouts = [collect(PPOEnv(loaded, amp=False), DrivingEvaluator(), batch)[0] for _ in range(2)]
    results = []
    for micro in (16, 4):
        model = random_single_frame(SMALL).model
        model.load_state_dict(loaded.model.state_dict())
        config = PPOConfig(
            minibatch_size=16, microbatch_size=micro, optimization_epochs=1, compile=False,
            amp=False, normalize_advantages=False,
        )  # fmt: skip
        torch.manual_seed(1)
        PPO(model, config, total_updates=1).update(rollouts)
        results.append([p.detach().clone() for p in model.parameters()])
    for a, b in zip(*results, strict=True):
        # Summation order differs; Adam rescales near-zero gradients, so allow float noise
        # well below one step (lr 3e-4).
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_a_slower_policy_records_one_sample_per_decision():
    scene = traffic_scene(
        ((0.0, 0.0), (20.0, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12
    )
    batch = batch_of(scene)
    loaded = random_single_frame(SingleFrameConfig(**{**SMALL.__dict__, "policy_interval_s": 0.2}))
    env = PPOEnv(loaded, amp=False, stochastic=False)
    rollout, _ = collect(env, DrivingEvaluator(), batch)
    # 11 steps at 10 Hz, deciding every second step: 6 decisions, the last holding one step.
    assert env.decision_steps == 2
    assert rollout.valid.shape == (6, 1) and rollout.reward.shape == (6, 1, 1)
    assert rollout.observation.lanes.shape[:2] == (6, 1)
    # Each sample's reward sums its steps: the total matches per-step scoring.
    env, evaluator = PPOEnv(loaded, amp=False, stochastic=False), DrivingEvaluator()
    _start(env, evaluator, batch, ego_controls(batch))
    rewards = [float(_step(env, evaluator)[1].reward[0, 0]) for _ in range(11)]
    expected = torch.tensor([sum(rewards[k : k + 2]) for k in range(0, 11, 2)])
    torch.testing.assert_close(rollout.reward[:, 0, 0], expected)


def test_mixed_worlds_train_every_policy_vehicle():
    from cs5478_project.training.mixing import WorldMode, mixed_controls

    torch.manual_seed(0)
    scene = traffic_scene(
        ((0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (10.0, 8.0)),
        (5.0, 5.0, 5.0, 1.0),
        (ActorType.VEHICLE,) * 3 + (ActorType.PEDESTRIAN,),
        frames=12,
    )
    batch = batch_of(scene, scene, scene)
    modes = torch.tensor([WorldMode.LOG, WorldMode.REACTIVE, WorldMode.SELF_PLAY])
    controls = mixed_controls(batch, modes, self_play_agents=2)
    loaded = random_single_frame(SMALL)
    rollout, metrics = collect(PPOEnv(loaded, amp=False), DrivingEvaluator(), batch, controls)
    # One ego per world plus two self-play vehicles in the last world.
    assert rollout.valid.shape == (11, 5) and metrics.score.shape == (5,)
    config = PPOConfig(worlds=3, minibatch_size=8, microbatch_size=8, compile=False, amp=False)
    stats = PPO(loaded.model, config, total_updates=1).update(rollout)
    assert all(isinstance(v, (int, float)) for v in stats.values())


def test_training_goals_are_log_points_in_time_order():
    from cs5478_project.agents.goals import sample_training_goals

    torch.manual_seed(0)
    batch = batch_of(traffic_scene(((0.0, 0.0),), (5.0,), (ActorType.VEHICLE,), frames=12))
    log = batch.reference_state.position[0, 1:, 0]  # the actor's future positions
    for count in (1, 2):
        goals = sample_training_goals(batch, count)
        assert goals.shape == (1, batch.actor_mask.shape[1], count, 2)
        for goal in goals[0, 0]:  # no lane to snap to: each is a logged position
            assert bool(((log - goal).norm(dim=-1) < 1e-5).any())
        # In time order, so the x coordinates of this straight drive never decrease.
        assert bool((goals[0, 0, 1:, 0] >= goals[0, 0, :-1, 0]).all())


def test_policy_observes_its_previous_action():
    from cs5478_project.datatypes.action import VehicleAction

    batch = batch_of(traffic_scene(((0.0, 0.0),), (5.0,), (ActorType.VEHICLE,), frames=12))
    config = SingleFrameConfig(**{**SMALL.__dict__, "last_action_input": True})
    env = PPOEnv(random_single_frame(config), amp=False, stochastic=False)
    env.reset(batch, control_types=ego_controls(batch))
    observation, _ = env._forward()
    assert observation.kinematics.shape[-1] == 10
    torch.testing.assert_close(observation.kinematics[0, 7:9], torch.tensor([2 * 8 / 13 - 1, 0.0]))
    held = torch.full_like(batch.actor_mask, 0.0, dtype=torch.float32)
    env._last_action = VehicleAction(held + config.max_jerk, held - config.max_steering_rate)
    observation, _ = env._forward()
    torch.testing.assert_close(observation.kinematics[0, 7:9], torch.tensor([1.0, -1.0]))


def test_sim_train_continues_at_start_epoch(tmp_path):
    from cs5478_project.data import save_scene
    from cs5478_project.training.train import main

    data = tmp_path / "scenes"
    data.mkdir()
    for index in range(4):
        scene = traffic_scene(
            ((0.0, 0.0), (20.0 + index, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12
        )
        save_scene(scene, data / f"scene{index}.pt")
    common = ["--train", str(data), "--eval", f"set={data}", "--device", "cpu",
              "--worlds", "2", "--epochs", "3", "--minibatch-size", "8",
              "--microbatch-size", "8", "--eval-worlds", "2", "--no-compile", "--no-amp",
              "--no-cache-scenes", "--loader-workers", "0", "--no-prefetch"]  # fmt: skip
    small = ["--model", "embed_dim=16", "--model", "num_heads=2", "--model", "max_agents=4",
             "--model", "max_lanes=4"]  # fmt: skip
    first, second = tmp_path / "first", tmp_path / "second"
    main([*common, "--output", str(first), *small])
    main([*common, "--output", str(second), "--start-epoch", "3",
          "--init", str(first / "checkpoints" / "epoch_002.pt")])  # fmt: skip

    def records(run):
        lines = (run / "metrics.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    full, continued = records(first), records(second)
    train = [r for r in continued if r["kind"] == "train"]
    # Two updates per epoch: the continued run makes epoch 3's updates 5 and 6 only.
    assert [r["step"] for r in train] == [5, 6]
    assert {r["epoch"] for r in continued if r["kind"] == "eval"} == {3}
    rates = {r["step"]: r["learning_rate"] for r in full if r["kind"] == "train"}
    assert [r["learning_rate"] for r in train] == pytest.approx([rates[5], rates[6]])
    assert (second / "checkpoints" / "epoch_003.pt").exists()


def test_sim_train_evaluates_every_n_updates(tmp_path):
    from cs5478_project.data import save_scene
    from cs5478_project.training.train import main

    data = tmp_path / "scenes"
    data.mkdir()
    for index in range(4):
        scene = traffic_scene(
            ((0.0, 0.0), (20.0 + index, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12
        )
        save_scene(scene, data / f"scene{index}.pt")
    run = tmp_path / "run"
    main(["--train", str(data), "--eval", f"set={data}", "--device", "cpu", "--output", str(run),
          "--worlds", "2", "--epochs", "2", "--eval-updates", "3", "--minibatch-size", "8",
          "--microbatch-size", "8", "--eval-worlds", "2", "--no-compile", "--no-amp",
          "--no-cache-scenes", "--loader-workers", "0", "--no-prefetch",
          "--model", "embed_dim=16", "--model", "num_heads=2", "--model", "max_agents=4",
          "--model", "max_lanes=4"])  # fmt: skip
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    # Two updates per epoch: evaluated before training, after update 3, and at the end (4).
    assert [r["update"] for r in records if r["kind"] == "eval"] == [0, 3, 4]
    names = sorted(path.name for path in (run / "checkpoints").iterdir())
    assert names == ["epoch_002.pt", "update_00003.pt"]
