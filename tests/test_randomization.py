"""Domain randomization: reward weights the policy observes, blind actors, sizes, starts."""

import math

import torch
from test_rule_agents import batch_of, traffic_scene
from test_training import SMALL

from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.observation import DEFAULT_REWARD_CONDITIONS
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.models.checkpoint import random_single_frame
from cs5478_project.models.config import SingleFrameConfig
from cs5478_project.training.config import PPOConfig
from cs5478_project.training.ppo import PPO
from cs5478_project.training.randomization import (
    conditions,
    default_weights,
    randomize,
)
from cs5478_project.training.rollout import PPOEnv, _start, _step, collect, ego_controls


def scene():
    return traffic_scene(
        ((0.0, 0.0), (20.0, 0.0), (40.0, 0.0)),
        (5.0, 5.0, 5.0),
        (ActorType.VEHICLE,) * 3,
        frames=12,
    )


def test_default_weights_are_the_default_conditions_and_reward():
    defaults = default_weights((2, 3), "cpu")
    assert torch.allclose(conditions(defaults), torch.tensor(DEFAULT_REWARD_CONDITIONS))
    batch = batch_of(scene())
    loaded = random_single_frame(SMALL)
    totals = []
    for weights in (None, default_weights(batch.actor_mask.shape, "cpu")):
        env, evaluator = PPOEnv(loaded, amp=False, stochastic=False), DrivingEvaluator()
        _start(env, evaluator, batch, ego_controls(batch), weights=weights)
        totals.append(torch.stack([_step(env, evaluator)[1].reward for _ in range(11)]))
    assert torch.equal(totals[0], totals[1])


def test_randomize_perturbs_within_the_configured_ranges():
    batch = batch_of(scene(), scene())
    controls = ego_controls(batch)
    controls[:, 1] = ControlType.SINGLE_FRAME  # one self-play vehicle per world
    torch.manual_seed(0)
    randomized, blind, weights = randomize(batch, controls, blind_probability=0.5)
    scale = randomized.size / batch.size
    real = batch.actor_mask
    assert bool(((scale[real] >= 0.9) & (scale[real] <= 1.1)).all())
    assert not blind[torch.arange(2), batch.ego_index].any()
    driven = controls == ControlType.SINGLE_FRAME
    moved = (randomized.initial_state.position - batch.initial_state.position).norm(dim=-1)
    assert bool((moved[driven] > 0).all()) and bool((moved[driven] < 3 * 0.56).all())
    assert bool((moved[~driven] == 0).all())
    turned = (randomized.initial_state.heading - batch.initial_state.heading).abs()
    assert bool((turned[driven] <= math.radians(6) + 1e-6).all())
    assert bool(((weights.collision >= -5) & (weights.collision <= 0)).all())
    assert bool(((weights.goal_threshold >= 1.5) & (weights.goal_threshold <= 3)).all())


def test_a_conditioned_policy_trains_on_randomized_episodes():
    torch.manual_seed(0)
    batch = batch_of(scene(), scene())
    config = SingleFrameConfig(**{**SMALL.__dict__, "reward_conditions": True})
    loaded = random_single_frame(config)
    controls = ego_controls(batch)
    randomized, blind, weights = randomize(batch, controls)
    env = PPOEnv(loaded, amp=False)
    rollout, _ = collect(env, DrivingEvaluator(), randomized, controls, blind, weights)
    width = rollout.observation.kinematics.shape[-1]
    # Own state (7), the previous action and command (3), then the conditions.
    assert width == 7 + 3 + len(DEFAULT_REWARD_CONDITIONS)
    # The observed conditions are the sampled weights of each query.
    expected = conditions(weights)[env._query_world, env._query_slot]
    torch.testing.assert_close(rollout.observation.kinematics[0, :, 10:], expected)
    ppo_config = PPOConfig(worlds=2, minibatch_size=8, microbatch_size=8, compile=False, amp=False)
    stats = PPO(loaded.model, ppo_config, total_updates=1).update(rollout)
    assert all(isinstance(v, (int, float)) for v in stats.values())
    # Without weights (evaluation) the policy sees the default conditions.
    env, evaluator = PPOEnv(loaded, amp=False, stochastic=False), DrivingEvaluator()
    _start(env, evaluator, batch, controls)
    observation, _ = env._forward()
    torch.testing.assert_close(
        observation.kinematics[:, 10:], torch.tensor(DEFAULT_REWARD_CONDITIONS).expand(2, -1)
    )
