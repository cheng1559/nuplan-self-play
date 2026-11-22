"""Collect PPO rollouts and run closed-loop evaluations with a SingleFrameNet ego.

``PPOEnv`` is a ``LearnedEnv`` whose SingleFrameNet actors sample their Beta policy
(or take its mode for evaluation) and keep the observation, normalized action,
log-probability, and value of the last decision. Rewards, terminations, and metrics
come from ``DrivingEvaluator``.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.agents.goals import sample_training_goals
from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.observation import Observation
from cs5478_project.env.learned import LearnedEnv
from cs5478_project.env.observations import build_observation
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.models.distributions import (
    beta_log_prob,
    beta_mean,
    beta_mode,
    decode_unit,
    sample_beta,
)
from cs5478_project.training.randomization import conditions

# Reach of a goal fixed by hand: the reward's default goal_threshold, in meters.


class PPOEnv(LearnedEnv):
    """SingleFrameNet actors decide every ``decision_steps`` simulation steps (the
    checkpoint's policy interval) and hold their targets in between; ``record`` holds the
    last decision and ``decided`` says whether the latest step made one."""

    def __init__(
        self,
        loaded,
        *,
        compile_step=False,
        amp=True,
        stochastic=True,
        fixed_goals=None,
        deterministic="mean",
    ):
        # Without sampling, act with the Beta mode (the report's evaluation) or its mean.
        # Training fixes log goals per episode (sample_training_goals); evaluation moves
        # the goal along the route every step.
        super().__init__(
            loaded,
            compile_step=compile_step,
            fixed_goals=stochastic if fixed_goals is None else fixed_goals,
        )
        if self.learned_type != ControlType.SINGLE_FRAME:
            raise ValueError("PPO training needs a SingleFrameNet")
        self.stochastic = stochastic
        if deterministic not in ("mean", "mode"):
            raise ValueError("deterministic must be 'mean' or 'mode'")
        self._deterministic = beta_mean if deterministic == "mean" else beta_mode
        self.amp = amp
        # dynamic=None: the policy-actor count varies by batch in self-play; after one
        # recompile the kernel accepts any count (fixed shapes recompiled every update).
        self._decide = (
            torch.compile(self._observe, fullgraph=True, dynamic=None)
            if compile_step
            else self._observe
        )
        self.record = None
        # The last decision's (alpha, beta, unit, value) [Q, ...] (the visualizer reads it).
        self.output = None
        # Reward-weight conditions [Q, C] of the current episode (None: the defaults).
        self.conditions = None
        self.decided = False
        self.decision_steps = 1
        self._tick = 0

    @torch.no_grad()
    def reset(self, scene, mask=None, *, control_types=None, blind=None):
        if mask is not None:
            raise ValueError("PPOEnv supports full resets only")
        result = super().reset(scene, mask, control_types=control_types, blind=blind)
        gaps = self._policy_gap
        if int(gaps.min()) != int(gaps.max()):
            raise ValueError("PPOEnv needs one policy interval (in steps) across worlds")
        self.decision_steps, self._tick = int(gaps[0]), 0
        if self._fixed_goals:
            # Training goals: a point of the actor's own logged future, sometimes snapped
            # to a nearby lane centerline, fixed for the episode.
            goals = sample_training_goals(self.scene)
            if mask is not None:
                goals = torch.where(mask[:, None, None, None], goals, self._goals)
            self._goals = goals
        return result

    def _observe(
        self, state, scene, step, actor_mask, control_types, goals, query_world, query_slot, blind,
        conditions, last_action,
    ):  # fmt: skip
        config = self.policy.config
        observation = build_observation(
            state, scene, step, actor_mask, control_types, goals, query_world, query_slot,
            config, blind=blind, conditions=conditions, last_action=last_action,
        )  # fmt: skip
        with torch.autocast(state.position.device.type, torch.bfloat16, enabled=self.amp):
            output = self.policy.model(observation)
        return observation, output

    def _forward(self):
        return self._decide(
            self.state, self.scene, self.episode.step, self.actor_mask, self.control_types,
            self.goals, self._query_world, self._query_slot, self.blind, self.conditions,
            self._last_action,
        )  # fmt: skip

    def _act(self):
        # Between decisions the held targets apply (LearnedEnv masks new actions), so
        # skip the network.
        self.decided = self._tick % self.decision_steps == 0
        self._tick += 1
        if not self.decided:
            return self._held
        observation, output = self._forward()
        alpha, beta = output.alpha.float(), output.beta.float()
        unit = sample_beta(alpha, beta) if self.stochastic else self._deterministic(alpha, beta)
        value = output.value_parts.float()
        self.record = (observation, unit, beta_log_prob(alpha, beta, unit), value)
        self.output = (alpha, beta, unit, value.sum(-1))
        jerk, steering_rate = decode_unit(unit, self.policy.config)
        shape = self.scene.actor_mask.shape
        index = self._query_world * shape[1] + self._query_slot
        dense = jerk.new_zeros((shape[0] * shape[1], 2)).index_copy(
            0, index, torch.stack((jerk, steering_rate), -1)
        )
        dense = dense.reshape(*shape, 2)
        self._held = VehicleAction(dense[..., 0], dense[..., 1]), value.sum(-1)
        return self._held

    @torch.no_grad()
    def value(self) -> Tensor:
        """Value parts [Q, C] of the current state (the bootstrap at time limits)."""
        return self._forward()[1].value_parts.float()


class Rollout(NamedTuple):
    """[T, Q] per decision and query, except the observation's own trailing axes; value
    and reward add the reward-component axis C (see reward_parts)."""

    observation: Observation
    unit: Tensor
    log_prob: Tensor
    value: Tensor
    reward: Tensor
    done: Tensor
    valid: Tensor
    last_value: Tensor  # [Q, C] value after the last step, zero once terminated


def reward_parts(reward, qw, qs, components: int = 1) -> Tensor:
    """[Q, 1] step reward of the query actors (one reward component)."""
    return reward.reward[qw, qs][:, None]


def ego_controls(scene):
    """Non-reactive control: ego is the SingleFrameNet, everyone else replays the log."""
    controls = torch.zeros_like(scene.actor_type)
    world = torch.arange(controls.shape[0], device=controls.device)
    controls[world, scene.ego_index] = ControlType.SINGLE_FRAME
    return controls


def _start(env, evaluator, scene, controls, blind=None, weights=None):
    """Reset both; ``blind`` [B, A] and reward ``weights`` (RewardWeights of [B, A]) come
    from domain randomization, and the policy observes the weights."""
    result = env.reset(scene, control_types=controls, blind=blind)
    env.conditions = (
        None if weights is None else conditions(weights)[env._query_world, env._query_slot]
    )
    # With fixed goals, the goal reward targets the same point the policy observes.
    goals = env.goals if env._fixed_goals else None
    evaluator.reset(
        env.scene, env.state, env.episode.step, result.actor_mask, result.static_mask,
        env.control_types, env.control_types == env.learned_type, goals, weights,
    )  # fmt: skip


def _step(env, evaluator):
    result = env.step()
    reward = evaluator.step(
        env.state, env.episode.step, result.actor_mask, result.static_mask,
        env.control_types, result.advanced,
    )  # fmt: skip
    return result, reward


@torch.no_grad()
def collect(
    env: PPOEnv, evaluator: DrivingEvaluator, scene, controls=None, blind=None, weights=None
):
    """Run one full episode on every world; return the rollout and its metrics.

    One sample per decision: its reward sums the simulation steps until the next one.
    """
    controls = ego_controls(scene) if controls is None else controls
    _start(env, evaluator, scene, controls, blind, weights)
    qw, qs = env._query_world, env._query_slot
    steps = int(scene.num_frames.max()) - 1
    gap = env.decision_steps
    decisions = -(-steps // gap)
    terminated = torch.zeros_like(qw, dtype=torch.bool)
    store = None
    for t in range(steps):
        result, reward = _step(env, evaluator)
        k = t // gap
        if env.decided:
            observation, unit, log_prob, value = env.record
            if store is None:
                store = {
                    "observation": [
                        torch.empty((decisions, *x.shape), dtype=x.dtype, device=x.device)
                        for x in observation
                    ],
                    **{name: torch.empty((decisions, *x.shape), dtype=x.dtype, device=x.device)
                       for name, x in (("unit", unit), ("log_prob", log_prob), ("value", value))},
                    "reward": torch.zeros((decisions, *value.shape), device=qw.device),
                    **{name: torch.zeros((decisions, qw.shape[0]), dtype=torch.bool,
                                         device=qw.device) for name in ("done", "valid")},
                }  # fmt: skip
            for buffer, x in zip(store["observation"], observation, strict=True):
                buffer[k] = x
            store["unit"][k], store["log_prob"][k], store["value"][k] = unit, log_prob, value
            store["valid"][k] = ~terminated & result.advanced[qw]
        done = reward.done[qw, qs]
        store["reward"][k] += reward_parts(reward, qw, qs, store["reward"].shape[-1])
        store["done"][k] |= done
        terminated |= done
    last_value = torch.where(terminated[:, None], 0.0, env.value())
    rollout = Rollout(
        Observation(*store["observation"]),
        store["unit"],
        store["log_prob"],
        store["value"],
        store["reward"],
        store["done"],
        store["valid"],
        last_value,
    )
    return rollout, evaluator.metrics()


@torch.no_grad()
def evaluate(env: PPOEnv, evaluator: DrivingEvaluator, scene, controls=None, conditions=None):
    """Deterministic closed-loop episode; returns the evaluator's metrics.

    ``conditions`` [REWARD_CONDITIONS] replaces the default reward weights that a policy
    trained with ``reward_conditions`` observes (the metrics do not depend on them)."""
    _start(env, evaluator, scene, ego_controls(scene) if controls is None else controls)
    if conditions is not None:
        env.conditions = conditions.expand(env._query_world.shape[0], -1)
    for _ in range(int(scene.num_frames.max()) - 1):
        _step(env, evaluator)
    return evaluator.metrics()
