"""Capture complete LogAgent batches from actual environment transitions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from cs5478_project.datatypes import SceneBatch, WorldState
from cs5478_project.env import ReplayEnv, RuleBasedEnv


@dataclass(frozen=True)
class BatchTrace:
    """CPU snapshots with time first, including each world's initial state."""

    state: WorldState
    step: Tensor
    done: Tensor
    actor_mask: Tensor
    goals: Tensor | None = None
    goal_mask: Tensor | None = None
    # The ego's episode scores when an evaluator ran alongside (Metrics of [Q] tensors).
    metrics: object | None = None
    # PPOEnv decisions per frame (the action applied from that frame): alpha, beta, unit
    # [F, Q, 2] and value [F, Q] of the query actors query_world/query_slot [Q]; NaN where
    # none. ``rewards``: each REWARD_FIELDS term [F, B] of every world's ego, per step.
    policy: dict | None = None
    rewards: dict | None = None
    # Evaluator events of each world's ego per frame: name -> bool [F, B].
    events: dict | None = None


REWARD_FIELDS = (
    "reward", "hard", "goal", "ttc", "speed", "comfort", "centerline", "direction", "pace",
)  # fmt: skip


@torch.inference_mode()
def rollout_log_batch(
    scene: SceneBatch, *, num_steps: int, compile_step: bool = False, evaluator=None
) -> BatchTrace:
    """Run one fixed-capacity batch and transfer its trace only after completion.

    The caller supplies the longest input horizon minus one from CPU metadata.
    This keeps control flow independent of device-side completion reads. Shorter
    worlds freeze in the environment and are trimmed when forming responses.
    """
    return _rollout(scene, num_steps, ReplayEnv(compile_step=compile_step), evaluator=evaluator)


@torch.inference_mode()
def rollout_rule_based_batch(
    scene: SceneBatch,
    *,
    num_steps: int,
    compile_step: bool = False,
    control_ego: bool = True,
    control_types: Tensor | None = None,
    blind: Tensor | None = None,
    evaluator=None,
) -> BatchTrace:
    """Capture reactive IDM vehicles and rule pedestrians with the same trace contract."""
    return _rollout(
        scene,
        num_steps,
        RuleBasedEnv(
            compile_step=compile_step,
            control_ego=control_ego,
        ),
        control_types,
        blind,
        evaluator,
    )


def _rollout(
    scene: SceneBatch,
    num_steps: int,
    env: ReplayEnv,
    control_types: Tensor | None = None,
    blind: Tensor | None = None,
    evaluator=None,
) -> BatchTrace:
    """``evaluator`` (a DrivingEvaluator) scores each world's ego along the way."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    result = (
        env.reset(scene, control_types=control_types, blind=blind)
        if isinstance(env, RuleBasedEnv)
        else env.reset(scene)
    )
    frames = num_steps + 1
    states = WorldState(*(value.new_empty((frames, *value.shape)) for value in result.state))
    steps = result.episode.step.new_empty((frames, *result.episode.step.shape))
    dones = result.episode.done.new_empty((frames, *result.episode.done.shape))
    actor_masks = result.actor_mask.new_empty((frames, *result.actor_mask.shape))
    goals = env.goals.new_empty((frames, *env.goals.shape)) if hasattr(env, "goals") else None
    goal_masks = (
        env.goal_mask.new_empty((frames, *env.goal_mask.shape)) if goals is not None else None
    )

    def capture(index: int) -> None:
        for destination, value in zip(states, result.state, strict=True):
            destination[index].copy_(value)
        steps[index].copy_(result.episode.step)
        dones[index].copy_(result.episode.done)
        actor_masks[index].copy_(result.actor_mask)
        if goals is not None:
            goals[index].copy_(env.goals)
            goal_masks[index].copy_(env.goal_mask)

    controls = getattr(env, "control_types", None)
    if controls is None:  # log replay
        controls = torch.zeros_like(scene.actor_type)
    if evaluator is not None:
        world = torch.arange(scene.actor_mask.shape[0], device=scene.actor_mask.device)
        ego = torch.zeros_like(scene.actor_mask)
        ego[world, scene.ego_index] = True
        evaluator.reset(
            env.scene, env.state, env.episode.step, result.actor_mask, result.static_mask,
            controls, ego & scene.actor_mask,
        )  # fmt: skip
    policy, rewards, events = None, None, None
    worlds = torch.arange(scene.actor_mask.shape[0], device=scene.actor_mask.device)
    capture(0)
    for index in range(1, frames):
        result = env.step()
        if evaluator is not None:
            reward = evaluator.step(
                env.state, env.episode.step, result.actor_mask, result.static_mask, controls,
                result.advanced,
            )  # fmt: skip
            if rewards is None:
                rewards = {
                    name: torch.full((frames, worlds.shape[0]), torch.nan, device=worlds.device)
                    for name in REWARD_FIELDS
                }
            for name in REWARD_FIELDS:
                rewards[name][index - 1] = getattr(reward, name)[worlds, scene.ego_index]
            if events is None:
                events = {
                    name: torch.zeros((frames, worlds.shape[0]), dtype=torch.bool,
                                      device=worlds.device)
                    for name in evaluator.events
                }  # fmt: skip
            for name, flags in evaluator.events.items():
                events[name][index, evaluator.query_world] = flags
        output = getattr(env, "output", None)
        if output is not None:
            if policy is None:
                policy = {
                    name: torch.full((frames, *value.shape), torch.nan, device=value.device)
                    for name, value in zip(("alpha", "beta", "unit", "value"), output, strict=True)
                }
                policy["query_world"], policy["query_slot"] = env._query_world, env._query_slot
            for name, value in zip(("alpha", "beta", "unit", "value"), output, strict=True):
                policy[name][index - 1] = value
        capture(index)
    if not bool(result.episode.done.all()):
        raise RuntimeError("Batch rollout did not complete all worlds")
    return BatchTrace(
        state=WorldState(*(value.cpu() for value in states)),
        step=steps.cpu(),
        done=dones.cpu(),
        actor_mask=actor_masks.cpu(),
        goals=goals.cpu() if goals is not None else None,
        goal_mask=goal_masks.cpu() if goal_masks is not None else None,
        metrics=evaluator.metrics() if evaluator is not None else None,
        policy={name: value.cpu() for name, value in policy.items()} if policy else None,
        rewards={name: value.cpu() for name, value in rewards.items()} if rewards else None,
        events={name: value.cpu() for name, value in events.items()} if events else None,
    )


@torch.inference_mode()
def rollout_learned_batch(
    scene,
    *,
    loaded_policy,
    num_steps,
    compile_step=False,
    control_types=None,
    sampling_method="argmax",
    blind=None,
    evaluator=None,
    goal_overrides=None,
):
    """Roll out a loaded SingleFrameNet and capture its actual per-frame goals.

    The policy runs in the training evaluation's environment (PPOEnv, deterministic: the
    Beta mode for ``argmax``, its mean for ``expectation``), so scores match it.
    ``goal_overrides`` (mask [B, A], scene points [B, A, 2]) fixes those actors' goals.
    """
    from cs5478_project.training.rollout import PPOEnv

    env = PPOEnv(
        loaded_policy,
        compile_step=compile_step,
        amp=scene.dt.device.type == "cuda",
        stochastic=False,
        deterministic="mode" if sampling_method == "argmax" else "mean",
    )
    env.goal_overrides = goal_overrides
    return _rollout(scene, num_steps, env, control_types, blind, evaluator)
