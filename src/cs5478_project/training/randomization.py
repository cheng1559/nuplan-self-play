"""Domain randomization for training episodes (the distributions of the report's Table 4).

- Reward weights per actor, observed by the policy (``reward_conditions``): collision
  U[-5, 0], off-road U[-2, 0], TTC and comfort penalty strengths U[0, 1], goal bonus
  U[0, 1] and goal radius U[1.5, 3] m, overspeed penalty strength U[0.2, 1].
- Perception: each actor except ego is blind with probability ``blind_probability``.
- Agent size: length and width each scaled by U[0.9, 1.1].
- Initial state of policy-driven actors: position noise along / across the heading
  (sigma 0.5 / 0.25 m), heading (2 deg), speed (1 m/s), and acceleration (0.5 m/s^2),
  each clipped at three sigma.

Evaluation uses the default weights, which reproduce the unrandomized reward.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.datatypes.control import ControlType
from cs5478_project.dynamics.bicycle import heading_vector
from cs5478_project.evaluation.config import EvaluationConfig

# Policy inputs, in this order (see conditions()).
REWARD_CONDITIONS = 7


class RewardWeights(NamedTuple):
    """Per-actor reward weights [B, A] (or per-query [Q])."""

    collision: Tensor  # reward for a collision (default -1)
    offroad: Tensor  # off-road reward before the speed scale (default -1)
    ttc: Tensor  # TTC penalty strength in [0, 1] (default 1)
    comfort: Tensor  # comfort penalty strength in [0, 1] (default 1)
    goal: Tensor  # goal bonus (default 0.5)
    goal_threshold: Tensor  # goal radius in meters (default 1.5)
    speed: Tensor  # overspeed penalty strength (default 1)


def default_weights(shape, device, config: EvaluationConfig = EvaluationConfig()):
    def full(value):
        return torch.full(shape, float(value), device=device)

    return RewardWeights(
        full(config.collision_reward),
        full(config.offroad_reward),
        full(1.0),
        full(1.0),
        full(config.goal_reward),
        full(config.goal_threshold),
        full(1.0),
    )


def sample_weights(shape, device, generator=None) -> RewardWeights:
    def uniform(low, high):
        u = torch.rand(shape, device=device, generator=generator)
        return low + (high - low) * u

    return RewardWeights(
        uniform(-5.0, 0.0),
        uniform(-2.0, 0.0),
        uniform(0.0, 1.0),
        uniform(0.0, 1.0),
        uniform(0.0, 1.0),
        uniform(1.5, 3.0),
        uniform(0.2, 1.0),
    )


def conditions(weights: RewardWeights) -> Tensor:
    """[..., REWARD_CONDITIONS] policy inputs, each scaled to about [0, 1]."""
    return torch.stack(
        (
            -weights.collision / 5,
            -weights.offroad / 2,
            weights.ttc,
            weights.comfort,
            weights.goal,
            (weights.goal_threshold - 1.5) / 1.5,
            weights.speed,
        ),
        -1,
    )


def default_conditions(shape, device) -> Tensor:
    return conditions(default_weights(shape, device))


def conditions_from(overrides: dict, device="cpu") -> Tensor:
    """[REWARD_CONDITIONS] policy inputs for the default reward weights with the given
    ones replaced, e.g. {"collision": -3.0, "goal": 1.0} (RewardWeights field names)."""
    unknown = set(overrides) - set(RewardWeights._fields)
    if unknown:
        raise ValueError(f"Unknown reward weights {sorted(unknown)}; use {RewardWeights._fields}")
    weights = default_weights((), device)._asdict()
    for name, value in overrides.items():
        weights[name] = torch.tensor(float(value), device=device)
    return conditions(RewardWeights(**weights))


class Randomized(NamedTuple):
    scene: object  # SceneBatch with randomized sizes and initial states
    blind: Tensor  # [B, A]
    weights: RewardWeights  # [B, A] each


def randomize(scene, controls: Tensor, blind_probability: float = 0.1, generator=None):
    """Randomize one training batch; ``controls`` [B, A] marks the policy-driven actors."""
    b, a = scene.actor_mask.shape
    device = scene.actor_mask.device

    def normal(*shape, sigma=1.0):
        noise = torch.randn(*shape, device=device, generator=generator) * sigma
        return noise.clamp(-3 * sigma, 3 * sigma)

    slots = torch.arange(a, device=device)
    ego = slots[None] == scene.ego_index[:, None]
    blind = (torch.rand(b, a, device=device, generator=generator) < blind_probability) & ~ego
    blind &= scene.actor_mask

    scale = 0.9 + 0.2 * torch.rand(b, a, 2, device=device, generator=generator)
    size = scene.size * scale
    reference_size = scene.reference_size * scale[:, None]
    rails = scene.rails._replace(rail_size=scene.rails.rail_size * scale)

    driven = (controls == ControlType.SINGLE_FRAME) & scene.actor_mask
    state = scene.initial_state
    direction = heading_vector(state.heading)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    shift = normal(b, a, 1, sigma=0.5) * direction + normal(b, a, 1, sigma=0.25) * lateral
    heading = state.heading + normal(b, a, sigma=math.radians(2.0))
    speed = (state.velocity.norm(dim=-1) + normal(b, a, sigma=1.0)).clamp_min(0)
    acceleration = state.acceleration + normal(b, a, sigma=0.5)
    noisy = state._replace(
        position=state.position + shift,
        heading=heading,
        velocity=speed[..., None] * heading_vector(heading),
        acceleration=acceleration,
    )
    initial_state = type(state)(
        *(
            torch.where(driven.view(b, a, *([1] * (new.ndim - 2))), new, old)
            for new, old in zip(noisy, state, strict=True)
        )
    )
    # The log's first frame is the episode start, so it moves with the initial state.
    reference_state = type(state)(
        *(
            torch.cat((start[:, None], log[:, 1:]), 1)
            for start, log in zip(initial_state, scene.reference_state, strict=True)
        )
    )
    scene = scene._replace(
        size=size,
        reference_size=reference_size,
        rails=rails,
        initial_state=initial_state,
        reference_state=reference_state,
    )
    return Randomized(scene, blind, sample_weights((b, a), device, generator))
