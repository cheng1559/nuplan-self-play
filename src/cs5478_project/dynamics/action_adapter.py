"""Jerk and steering-rate actions over nuPlan's kinematic bicycle."""

import torch

from cs5478_project.datatypes.state import WorldState
from cs5478_project.dynamics.bicycle import no_reverse_command, nuplan_bicycle_step


def step_jerk_bicycle(state, action, wheelbase, dt, config, policy_forward):
    """Turn jerk into a held acceleration target, then step nuPlan's bicycle.

    Action layer: once per policy interval, target = measured acceleration + jerk *
    interval, with positive jerk capped while accelerating; the target is held until
    the next tick. Each step, the no-reverse floor raises the ideal acceleration just
    enough that the next speed is nonnegative. The motion model itself is nuPlan's.
    """
    jerk = action.jerk.clamp(config.min_jerk, config.max_jerk)
    positive_limit = torch.where(
        state.acceleration >= 0, config.positive_jerk_limit, config.max_jerk
    )
    jerk = torch.minimum(jerk, positive_limit)
    target = (state.acceleration + jerk * config.policy_interval_s).clamp(
        -config.max_acceleration, config.max_acceleration
    )
    target = torch.where(policy_forward, target, state.acceleration_command)
    ideal = no_reverse_command(state, target, dt, config.acceleration_time_constant)
    steering_rate = action.steering_rate.clamp(-config.max_steering_rate, config.max_steering_rate)
    result = nuplan_bicycle_step(
        state,
        ideal,
        steering_rate,
        wheelbase,
        dt,
        acceleration_time_constant=config.acceleration_time_constant,
        steering_time_constant=config.steering_time_constant,
        max_steering=config.max_steering_angle,
    )
    # Keep the policy's held target, not the per-step floor, as the command state.
    # (Explicit construction: dynamo mis-reconstructs NamedTuple._replace outputs.)
    return WorldState(
        result.position,
        result.heading,
        result.velocity,
        result.acceleration,
        target,
        result.steering,
        result.yaw_rate,
    )
