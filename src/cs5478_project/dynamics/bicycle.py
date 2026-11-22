"""nuPlan's rear-axle kinematic bicycle, plus the action layer that feeds it."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.datatypes.state import WorldState


@dataclass(frozen=True)
class BicycleConfig:
    """Physical limits and nuPlan actuator time constants, in SI units."""

    max_jerk: float = 5.0
    max_steering_rate: float = 0.6
    min_acceleration: float = -6.0
    max_acceleration: float = 3.0
    max_steering: float = math.pi / 3
    acceleration_time_constant: float = 0.2
    steering_time_constant: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("time_constant"):
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif name == "min_acceleration":
                if value >= 0:
                    raise ValueError("min_acceleration must be negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")


class AdaptedControl(NamedTuple):
    acceleration: Tensor
    steering_rate: Tensor


def adapt_action(
    state: WorldState, action: VehicleAction, dt: Tensor, config: BicycleConfig
) -> AdaptedControl:
    """Integrate bounded jerk into acceleration commands, independently of actuator lag."""
    acceleration = (
        state.acceleration_command + action.jerk.clamp(-config.max_jerk, config.max_jerk) * dt
    ).clamp(config.min_acceleration, config.max_acceleration)
    return AdaptedControl(
        acceleration,
        action.steering_rate.clamp(-config.max_steering_rate, config.max_steering_rate),
    )


def heading_vector(heading: Tensor) -> Tensor:
    return torch.stack((heading.cos(), heading.sin()), dim=-1)


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(angle.sin(), angle.cos())


def no_reverse_command(
    state: WorldState, acceleration: Tensor, dt: Tensor, time_constant: float
) -> Tensor:
    """Action-layer floor on the ideal acceleration so nuPlan's next speed stays >= 0.

    nuPlan filters the ideal command, a' = a + k (u - a) with k = dt / (dt + tau), and
    integrates v' = v + a' dt. Requiring v' >= 0 gives u >= a + (-v / dt - a) / k. A
    stopped vehicle is held at rest instead of reversing; forward commands pass through.
    """
    speed = (state.velocity * heading_vector(state.heading)).sum(dim=-1)
    gain = dt / (dt + time_constant)
    minimum = state.acceleration + (-speed / dt - state.acceleration) / gain
    return torch.maximum(acceleration, minimum)


def nuplan_bicycle_step(
    state: WorldState,
    ideal_acceleration: Tensor,
    steering_rate: Tensor,
    wheelbase: Tensor,
    dt: Tensor,
    *,
    acceleration_time_constant: float,
    steering_time_constant: float,
    max_steering: float,
) -> WorldState:
    """nuPlan ``KinematicBicycleModel.propagate_state``, unchanged: first-order actuator
    filters, then forward Euler with the current speed and steering. Speed is signed and
    only the steering angle is clipped; reversing is prevented in the action layer.
    """
    direction = heading_vector(state.heading)
    speed = (state.velocity * direction).sum(dim=-1)
    acceleration = state.acceleration + dt / (dt + acceleration_time_constant) * (
        ideal_acceleration - state.acceleration
    )
    next_speed = speed + acceleration * dt
    next_steering = (
        state.steering + dt / (dt + steering_time_constant) * steering_rate * dt
    ).clamp(-max_steering, max_steering)
    next_heading = wrap_angle(state.heading + speed * state.steering.tan() / wheelbase * dt)
    return WorldState(
        position=state.position + speed[..., None] * direction * dt[..., None],
        heading=next_heading,
        velocity=next_speed[..., None] * heading_vector(next_heading),
        acceleration=acceleration,
        acceleration_command=ideal_acceleration,
        steering=next_steering,
        yaw_rate=next_speed * next_steering.tan() / wheelbase,
    )


class KinematicBicycleModel:
    """Jerk/steering-rate actions -> bounded command -> no-reverse floor -> nuPlan bicycle."""

    @staticmethod
    def step(
        state: WorldState,
        action: VehicleAction,
        wheelbase: Tensor,
        dt: Tensor,
        config: BicycleConfig = BicycleConfig(),
    ) -> WorldState:
        control = adapt_action(state, action, dt, config)
        # The stored command is the floored one, so the jerk integrator cannot wind up.
        command = no_reverse_command(
            state, control.acceleration, dt, config.acceleration_time_constant
        )
        return nuplan_bicycle_step(
            state,
            command,
            control.steering_rate,
            wheelbase,
            dt,
            acceleration_time_constant=config.acceleration_time_constant,
            steering_time_constant=config.steering_time_constant,
            max_steering=config.max_steering,
        )
