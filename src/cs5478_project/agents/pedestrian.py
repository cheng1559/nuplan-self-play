"""Holonomic route-following pedestrians with oracle speed and vehicle yielding."""

import torch
from torch import Tensor

from cs5478_project.agents.config import PedestrianConfig
from cs5478_project.agents.neighbors import Neighbors
from cs5478_project.agents.route import RouteCache, sample_route
from cs5478_project.datatypes.action import PedestrianOracle
from cs5478_project.datatypes.state import WorldState
from cs5478_project.dynamics.bicycle import heading_vector, wrap_angle


def pedestrian_clearance(
    position: Tensor,
    velocity: Tensor,
    direction: Tensor,
    radius: Tensor,
    neighbors: Neighbors,
    config: PedestrianConfig,
) -> Tensor:
    """Predict closest approach to vehicle-only candidates from query_neighbors."""
    relative = neighbors.position - position[..., None, :]
    relative_velocity = neighbors.velocity - velocity[..., None, :]
    time = (
        -(relative * relative_velocity).sum(dim=-1)
        / relative_velocity.square().sum(dim=-1).clamp_min(1e-5)
    ).clamp(0, config.prediction_horizon)
    predicted = relative + relative_velocity * time[..., None]
    obstacle_direction = heading_vector(neighbors.heading)

    def box_clearance(offset: Tensor) -> Tensor:
        longitudinal = (offset * obstacle_direction).sum(dim=-1)
        lateral = (
            offset[..., 0] * obstacle_direction[..., 1]
            - offset[..., 1] * obstacle_direction[..., 0]
        )
        local = torch.stack((longitudinal, lateral), dim=-1)
        outside = (local.abs() - neighbors.size * 0.5).clamp_min(0)
        return outside.square().sum(dim=-1).sqrt() - radius[..., None]

    distance = torch.minimum(box_clearance(relative), box_clearance(predicted))
    ahead = (relative * direction[..., None, :]).sum(dim=-1) >= -radius[..., None]
    approaching = (relative * relative_velocity).sum(dim=-1) < 0
    relevant = neighbors.valid & ahead & (approaching | (time == 0))
    return distance.masked_fill(~relevant, torch.inf).amin(dim=-1)


class RuleBasedPedestrianAgent:
    """Walk on a route, yield to vehicles, and stop at its finite endpoint."""

    @staticmethod
    def step(
        state: WorldState,
        cache: RouteCache,
        progress: Tensor,
        neighbors: Neighbors,
        size: Tensor,
        dt: Tensor,
        oracle: PedestrianOracle,
        config: PedestrianConfig = PedestrianConfig(),
        reference_to_center: Tensor | None = None,
    ) -> WorldState:
        target = sample_route(cache, progress + 0.5)
        displacement = target - state.position
        direction = displacement / torch.linalg.vector_norm(displacement, dim=-1)[
            ..., None
        ].clamp_min(1e-5)
        preferred = torch.where(
            oracle.valid, oracle.preferred_speed, torch.full_like(progress, config.desired_speed)
        )
        preferred = preferred.clamp(0, config.max_speed)
        remaining = (cache.total - progress).clamp_min(0)
        preferred = torch.minimum(preferred, (2 * config.decel_max * remaining).sqrt())
        center = state.position
        if reference_to_center is not None:
            center = center + reference_to_center[..., None] * heading_vector(state.heading)
        clearance = pedestrian_clearance(
            center,
            preferred[..., None] * direction,
            direction,
            size.amax(dim=-1) * 0.5,
            neighbors,
            config,
        )
        preferred = preferred * ((clearance - config.clearance) / config.slowing_distance).clamp(
            0, 1
        )
        speed = torch.linalg.vector_norm(state.velocity, dim=-1)
        next_speed = speed + torch.minimum(
            torch.maximum(preferred - speed, -config.decel_max * dt), config.accel_max * dt
        )
        distance = torch.minimum(next_speed.clamp_min(0) * dt, remaining)
        distance = torch.minimum(distance, torch.linalg.vector_norm(displacement, dim=-1))
        position = state.position + distance[..., None] * direction
        velocity = (position - state.position) / dt[..., None]
        actual_speed = torch.linalg.vector_norm(velocity, dim=-1)
        heading = torch.where(
            actual_speed > 1e-4, torch.atan2(velocity[..., 1], velocity[..., 0]), state.heading
        )
        acceleration = (actual_speed - speed) / dt
        zeros = torch.zeros_like(progress)
        return WorldState(
            position=position,
            heading=heading,
            velocity=velocity,
            acceleration=acceleration,
            acceleration_command=acceleration,
            steering=zeros,
            yaw_rate=wrap_angle(heading - state.heading) / dt,
        )
