"""Deterministic normalized traffic records for development without nuPlan."""

import math
from dataclasses import replace

import torch

from cs5478_project.data.preparation import prepare_scene
from cs5478_project.data.rails import with_route_rails
from cs5478_project.data.routes import build_gt_routes
from cs5478_project.datatypes import ActorType, Scene, WorldState
from cs5478_project.defaults import EPISODE_FRAMES, NPC_ROUTE_HORIZON_S


def make_synthetic_scene(
    *,
    scenario_id: str = "synthetic",
    num_frames: int = EPISODE_FRAMES,
    num_actors: int = 3,
    dt: float = 0.1,
) -> Scene:
    """Create a curved vehicle, walking pedestrian, and static-object pattern.

    Additional actors repeat the pattern at different starting positions. The
    vehicle positions already use the rear-axle reference convention.
    """
    for name, value in (("num_frames", num_frames), ("num_actors", num_actors)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number")
    episode_frames = num_frames
    num_frames = max(num_frames, math.ceil(NPC_ROUTE_HORIZON_S / dt) + 1)
    t = torch.arange(num_frames, dtype=torch.float32)[:, None] * dt
    actor_type = torch.arange(num_actors, dtype=torch.int64) % 3
    vehicles = (actor_type == ActorType.VEHICLE).float()[None, :]
    pedestrians = (actor_type == ActorType.PEDESTRIAN).float()[None, :]
    offsets = torch.arange(num_actors, dtype=torch.float32)[None, :] * 8.0
    omega = 0.15
    vehicle_heading = omega * t
    x = vehicles * (4.0 / omega) * torch.sin(vehicle_heading) + pedestrians * t + offsets
    y = vehicles * (4.0 / omega) * (1.0 - torch.cos(vehicle_heading)) + pedestrians * 0.5 * t
    heading = vehicles * vehicle_heading + pedestrians * math.atan2(0.5, 1.0)
    vx = vehicles * 4.0 * torch.cos(vehicle_heading) + pedestrians
    vy = vehicles * 4.0 * torch.sin(vehicle_heading) + pedestrians * 0.5
    zeros = torch.zeros((num_frames, num_actors), dtype=torch.float32)
    reference = WorldState(
        position=torch.stack((x, y), dim=-1),
        heading=heading,
        velocity=torch.stack((vx, vy), dim=-1),
        acceleration=zeros.clone(),
        acceleration_command=zeros.clone(),
        steering=vehicles.expand_as(zeros) * math.atan(omega * 3.0 / 4.0),
        yaw_rate=vehicles.expand_as(zeros) * omega,
    )
    dimensions = torch.tensor([[4.8, 2.0], [0.6, 0.6], [1.0, 1.0]], dtype=torch.float32)
    scene = Scene(
        scenario_id=scenario_id,
        dt=float(dt),
        actor_type=actor_type,
        actor_mask=torch.ones(num_actors, dtype=torch.bool),
        size=dimensions[actor_type],
        wheelbase=torch.where(actor_type == ActorType.VEHICLE, 3.0, 1.0).float(),
        reference_to_center=torch.where(actor_type == ActorType.VEHICLE, 1.4, 0.0).float(),
        initial_state=WorldState(*(value[0].clone() for value in reference)),
        reference_state=WorldState(*(value[:episode_frames].clone() for value in reference)),
        reference_valid=torch.ones((episode_frames, num_actors), dtype=torch.bool),
    )
    scene = replace(
        scene,
        routes=build_gt_routes(
            scene,
            future_positions=reference.position,
            future_valid=torch.ones(reference.heading.shape, dtype=torch.bool),
        ),
    )
    scene.validate()
    return prepare_scene(with_route_rails(scene))
