"""Shared fixed-K dynamic/static neighbor queries for rule policies."""

from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.agents.config import InteractionConfig
from cs5478_project.datatypes.scene import ActorType, SceneBatch
from cs5478_project.datatypes.state import WorldState
from cs5478_project.dynamics.bicycle import heading_vector


class Neighbors(NamedTuple):
    position: Tensor
    heading: Tensor
    velocity: Tensor
    size: Tensor
    valid: Tensor


def query_neighbors(
    state: WorldState,
    scene: SceneBatch,
    actor_mask: Tensor,
    static_mask: Tensor,
    query_mask: Tensor,
    config: InteractionConfig,
    vehicle_only_queries: Tensor | None = None,
) -> Neighbors:
    """Use O(B*A*(A+S)) distances, then bound route work to K candidates.

    Static geometry remains in its compact scene buffer. Padding, absent tracks,
    the query actor itself, and objects outside the radius cannot become leads.
    Pedestrian-controller queries admit only vehicles before top-K selection, so nearby
    pedestrians and static objects cannot hide a relevant vehicle.
    """
    center = state.position + scene.reference_to_center[..., None] * heading_vector(state.heading)
    position = torch.cat((center, scene.static.position), dim=1)
    heading = torch.cat((state.heading, scene.static.heading), dim=1)
    velocity = torch.cat((state.velocity, torch.zeros_like(scene.static.position)), dim=1)
    size = torch.cat((scene.size, scene.static.size), dim=1)
    valid = torch.cat((actor_mask, static_mask), dim=1)
    distance_sq = (center[:, :, None] - position[:, None]).square().sum(dim=-1)
    actors = torch.arange(center.shape[1], device=center.device)
    objects = torch.arange(position.shape[1], device=center.device)
    eligible = valid[:, None] & query_mask[:, :, None] & (actors[:, None] != objects)
    vehicle_candidates = torch.cat(
        (scene.actor_type == ActorType.VEHICLE, torch.zeros_like(static_mask)), dim=1
    )
    if vehicle_only_queries is None:
        vehicle_only_queries = torch.zeros_like(query_mask)
    eligible = eligible & (~vehicle_only_queries[:, :, None] | vehicle_candidates[:, None])
    distance_sq = distance_sq.masked_fill(~eligible, torch.inf)
    distances, indices = distance_sq.topk(
        min(config.neighbors, position.shape[1]), dim=-1, largest=False, sorted=False
    )
    world = torch.arange(center.shape[0], device=center.device)[:, None, None]
    return Neighbors(
        position[world, indices],
        heading[world, indices],
        velocity[world, indices],
        size[world, indices],
        distances <= config.radius**2,
    )
