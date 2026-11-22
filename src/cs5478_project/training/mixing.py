"""Per-world controller assignment for mixed training and reactive evaluation.

Every world is one of three modes; the ego vehicle is always the policy:

- ``LOG``: everyone else replays the log (nuPlan's non-reactive setting).
- ``REACTIVE``: vehicles with an exported rail follow IDM, other vehicles replay the
  log, and pedestrians follow the rule-based model (nuPlan's reactive setting).
- ``SELF_PLAY``: the policy also drives up to ``self_play_agents`` vehicles near ego;
  the other vehicles with a rail follow IDM, those without one are not simulated
  (ABSENT, like vehicles nuPlan's IDM builder never creates), and pedestrians follow
  the rule-based model. Nothing replays the log.

Self-play vehicles exist at the first frame and later, have a
route for their training goal, and neither overlap another object nor touch a road
boundary at the start; the ones nearest to ego are taken.
"""

from enum import IntEnum

import torch
from torch import Tensor

from cs5478_project.agents.config import IDMConfig
from cs5478_project.agents.geometry import box_corners
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.map import MapElementType
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.dynamics.bicycle import heading_vector
from cs5478_project.evaluation.geometry import boundary_crossing, box_overlap_over_time

# Candidates checked for start overlaps per world, and neighbors checked per candidate.
CANDIDATES = 32
OVERLAP_NEIGHBORS = 16
# Queries per road-boundary check (each gathers its world's map segments).
BOUNDARY_CHUNK = 2048


class WorldMode(IntEnum):
    LOG = 0
    REACTIVE = 1
    SELF_PLAY = 2


def sample_modes(worlds: int, weights, device, generator=None) -> Tensor:
    """[B] WorldMode drawn with the given (log, reactive, self-play) weights."""
    weights = torch.tensor(weights, dtype=torch.float32)
    if weights.shape != (3,) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ValueError("World mode weights must be three nonnegative numbers, not all zero")
    modes = torch.multinomial(weights, worlds, replacement=True, generator=generator)
    return modes.to(device)


def _initial_boxes(scene):
    state = scene.initial_state
    center = state.position + scene.reference_to_center[..., None] * heading_vector(state.heading)
    return center, box_corners(center, state.heading, scene.size)


def self_play_candidates(scene, count: int) -> Tensor:
    """[B, A] bool: up to ``count`` vehicles per world that the policy can drive."""
    b, a = scene.actor_mask.shape
    device = scene.actor_mask.device
    if count <= 0:
        return torch.zeros_like(scene.actor_mask)
    world = torch.arange(b, device=device)
    slots = torch.arange(a, device=device)
    present = scene.actor_mask & scene.reference_valid[:, 0]
    eligible = (
        present
        & (scene.actor_type == ActorType.VEHICLE)
        & (slots[None] != scene.ego_index[:, None])
        & scene.reference_valid[:, 1:].any(1)
        & scene.routes.mask.any(-1)
    )
    center, box = _initial_boxes(scene)
    distance = (center - center[world, scene.ego_index][:, None]).norm(dim=-1)
    distance = distance.masked_fill(~eligible, torch.inf)
    m = min(CANDIDATES, a)
    order = distance.topk(m, largest=False).indices  # [B, M] nearest first
    ok = eligible.gather(1, order)

    # Start overlaps with other actors and static objects.
    static = scene.static
    others_center = torch.cat((center, static.position), 1)  # [B, A + S, 2]
    others_box = torch.cat((box, box_corners(static.position, static.heading, static.size)), 1)
    others_valid = torch.cat((present, static.mask & static.valid[:, 0]), 1)
    rows = world[:, None]
    cand_center, cand_box = center[rows, order], box[rows, order]  # [B, M, ...]
    gap = (cand_center[:, :, None] - others_center[:, None]).norm(dim=-1)  # [B, M, A + S]
    gap = gap.masked_fill(~others_valid[:, None], torch.inf)
    gap.scatter_(2, order[..., None], torch.inf)  # not itself
    k = min(OVERLAP_NEIGHBORS, gap.shape[2])
    near = gap.topk(k, largest=False).indices  # [B, M, K]
    near_valid = others_valid[:, None].expand(-1, m, -1).gather(2, near)
    near_valid &= gap.gather(2, near).isfinite()
    near_box = others_box[rows[..., None], near]  # [B, M, K, 4, 2]
    flat_box = cand_box.reshape(b * m, 4, 2)
    overlap = box_overlap_over_time(
        flat_box, near_box.reshape(b * m, k, 4, 2), torch.zeros(b * m, k, 2, device=device), 0
    )[..., 0].view(b, m, k)
    ok &= ~(overlap & near_valid).any(-1)

    # Start contact with a road boundary.
    # In chunks: each query gathers its world's segments.
    query_world = rows.expand(-1, m).reshape(-1)
    curb = torch.cat(
        [
            boundary_crossing(
                scene.map,
                query_world[start : start + BOUNDARY_CHUNK],
                flat_box[start : start + BOUNDARY_CHUNK],
                flat_box[start : start + BOUNDARY_CHUNK],
                MapElementType.ROAD_BOUNDARY,
            )
            for start in range(0, b * m, BOUNDARY_CHUNK)
        ]
    ).view(b, m)
    ok &= ~curb

    # The nearest ``count`` that pass.
    rank = ok.long().cumsum(1)
    keep = ok & (rank <= count)
    selected = torch.zeros_like(scene.actor_mask)
    selected.scatter_(1, order, keep)
    return selected


def mixed_controls(
    scene, modes: Tensor, self_play_agents: int = 10, max_idm_agents: int = IDMConfig().max_agents
) -> Tensor:
    """[B, A] int64 control types for worlds in the given WorldMode [B].

    At most ``max_idm_agents`` vehicles per world (the IDM capacity) follow IDM, the
    ones nearest to ego; the rest are treated like vehicles without a rail.
    """
    b, a = scene.actor_mask.shape
    world = torch.arange(b, device=modes.device)
    slots = torch.arange(a, device=modes.device)
    vehicle = scene.actor_type == ActorType.VEHICLE
    is_ego = slots[None] == scene.ego_index[:, None]
    reactive = (modes == WorldMode.REACTIVE)[:, None]
    self_play = (modes == WorldMode.SELF_PLAY)[:, None]
    drivers = torch.zeros_like(scene.actor_mask)
    if self_play_agents > 0 and bool(self_play.any()):
        drivers = self_play_candidates(scene, self_play_agents) & self_play
    # IDM for railed vehicles, nearest to ego first, up to the capacity.
    center, _ = _initial_boxes(scene)
    distance = (center - center[world, scene.ego_index][:, None]).norm(dim=-1)
    railed = (
        vehicle & scene.actor_mask & scene.rails.rail_valid & ~drivers & ~is_ego
        & (reactive | self_play)
    )  # fmt: skip
    order = distance.masked_fill(~railed, torch.inf).argsort(1)
    rank = torch.empty_like(order).scatter_(1, order, slots.expand(b, -1))
    idm = railed & (rank < max_idm_agents)
    controls = torch.full_like(scene.actor_type, ControlType.LOG)
    # Vehicles without IDM replay the log (reactive) or are not simulated (self-play).
    controls = torch.where(vehicle & self_play, ControlType.ABSENT, controls)
    controls = torch.where(idm, ControlType.IDM, controls)
    pedestrians = (scene.actor_type == ActorType.PEDESTRIAN) & (reactive | self_play)
    controls = torch.where(pedestrians, ControlType.PEDESTRIAN, controls)
    controls = torch.where(drivers, ControlType.SINGLE_FRAME, controls)
    controls[world, scene.ego_index] = ControlType.SINGLE_FRAME
    return torch.where(scene.actor_mask, controls, ControlType.LOG)
