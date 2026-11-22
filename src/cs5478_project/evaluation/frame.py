"""One frame of nuPlan-style checks for a fixed set of evaluated actors (queries).

Each query is judged as nuPlan judges ego: every other present dynamic actor and every
visible static object is a track. All outputs are [Q] or [Q, K] over the K nearest
tracks, so the function has fixed shapes and compiles with ``fullgraph=True``.
"""

import functools
import math
from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.agents.geometry import box_corners
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.map import (
    CHUNK_POINTS,
    LANE_INFO_IN_INTERSECTION,
    LANE_INFO_IN_LANE,
    LANE_INFO_SPEED_UNIT,
    MapElementType,
)
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.dynamics.bicycle import heading_vector, wrap_angle
from cs5478_project.evaluation.geometry import (
    box_overlap_over_time,
    drivable_distance,
    lane_info,
    segment_hits_box,
)

# Collision partner classes, in nuPlan's order: VRUs, vehicles, objects.
COLLISION_CLASSES = 3
LANE_CANDIDATES = 8
PROBES = (0, CHUNK_POINTS // 2, CHUNK_POINTS - 1)
CENTERLINES = (MapElementType.LANE_CENTERLINE, MapElementType.LANE_CONNECTOR_CENTERLINE)
# Lines for the reward's lane-centering score: lane and connector boundaries, stop
# lines, and crosswalk outlines.
LANE_LINES = (MapElementType.LANE_BOUNDARY, MapElementType.STOP_LINE, MapElementType.CROSSWALK)


class Frame(NamedTuple):
    present: Tensor  # [Q] the query exists at this frame
    box: Tensor  # [Q, 4, 2] the query's box corners
    track: Tensor  # [Q, K] index on the [dynamic actors, static objects] axis
    new_collision: Tensor  # [Q, K] first overlap with this track in the episode
    at_fault: Tensor  # [Q, K] new collisions nuPlan attributes to the query
    at_fault_by_class: Tensor  # [Q, 3] new at-fault collisions with VRUs, vehicles, objects
    ttc: Tensor  # [Q] seconds; ttc_horizon when nothing is hit, 0 on an at-fault collision
    # Reward versions: any overlap while the query moves (no fault rules), and TTC over
    # every track not behind the query.
    moving_collision: Tensor  # [Q]
    ttc_wide: Tensor  # [Q] seconds, as ttc
    centerline_offset: Tensor  # [Q] box center to the nearest lane or connector centerline
    centerline_direction: Tensor  # [Q, 2] unit direction of that centerline
    centerline_chunk: Tensor  # [Q] map chunk index of that centerline
    boundary_offset: Tensor  # [Q] box center to the nearest lane line (LANE_LINES)
    outside: Tensor  # [Q] largest corner distance outside the drivable area
    overspeed: Tensor  # [Q] speed above the lane speed limit, m/s
    speed_limit: Tensor  # [Q] nuPlan speed limit at the box center, 0 when unknown
    straddling: Tensor  # [Q] corners not all in one lane (or off lanes)
    in_intersection: Tensor  # [Q] rear axle inside an intersection polygon


def _nearest_polyline(scene, query_world, point, kinds):
    """Distance from each query point to the nearest map chunk of the given kinds, the
    unit direction of its nearest segment, and the chunk index (inf, zeros, and an
    arbitrary index when there is none)."""
    kind = scene.map.chunk_type[query_world]
    lane = scene.map.chunk_mask[query_world] & functools.reduce(
        torch.logical_or, [kind == k for k in kinds]
    )
    # Rank chunks by three probe points, then measure exact distances to a few.
    probes = scene.map.chunk_points[:, :, list(PROBES)][query_world]  # [Q, C, 3, 2]
    probe = (probes - point[:, None, None]).norm(dim=-1).amin(-1).masked_fill(~lane, torch.inf)
    candidate = probe.topk(min(LANE_CANDIDATES, probe.shape[1]), largest=False).indices
    points = scene.map.chunk_points[query_world[:, None], candidate]  # [Q, L, P, 2]
    a, b = points[:, :, :-1], points[:, :, 1:]
    delta = b - a
    p = point[:, None, None]
    fraction = (((p - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    distance = (p - a - fraction[..., None] * delta).norm(dim=-1)  # [Q, L, P - 1]
    distance = distance.masked_fill(~lane.gather(1, candidate)[..., None], torch.inf)
    distance, nearest = distance.flatten(1).min(-1)
    rows = torch.arange(point.shape[0], device=point.device)
    direction = delta.flatten(1, 2)[rows, nearest]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    chunk = candidate[rows, nearest // delta.shape[2]]
    return distance, direction, chunk


def evaluate_frame(
    state, scene, step, actor_mask, static_mask, control_types, query_world, query_slot,
    collided, config,
):  # fmt: skip
    """Check queries at ``step``. ``collided`` [Q, A + S] marks tracks hit earlier."""
    world = torch.arange(step.shape[0], device=step.device)
    direction = heading_vector(state.heading)
    center = state.position + scene.reference_to_center[..., None] * direction
    sizes = torch.where(
        (control_types == ControlType.LOG)[..., None], scene.reference_size[world, step], scene.size
    )
    static = scene.static
    position = torch.cat((center, static.position), 1)
    heading = torch.cat((state.heading, static.heading), 1)
    size = torch.cat((sizes, static.size), 1)
    speed = torch.cat((state.velocity.norm(dim=-1), torch.zeros_like(static.heading)), 1)
    valid = torch.cat((actor_mask, static_mask), 1)
    kind = torch.cat(
        (scene.actor_type, torch.full_like(static.heading, ActorType.STATIC, dtype=torch.int64)),
        1,
    )

    q_center = center[query_world, query_slot]
    q_heading = state.heading[query_world, query_slot]
    q_speed = speed[query_world, query_slot]
    q_rear = state.position[query_world, query_slot]
    present = actor_mask[query_world, query_slot]
    ego_box = box_corners(q_center, q_heading, sizes[query_world, query_slot])  # [Q, 4, 2]

    # The K nearest tracks by center distance.
    distance_sq = (position[query_world] - q_center[:, None]).square().sum(-1)
    slots = torch.arange(position.shape[1], device=step.device)
    eligible = (
        valid[query_world]
        & (slots[None] != query_slot[:, None])
        & (distance_sq <= config.neighbor_radius**2)
    )
    track = (
        distance_sq.masked_fill(~eligible, torch.inf)
        .topk(min(config.neighbors, position.shape[1]), largest=False)
        .indices
    )  # [Q, K]
    track_valid = eligible.gather(1, track) & present[:, None]
    rows = query_world[:, None]
    t_center, t_heading = position[rows, track], heading[rows, track]
    t_speed, t_kind = speed[rows, track], kind[rows, track]
    t_box = box_corners(t_center, t_heading, size[rows, track])  # [Q, K, 4, 2]

    # Lane context from the exported lane-info grid (nuPlan's lane polygons). A query
    # "straddles" (nuPlan: corners not in common or connected lanes) when a corner is
    # outside every lane, or when its center is far from the nearest centerline.
    points = torch.cat((q_center[:, None], q_rear[:, None], ego_box), 1)
    info = lane_info(scene.map, query_world, points)  # [Q, 6]
    corners_in_lane = ((info[:, 2:] & LANE_INFO_IN_LANE) > 0).all(-1)
    offset, tangent, centerline_chunk = _nearest_polyline(scene, query_world, q_center, CENTERLINES)
    boundary_offset, _, _ = _nearest_polyline(scene, query_world, q_center, LANE_LINES)
    straddling = ~corners_in_lane | (offset > config.lane_straddle_offset)
    in_intersection = (info[:, 1] & LANE_INFO_IN_INTERSECTION) > 0
    units = (info[:, 0] & (LANE_INFO_IN_LANE - 1)).to(q_speed.dtype)
    in_lane = (info[:, 0] & LANE_INFO_IN_LANE) > 0
    speed_limit = torch.where(in_lane, units * LANE_INFO_SPEED_UNIT, 0.0)
    overspeed = torch.where(present & (speed_limit > 0), (q_speed - speed_limit).clamp_min(0), 0.0)

    # Collisions: overlap now, with tracks not hit before (nuPlan counts each track once).
    no_shift = torch.zeros_like(t_center)
    overlap = box_overlap_over_time(ego_box, t_box, no_shift, 0)[..., 0] & track_valid
    new_collision = overlap & ~collided.gather(1, track)
    relative = t_center - q_rear[:, None]
    angle = wrap_angle(torch.atan2(relative[..., 1], relative[..., 0]) - q_heading[:, None]).abs()
    behind = angle > math.radians(config.behind_angle_deg)
    ahead = angle < math.radians(config.ahead_angle_deg)
    ego_moving = (q_speed > config.stopped_speed)[:, None]
    track_stopped = (t_speed <= config.stopped_speed) | (t_kind == ActorType.STATIC)
    front = segment_hits_box(ego_box[:, None, 0], ego_box[:, None, 1], t_box)
    at_fault = (
        new_collision & ego_moving & (track_stopped | (~behind & (front | straddling[:, None])))
    )
    classes = torch.stack(
        (t_kind == ActorType.PEDESTRIAN, t_kind == ActorType.VEHICLE, t_kind == ActorType.STATIC),
        -1,
    )
    at_fault_by_class = (at_fault[..., None] & classes).sum(1)

    # Time to collision: constant-velocity boxes every ttc_step, tracks not hit so far.
    hit_before = collided.gather(1, track) | new_collision
    widen = (straddling | in_intersection)[:, None]
    candidate = track_valid & ~hit_before
    step_size = config.ttc_step
    steps = round(config.ttc_horizon / step_size) - 1
    ego_shift = (q_speed * step_size)[:, None] * heading_vector(q_heading)
    track_shift = (t_speed * step_size)[..., None] * heading_vector(t_heading)
    hits = box_overlap_over_time(ego_box, t_box, track_shift - ego_shift[:, None], steps)[..., 1:]

    def time_to_collision(relevant):
        hit = (hits & relevant[..., None]).any(1)  # [Q, steps]
        first = hit.float().argmax(-1)
        ttc = torch.where(
            hit.any(-1), (first + 1).to(q_speed.dtype) * step_size, config.ttc_horizon
        )
        ttc = torch.where(q_speed <= config.ttc_stopped_speed, config.ttc_horizon, ttc)
        return torch.where(at_fault.any(-1), 0.0, ttc)

    ttc = time_to_collision(candidate & (ahead | (widen & ~behind)))
    ttc_wide = time_to_collision(candidate & ~behind)
    moving_collision = overlap.any(-1) & ego_moving[:, 0]

    outside, _ = drivable_distance(scene.map, query_world, ego_box)
    outside = torch.where(present, outside.amax(-1), 0.0)

    return Frame(
        present,
        ego_box,
        track,
        new_collision,
        at_fault,
        at_fault_by_class,
        ttc,
        moving_collision,
        ttc_wide,
        offset,
        tangent,
        centerline_chunk,
        boundary_offset,
        outside,
        overspeed,
        speed_limit,
        straddling,
        in_intersection,
    )
