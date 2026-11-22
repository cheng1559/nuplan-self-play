"""nuPlan reactive IDM agents (the IDMAgents observation) on exported map rails, on GPU.

Per step, mirroring ``IDMAgentManager.propagate_agents`` and ``IDMAgent``:

1. Agents farther than ``radius`` from the pre-step ego center are removed for good.
2. ``plan_route``: while the path to go is shorter than ``minimum_path_length +
   target * headway``, append the lowest-curvature allowed exit (green for signalized
   connectors, otherwise not red), rebuilding the path from the current progress.
3. Red connectors among the last five route edges contribute their stop lines.
4. Geometry touching the path to go, buffered by half the agent width with flat caps,
   are leads; the nearest one to the agent's projected footprint sets the IDM gap.
   Ego and vehicles lead with their projected speed; stop lines and open-loop objects
   with zero. A projected footprint is the box plus the flat-capped buffer of the path
   from ``progress - length/2`` to ``progress + length/2 + velocity * headway``.
5. Target velocity becomes the end segment's speed limit; forward-Euler IDM with
   clipped acceleration advances progress, and the agent is drawn at its clamped
   progress along the path.

All agents update simultaneously from the same pre-step world. nuPlan updates them
sequentially, so later agents there already see earlier agents' new footprints.
Signals and open-loop objects are read at the next frame, ego at the current one.
Flat-capped buffers are unions of one rectangle per polyline segment; the round
joins between them are omitted. Heavy geometry runs only on a fixed-capacity table
of IDM agents per world, built at reset.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.agents.config import IDMConfig
from cs5478_project.agents.geometry import box_corners
from cs5478_project.agents.polygons import INF, polygon_distance, polygon_edges
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.dynamics.bicycle import heading_vector, wrap_angle


class RailState(NamedTuple):
    """Runtime IDM agent state for every [B, N] slot; inert outside IDM agents."""

    edges: Tensor  # [B, N, R] route window of edge indices, -1 padding
    edge_count: Tensor  # [B, N] edges used in the window
    arc: Tensor  # [B, N] unclamped progress from the window's first edge start
    path_start: Tensor  # [B, N] where the current path begins (last rebuild)
    velocity: Tensor  # [B, N]
    target: Tensor  # [B, N] IDM target velocity
    removed: Tensor  # [B, N] left the radius around ego
    started: Tensor  # [B] footprints exist after the first propagation
    query: Tensor  # [B, Q] slots of IDM agents, -1 padding


class RailPose(NamedTuple):
    center: Tensor
    heading: Tensor
    velocity: Tensor


PER_AGENT = ("edges", "edge_count", "arc", "path_start", "velocity", "target")


class _Agents(NamedTuple):
    """Per-agent rail fields of a RailState, viewed at the query table."""

    edges: Tensor
    edge_count: Tensor
    arc: Tensor
    path_start: Tensor
    velocity: Tensor
    target: Tensor


def idm_acceleration(
    speed: Tensor,
    gap: Tensor,
    lead_speed: Tensor,
    config: IDMConfig = IDMConfig(),
    target_velocity: Tensor | None = None,
) -> Tensor:
    """nuPlan IDMPolicy.idm_model with its clipped forward-Euler acceleration."""
    target = config.target_velocity if target_velocity is None else target_velocity.clamp_min(0.1)
    desired_gap = (
        config.min_gap
        + speed * config.headway_time
        + speed * (speed - lead_speed) / (2 * math.sqrt(config.accel_max * config.decel_max))
    )
    acceleration = config.accel_max * (
        1 - (speed / target) ** 4 - (desired_gap / gap.clamp_min(config.min_gap)) ** 2
    )
    return acceleration.clamp(-config.decel_max, config.accel_max)


def initial_rail_state(scene, control_types: Tensor, config: IDMConfig = IDMConfig()) -> RailState:
    """Start every rail; IDM-controlled railed slots form each world's query table."""
    rails = scene.rails
    b, n = rails.rail_valid.shape
    device = rails.rail_edge.device
    edges = torch.full((b, n, config.route_edges), -1, dtype=torch.int64, device=device)
    edges[..., 0] = torch.where(rails.rail_valid, rails.rail_edge, -1)
    agents = (control_types == ControlType.IDM) & rails.rail_valid
    most = int(agents.sum(-1).max())
    if most > config.max_agents:
        raise ValueError(f"A world has {most} IDM agents; max_agents={config.max_agents}")
    order = torch.where(agents, torch.arange(n, device=device), n)
    query = order.sort(-1).values[:, : min(config.max_agents, n)]
    return RailState(
        edges,
        rails.rail_valid.long(),
        rails.rail_progress.clone(),
        torch.zeros_like(rails.rail_progress),
        rails.rail_speed.clone(),
        rails.rail_target.clone(),
        torch.zeros_like(rails.rail_valid),
        torch.zeros(b, dtype=torch.bool, device=device),
        torch.where(query < n, query, -1).contiguous(),
    )


def _world(tensor: Tensor) -> Tensor:
    return torch.arange(tensor.shape[0], device=tensor.device).view(-1, *([1] * (tensor.ndim - 1)))


def _gather(value: Tensor, index: Tensor) -> Tensor:
    """value [B, N, ...] at index [B, Q] -> [B, Q, ...]."""
    shape = (*index.shape, *([1] * (value.ndim - 2)))
    return value.gather(1, index.reshape(shape).expand(-1, -1, *value.shape[2:]))


def _scatter(value: Tensor, index: Tensor, update: Tensor, valid: Tensor) -> Tensor:
    """Write update [B, Q, ...] into value [B, N, ...] where valid, without collisions."""
    n = value.shape[1]
    padded = torch.cat((value, value[:, :1]), 1)
    target = torch.where(valid, index, n)
    shape = (*target.shape, *([1] * (value.ndim - 2)))
    index = target.reshape(shape).expand(-1, -1, *value.shape[2:])
    # Contiguous, so state strides match reset tensors and compiled graphs are reused.
    result = padded.scatter(1, index, update.to(value.dtype))[:, :n]
    return result.clone(memory_format=torch.contiguous_format)


def _cumulative(rails, agents) -> tuple[Tensor, Tensor]:
    """[..., R + 1] arc offsets of window edges, and the path end."""
    slots = torch.arange(agents.edges.shape[-1], device=agents.edges.device)
    lengths = rails.edge_length[_world(agents.edges), agents.edges.clamp_min(0)]
    lengths = lengths * (slots < agents.edge_count[..., None])
    # Prefix sums as a masked sum: Triton fails to fuse a scan with plan_route's gathers.
    before = slots[None, :] < torch.arange(slots.shape[0] + 1, device=slots.device)[:, None]
    cumulative = (lengths[..., None, :] * before).sum(-1)
    return cumulative, cumulative.gather(-1, agents.edge_count[..., None]).squeeze(-1)


def _sample(rails, agents, cumulative: Tensor, arcs: Tensor) -> tuple[Tensor, Tensor]:
    """Positions [..., K, 2] and headings [..., K] at window arcs [..., K]."""
    r = agents.edges.shape[-1]
    inner = cumulative[..., None, 1:r]
    usable = torch.arange(1, r, device=arcs.device) < agents.edge_count[..., None, None]
    slot = ((arcs[..., None] >= inner) & usable).sum(-1)
    edge = agents.edges.gather(-1, slot).clamp_min(0)
    world = _world(edge)
    length = rails.edge_length[world, edge]
    points = rails.edge_points.shape[-2]
    u = ((arcs - cumulative.gather(-1, slot)) / length).clamp(0, 1) * (points - 1)
    j = u.floor().long().clamp(max=points - 2)
    t = u - j
    p0, p1 = rails.edge_points[world, edge, j], rails.edge_points[world, edge, j + 1]
    h0, h1 = rails.edge_heading[world, edge, j], rails.edge_heading[world, edge, j + 1]
    return p0 + t[..., None] * (p1 - p0), h0 + t * wrap_angle(h1 - h0)


def _clamped(agents, end: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(agents.arc, agents.path_start), end)


def _pose(rails, agents) -> tuple[Tensor, Tensor]:
    cumulative, end = _cumulative(rails, agents)
    center, heading = _sample(rails, agents, cumulative, _clamped(agents, end)[..., None])
    return center[..., 0, :], wrap_angle(heading[..., 0])


def rail_pose(scene, state: RailState) -> RailPose:
    center, heading = _pose(scene.rails, state)
    return RailPose(center, heading, state.velocity)


def _polyline(rails, agents, cumulative, start, end, samples):
    fraction = torch.linspace(0, 1, samples + 1, device=start.device)
    arcs = start[..., None] + (end - start)[..., None] * fraction
    return _sample(rails, agents, cumulative, arcs)[0]


def _rectangles(points: Tensor, half_width: Tensor) -> tuple[Tensor, Tensor]:
    """Flat-capped buffer of a polyline [..., K + 1, 2] as K quads, with validity."""
    a, b = points[..., :-1, :], points[..., 1:, :]
    delta = b - a
    length = delta.norm(dim=-1)
    unit = delta / length.clamp_min(1e-9)[..., None]
    normal = torch.stack((-unit[..., 1], unit[..., 0]), -1) * half_width[..., None, None]
    quads = torch.stack((a + normal, b + normal, b - normal, a - normal), -2)
    return quads, length > 1e-6


def _pieces_distance(first, first_valid, second, second_valid) -> Tensor:
    """Distance between unions of polygons [..., P, V, 2] and [..., Q, W, 2]."""
    p, q = first.shape[-3], second.shape[-3]
    a = first[..., :, None, :, :].expand(*first.shape[:-3], p, q, *first.shape[-2:])
    b = second[..., None, :, :, :].expand(*second.shape[:-3], p, q, *second.shape[-2:])
    a_mask = torch.ones(a.shape[:-1], dtype=torch.bool, device=a.device)
    b_mask = torch.ones(b.shape[:-1], dtype=torch.bool, device=b.device)
    distance = polygon_distance(
        polygon_edges(a, a_mask), polygon_edges(b, b_mask), a[..., 0, :], b[..., 0, :]
    )
    valid = first_valid[..., :, None] & second_valid[..., None, :]
    return distance.masked_fill(~valid, INF).flatten(-2).amin(-1)


def _rectangles_touch(first, first_valid, second, second_valid, tolerance=1e-6) -> Tensor:
    """Whether any valid rectangle of [..., P, 4, 2] comes within ``tolerance`` of any
    valid one of [..., Q, 4, 2], by separating axes.

    The same test as ``_pieces_distance(...) <= tolerance`` for rectangles (corners in
    order), except for corner-to-corner gaps under the tolerance, at a fraction of the
    cost: four projections per pair instead of sixteen segment distances.
    """

    def axes(rect):
        edge = torch.stack(
            (rect[..., 1, :] - rect[..., 0, :], rect[..., 3, :] - rect[..., 0, :]), -2
        )
        return edge / edge.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    a, b = first[..., :, None, :, :], second[..., None, :, :, :]  # [..., P, 1 | 1, Q, 4, 2]
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    axis = torch.cat((axes(a).expand(*shape, 2, 2), axes(b).expand(*shape, 2, 2)), -2)
    project_a = (a[..., None, :, :] * axis[..., :, None, :]).sum(-1)  # [..., P, Q, 4, 4]
    project_b = (b[..., None, :, :] * axis[..., :, None, :]).sum(-1)
    separated = (project_b.amin(-1) - project_a.amax(-1) > tolerance) | (
        project_a.amin(-1) - project_b.amax(-1) > tolerance
    )
    touch = ~separated.any(-1) & first_valid[..., :, None] & second_valid[..., None, :]
    return touch.flatten(-2).any(-1)


def _plan_route(rails, agents, active, signal, follow_route, config):
    """Unrolled ``IDMAgent.plan_route``; signal = (green, red) [B, L] at the next frame."""
    green, red = signal
    world = _world(agents.edge_count)
    r = agents.edges.shape[-1]
    for _ in range(config.extensions_per_step):
        cumulative, end = _cumulative(rails, agents)
        threshold = config.minimum_path_length + agents.target * config.headway_time
        need = active & (agents.edge_count > 0) & (end - agents.arc < threshold)
        last = agents.edges.gather(-1, (agents.edge_count - 1).clamp_min(0)[..., None]).squeeze(-1)
        outgoing = rails.edge_outgoing[world, last.clamp_min(0)]  # [B, Q, M]
        exits = outgoing.clamp_min(0)
        exit_world = _world(exits)
        signal_slot = rails.edge_signal[exit_world, exits]
        has_signal = signal_slot >= 0
        slot = signal_slot.clamp_min(0)
        is_green = green[exit_world, slot] & has_signal
        is_red = red[exit_world, slot] & has_signal
        allowed = (outgoing >= 0) & torch.where(
            rails.edge_has_lights[exit_world, exits], is_green, ~is_red
        )
        off_route = follow_route[..., None] & ~rails.edge_on_route[exit_world, exits]
        score = rails.edge_curvature[exit_world, exits] + off_route * 1e6
        choice = outgoing.gather(-1, score.masked_fill(~allowed, INF).argmin(-1, keepdim=True))
        extend = need & allowed.any(-1)
        # nuPlan rebuilds the path from the clamped progress and resets progress to zero.
        bounded = _clamped(agents, end)
        full = extend & (agents.edge_count == r)
        dropped = torch.where(full, cumulative[..., 1], 0.0)
        edges = torch.where(full[..., None], agents.edges.roll(-1, -1), agents.edges)
        count = agents.edge_count - full.long()
        position = torch.arange(r, device=edges.device) == count[..., None]
        edges = torch.where(extend[..., None] & position, choice, edges)
        bounded = (bounded - dropped).clamp_min(0)
        # Explicit construction: dynamo mis-reconstructs NamedTuple._replace outputs.
        agents = _Agents(
            edges,
            count + extend.long(),
            torch.where(extend, bounded, agents.arc),
            torch.where(extend, bounded, agents.path_start),
            agents.velocity,
            agents.target,
        )
    return agents


def reactive_idm_step(
    rail_state: RailState,
    state,
    scene,
    next_step: Tensor,
    control_types: Tensor,
    current_valid: Tensor,
    advanced: Tensor,
    dt: Tensor,
    config: IDMConfig = IDMConfig(),
    blind: Tensor | None = None,
) -> tuple[RailPose, RailState]:
    """Advance every active IDM agent one step; returns all slots' poses and next state."""
    rails = scene.rails
    b, n = control_types.shape
    device = dt.device
    world = torch.arange(b, device=device)
    query = rail_state.query
    slot = query.clamp_min(0)
    agents = _Agents(*(_gather(getattr(rail_state, name), slot) for name in PER_AGENT))
    is_ego = slot == scene.ego_index[:, None]
    idm_slot = (control_types == ControlType.IDM) & rails.rail_valid
    active = (
        (query >= 0)
        & _gather(idm_slot & ~rail_state.removed & current_valid, slot)
        & advanced[:, None]
    )
    size = _gather(rails.rail_size, slot)
    length, half_width = size[..., 0], size[..., 1] / 2

    # 1. Radius filter around the pre-step ego center.
    ego = scene.ego_index
    ego_center = state.position[world, ego] + scene.reference_to_center[
        world, ego, None
    ] * heading_vector(state.heading[world, ego])
    center, heading = _pose(rails, agents)
    remove = active & ~is_ego & ((center - ego_center[:, None]).norm(dim=-1) > config.radius)
    active = active & ~remove
    removed = rail_state.removed | _scatter(rail_state.removed, slot, remove, remove)

    # Projected footprints from the current state, before any path rebuild.
    cumulative, end = _cumulative(rails, agents)
    front = agents.arc + length / 2 + agents.velocity * config.headway_time
    chain = _polyline(
        rails,
        agents,
        cumulative,
        torch.minimum(torch.maximum(agents.arc - length / 2, agents.path_start), end),
        torch.minimum(torch.maximum(front, agents.path_start), end),
        config.footprint_samples,
    )
    chain_rects, chain_ok = _rectangles(chain, half_width)
    chain_ok = chain_ok & (active & rail_state.started[:, None])[..., None]
    own_box = box_corners(center, heading, size)
    own_pieces = torch.cat((own_box[..., None, :, :], chain_rects), -3)  # [B, Q, P, 4, 2]
    own_valid = torch.cat((torch.ones_like(chain_ok[..., :1]), chain_ok), -1)
    pieces_count = own_pieces.shape[-3]

    # 2. Route planning with next-frame signals, then the buffered path to go.
    signal = (rails.signal_green[world, next_step], rails.signal_red[world, next_step])
    planned = _plan_route(rails, agents, active, signal, is_ego & config.ego_follows_route, config)
    cumulative, end = _cumulative(rails, planned)
    corridor = _polyline(
        rails, planned, cumulative, _clamped(planned, end), end, config.corridor_samples
    )
    corridor_rects, corridor_ok = _rectangles(corridor, half_width)

    # Entities: every dynamic slot, then static objects; IDM agents use footprints.
    ego_slot = torch.arange(n, device=device)[None] == ego[:, None]
    active_slot = _scatter(torch.zeros_like(idm_slot), slot, active, active)
    logged = (control_types == ControlType.LOG) & ~ego_slot
    reference = scene.reference_state
    ref_heading = reference.heading[world, next_step]
    offset = scene.reference_to_center[..., None]
    ref_center = reference.position[world, next_step] + offset * heading_vector(ref_heading)
    entity_center = state.position + offset * heading_vector(state.heading)
    entity_center = torch.where(logged[..., None], ref_center, entity_center)
    entity_center = _scatter(entity_center, slot, center, active)
    entity_heading = torch.where(logged, ref_heading, state.heading)
    entity_heading = _scatter(entity_heading, slot, heading, active)
    velocity = torch.where(logged[..., None], reference.velocity[world, next_step], state.velocity)
    speed = _scatter(velocity.norm(dim=-1), slot, agents.velocity, active)
    entity_size = torch.where(logged[..., None], scene.reference_size[world, next_step], scene.size)
    entity_size = torch.where(idm_slot[..., None], rails.rail_size, entity_size)
    present = torch.where(logged, scene.reference_valid[world, next_step], current_valid)
    # Railed IDM agents are present only while active; rail-less IDM actors replay logs.
    present = torch.where(idm_slot, active_slot, present)
    # nuPlan gives open-loop objects zero velocity; ego and vehicles lead with their speed.
    lead_speed = torch.where((scene.actor_type == ActorType.VEHICLE) | ego_slot, speed, 0.0)
    box = box_corners(entity_center, entity_heading, entity_size)[..., None, :, :]
    dynamic_pieces = _scatter(box.expand(-1, -1, pieces_count, -1, -1), slot, own_pieces, active)
    box_only = torch.zeros((b, n, pieces_count), dtype=torch.bool, device=device)
    box_only[..., 0] = True
    dynamic_valid = _scatter(box_only, slot, own_valid, active)
    static = scene.static
    static_box = box_corners(static.position, static.heading, static.size)[..., None, :, :]
    static_valid = torch.zeros(
        (b, static.heading.shape[1], pieces_count), dtype=torch.bool, device=device
    )
    static_valid[..., 0] = True
    pieces = torch.cat((dynamic_pieces, static_box.expand(-1, -1, pieces_count, -1, -1)), 1)
    pieces_valid = torch.cat((dynamic_valid, static_valid), 1)  # [B, J, P]
    entity_present = torch.cat((present, static.valid[world, next_step] & static.mask), 1)
    entity_speed = torch.cat((lead_speed, torch.zeros_like(static.heading)), 1)
    entity_heading = torch.cat((entity_heading, static.heading), 1)
    entity_point = torch.cat((entity_center, static.position), 1)
    corners = (pieces - entity_point[:, :, None, None]).norm(dim=-1)  # [B, J, P, 4]
    reach = corners.masked_fill(~pieces_valid[..., None], 0).flatten(-2).amax(-1)

    # Broad phase: entity bounding circles near the path to go.
    near = torch.full((b, query.shape[1], entity_point.shape[1]), INF, device=device)
    points = entity_point[:, None]
    for k in range(config.corridor_samples):
        a, c = corridor[:, :, k, None], corridor[:, :, k + 1, None]
        delta = c - a
        t = (((points - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
        near = torch.minimum(near, (points - a - t[..., None] * delta).norm(dim=-1))
    own = torch.zeros_like(near, dtype=torch.bool).scatter(-1, slot[..., None], True)
    eligible = (near - reach[:, None] <= half_width[..., None]) & entity_present[:, None] & ~own
    eligible = eligible & active[..., None]
    if blind is not None:
        # Blind agents ignore every object but still stop for red lines and path ends.
        eligible = eligible & ~_gather(blind, slot)[..., None]
    ranking = (entity_point[:, None] - center[:, :, None]).norm(dim=-1)
    count = min(config.candidates, entity_point.shape[1])
    candidate = ranking.masked_fill(~eligible, INF).topk(count, largest=False).indices
    candidate_ok = eligible.gather(-1, candidate)

    def take(value):
        index = candidate.reshape(b, -1, *([1] * (value.ndim - 2)))
        result = value.gather(1, index.expand(-1, -1, *value.shape[2:]))
        return result.reshape(*candidate.shape, *value.shape[2:])

    def expand(value, k):
        return value[:, :, None].expand(-1, -1, k, *value.shape[2:])

    candidate_pieces, candidate_valid = take(pieces), take(pieces_valid)
    touches = candidate_ok & _rectangles_touch(
        expand(corridor_rects, count), expand(corridor_ok, count), candidate_pieces, candidate_valid
    )
    gap = _pieces_distance(
        expand(own_pieces, count), expand(own_valid, count), candidate_pieces, candidate_valid
    ).masked_fill(~touches, INF)
    candidate_speed = take(entity_speed) * torch.cos(take(entity_heading) - heading[..., None])

    # 3. Red stop lines among the last ``route_memory`` route edges.
    memory = config.route_memory
    recent = planned.edge_count[..., None] - memory + torch.arange(memory, device=device)
    recent_edge = planned.edges.gather(-1, recent.clamp(0, planned.edges.shape[-1] - 1))
    recent_edge = torch.where(recent >= 0, recent_edge, -1)
    edge_index = recent_edge.clamp_min(0)
    edge_world = _world(edge_index)
    signal_slot = rails.edge_signal[edge_world, edge_index]
    red_now = signal[1][edge_world, signal_slot.clamp_min(0)]
    stop_red = (recent_edge >= 0) & (signal_slot >= 0) & red_now
    per_edge = rails.stop_points.shape[-3]
    stops = rails.stop_points[edge_world, edge_index].flatten(2, 3)  # [B, Q, L, V, 2]
    stop_vertices = rails.stop_mask[edge_world, edge_index].flatten(2, 3)
    stop_ok = stop_vertices.any(-1) & stop_red.repeat_interleave(per_edge, -1) & active[..., None]
    # Only red stop lines matter: keep up to ``red_stop_lines`` of them per agent (all of
    # them unless more are red at once) instead of every slot of the remembered edges.
    keep = min(config.red_stop_lines, stop_ok.shape[-1])
    pick = stop_ok.to(torch.int8).topk(keep, -1).indices  # [B, Q, K]
    stop_ok = stop_ok.gather(-1, pick)
    stops = stops.gather(2, pick[..., None, None].expand(-1, -1, -1, *stops.shape[-2:]))
    stop_vertices = stop_vertices.gather(2, pick[..., None].expand(-1, -1, -1, stops.shape[-2]))

    def stop_distance(quads, quads_valid):
        """Distance from each stop polygon to a union of quads [B, Q, P, 4, 2]."""
        lines, p = stops.shape[2], quads.shape[2]
        polygon = stops[:, :, :, None].expand(-1, -1, -1, p, -1, -1)
        polygon_mask = stop_vertices[:, :, :, None].expand(-1, -1, -1, p, -1)
        quad = expand(quads, lines)
        quad_mask = torch.ones(quad.shape[:-1], dtype=torch.bool, device=device)
        distance = polygon_distance(
            polygon_edges(polygon, polygon_mask),
            polygon_edges(quad, quad_mask),
            polygon[..., 0, :],
            quad[..., 0, :],
        )
        return distance.masked_fill(~expand(quads_valid, lines), INF).amin(-1)

    stop_touch = stop_ok & (stop_distance(corridor_rects, corridor_ok) <= 1e-6)
    stop_gap = stop_distance(own_pieces, own_valid).masked_fill(~stop_touch, INF)

    # 4. Nearest touching geometry, else the free road to the path end.
    gaps = torch.cat((gap, stop_gap), -1)
    speeds = torch.cat((candidate_speed, torch.zeros_like(stop_gap)), -1)
    nearest, index = gaps.min(-1)
    has_lead = nearest.isfinite()
    lead_gap = torch.where(has_lead, nearest, end - planned.arc - length / 2)
    lead_speed = torch.where(has_lead, speeds.gather(-1, index[..., None]).squeeze(-1), 0.0)

    # 5. End-segment speed limit, then forward-Euler IDM.
    last = planned.edges.gather(-1, (planned.edge_count - 1).clamp_min(0)[..., None]).squeeze(-1)
    limit = rails.edge_speed_limit[world[:, None], last.clamp_min(0)]
    target = torch.where(limit > 0, limit, planned.target)
    acceleration = idm_acceleration(planned.velocity, lead_gap, lead_speed, config, target)
    step_dt = dt[:, None]
    updated = _Agents(
        planned.edges,
        planned.edge_count,
        planned.arc + step_dt * planned.velocity,
        planned.path_start,
        (planned.velocity + step_dt * acceleration).clamp_min(0),
        target,
    )
    next_state = RailState(
        *(
            _scatter(getattr(rail_state, name), slot, getattr(updated, name), active)
            for name in PER_AGENT
        ),
        removed,
        rail_state.started | advanced,
        rail_state.query,
    )
    return rail_pose(scene, next_state), next_state
