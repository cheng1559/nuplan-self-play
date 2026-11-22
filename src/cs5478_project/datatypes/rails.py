"""Exported nuPlan reactive-agent inputs: a local lane graph, signals, and per-actor rails.

Everything here is computed once at export time with nuPlan's own map API and IDM agent
builder, so the GPU runtime only follows edges, reads signal states, and integrates IDM.
Coordinates are scene-local meters. Arc lengths use nuPlan's baseline-path progress.
"""

from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor

RAIL_EDGES = 512
EDGE_POINTS = 64
EDGE_OUTGOING = 8
STOP_LINES = 4
STOP_LINE_VERTICES = 16
# Lane connectors with a signal status anywhere in the episode.
SIGNALS = 64


class ReactiveRole(IntEnum):
    """What nuPlan's reactive (IDMAgents) observation does with each dynamic slot."""

    ABSENT = 0  # not simulated: later vehicles, rejected vehicles, bicycles
    IDM = 1  # IDM agent on map rails
    OPEN_LOOP = 2  # replayed from the log (pedestrians and other open-loop types)
    EGO = 3  # the planner under test


class RailData(NamedTuple):
    """Optional leading batch axis; E edges, P points, M outgoing, S stop lines, V vertices."""

    edge_points: Tensor  # [E, P, 2] samples at uniform nuPlan progress
    edge_heading: Tensor  # [E, P] nuPlan discrete-path heading at each sample
    edge_length: Tensor  # [E] nuPlan baseline progress length
    edge_speed_limit: Tensor  # [E] m/s, 0 when the map has none
    edge_curvature: Tensor  # [E] |curvature| at arc 0, as plan_route compares
    edge_outgoing: Tensor  # [E, M] int64 edge indices in nuPlan order, -1 padding
    edge_has_lights: Tensor  # [E] bool, nuPlan has_traffic_lights (stop lines exist)
    edge_signal: Tensor  # [E] int64 index into the signal table, -1 without a status
    edge_on_route: Tensor  # [E] bool, part of the ego mission route
    edge_mask: Tensor  # [E] bool
    stop_points: Tensor  # [E, S, V, 2] traffic-light stop line polygons
    stop_mask: Tensor  # [E, S, V] bool prefixes
    signal_green: Tensor  # [F, L] bool
    signal_red: Tensor  # [F, L] bool
    speed_limit: Tensor  # [] ego starting lane speed limit (10 m/s without one)
    rail_valid: Tensor  # [N] bool, accepted by nuPlan's IDM builder (or ego's start lane)
    rail_edge: Tensor  # [N] int64 starting edge
    rail_progress: Tensor  # [N] arc length along the starting edge
    rail_pose: Tensor  # [N, 3] snapped box-center x, y, heading
    rail_speed: Tensor  # [N] initial speed along the path
    rail_target: Tensor  # [N] initial IDM target velocity
    rail_size: Tensor  # [N, 2] initial box length, width
    role: Tensor  # [N] int64 ReactiveRole


def empty_rails(shape: tuple[int, ...], frames: int, actors: int) -> RailData:
    e, p, m, s, v = RAIL_EDGES, EDGE_POINTS, EDGE_OUTGOING, STOP_LINES, STOP_LINE_VERTICES
    return RailData(
        torch.zeros(shape + (e, p, 2)),
        torch.zeros(shape + (e, p)),
        torch.ones(shape + (e,)),
        torch.zeros(shape + (e,)),
        torch.zeros(shape + (e,)),
        torch.full(shape + (e, m), -1, dtype=torch.int64),
        torch.zeros(shape + (e,), dtype=torch.bool),
        torch.full(shape + (e,), -1, dtype=torch.int64),
        torch.zeros(shape + (e,), dtype=torch.bool),
        torch.zeros(shape + (e,), dtype=torch.bool),
        torch.zeros(shape + (e, s, v, 2)),
        torch.zeros(shape + (e, s, v), dtype=torch.bool),
        torch.zeros(shape + (frames, SIGNALS), dtype=torch.bool),
        torch.zeros(shape + (frames, SIGNALS), dtype=torch.bool),
        torch.full(shape, 10.0),
        torch.zeros(shape + (actors,), dtype=torch.bool),
        torch.zeros(shape + (actors,), dtype=torch.int64),
        torch.zeros(shape + (actors,)),
        torch.zeros(shape + (actors, 3)),
        torch.zeros(shape + (actors,)),
        torch.full(shape + (actors,), 10.0),
        torch.ones(shape + (actors, 2)),
        torch.zeros(shape + (actors,), dtype=torch.int64),
    )


# Fields indexed by actor slot, sliced/padded with the dynamic actors.
ACTOR_FIELDS = (
    "rail_valid",
    "rail_edge",
    "rail_progress",
    "rail_pose",
    "rail_speed",
    "rail_target",
    "rail_size",
    "role",
)
FRAME_FIELDS = ("signal_green", "signal_red")


def validate_rails(data: RailData, shape: tuple[int, ...], frames: int, actors: int, device):
    template = empty_rails(shape, frames, actors)
    for name, actual, expected in zip(RailData._fields, data, template, strict=True):
        if (
            not isinstance(actual, Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device != device
        ):
            raise ValueError(f"rails.{name} has invalid shape, dtype, or device")
        if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
            raise ValueError(f"rails.{name} must be finite")
    if not bool((data.edge_length > 0).all()):
        raise ValueError("rails.edge_length must be positive, including padding")
    edges = data.edge_mask.shape[-1]
    outgoing = data.edge_outgoing
    if bool(((outgoing < -1) | (outgoing >= edges)).any()):
        raise ValueError("rails.edge_outgoing indexes outside the edge table")
    if bool((data.stop_mask[..., 1:] & ~data.stop_mask[..., :-1]).any()):
        raise ValueError("rails.stop_mask must form prefixes")
    if bool(((data.edge_signal < -1) | (data.edge_signal >= SIGNALS)).any()):
        raise ValueError("rails.edge_signal indexes outside the signal table")
    if bool((data.speed_limit <= 0).any()):
        raise ValueError("rails.speed_limit must be positive")
    if bool(((data.rail_edge < 0) | (data.rail_edge >= edges)).any()):
        raise ValueError("rails.rail_edge indexes outside the edge table")
    start = data.rail_edge.clamp(0, edges - 1)
    used = data.edge_mask.gather(-1, start)
    if bool((data.rail_valid & ~used).any()):
        raise ValueError("valid rails must start on a valid edge")
    if bool(((data.role < ReactiveRole.ABSENT) | (data.role > ReactiveRole.EGO)).any()):
        raise ValueError("rails.role contains an unknown reactive role")
    if bool((data.rail_size <= 0).any()):
        raise ValueError("rails.rail_size must be positive")
