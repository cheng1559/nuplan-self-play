"""Batched oriented-box tests and drivable-grid lookups for metrics and rewards."""

import torch
from torch import Tensor

from cs5478_project.datatypes.map import DRIVABLE_RESOLUTION_M


def _axes(box: Tensor) -> Tensor:
    """[..., 2, 2] unit edge directions of a rectangle with corners [..., 4, 2]."""
    edges = torch.stack((box[..., 1, :] - box[..., 0, :], box[..., 2, :] - box[..., 1, :]), -2)
    return edges / edges.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _project(points: Tensor, axes: Tensor) -> tuple[Tensor, Tensor]:
    """Min and max of points [..., P, 2] along axes [..., X, 2], each [..., X]."""
    projection = (points[..., None, :, :] * axes[..., :, None, :]).sum(-1)
    return projection.amin(-1), projection.amax(-1)


def box_overlap_over_time(ego: Tensor, other: Tensor, shift: Tensor, steps: int) -> Tensor:
    """Whether rectangle ``ego`` overlaps ``other`` translated by ``k * shift``.

    ``ego`` is [..., 4, 2], ``other`` is [..., K, 4, 2], and ``shift`` [..., K, 2] is the
    relative displacement per step. Returns [..., K, steps + 1] for k = 0 .. steps, using
    the separating-axis test (touching counts as overlap, as in shapely's intersects).
    Translation moves each projection interval by k * (shift . axis), so all steps share
    one set of projections.
    """
    ego = ego[..., None, :, :].expand_as(other)
    axes = torch.cat((_axes(ego), _axes(other)), -2)  # [..., K, 4, 2]
    ego_min, ego_max = _project(ego, axes)
    other_min, other_max = _project(other, axes)
    rate = (shift[..., None, :] * axes).sum(-1)  # [..., K, 4]
    k = torch.arange(steps + 1, device=ego.device, dtype=ego.dtype)
    moved = rate[..., None] * k  # [..., K, 4, S]
    separated = (other_min[..., None] + moved > ego_max[..., None]) | (
        other_max[..., None] + moved < ego_min[..., None]
    )
    return ~separated.any(-2)


def segment_hits_box(start: Tensor, end: Tensor, box: Tensor) -> Tensor:
    """Whether segments [..., 2] intersect rectangles [..., 4, 2] (separating axes)."""
    start, end = (point.expand(*box.shape[:-2], 2) for point in (start, end))
    direction = end - start
    normal = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axes = torch.cat((_axes(box), normal[..., None, :]), -2)  # [..., 3, 2]
    segment = torch.stack((start, end), -2)
    seg_min, seg_max = _project(segment, axes)
    box_min, box_max = _project(box, axes)
    return ~((seg_min > box_max) | (seg_max < box_min)).any(-1)


def segments_intersect(p0: Tensor, p1: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Whether segments p0-p1 and q0-q1 (broadcast [..., 2]) intersect, touching included."""

    def cross(o, a, b):
        return (a - o)[..., 0] * (b - o)[..., 1] - (a - o)[..., 1] * (b - o)[..., 0]

    d1, d2 = cross(q0, q1, p0), cross(q0, q1, p1)
    d3, d4 = cross(p0, p1, q0), cross(p0, p1, q1)
    straddle = (d1 * d2 <= 0) & (d3 * d4 <= 0)
    # Collinear (or degenerate) pairs also need overlapping extents.
    lower = torch.maximum(torch.minimum(p0, p1), torch.minimum(q0, q1))
    upper = torch.minimum(torch.maximum(p0, p1), torch.maximum(q0, q1))
    overlap = (lower <= upper).all(-1)
    collinear = (d1 == 0) & (d2 == 0)
    return straddle & (~collinear | overlap)


def boundary_crossing(map_batch, world: Tensor, box: Tensor, previous: Tensor, kind, k=12):
    """Road-boundary check: whether the edges of boxes [Q, 4, 2], or their corners' paths
    from ``previous`` [Q, 4, 2], touch one of the k map segments of ``kind`` nearest to
    the box center."""
    segments = map_batch.segments[world]  # [Q, S, 2, 2]
    valid = map_batch.segment_mask[world] & (map_batch.segment_type[world] == kind)
    center = box.mean(-2)[:, None]
    a, delta = segments[..., 0, :], segments[..., 1, :] - segments[..., 0, :]
    t = (((center - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    distance = (center - a - t[..., None] * delta).norm(dim=-1).masked_fill(~valid, torch.inf)
    near = distance.topk(min(k, distance.shape[1]), largest=False).indices  # [Q, k]
    rows = torch.arange(box.shape[0], device=box.device)[:, None]
    lines, line_valid = segments[rows, near], valid[rows, near]  # [Q, k, 2, 2], [Q, k]
    vehicle = torch.cat(
        (torch.stack((box, box.roll(-1, -2)), -2), torch.stack((previous, box), -2)), 1
    )  # [Q, 8, 2, 2]: edges now, then corner paths
    hit = segments_intersect(
        vehicle[:, :, None, 0], vehicle[:, :, None, 1], lines[:, None, :, 0], lines[:, None, :, 1]
    )
    return (hit & line_valid[:, None]).flatten(1).any(-1)


def drivable_distance(map_batch, world: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    """Bilinear distance outside the drivable area at points [Q, P, 2] of worlds [Q].

    Returns (distance [Q, P], known [Q, P]); points off the grid, or in worlds without
    one, are unknown and read as distance 0.
    """
    grid = map_batch.drivable_distance
    rows, columns = map_batch.drivable_size[world].unbind(-1)  # [Q]
    local = (points - map_batch.drivable_origin[world, None]) / DRIVABLE_RESOLUTION_M
    column, row = local.unbind(-1)
    known = (
        (column >= 0)
        & (row >= 0)
        & (column <= (columns - 1)[:, None])
        & (row <= (rows - 1)[:, None])
    )
    c0 = column.floor().clamp(0, grid.shape[2] - 2)
    r0 = row.floor().clamp(0, grid.shape[1] - 2)
    fc, fr = (column - c0).clamp(0, 1), (row - r0).clamp(0, 1)
    width = grid.shape[2]
    # Index the flattened batch grid directly; never copy a grid per query.
    base = (world[:, None] * grid.shape[1] + r0.long()) * width + c0.long()
    flat = grid.reshape(-1)

    def at(offset):
        return flat[base + offset].float()

    value = (
        at(0) * (1 - fc) * (1 - fr)
        + at(1) * fc * (1 - fr)
        + at(width) * (1 - fc) * fr
        + at(width + 1) * fc * fr
    )
    return torch.where(known, value, 0.0), known


def lane_info(map_batch, world: Tensor, points: Tensor) -> Tensor:
    """Nearest-sample lane-info grid values at points [Q, P, 2]; 0 off the grid."""
    grid = map_batch.lane_info
    rows, columns = map_batch.drivable_size[world].unbind(-1)
    local = (points - map_batch.drivable_origin[world, None]) / DRIVABLE_RESOLUTION_M
    column, row = local.round().long().unbind(-1)
    known = (column >= 0) & (row >= 0) & (column < columns[:, None]) & (row < rows[:, None])
    index = (world[:, None] * grid.shape[1] + row.clamp(0, grid.shape[1] - 1)) * grid.shape[
        2
    ] + column.clamp(0, grid.shape[2] - 1)
    return torch.where(known, grid.reshape(-1)[index].long(), 0)
