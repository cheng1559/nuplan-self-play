"""Broadcast polygon/polyline distances for occupancy queries, masked and fixed-shape.

Polygons are closed vertex lists [..., V, 2] with prefix masks; polylines are open
point lists [..., K, 2]. Distances are zero when shapes touch or one contains the other.
"""

import torch
from torch import Tensor

INF = float("inf")


def _cross(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_segment(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    delta = b - a
    t = (((p - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - a - t[..., None] * delta).norm(dim=-1)


def segment_distance(p0: Tensor, p1: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Distance between segments p and q (broadcast [..., 2]); zero if they cross."""
    d1, d2 = p1 - p0, q1 - q0
    o1, o2 = _cross(d1, q0 - p0), _cross(d1, q1 - p0)
    o3, o4 = _cross(d2, p0 - q0), _cross(d2, p1 - q0)
    collinear = (o1 == 0) & (o2 == 0)
    crossing = (o1 * o2 <= 0) & (o3 * o4 <= 0) & ~collinear
    distance = torch.minimum(
        torch.minimum(_point_segment(p0, q0, q1), _point_segment(p1, q0, q1)),
        torch.minimum(_point_segment(q0, p0, p1), _point_segment(q1, p0, p1)),
    )
    return torch.where(crossing, 0.0, distance)


def polygon_edges(points: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-ring edges (start, end, valid) for prefix-masked vertices."""
    following = torch.where(mask.roll(-1, -1)[..., None], points.roll(-1, -2), points[..., :1, :])
    return points, following, mask


def polyline_edges(points: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return points[..., :-1, :], points[..., 1:, :], mask[..., :-1] & mask[..., 1:]


def edge_set_distance(first, second) -> Tensor:
    """Minimum distance between two masked edge sets with matching leading shapes."""
    a0, a1, am = first
    b0, b1, bm = second
    distance = segment_distance(
        a0[..., :, None, :], a1[..., :, None, :], b0[..., None, :, :], b1[..., None, :, :]
    )
    valid = am[..., :, None] & bm[..., None, :]
    return distance.masked_fill(~valid, INF).flatten(-2).amin(-1)


def point_in_polygon(point: Tensor, edges) -> Tensor:
    """Even-odd test of point [..., 2] against polygon edges [..., V, 2]."""
    a, b, mask = edges
    p = point[..., None, :]
    dy = b[..., 1] - a[..., 1]
    safe = torch.where(dy.abs() > 1e-12, dy, 1.0)
    x = a[..., 0] + (p[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) / safe
    hits = ((a[..., 1] > p[..., 1]) != (b[..., 1] > p[..., 1])) & (p[..., 0] < x) & mask
    return hits.sum(-1).remainder(2).bool()


def polygon_distance(first, second, first_point: Tensor, second_point: Tensor) -> Tensor:
    """Polygon-polygon distance; ``*_point`` is any vertex of each polygon."""
    distance = edge_set_distance(first, second)
    inside = point_in_polygon(first_point, second) | point_in_polygon(second_point, first)
    return torch.where(inside, 0.0, distance)


def polygon_polyline_distance(polygon, line, line_point: Tensor) -> Tensor:
    """Polygon-to-polyline distance, zero when the polyline starts inside the polygon."""
    distance = edge_set_distance(polygon, line)
    return torch.where(point_in_polygon(line_point, polygon), 0.0, distance)
