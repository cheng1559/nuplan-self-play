"""Tensor polygon distances and swept-path intersection for ego IDM queries."""

import torch
from torch import Tensor

from cs5478_project.dynamics.bicycle import heading_vector


def cross(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def box_corners(center: Tensor, heading: Tensor, size: Tensor) -> Tensor:
    direction = heading_vector(heading)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    signs = center.new_tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    return (
        center[..., None, :]
        + signs[:, :1] * size[..., None, :1] * 0.5 * direction[..., None, :]
        + signs[:, 1:] * size[..., None, 1:] * 0.5 * lateral[..., None, :]
    )


def polygon_ends(points: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask.roll(-1, -1)[..., None], points.roll(-1, -2), points[..., :1, :])


def inside(points: Tensor, polygon: Tensor, mask: Tensor) -> Tensor:
    """Ray crossing: points [B, R, 2], polygons [B, O, V, 2]."""
    a, b = polygon[:, :, None], polygon_ends(polygon, mask)[:, :, None]
    p = points[:, None, :, None]
    dy = b[..., 1] - a[..., 1]
    safe_dy = torch.where(dy.abs() > 1e-12, dy, 1.0)
    crossing_x = a[..., 0] + (p[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) / safe_dy
    hits = ((a[..., 1] > p[..., 1]) != (b[..., 1] > p[..., 1])) & (p[..., 0] < crossing_x)
    return (hits & mask[:, :, None]).sum(-1).remainder(2).bool()


def point_segment_sq(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    delta = b - a
    fraction = ((p - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-20)
    return (p - a - fraction.clamp(0, 1)[..., None] * delta).square().sum(-1)


@torch.compiler.nested_compile_region
def segment_polygon_distance(a: Tensor, b: Tensor, polygon: Tensor, mask: Tensor) -> Tensor:
    """Exact Euclidean segment/polygon distances, output [B, O, R]."""
    u, v = polygon[:, :, None], polygon_ends(polygon, mask)[:, :, None]
    start, end = a[:, None, :, None], b[:, None, :, None]
    r, s = end - start, v - u
    denominator = cross(r, s)
    safe = torch.where(denominator.abs() > 1e-12, denominator, 1.0)
    t, q = cross(u - start, s) / safe, cross(u - start, r) / safe
    intersects = (denominator.abs() > 1e-12) & (t >= 0) & (t <= 1) & (q >= 0) & (q <= 1)
    distance = torch.minimum(point_segment_sq(start, u, v), point_segment_sq(end, u, v))
    distance = torch.minimum(distance, point_segment_sq(u, start, end))
    distance = torch.minimum(distance, point_segment_sq(v, start, end))
    distance = torch.where(intersects, 0.0, distance)
    distance = distance.masked_fill(~mask[:, :, None], torch.inf).amin(-1).sqrt()
    return torch.where(inside(a, polygon, mask), 0.0, distance)


def polygon_distance(points: Tensor, polygon: Tensor, mask: Tensor) -> Tensor:
    """Distances from one quadrilateral [B,4,2] to each polygon [B,O,V,2]."""
    distance = segment_polygon_distance(points, points.roll(-1, -2), polygon, mask).amin(-1)
    outer_mask = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
    contained = inside(polygon[:, :, 0], points[:, None], outer_mask[:, None]).squeeze(1)
    return torch.where(contained & mask.any(-1), 0.0, distance)


def rotate(points, heading):
    """Express scene-frame vectors [..., 2] in a frame rotated by ``heading``."""
    cosine, sine = heading.cos(), heading.sin()
    return torch.stack(
        (
            points[..., 0] * cosine + points[..., 1] * sine,
            -points[..., 0] * sine + points[..., 1] * cosine,
        ),
        -1,
    )
