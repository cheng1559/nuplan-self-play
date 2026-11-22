"""Cached fixed-shape polyline projection and arc-length interpolation."""

from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.datatypes.route import RouteData


class RouteCache(NamedTuple):
    start: Tensor
    delta: Tensor
    length: Tensor
    arc_start: Tensor
    total: Tensor
    valid: Tensor


class RouteProjection(NamedTuple):
    progress: Tensor
    position: Tensor
    tangent: Tensor
    distance: Tensor


def cache_routes(routes: RouteData) -> RouteCache:
    """Compute immutable segment geometry once per reset, outside the step graph."""
    delta = routes.points[..., 1:, :] - routes.points[..., :-1, :]
    length = torch.linalg.vector_norm(delta, dim=-1)
    valid = routes.mask[..., 1:] & routes.mask[..., :-1] & (length > 1e-5)
    length = torch.where(valid, length, 0.0)
    cumulative = length.cumsum(dim=-1)
    return RouteCache(
        routes.points[..., :-1, :],
        delta,
        length,
        cumulative - length,
        cumulative[..., -1],
        valid,
    )


def project_route(
    cache: RouteCache,
    position: Tensor,
    minimum: Tensor | None = None,
    maximum: Tensor | None = None,
) -> RouteProjection:
    """Project [B, A, 2] or [B, A, K, 2] points without an actor loop."""
    extra = position.ndim == 4
    start = cache.start.unsqueeze(-3) if extra else cache.start
    delta = cache.delta.unsqueeze(-3) if extra else cache.delta
    length = cache.length.unsqueeze(-2) if extra else cache.length
    arc_start = cache.arc_start.unsqueeze(-2) if extra else cache.arc_start
    valid = cache.valid.unsqueeze(-2) if extra else cache.valid
    relative = position.unsqueeze(-2) - start
    fraction = ((relative * delta).sum(dim=-1) / length.square().clamp_min(1e-10)).clamp(0, 1)
    projected = start + fraction[..., None] * delta
    distance_sq = (position.unsqueeze(-2) - projected).square().sum(dim=-1)
    if minimum is not None:
        valid = valid & (arc_start + length >= minimum[..., None])
    if maximum is not None:
        valid = valid & (arc_start <= maximum[..., None])
    index = distance_sq.masked_fill(~valid, torch.inf).argmin(dim=-1)
    scalar_index = index[..., None]
    vector_index = index[..., None, None].expand(*index.shape, 1, 2)
    position_on_route = projected.gather(-2, vector_index).squeeze(-2)
    tangent = (delta / length.clamp_min(1e-5)[..., None]).expand_as(projected)
    return RouteProjection(
        (arc_start + fraction * length).gather(-1, scalar_index).squeeze(-1),
        position_on_route,
        tangent.gather(-2, vector_index).squeeze(-2),
        (position - position_on_route).square().sum(dim=-1).sqrt(),
    )


def sample_route(cache: RouteCache, progress: Tensor) -> Tensor:
    """Interpolate rear-axle/center route coordinates, clamped at both endpoints."""
    progress = torch.minimum(progress.clamp_min(0), cache.total)
    eligible = cache.valid & (cache.arc_start <= progress[..., None] + 1e-5)
    # The last eligible segment also handles repeated points and padded tails.
    indices = torch.arange(cache.length.shape[-1], device=progress.device)
    index = torch.where(eligible, indices, 0).amax(dim=-1)
    gather = index[..., None]
    vector_gather = index[..., None, None].expand(*index.shape, 1, 2)
    start = cache.start.gather(-2, vector_gather).squeeze(-2)
    delta = cache.delta.gather(-2, vector_gather).squeeze(-2)
    arc_start = cache.arc_start.gather(-1, gather).squeeze(-1)
    length = cache.length.gather(-1, gather).squeeze(-1)
    fraction = ((progress - arc_start) / length.clamp_min(1e-5)).clamp(0, 1)
    return start + fraction[..., None] * delta
