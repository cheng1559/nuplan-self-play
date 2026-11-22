"""Fixed-size actor routes in scene coordinates."""

from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor

ROUTE_POINTS = 100


class RouteSource(IntEnum):
    NONE = 0
    MAP = 1
    GT = 2


class RouteData(NamedTuple):
    """CPU [N, 100, 2] routes, or device [B, N, 100, 2] routes in batches.

    Valid points form a prefix. Stationary actors have one valid point; finite
    tail padding repeats the last point. Route coordinates use the same actor
    reference as physical state. A map route can extend beyond the map crop.
    GT timing records the first observation and continuous coverage in seconds;
    map routes and absent routes have zero timing values.
    """

    points: Tensor
    mask: Tensor
    source: Tensor
    start_time_s: Tensor
    horizon_s: Tensor


def validate_routes(routes: RouteData, actor_mask: Tensor) -> None:
    """Validate routes outside compiled kernels on the actor tensor device."""
    if not isinstance(routes, RouteData):
        raise ValueError("routes must be RouteData")
    shape = tuple(actor_mask.shape)
    for name, value, expected, dtype in (
        ("points", routes.points, shape + (ROUTE_POINTS, 2), torch.float32),
        ("mask", routes.mask, shape + (ROUTE_POINTS,), torch.bool),
        ("source", routes.source, shape, torch.int64),
        ("start_time_s", routes.start_time_s, shape, torch.float32),
        ("horizon_s", routes.horizon_s, shape, torch.float32),
    ):
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.dtype != dtype
            or value.device != actor_mask.device
        ):
            raise ValueError(f"routes.{name} has invalid shape, dtype, or device")
    if not bool(torch.isfinite(routes.points).all()):
        raise ValueError("routes.points must be finite, including padding")
    if bool((routes.mask[..., 1:] & ~routes.mask[..., :-1]).any()):
        raise ValueError("route validity must form a prefix")
    if bool((routes.mask & ~actor_mask[..., None]).any()):
        raise ValueError("padding actors cannot have routes")
    if not bool(((routes.source >= 0) & (routes.source <= RouteSource.GT)).all()):
        raise ValueError("unknown route source")
    if not torch.equal(routes.mask.any(dim=-1), routes.source != RouteSource.NONE):
        raise ValueError("route source must match point validity")
    for value in (routes.start_time_s, routes.horizon_s):
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("route times must be finite and nonnegative")
        if bool((value[routes.source != RouteSource.GT] != 0).any()):
            raise ValueError("only GT routes have temporal horizons")


def empty_routes(shape: tuple[int, ...]) -> RouteData:
    """Allocate finite CPU route padding."""
    return RouteData(
        torch.zeros(shape + (ROUTE_POINTS, 2), dtype=torch.float32),
        torch.zeros(shape + (ROUTE_POINTS,), dtype=torch.bool),
        torch.zeros(shape, dtype=torch.int64),
        torch.zeros(shape, dtype=torch.float32),
        torch.zeros(shape, dtype=torch.float32),
    )
