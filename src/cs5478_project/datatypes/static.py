"""Static geometry stored once, with temporal observation validity."""

from typing import NamedTuple

import torch
from torch import Tensor


class StaticObjects(NamedTuple):
    """Scene [S] or batch [B, S] geometry; valid is [F, S] or [B, F, S]."""

    ids: Tensor
    position: Tensor
    heading: Tensor
    size: Tensor
    mask: Tensor
    valid: Tensor


def empty_static(shape: tuple[int, ...], frames: int) -> StaticObjects:
    """Allocate CPU padding without repeating geometry over time."""
    return StaticObjects(
        ids=torch.full(shape, -1, dtype=torch.int64),
        position=torch.zeros(shape + (2,), dtype=torch.float32),
        heading=torch.zeros(shape, dtype=torch.float32),
        size=torch.ones(shape + (2,), dtype=torch.float32),
        mask=torch.zeros(shape, dtype=torch.bool),
        valid=torch.zeros(shape[:-1] + (frames, shape[-1]), dtype=torch.bool),
    )


def validate_static(objects: StaticObjects, shape: tuple[int, ...], frames: int, device) -> None:
    """Validate static geometry at scene and batch boundaries."""
    if not isinstance(objects, StaticObjects):
        raise ValueError("static must be StaticObjects")
    for name, tensor, expected, dtype in (
        ("ids", objects.ids, shape, torch.int64),
        ("position", objects.position, shape + (2,), torch.float32),
        ("heading", objects.heading, shape, torch.float32),
        ("size", objects.size, shape + (2,), torch.float32),
        ("mask", objects.mask, shape, torch.bool),
        ("valid", objects.valid, shape[:-1] + (frames, shape[-1]), torch.bool),
    ):
        if tensor.shape != expected or tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"static.{name} has invalid shape, dtype, or device")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"static.{name} must be finite")
    if not bool((objects.size > 0).all()):
        raise ValueError("static sizes must be positive")
    if not torch.equal(objects.valid.any(dim=-2), objects.mask):
        raise ValueError("static validity must match object membership")
    if bool((objects.ids[objects.mask] < 0).any()):
        raise ValueError("static object IDs must be nonnegative")
