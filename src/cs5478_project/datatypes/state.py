"""Fixed-structure tensor state shared by data producers and simulation."""

from typing import NamedTuple

import torch
from torch import Tensor


class WorldState(NamedTuple):
    """Actor state with arbitrary leading dimensions and a final actor axis.

    Position and velocity add a final coordinate axis of size two. Vehicle
    positions refer to the rear axle; other actor positions refer to their center.
    """

    position: Tensor
    heading: Tensor
    velocity: Tensor
    acceleration: Tensor
    acceleration_command: Tensor
    steering: Tensor
    yaw_rate: Tensor


class EpisodeState(NamedTuple):
    """Per-world frame counters and sticky completion flags, both shaped [B]."""

    step: Tensor
    done: Tensor


def clone_state(state: WorldState) -> WorldState:
    """Copy every state tensor without changing the container structure."""
    return WorldState(*(value.clone() for value in state))


def select_state(mask: Tensor, when_true: WorldState, when_false: WorldState) -> WorldState:
    """Select complete state rows using a mask over their leading dimensions."""
    return WorldState(
        *(
            torch.where(
                mask.reshape(mask.shape + (1,) * (true_value.ndim - mask.ndim)),
                true_value,
                false_value,
            )
            for true_value, false_value in zip(when_true, when_false, strict=True)
        )
    )
