"""Tensor outputs from one replay transition."""

from typing import NamedTuple

from torch import Tensor

from cs5478_project.datatypes.state import EpisodeState, WorldState


class StepResult(NamedTuple):
    """Post-step state and [B] transition flags.

    ``advanced`` identifies worlds that consumed a transition. ``just_finished``
    is true only on the transition that first reaches a scenario's final frame.
    ``actor_mask[B, N]`` marks actors present in the returned simulated world.
    Log actors follow recorded visibility; controlled actors persist after arrival.
    ``static_mask[B, S]`` selects visible static geometry from the scene buffer.
    """

    state: WorldState
    episode: EpisodeState
    advanced: Tensor
    just_finished: Tensor
    actor_mask: Tensor
    static_mask: Tensor
