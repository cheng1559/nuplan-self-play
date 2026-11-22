"""Select recorded states without integrating controls or dynamics."""

import torch

from cs5478_project.datatypes.state import WorldState


class LogAgent:
    """Replay every actor at each world's selected reference frame."""

    @staticmethod
    def get_state(reference_state: WorldState, frame_index: torch.Tensor) -> WorldState:
        """Gather one frame per world while preserving the fixed actor capacity."""
        world_index = torch.arange(frame_index.shape[0], device=frame_index.device)
        return WorldState(*(value[world_index, frame_index] for value in reference_state))
