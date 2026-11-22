"""Cache logged membership and persistent closed-loop membership at reset."""

import torch

from cs5478_project.datatypes.control import ControlType


def cache_actor_presence(scene, control_types):
    """Spawn controlled actors once, retaining logged visibility for playback actors."""
    has_input = torch.where(
        control_types == ControlType.IDM, scene.rails.rail_valid, scene.routes.mask.any(-1)
    )
    absent = control_types == ControlType.ABSENT
    controlled = (control_types != ControlType.LOG) & ~absent & has_input & scene.actor_mask
    spawned = scene.reference_valid.to(torch.int32).cumsum(dim=1) > 0
    frames = torch.arange(scene.reference_valid.shape[1], device=scene.actor_mask.device)
    within_episode = frames[None, :, None] < scene.num_frames[:, None, None]
    presence = torch.where(controlled[:, None], spawned, scene.reference_valid)
    return presence & within_episode & ~absent[:, None]
