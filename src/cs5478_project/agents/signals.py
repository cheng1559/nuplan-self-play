"""Per-frame signal lookups from the exported signal table."""

import torch


def _red(scene, step, index):
    world = torch.arange(step.shape[0], device=step.device)
    red = scene.rails.signal_red[world, step]  # [B, L]
    hits = red.gather(1, index.clamp_min(0).flatten(1)).view_as(index)
    return (hits & (index >= 0)).any(-1)


def segment_red(scene, step):
    """[B, S] map segments touching a lane connector whose signal is red at ``step``."""
    return _red(scene, step, scene.map.segment_signal)


def chunk_red(scene, step):
    """[B, C] map chunks touching a lane connector whose signal is red at ``step``."""
    return _red(scene, step, scene.map.chunk_signal)
