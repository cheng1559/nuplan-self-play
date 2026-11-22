"""Cut map polylines into fixed-size chunks, the SingleFrameNet map tokens.

Two-point segments are a few meters long, so a few hundred of them cover only the
nearest tens of meters. A chunk covers up to CHUNK_LENGTH_M meters of one polyline
with CHUNK_POINTS ordered points, so the same token budget sees much farther.
"""

import math
from dataclasses import replace

import numpy as np
import torch

from cs5478_project.datatypes.map import CHUNK_LENGTH_M, CHUNK_POINTS, MapData


def polyline_chunks(
    data: MapData, length: float = CHUNK_LENGTH_M, points: int = CHUNK_POINTS
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    """Split each polyline into equal pieces of at most ``length`` meters.

    Each piece is resampled at ``points`` evenly spaced arc-length positions, both ends
    included. Returns points [C, points, 2], types [C], and each piece's exact source
    geometry (float64 vertices) for geometric tests such as signal association.
    """
    if not math.isfinite(length) or length <= 0 or points < 2:
        raise ValueError("Chunks need a positive length and at least two points")
    segments = data.segments.double().numpy()
    line = data.polyline_index.numpy()
    kinds = data.segment_type.numpy()
    bounds = np.r_[0, np.flatnonzero(line[1:] != line[:-1]) + 1, len(line)]
    chunk_points, chunk_type, pieces = [], [], []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        vertices = np.concatenate((segments[start : start + 1, 0], segments[start:end, 1]))
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
        count = max(1, math.ceil(arc[-1] / length - 1e-9))
        cuts = np.linspace(0.0, arc[-1], count + 1)
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            at = np.linspace(a, b, points)
            sampled = np.stack(
                (np.interp(at, arc, vertices[:, 0]), np.interp(at, arc, vertices[:, 1])), -1
            )
            inner = vertices[(arc > a) & (arc < b)]
            chunk_points.append(sampled)
            chunk_type.append(int(kinds[start]))
            pieces.append(np.concatenate((sampled[:1], inner, sampled[-1:])))
    return (
        torch.tensor(np.asarray(chunk_points), dtype=torch.float32).reshape(-1, points, 2),
        torch.tensor(chunk_type, dtype=torch.int64),
        pieces,
    )


def with_chunks(data: MapData) -> MapData:
    """Attach chunks cut from this map's own segments; signals are left unset."""
    points, kinds, _ = polyline_chunks(data)
    result = replace(data, chunk_points=points, chunk_type=kinds, chunk_signal=None)
    result.validate()
    return result
