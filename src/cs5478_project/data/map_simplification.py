"""Budgeted RDP map simplification using global chord-error refinement."""

import heapq
from dataclasses import replace

import numpy as np
import torch

from cs5478_project.datatypes import MapData


def simplify_map(data: MapData, *, max_segments: int = 2048) -> MapData:
    """Refine the worst curved span first while preserving source endpoints.

    All polylines survive. Closed outlines start with at least three segments
    when available in the source. Selected vertices keep their original order
    and coordinates. Straight spans do not spend unused capacity. Inputs are
    validated on CPU and are never mutated.
    """
    if isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments < 1:
        raise ValueError("max_segments must be a positive integer")
    data.validate()
    if len(data.segments) <= max_segments:
        return data
    if data.segment_signal is not None:
        raise ValueError("Simplify the map before associating segments with signals")
    indices = data.polyline_index.numpy()
    boundaries = np.r_[0, np.flatnonzero(indices[1:] != indices[:-1]) + 1, len(indices)]
    source = data.segments.numpy()
    points = [
        np.concatenate((source[start : start + 1, 0], source[start:end, 1])).astype(np.float64)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    selected: list[set[int]] = []
    heap: list[tuple[float, int, int, int, int]] = []

    def candidate(line: int, start: int, end: int) -> tuple[float, int] | None:
        if end - start <= 1:
            return None
        vertices = points[line]
        offset = vertices[start + 1 : end] - vertices[start]
        chord = vertices[end] - vertices[start]
        squared_length = float(chord @ chord)
        if squared_length > 0:
            projection = np.clip((offset @ chord) / squared_length, 0.0, 1.0)
            offset = offset - projection[:, None] * chord
        squared_distance = np.einsum("ij,ij->i", offset, offset)
        local = int(squared_distance.argmax())
        return float(squared_distance[local]), start + 1 + local

    def enqueue(line: int, start: int, end: int) -> None:
        result = candidate(line, start, end)
        if result is not None:
            error, split = result
            if error > 0:
                heapq.heappush(heap, (-error, line, start, end, split))

    count = 0
    for line, vertices in enumerate(points):
        last = len(vertices) - 1
        closed = np.array_equal(vertices[0], vertices[-1])
        minimum = min(3, last) if closed else 1
        anchors = [0, last]
        while len(anchors) <= minimum:
            options = []
            for start, end in zip(anchors[:-1], anchors[1:], strict=True):
                result = candidate(line, start, end)
                if result is not None:
                    error, split = result
                    options.append((error, split))
            _, split = max(options)
            anchors = sorted([*anchors, split])
        selected.append(set(anchors))
        count += minimum
        for start, end in zip(anchors[:-1], anchors[1:], strict=True):
            enqueue(line, start, end)
    if count > max_segments:
        raise ValueError(
            f"max_segments={max_segments} cannot preserve all polylines; at least {count} required"
        )
    while heap and count < max_segments:
        _, line, start, end, split = heapq.heappop(heap)
        selected[line].add(split)
        count += 1
        enqueue(line, start, split)
        enqueue(line, split, end)

    segments, types, groups = [], [], []
    for line, vertices in enumerate(points):
        retained = vertices[sorted(selected[line])].astype(np.float32)
        pairs = np.stack((retained[:-1], retained[1:]), axis=1)
        segments.append(pairs)
        types.append(np.full(len(pairs), data.segment_type[boundaries[line]].item(), np.int64))
        groups.append(np.full(len(pairs), line, np.int64))
    result = replace(
        data,
        segments=torch.from_numpy(np.concatenate(segments)),
        segment_type=torch.from_numpy(np.concatenate(types)),
        polyline_index=torch.from_numpy(np.concatenate(groups)),
    )
    result.validate()
    return result
