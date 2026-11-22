"""Scene-owned static map geometry and fixed-capacity simulation tensors."""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor


class MapElementType(IntEnum):
    """Polyline semantics shared by data adapters and tensor consumers."""

    LANE_CENTERLINE = 0
    LANE_CONNECTOR_CENTERLINE = 1
    LANE_BOUNDARY = 2
    ROAD_BOUNDARY = 3
    CROSSWALK = 4
    STOP_LINE = 5


# Signalized lane connectors whose polygon touches one map segment or chunk (20 m chunks
# in dense intersections touch up to 17).
SEGMENT_SIGNALS = 32
# SingleFrameNet map tokens: each polyline is cut into equal pieces of at most
# CHUNK_LENGTH_M meters, each resampled at CHUNK_POINTS evenly spaced points.
CHUNK_LENGTH_M = 20.0
CHUNK_POINTS = 8
MAX_MAP_CHUNKS = 2048
# Drivable-area distance grid: one sample every DRIVABLE_RESOLUTION_M meters, padded to
# MAX_DRIVABLE_CELLS per side in batches.
DRIVABLE_RESOLUTION_M = 0.5
MAX_DRIVABLE_CELLS = 1024
# Lane-info grid (same samples): low bits hold the nuPlan speed limit in units of
# LANE_INFO_SPEED_UNIT m/s; flag bits mark lanes/connectors and intersections.
LANE_INFO_SPEED_UNIT = 0.01
LANE_INFO_IN_LANE = 1 << 12
LANE_INFO_IN_INTERSECTION = 1 << 13


class MapBatch(NamedTuple):
    """Tensor-only map input with leading [B, S] capacity axes."""

    segments: Tensor
    segment_type: Tensor
    segment_mask: Tensor
    segment_signal: Tensor  # [B, S, SEGMENT_SIGNALS] int64 signal-table indices, -1 padding
    chunk_points: Tensor  # [B, C, CHUNK_POINTS, 2]
    chunk_type: Tensor  # [B, C]
    chunk_mask: Tensor  # [B, C]
    chunk_signal: Tensor  # [B, C, SEGMENT_SIGNALS]
    drivable_distance: Tensor  # [B, H, W] float16 meters outside the drivable area
    drivable_origin: Tensor  # [B, 2] scene coordinates of grid sample [0, 0]
    drivable_size: Tensor  # [B, 2] valid rows, columns; 0 when a world has no grid
    lane_info: Tensor  # [B, H, W] int16 lane-info grid on the same samples


@dataclass(frozen=True)
class MapData:
    """Unpadded CPU segments in scene coordinates, grouped by source polyline.

    Each segment is [[x0, y0], [x1, y1]] in meters. Polyline groups preserve
    source point order, including the closing edge of closed outlines. String
    identifiers and world transforms stay outside the compiled batch.
    """

    map_name: str
    world_from_scene: tuple[float, float, float]
    bounds: tuple[float, float, float, float]
    segments: Tensor
    segment_type: Tensor
    polyline_index: Tensor
    polyline_ids: tuple[str, ...]
    # [S, SEGMENT_SIGNALS] signal-table indices (see RailData), computed after simplification.
    segment_signal: Tensor | None = None
    # Polyline chunks cut from the unsimplified source geometry (see data/map_chunks.py):
    # [C, CHUNK_POINTS, 2] points, [C] types, and [C, SEGMENT_SIGNALS] signal indices.
    chunk_points: Tensor | None = None
    chunk_type: Tensor | None = None
    chunk_signal: Tensor | None = None
    # [H, W] float16 distance outside the drivable area, sampled every
    # DRIVABLE_RESOLUTION_M meters from ``drivable_origin`` (scene x, y).
    drivable_distance: Tensor | None = None
    drivable_origin: tuple[float, float] | None = None
    # [H, W] int16 lane facts on the same samples (see LANE_INFO_* constants).
    lane_info: Tensor | None = None

    def validate(self) -> None:
        """Validate finite CPU geometry and lossless polyline reconstruction."""
        if not isinstance(self.map_name, str) or not self.map_name.strip():
            raise ValueError("map_name must be a nonempty string")
        for name, values, length in (
            ("world_from_scene", self.world_from_scene, 3),
            ("bounds", self.bounds, 4),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != length
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"map.{name} must contain {length} finite numbers")
        xmin, ymin, xmax, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("map.bounds must be ordered [xmin, ymin, xmax, ymax]")
        if not isinstance(self.segments, Tensor) or self.segments.ndim != 3:
            raise ValueError("map.segments must have shape [S, 2, 2]")
        count = self.segments.shape[0]
        for name, value, shape, dtype in (
            ("segments", self.segments, (count, 2, 2), torch.float32),
            ("segment_type", self.segment_type, (count,), torch.int64),
            ("polyline_index", self.polyline_index, (count,), torch.int64),
        ):
            if (
                not isinstance(value, Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
            ):
                raise ValueError(f"map.{name} must be a CPU {dtype} tensor with shape {shape}")
        if not bool(torch.isfinite(self.segments).all()):
            raise ValueError("map.segments must contain only finite coordinates")
        if not bool(((self.segment_type >= 0) & (self.segment_type < len(MapElementType))).all()):
            raise ValueError("map.segment_type contains an unknown map element type")
        if (
            not isinstance(self.polyline_ids, tuple)
            or any(not isinstance(value, str) or not value for value in self.polyline_ids)
            or len(set(self.polyline_ids)) != len(self.polyline_ids)
        ):
            raise ValueError("map.polyline_ids must be unique nonempty strings")
        groups = torch.unique_consecutive(self.polyline_index)
        if not torch.equal(groups, torch.arange(len(self.polyline_ids))):
            raise ValueError("map.polyline_index must form ordered groups covering polyline_ids")
        same_line = self.polyline_index[1:] == self.polyline_index[:-1]
        if not torch.equal(self.segments[1:, 0][same_line], self.segments[:-1, 1][same_line]):
            raise ValueError("map segments within each polyline must be connected in order")
        if not torch.equal(self.segment_type[1:][same_line], self.segment_type[:-1][same_line]):
            raise ValueError("map segment types must be consistent within each polyline")
        if self.segment_signal is not None:
            signal = self.segment_signal
            if (
                not isinstance(signal, Tensor)
                or signal.shape != (count, SEGMENT_SIGNALS)
                or signal.dtype != torch.int64
                or signal.device.type != "cpu"
            ):
                raise ValueError("map.segment_signal must be CPU int64 [S, SEGMENT_SIGNALS]")
            if bool((signal < -1).any()):
                raise ValueError("map.segment_signal must use -1 padding")
        if self.chunk_points is None:
            if self.chunk_type is not None or self.chunk_signal is not None:
                raise ValueError("map chunk types and signals need chunk_points")
            return
        chunks = self.chunk_points.shape[0] if isinstance(self.chunk_points, Tensor) else -1
        for name, value, shape, dtype in (
            ("chunk_points", self.chunk_points, (chunks, CHUNK_POINTS, 2), torch.float32),
            ("chunk_type", self.chunk_type, (chunks,), torch.int64),
            ("chunk_signal", self.chunk_signal, (chunks, SEGMENT_SIGNALS), torch.int64),
        ):
            if name == "chunk_signal" and value is None:
                continue
            if (
                not isinstance(value, Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
            ):
                raise ValueError(f"map.{name} must be a CPU {dtype} tensor with shape {shape}")
        if not bool(torch.isfinite(self.chunk_points).all()):
            raise ValueError("map.chunk_points must contain only finite coordinates")
        if not bool(((self.chunk_type >= 0) & (self.chunk_type < len(MapElementType))).all()):
            raise ValueError("map.chunk_type contains an unknown map element type")
        if self.chunk_signal is not None and bool((self.chunk_signal < -1).any()):
            raise ValueError("map.chunk_signal must use -1 padding")
        grid = self.drivable_distance
        if (grid is None) != (self.drivable_origin is None):
            raise ValueError("map.drivable_distance and drivable_origin go together")
        if grid is not None and (
            not isinstance(grid, Tensor)
            or grid.ndim != 2
            or min(grid.shape) < 1
            or grid.dtype != torch.float16
            or not bool(torch.isfinite(grid).all())
            or len(self.drivable_origin) != 2
        ):
            raise ValueError("map.drivable_distance must be a finite float16 [H, W] grid")
        if self.lane_info is not None and (
            grid is None
            or not isinstance(self.lane_info, Tensor)
            or self.lane_info.shape != grid.shape
            or self.lane_info.dtype != torch.int16
        ):
            raise ValueError("map.lane_info must be int16 shaped like drivable_distance")


def validate_map_batch(batch: MapBatch, batch_size: int, device: torch.device) -> None:
    """Validate fixed-capacity map input outside compiled environment steps."""
    if not isinstance(batch, MapBatch) or batch.segment_mask.ndim != 2:
        raise ValueError("map must be a MapBatch with segment_mask shaped [B, S]")
    b, s = batch.segment_mask.shape
    if b != batch_size or s < 1:
        raise ValueError("map batch size must match actors and segment capacity must be positive")
    c = batch.chunk_mask.shape[-1] if batch.chunk_mask.ndim == 2 else -1
    for name, value, shape, dtype in (
        ("segments", batch.segments, (b, s, 2, 2), torch.float32),
        ("segment_type", batch.segment_type, (b, s), torch.int64),
        ("segment_mask", batch.segment_mask, (b, s), torch.bool),
        ("segment_signal", batch.segment_signal, (b, s, SEGMENT_SIGNALS), torch.int64),
        ("chunk_points", batch.chunk_points, (b, c, CHUNK_POINTS, 2), torch.float32),
        ("chunk_type", batch.chunk_type, (b, c), torch.int64),
        ("chunk_mask", batch.chunk_mask, (b, c), torch.bool),
        ("chunk_signal", batch.chunk_signal, (b, c, SEGMENT_SIGNALS), torch.int64),
        (
            "drivable_distance",
            batch.drivable_distance,
            (b, *batch.drivable_distance.shape[1:]),
            torch.float16,
        ),
        ("drivable_origin", batch.drivable_origin, (b, 2), torch.float32),
        ("drivable_size", batch.drivable_size, (b, 2), torch.int64),
        ("lane_info", batch.lane_info, batch.drivable_distance.shape, torch.int16),
    ):
        if value.shape != shape or value.dtype != dtype or value.device != device:
            raise ValueError(f"map.{name} has invalid shape, dtype, or device")
    if c < 1:
        raise ValueError("map chunk capacity must be positive")
    if not bool(torch.isfinite(batch.segments).all()) or not bool(
        torch.isfinite(batch.chunk_points).all()
    ):
        raise ValueError("map segments and chunks must contain finite values including padding")
    if not bool(((batch.segment_type >= 0) & (batch.segment_type < len(MapElementType))).all()):
        raise ValueError("map.segment_type contains an unknown map element type")
