"""CPU preparation of actor routes, with explicit validity at the scene origin."""

import math

import numpy as np
import torch

from cs5478_project.datatypes.route import ROUTE_POINTS, RouteData, RouteSource, empty_routes
from cs5478_project.datatypes.scene import Scene
from cs5478_project.defaults import NPC_ROUTE_HORIZON_S


def fit_route(points: np.ndarray, *, smooth: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Hermes' arc-length sampling and optional five-point smoothing.

    Unlike zero-sentinel route readers, a point at (0, 0) remains valid. GT routes
    are sampled without smoothing. Endpoints survive smoothing and resampling.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all() or len(points) == 0:
        raise ValueError("route input must contain finite points")
    points = points[
        np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > (1e-3 if smooth else 0.0)]
    ]
    if len(points) == 1:
        result = np.repeat(points, ROUTE_POINTS, axis=0)
        mask = torch.arange(ROUTE_POINTS) == 0
        return torch.tensor(result, dtype=torch.float32), mask

    def resample(vertices, count):
        progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
        target = np.linspace(0.0, progress[-1], count)
        return np.stack(
            [np.interp(target, progress, vertices[:, axis]) for axis in range(2)], axis=-1
        )

    if smooth and len(points) >= 3:
        fitted = resample(points, len(points))
        window = min(5, len(fitted))
        window -= 1 - window % 2
        padded = np.pad(fitted, ((window // 2, window // 2), (0, 0)), mode="edge")
        fitted = np.stack(
            [
                np.convolve(padded[:, axis], np.ones(window) / window, mode="valid")
                for axis in range(2)
            ],
            axis=-1,
        )
        fitted[0], fitted[-1] = points[0], points[-1]
        points = fitted
    result = resample(points, ROUTE_POINTS)
    return torch.tensor(result, dtype=torch.float32), torch.ones(ROUTE_POINTS, dtype=torch.bool)


def build_gt_routes(
    scene: Scene,
    *,
    future_positions: torch.Tensor | None = None,
    future_valid: torch.Tensor | None = None,
    horizon_s: float = NPC_ROUTE_HORIZON_S,
) -> RouteData:
    """Sample up to 30 s of consecutive GT starting at each actor's first appearance.

    Extended inputs share scene actor slots, scene coordinates, and dt, but their
    time axis may exceed the episode. Missing observations terminate a route;
    later reappearances are never joined across the gap. No extrapolation or
    padding contributes to the measured horizon. Validity is explicit at zero.
    """
    if not math.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("route horizon must be finite and nonnegative")
    if (future_positions is None) != (future_valid is None):
        raise ValueError("extended GT positions and validity must be supplied together")
    positions = scene.reference_state.position if future_positions is None else future_positions
    valid = scene.reference_valid if future_valid is None else future_valid
    if (
        positions.ndim != 3
        or positions.shape[1:] != (scene.num_actors, 2)
        or positions.dtype != torch.float32
        or positions.device.type != "cpu"
        or not bool(torch.isfinite(positions).all())
    ):
        raise ValueError("GT positions must be finite CPU float32 [T, N, 2]")
    if (
        valid.shape != positions.shape[:2]
        or valid.dtype != torch.bool
        or valid.device.type != "cpu"
    ):
        raise ValueError("GT validity must be CPU bool [T, N]")
    if future_positions is not None and (
        len(valid) < scene.num_frames
        or not torch.equal(valid[: scene.num_frames], scene.reference_valid)
        or not torch.equal(
            positions[: scene.num_frames][scene.reference_valid],
            scene.reference_state.position[scene.reference_valid],
        )
    ):
        raise ValueError("extended GT must match episode actor slots and observed states")
    routes = empty_routes((scene.num_actors,))
    max_steps = int(math.floor(horizon_s / scene.dt + 1e-6))
    for slot in scene.actor_mask.nonzero().flatten().tolist():
        indices = valid[:, slot].nonzero().flatten()
        if not len(indices):
            raise ValueError("every episode actor needs a GT route observation")
        start = int(indices[0])
        stop = min(len(valid), start + max_steps + 1)
        missing = (~valid[start:stop, slot]).nonzero().flatten()
        if len(missing):
            stop = start + int(missing[0])
        points, mask = fit_route(positions[start:stop, slot].numpy())
        routes.points[slot] = points
        routes.mask[slot] = mask
        routes.source[slot] = RouteSource.GT
        routes.start_time_s[slot] = start * scene.dt
        routes.horizon_s[slot] = (stop - start - 1) * scene.dt
    return routes
