"""Synthetic rails without a nuPlan map: each vehicle's own route becomes one lane edge.

For tests and development scenes only. nuPlan scenes get real lane graphs from
``data.nuplan.rails`` at export time.
"""

from dataclasses import replace

import numpy as np
import torch

from cs5478_project.datatypes.rails import EDGE_POINTS, RAIL_EDGES, ReactiveRole, empty_rails
from cs5478_project.datatypes.scene import ActorType, Scene


def _resample(points: np.ndarray):
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, ~np.isclose(np.diff(progress), 0.0)]
    points, progress = points[keep], progress[keep]
    target = np.linspace(0.0, progress[-1], EDGE_POINTS)
    samples = np.stack([np.interp(target, progress, points[:, i]) for i in range(2)], -1)
    direction = np.diff(points, axis=0)
    heading = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
    heading = np.r_[heading, heading[-1]]
    return samples, np.interp(target, progress, heading), float(progress[-1])


def with_route_rails(scene: Scene, *, speed_limit: float = 0.0) -> Scene:
    """Give vehicles present at frame zero a one-edge rail along their (center) route."""
    scene.validate()
    if scene.routes is None:
        raise ValueError("Synthetic rails need actor routes")
    rails = empty_rails((), scene.num_frames, scene.num_actors)
    edge = 0
    for slot in range(scene.num_actors):
        kind = int(scene.actor_type[slot])
        rails.role[slot] = (
            ReactiveRole.OPEN_LOOP if kind != ActorType.VEHICLE else ReactiveRole.ABSENT
        )
        if slot == scene.ego_index:
            rails.role[slot] = ReactiveRole.EGO
        if kind != ActorType.VEHICLE or not bool(scene.reference_valid[0, slot]):
            continue
        route = scene.routes.points[slot][scene.routes.mask[slot]].double().numpy()
        if len(route) < 2 or np.linalg.norm(route[-1] - route[0]) < 1e-3:
            continue
        if edge >= RAIL_EDGES:
            raise ValueError("Too many synthetic rails")
        # Routes use the rear axle; rails carry box centers, as nuPlan agents do.
        direction = np.diff(route, axis=0)
        direction = np.r_[direction, direction[-1:]]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-9)
        center = route + float(scene.reference_to_center[slot]) * direction
        points, heading, length = _resample(center)
        rails.edge_points[edge] = torch.tensor(points, dtype=torch.float32)
        rails.edge_heading[edge] = torch.tensor(heading, dtype=torch.float32)
        rails.edge_length[edge] = length
        rails.edge_speed_limit[edge] = speed_limit
        rails.edge_mask[edge] = True
        rails.edge_on_route[edge] = slot == scene.ego_index
        start = center[0]
        rails.rail_valid[slot] = True
        rails.rail_edge[slot] = edge
        rails.rail_progress[slot] = 0.0
        rails.rail_pose[slot] = torch.tensor([start[0], start[1], heading[0]], dtype=torch.float32)
        rails.rail_speed[slot] = float(scene.initial_state.velocity[slot].norm())
        rails.rail_size[slot] = scene.size[slot]
        if slot != scene.ego_index:
            rails.role[slot] = ReactiveRole.IDM
        edge += 1
    return replace(scene, rails=rails)
