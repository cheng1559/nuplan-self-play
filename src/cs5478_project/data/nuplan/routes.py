"""Hermes map-route construction with local roadblock correction.

Reference: hermes/scripts/nuplan/export_hermes_webdataset.py. This module has no
runtime dependency on Hermes and runs only while preparing CPU scene inputs.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from cs5478_project.data.nuplan.route_correction import correct_route_roadblock_ids
from cs5478_project.data.routes import build_gt_routes, fit_route
from cs5478_project.datatypes import RouteData, RouteSource, Scene


@dataclass(frozen=True)
class RouteAnchor:
    x: float
    y: float
    yaw: float


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _line_coords(line_obj: object) -> list[tuple[float, float]]:
    linestring = getattr(line_obj, "linestring", None)
    if linestring is not None:
        return [(float(x), float(y)) for x, y, *_ in linestring.coords]
    discrete_path = getattr(line_obj, "discrete_path", None)
    if discrete_path is not None:
        return [(float(state.x), float(state.y)) for state in discrete_path]
    return []


def _edge_centerline_coords(edge: object) -> list[tuple[float, float]]:
    baseline_path = getattr(edge, "baseline_path", None)
    if baseline_path is None:
        return []
    return _line_coords(baseline_path)


def _route_roadblock_for_id(map_api: object, roadblock_id: str) -> object | None:
    roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
    if roadblock is None:
        roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    return roadblock


def _route_edge_heading(edge: object) -> float:
    coords = _edge_centerline_coords(edge)
    if len(coords) < 2:
        return 0.0
    return math.atan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0])


def _route_edge_anchor_score(edge: object, anchor: RouteAnchor) -> float:
    coords = _edge_centerline_coords(edge)
    if not coords:
        return float("inf")
    distance = min(math.hypot(x - anchor.x, y - anchor.y) for x, y in coords)
    heading_error = abs(float(_wrap_angle(_route_edge_heading(edge) - anchor.yaw)))
    return distance + 5.0 * heading_error


def _route_edge_continuity_score(edge: object, prev_edge: object) -> float:
    coords = _edge_centerline_coords(edge)
    prev_coords = _edge_centerline_coords(prev_edge)
    if not coords or not prev_coords:
        return float("inf")
    gap = math.hypot(coords[0][0] - prev_coords[-1][0], coords[0][1] - prev_coords[-1][1])
    heading_error = abs(
        float(_wrap_angle(_route_edge_heading(edge) - _route_edge_heading(prev_edge)))
    )
    return gap + 5.0 * heading_error


def _edge_ids(edges: Iterable[object]) -> set[str]:
    return {str(getattr(edge, "id", "")) for edge in edges}


def _choose_route_edge(
    edges: list[object], anchor: RouteAnchor, prev_edge: object | None
) -> object | None:
    candidates = [edge for edge in edges if _edge_centerline_coords(edge)]
    if not candidates:
        return None
    if prev_edge is None:
        return min(candidates, key=lambda edge: _route_edge_anchor_score(edge, anchor))

    outgoing_ids = _edge_ids(getattr(prev_edge, "outgoing_edges", []) or [])
    linked = [edge for edge in candidates if str(getattr(edge, "id", "")) in outgoing_ids]
    if not linked:
        prev_id = str(getattr(prev_edge, "id", ""))
        linked = [
            edge
            for edge in candidates
            if prev_id in _edge_ids(getattr(edge, "incoming_edges", []) or [])
        ]
    if linked:
        candidates = linked
    return min(candidates, key=lambda edge: _route_edge_continuity_score(edge, prev_edge))


def _stitch_route_edge_coords(edges: list[object]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for edge in edges:
        edge_coords = _edge_centerline_coords(edge)
        if not edge_coords:
            continue
        if (
            coords
            and math.hypot(coords[-1][0] - edge_coords[0][0], coords[-1][1] - edge_coords[0][1])
            < 1.0
        ):
            coords.extend(edge_coords[1:])
        else:
            coords.extend(edge_coords)
    return coords


def _route_start_index(roadblocks: list[object], anchor: RouteAnchor) -> int:
    scores = [
        min(
            (
                _route_edge_anchor_score(edge, anchor)
                for edge in getattr(roadblock, "interior_edges", []) or []
            ),
            default=float("inf"),
        )
        for roadblock in roadblocks
    ]
    if not scores:
        return 0
    return int(np.argmin(scores))


def _sample_route_points_from_map(
    *,
    map_api: object | None,
    route_roadblock_ids: list[str],
    anchor: RouteAnchor,
) -> np.ndarray | None:
    if map_api is None or SemanticMapLayer is None or not route_roadblock_ids:
        return None

    roadblocks = [
        roadblock
        for roadblock_id in route_roadblock_ids
        if (roadblock := _route_roadblock_for_id(map_api, roadblock_id)) is not None
    ]
    if not roadblocks:
        return None

    route_edges: list[object] = []
    prev_edge: object | None = None
    for roadblock in roadblocks[_route_start_index(roadblocks, anchor) :]:
        edge = _choose_route_edge(
            list(getattr(roadblock, "interior_edges", []) or []), anchor, prev_edge
        )
        if edge is None:
            continue
        route_edges.append(edge)
        prev_edge = edge

    coords = _stitch_route_edge_coords(route_edges)
    if not coords:
        return None
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    local = ((np.asarray(coords) - [anchor.x, anchor.y]) @ np.array([[c, -s], [s, c]])).astype(
        np.float32
    )
    return fit_route(local, smooth=True)[0].numpy()


def ego_map_route(map_api, route_roadblock_ids, x, y, yaw):
    """Ego's mission route from the map: (points [100, 2] in the frame of the pose
    ``x, y, yaw``, or None without usable baselines; corrected roadblock ids)."""
    ego = SimpleNamespace(rear_axle=StateSE2(x, y, yaw))
    corrected = correct_route_roadblock_ids(ego, map_api, route_roadblock_ids)
    points = _sample_route_points_from_map(
        map_api=map_api, route_roadblock_ids=corrected, anchor=RouteAnchor(x, y, yaw)
    )
    return points, corrected


def build_scene_routes(
    scene: Scene,
    map_api: object,
    route_roadblock_ids: list[str],
    *,
    gt_routes: RouteData | None = None,
) -> tuple[RouteData, list[str]]:
    """Generate ego from map topology and every other actor from unsmoothed GT.

    Repair route-roadblock connectivity, choose the starting block by ego pose,
    prefer linked lanes, stitch baselines, smooth at map resolution, then sample
    100 points by arc length. Missing ego map routes raise instead of silently
    substituting GT. Map-route preparation uses the full map, before RDP sampling.
    """
    scene.validate()
    if scene.map is None:
        raise ValueError("ego map route requires the scene world transform")
    origin_x, origin_y, origin_heading = scene.map.world_from_scene
    position = scene.initial_state.position[scene.ego_index].numpy()
    c, s = math.cos(origin_heading), math.sin(origin_heading)
    x = float(origin_x + c * position[0] - s * position[1])
    y = float(origin_y + s * position[0] + c * position[1])
    yaw = float(origin_heading + scene.initial_state.heading[scene.ego_index])
    points, corrected = ego_map_route(map_api, route_roadblock_ids, x, y, yaw)
    if points is None:
        raise ValueError("nuPlan route has no usable map baselines")
    # The helper's ego anchor differs from the scene origin when ego is not at zero.
    transform = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    world = points.astype(np.float64) @ transform.T + [x, y]
    scene_rotation = np.array([[c, -s], [s, c]])
    local = (world - [origin_x, origin_y]) @ scene_rotation
    routes = (
        build_gt_routes(scene)
        if gt_routes is None
        else RouteData(*(value.clone() for value in gt_routes))
    )
    routes.points[scene.ego_index] = torch.tensor(local, dtype=torch.float32)
    routes.mask[scene.ego_index] = True
    routes.source[scene.ego_index] = RouteSource.MAP
    routes.start_time_s[scene.ego_index] = 0.0
    routes.horizon_s[scene.ego_index] = 0.0
    return routes, corrected
