"""Scene-local lane edges, boundaries, crosswalks, stop lines, and drivable contours."""

import math
from collections import Counter

import numpy as np
import torch

from cs5478_project.datatypes.map import (
    DRIVABLE_RESOLUTION_M,
    LANE_INFO_IN_INTERSECTION,
    LANE_INFO_IN_LANE,
    LANE_INFO_SPEED_UNIT,
    MAX_DRIVABLE_CELLS,
    MapData,
    MapElementType,
)

# The drivable-distance grid extends this far past the map crop, so actors that leave
# the crop are still evaluated.
DRIVABLE_GRID_MARGIN_M = 25.0
# Grid samples closer than this to the drivable area get exact distances; farther ones
# keep the Euclidean distance transform, which is accurate to about one cell.
EXACT_DISTANCE_BAND_M = 3.0
MAX_DRIVABLE_DISTANCE_M = 20.0


def _remove_backtracks(coords, tolerance=1e-6):
    """Drop zero-width A-B-A spikes while keeping the surrounding line or ring."""

    def same(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance

    cleaned = list(coords)
    while len(cleaned) >= 4 and same(cleaned[0], cleaned[-1]) and same(cleaned[1], cleaned[-2]):
        cleaned = cleaned[1:-1]
    output = []
    for point in cleaned:
        if len(output) >= 2 and same(output[-2], point):
            output.pop()
        elif not output or not same(output[-1], point):
            output.append(point)
    return output


def _polygons(geometry):
    from shapely.validation import make_valid

    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [part for item in geometry.geoms for part in _polygons(item)]
    return []


def _lines(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [[(float(x), float(y)) for x, y, *_ in geometry.coords]]
    if geometry.geom_type in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        return [line for part in geometry.geoms for line in _lines(part)]
    if geometry.geom_type == "Polygon":
        return _lines(geometry.exterior) + [
            line for ring in geometry.interiors for line in _lines(ring)
        ]
    return []


def drivable_area(map_api, center, radius, layers):
    """Union of the given polygon layers within a radius (world frame), or None."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    region = Point(*center).buffer(radius)
    geometries = []
    for layer in layers:
        try:
            frame = map_api._get_vector_map_layer(layer)
        except (KeyError, ValueError):
            continue
        geometries.extend(frame[frame["geometry"].intersects(region)]["geometry"].tolist())
    polygons = [part for geometry in geometries for part in _polygons(geometry)]
    return unary_union(polygons) if polygons else None


def drivable_area_boundaries(area):
    """Outer and hole contours of a merged drivable area."""
    if area is None:
        return []
    boundaries = [_remove_backtracks(line) for line in _lines(area.boundary)]
    return [line for line in boundaries if len(line) >= 2]


def grid_window(lower, upper, focus):
    """Grid corners: the crop plus DRIVABLE_GRID_MARGIN_M, at most MAX_DRIVABLE_CELLS
    samples per side. A larger window is centered on ``focus`` [N, 2] (ego's log)."""
    lower = np.asarray(lower, dtype=np.float64) - DRIVABLE_GRID_MARGIN_M
    upper = np.asarray(upper, dtype=np.float64) + DRIVABLE_GRID_MARGIN_M
    span = (MAX_DRIVABLE_CELLS - 1) * DRIVABLE_RESOLUTION_M
    center = (np.asarray(focus).min(0) + np.asarray(focus).max(0)) / 2
    start = np.clip(center - span / 2, lower, np.maximum(upper - span, lower))
    return np.maximum(lower, start), np.minimum(upper, start + span)


def _grid_points(lower, upper):
    resolution = DRIVABLE_RESOLUTION_M
    origin = np.floor(np.asarray(lower) / resolution) * resolution
    count = np.floor((np.asarray(upper) - origin) / resolution).astype(int) + 1
    width, height = np.minimum(count, MAX_DRIVABLE_CELLS).tolist()
    xs = origin[0] + np.arange(width) * resolution
    ys = origin[1] + np.arange(height) * resolution
    return origin, xs, ys


def _rasterize(polygon, xs, ys):
    """(rows, columns) of grid samples inside or on ``polygon``."""
    import shapely

    x0, y0, x1, y1 = polygon.bounds
    columns = np.nonzero((xs >= x0) & (xs <= x1))[0]
    rows = np.nonzero((ys >= y0) & (ys <= y1))[0]
    if not len(columns) or not len(rows):
        return rows[:0], columns[:0]
    r, c = np.meshgrid(rows, columns, indexing="ij")
    x, y = xs[c], ys[r]
    inside = shapely.contains_xy(polygon, x, y) | shapely.intersects_xy(polygon, x, y)
    return r[inside], c[inside]


def lane_info_grid(map_api, center, radius, to_local, lower, upper):
    """int16 grid of nuPlan lane facts at the drivable grid's samples.

    Bits below LANE_INFO_IN_LANE hold the speed limit in LANE_INFO_SPEED_UNIT: a lane's
    own limit, or for a lane connector the largest limit of its incoming and outgoing
    lanes, as speed_limit_compliance uses. Lanes take precedence over connectors, and
    overlaps keep the largest limit. LANE_INFO_IN_LANE marks samples in any lane or
    connector; LANE_INFO_IN_INTERSECTION marks samples in an intersection polygon.
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Polygon

    _, xs, ys = _grid_points(lower, upper)
    lane_limit = np.zeros((len(ys), len(xs)))
    connector_limit = np.zeros_like(lane_limit)
    in_lane = np.zeros(lane_limit.shape, dtype=bool)
    in_intersection = np.zeros_like(in_lane)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.INTERSECTION]
    nearby = map_api.get_proximal_map_objects(Point2D(*center), radius, layers)

    def local(obj):
        return Polygon(to_local(np.asarray(obj.polygon.exterior.coords)[:, :2]))

    for lane in nearby[SemanticMapLayer.LANE]:
        r, c = _rasterize(local(lane), xs, ys)
        lane_limit[r, c] = np.maximum(lane_limit[r, c], lane.speed_limit_mps or 0.0)
        in_lane[r, c] = True
    for connector in nearby[SemanticMapLayer.LANE_CONNECTOR]:
        edges = connector.incoming_edges + connector.outgoing_edges
        limits = [edge.speed_limit_mps for edge in edges]
        limit = max(limits) if limits and all(limits) else 0.0
        r, c = _rasterize(local(connector), xs, ys)
        connector_limit[r, c] = np.maximum(connector_limit[r, c], limit)
        in_lane[r, c] = True
    for intersection in nearby[SemanticMapLayer.INTERSECTION]:
        r, c = _rasterize(local(intersection), xs, ys)
        in_intersection[r, c] = True
    limit = np.where(lane_limit > 0, lane_limit, connector_limit)
    units = np.clip(np.round(limit / LANE_INFO_SPEED_UNIT), 0, LANE_INFO_IN_LANE - 1)
    grid = units.astype(np.int32) + in_lane * LANE_INFO_IN_LANE
    grid += in_intersection * LANE_INFO_IN_INTERSECTION
    return torch.tensor(grid.astype(np.int16))


def drivable_distance_grid(area_local, lower, upper):
    """Distance outside the drivable area at grid samples, in meters (0 inside).

    ``lower`` and ``upper`` are the grid window (see grid_window).

    Samples sit at ``origin + (column, row) * DRIVABLE_RESOLUTION_M`` in scene coordinates.
    Returns (float16 [H, W] grid, origin). Distances are capped at MAX_DRIVABLE_DISTANCE_M.
    """
    import shapely
    from scipy.ndimage import distance_transform_edt

    resolution = DRIVABLE_RESOLUTION_M
    origin, xs, ys = _grid_points(lower, upper)
    x, y = np.meshgrid(xs, ys)
    if area_local is None:
        grid = np.full(x.shape, MAX_DRIVABLE_DISTANCE_M)
    else:
        shapely.prepare(area_local)
        inside = shapely.contains_xy(area_local, x, y) | shapely.intersects_xy(area_local, x, y)
        grid = distance_transform_edt(~inside) * resolution
        band = (grid > 0) & (grid < EXACT_DISTANCE_BAND_M)
        grid[band] = shapely.distance(shapely.points(x[band], y[band]), area_local)
    grid = np.minimum(grid, MAX_DRIVABLE_DISTANCE_M)
    return torch.tensor(grid, dtype=torch.float16), (float(origin[0]), float(origin[1]))


def build_map(scene, map_api, origin, *, points=None, focus=None, grids=True):
    """Crop map polylines around all actor positions plus a 25 m margin, in scene frame.

    ``origin`` is (x, y, yaw) of the scene frame in world coordinates. ``points`` [N, 2]
    replaces the logged actor positions as what the crop must cover, and ``focus`` [N, 2]
    replaces ego's log as the center of the drivable grid (both in scene frame): a
    planner knows no log. Without ``grids``, the drivable-distance and lane-info grids
    (read only by the evaluator) are left out.
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import LineString, box

    ox, oy, yaw = origin
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    if points is None:
        dynamic = scene.reference_state.position[scene.reference_valid].numpy()
        static = scene.static.position[scene.static.mask].numpy()
        points = np.concatenate((dynamic, static))
    points = np.asarray(points, dtype=np.float32)
    margin = 25 + float(torch.cat((scene.size, scene.static.size)).norm(dim=-1).max()) / 2
    lower, upper = points.min(0) - margin, points.max(0) + margin
    corners = np.array(
        [[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]
    ) @ rotation.T + [ox, oy]
    crop = box(*lower, *upper)
    midpoint = (corners.min(0) + corners.max(0)) / 2
    radius = float(np.linalg.norm(corners - midpoint, axis=-1).max())
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.STOP_LINE,
    ]
    nearby = map_api.get_proximal_map_objects(Point2D(*midpoint), radius, layers)
    segments, types, groups, ids, seen, counts = [], [], [], [], set(), Counter()

    def append(coords, kind, identity):
        if identity in seen or len(coords) < 2:
            return
        seen.add(identity)
        local = (np.asarray(coords, dtype=np.float64)[:, :2] - [ox, oy]) @ rotation
        clipped = LineString(local).intersection(crop)
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        for index, part in enumerate(parts):
            if part.geom_type != "LineString" or part.length < 1e-5:
                continue
            vertices = np.asarray(part.coords, dtype=np.float32)
            keep = np.r_[True, np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-5]
            vertices = vertices[keep]
            if len(vertices) < 2:
                continue
            pairs = np.stack((vertices[:-1], vertices[1:]), axis=1)
            segments.extend(pairs)
            types.extend([kind] * len(pairs))
            groups.extend([len(ids)] * len(pairs))
            ids.append(f"{identity}:{index}")
            counts[int(kind)] += 1

    for layer, kind in (
        (SemanticMapLayer.LANE, MapElementType.LANE_CENTERLINE),
        (SemanticMapLayer.LANE_CONNECTOR, MapElementType.LANE_CONNECTOR_CENTERLINE),
    ):
        for edge in sorted(nearby[layer], key=lambda e: str(e.id)):
            append(
                list(edge.baseline_path.linestring.coords),
                kind,
                f"center:{layer.name}:{edge.id}",
            )
            for side in ("left_boundary", "right_boundary"):
                boundary = getattr(edge, side, None)
                if boundary is not None:
                    append(
                        list(boundary.linestring.coords),
                        MapElementType.LANE_BOUNDARY,
                        f"lane-boundary:{boundary.id}",
                    )
    for layer, kind in (
        (SemanticMapLayer.CROSSWALK, MapElementType.CROSSWALK),
        (SemanticMapLayer.STOP_LINE, MapElementType.STOP_LINE),
    ):
        for obj in sorted(nearby[layer], key=lambda e: str(e.id)):
            append(list(obj.polygon.exterior.coords), kind, f"{layer.name}:{obj.id}")
    contour_area = drivable_area(
        map_api, midpoint, radius, (SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA)
    )
    for index, coords in enumerate(drivable_area_boundaries(contour_area)):
        append(coords, MapElementType.ROAD_BOUNDARY, f"drivable-contour:{index}")
    drivable = drivable_origin = lane_info = None
    if grids:
        # nuPlan's drivable_area_compliance accepts the drivable-area layer (which excludes car
        # parks) and any lane or lane connector. The grid reaches past the crop, so query
        # farther out.
        area = drivable_area(
            map_api,
            midpoint,
            radius + DRIVABLE_GRID_MARGIN_M * math.sqrt(2),
            (
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
            ),
        )
        area_local = None
        if area is not None:
            from shapely import affinity

            area_local = affinity.rotate(
                affinity.translate(area, -ox, -oy), -yaw, origin=(0.0, 0.0), use_radians=True
            )
        if focus is None:
            focus = scene.reference_state.position[:, scene.ego_index][
                scene.reference_valid[:, scene.ego_index]
            ].numpy()
        grid_lower, grid_upper = grid_window(lower, upper, focus)
        drivable, drivable_origin = drivable_distance_grid(area_local, grid_lower, grid_upper)
        lane_info = lane_info_grid(
            map_api,
            midpoint,
            radius + DRIVABLE_GRID_MARGIN_M * math.sqrt(2),
            lambda coords: (coords - [ox, oy]) @ rotation,
            grid_lower,
            grid_upper,
        )
    if not segments:
        raise ValueError("No map geometry in scene crop")
    data = MapData(
        map_name=map_api.map_name,
        world_from_scene=(float(ox), float(oy), float(yaw)),
        bounds=tuple(float(x) for x in (*lower, *upper)),
        segments=torch.tensor(np.asarray(segments), dtype=torch.float32),
        segment_type=torch.tensor(types, dtype=torch.int64),
        polyline_index=torch.tensor(groups, dtype=torch.int64),
        polyline_ids=tuple(ids),
        drivable_distance=drivable,
        drivable_origin=drivable_origin,
        lane_info=lane_info,
    )
    data.validate()
    info = {
        "original_segments": len(segments),
        "polylines": len(ids),
        "polyline_counts": {MapElementType(k).name: v for k, v in counts.items()},
        "crop_margin_m": margin,
    }
    return data, info
