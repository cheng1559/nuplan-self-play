"""Export nuPlan reactive-agent inputs with nuPlan's own IDM builder and map API.

``build_idm_agents_on_map_rails`` decides which vehicles become IDM agents, their
starting lane, snapped pose, speed, and initial target velocity, exactly as the
reactive (R) closed-loop simulation does. The lane graph reachable from those starts
is tabulated with nuPlan's progress, curvature, speed limits, signals and stop lines.
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cs5478_project.datatypes.map import SEGMENT_SIGNALS
from cs5478_project.datatypes.rails import (
    EDGE_OUTGOING,
    EDGE_POINTS,
    RAIL_EDGES,
    SIGNALS,
    STOP_LINE_VERTICES,
    STOP_LINES,
    ReactiveRole,
    empty_rails,
)

# nuPlan idm_agents_observation.yaml
IDM_TARGET_VELOCITY = 10.0
IDM_MIN_GAP = 1.0
IDM_HEADWAY = 1.5
IDM_ACCEL_MAX = 1.0
IDM_DECEL_MAX = 2.0
IDM_MINIMUM_PATH_LENGTH = 20.0
OPEN_LOOP_TYPES = ("PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT")
OPEN_LOOP_CATEGORIES = {name.lower() for name in OPEN_LOOP_TYPES}
# Reach of a 15 s episode at highway speed, plus plan_route's look-ahead.
GRAPH_BUDGET_M = 500.0


class _Frame:
    def __init__(self, origin):
        ox, oy, yaw = origin
        self.origin = np.array([ox, oy])
        self.yaw = yaw
        c, s = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([[c, -s], [s, c]])

    def points(self, xy):
        return (np.asarray(xy, dtype=np.float64) - self.origin) @ self.rotation

    def heading(self, heading):
        return math.atan2(math.sin(heading - self.yaw), math.cos(heading - self.yaw))


def _resample_edge(edge):
    """P samples at uniform nuPlan progress, with interpolated discrete-path headings."""
    states = edge.baseline_path.discrete_path
    xy = np.array([[state.x, state.y] for state in states])
    heading = np.unwrap([state.heading for state in states])
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, ~np.isclose(np.diff(progress), 0.0)]
    xy, heading, progress = xy[keep], heading[keep], progress[keep]
    target = np.linspace(0.0, progress[-1], EDGE_POINTS)
    points = np.stack([np.interp(target, progress, xy[:, i]) for i in range(2)], -1)
    return points, np.interp(target, progress, heading), float(progress[-1])


def _stop_polygons(edge, frame):
    polygons = []
    for stop_line in edge.stop_lines[:STOP_LINES]:
        polygon = stop_line.polygon
        coords = np.asarray(polygon.exterior.coords)[:-1]
        if len(coords) > STOP_LINE_VERTICES:
            # Stop lines are thin quadrilaterals; the hull keeps intersection extents.
            coords = np.asarray(polygon.convex_hull.exterior.coords)[:-1]
        if len(coords) > STOP_LINE_VERTICES:
            coords = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:-1]
        polygons.append(frame.points(coords))
    if len(edge.stop_lines) > STOP_LINES:
        raise ValueError(f"Lane connector {edge.id} has more than {STOP_LINES} stop lines")
    return polygons


# nuPlan's per-iteration scenario queries open the log database on every call, which
# dominates export time on network drives. These read the same rows in one query.


def _query_log(scenario, sql, tokens):
    import sqlite3

    uri = Path(scenario._log_file).as_uri() + "?mode=ro&immutable=1"
    marks = ",".join("?" * len(tokens))
    with sqlite3.connect(uri, uri=True) as conn:
        return conn.execute(sql.format(marks=marks), tokens).fetchall()


def _iteration_times(scenario):
    """Timestamps of every scenario iteration (as get_time_point returns them)."""
    tokens = [bytes.fromhex(token) for token in scenario._lidarpc_tokens]
    rows = dict(
        _query_log(
            scenario, "SELECT token, timestamp FROM lidar_pc WHERE token IN ({marks})", tokens
        )
    )
    return np.array([rows[token] for token in tokens], dtype=np.int64)


def _iteration_statuses(scenario, iterations):
    """Traffic lights at each iteration (as get_traffic_light_status_at_iteration)."""
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType

    kinds = {"green": TrafficLightStatusType.GREEN, "yellow": TrafficLightStatusType.YELLOW}
    kinds["red"] = TrafficLightStatusType.RED
    tokens = [bytes.fromhex(scenario._lidarpc_tokens[int(i)]) for i in iterations]
    rows = _query_log(
        scenario,
        "SELECT lp.token, tl.status, tl.lane_connector_id, lp.timestamp FROM lidar_pc AS lp "
        "INNER JOIN traffic_light_status AS tl ON lp.token = tl.lidar_pc_token "
        "WHERE lp.token IN ({marks})",
        list(set(tokens)),
    )
    by_token = {}
    for token, status, connector, timestamp in rows:
        by_token.setdefault(bytes(token), []).append(
            TrafficLightStatusData(
                kinds.get(status, TrafficLightStatusType.UNKNOWN), connector, timestamp
            )
        )
    return [by_token.get(token, []) for token in tokens]


def _signal_frames(scenario, scene):
    """Scenario iteration nearest each scene frame, like annotation masks."""
    times = _iteration_times(scenario)
    relative = (times - times[0]) / 1e6
    frames = np.arange(scene.num_frames) * scene.dt
    if relative[-1] + 0.051 < frames[-1]:
        raise ValueError("Scenario does not cover the scene's signal horizon")
    return np.abs(relative[:, None] - frames[None]).argmin(0)


def signal_links(map_api, frame, signal_ids, lines):
    """[len(lines), SEGMENT_SIGNALS] indices into ``signal_ids`` of the signalized lane
    connectors whose polygon each line (scene-frame vertices) touches; -1 padding."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import LineString, Polygon
    from shapely.strtree import STRtree

    polygons = []
    for identity in signal_ids:
        connector = map_api.get_map_object(identity, SemanticMapLayer.LANE_CONNECTOR)
        coords = np.asarray(connector.polygon.exterior.coords)
        polygons.append(Polygon(frame.points(coords)))
    table = torch.full((len(lines), SEGMENT_SIGNALS), -1)
    if not polygons:
        return table
    tree = STRtree(polygons)
    for row, coords in enumerate(lines):
        line = LineString(coords)
        hits = [int(i) for i in tree.query(line) if polygons[int(i)].distance(line) <= 1e-5]
        if len(hits) > SEGMENT_SIGNALS:
            raise ValueError(f"A map element touches {len(hits)} signalized connectors")
        table[row, : len(hits)] = torch.tensor(sorted(hits), dtype=torch.int64)
    return table


def build_rails(scene, scenario, track_tokens, categories, route_roadblock_ids, chunk_pieces):
    """Return (RailData, segment signals [S, K], chunk signals [C, K], info) in scene frame.

    ``chunk_pieces`` holds each map chunk's exact source vertices (see map_chunks).
    """
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
    from nuplan.planning.simulation.observation.idm.idm_agents_builder import (
        build_idm_agents_on_map_rails,
        get_starting_segment,
    )

    if scene.map is None:
        raise ValueError("Rails require the scene map transform")
    frame = _Frame(scene.map.world_from_scene)
    map_api = scenario.map_api
    agents, _ = build_idm_agents_on_map_rails(
        IDM_TARGET_VELOCITY,
        IDM_MIN_GAP,
        IDM_HEADWAY,
        IDM_ACCEL_MAX,
        IDM_DECEL_MAX,
        IDM_MINIMUM_PATH_LENGTH,
        scenario,
        [TrackedObjectType[name] for name in OPEN_LOOP_TYPES],
    )
    ego_state = scenario.initial_ego_state
    ego_edge, ego_progress = get_starting_segment(ego_state.agent, map_api)

    starts = [(agent._route[0], agent._state.progress) for agent in agents.values()]
    if ego_edge is not None:
        starts.append((ego_edge, ego_progress))
    # Breadth-first reach within the budget, measured from each start position.
    best, edges, queue = {}, {}, deque((edge, -progress) for edge, progress in starts)
    while queue:
        edge, offset = queue.popleft()
        if offset > GRAPH_BUDGET_M or best.get(edge.id, math.inf) <= offset:
            continue
        best[edge.id] = offset
        edges[edge.id] = edge
        for outgoing in edge.outgoing_edges:
            queue.append((outgoing, offset + edge.baseline_path.length))
    if len(edges) > RAIL_EDGES:
        raise ValueError(f"Rail graph needs {len(edges)} edges; capacity is {RAIL_EDGES}")
    order = sorted(edges, key=lambda identity: (best[identity], identity))
    index = {identity: slot for slot, identity in enumerate(order)}

    # Signal table: every lane connector with a status during the episode.
    iterations = _signal_frames(scenario, scene)
    statuses = _iteration_statuses(scenario, iterations)
    signal_ids = sorted({str(light.lane_connector_id) for lights in statuses for light in lights})
    if len(signal_ids) > SIGNALS:
        raise ValueError(f"Episode has {len(signal_ids)} signalized connectors; capacity {SIGNALS}")
    signal_index = {identity: slot for slot, identity in enumerate(signal_ids)}

    rails = empty_rails((), scene.num_frames, scene.num_actors)
    route = set(route_roadblock_ids)
    gaps = []
    for identity in order:
        e = index[identity]
        edge = edges[identity]
        points, heading, length = _resample_edge(edge)
        rails.edge_points[e] = torch.tensor(frame.points(points), dtype=torch.float32)
        rails.edge_heading[e] = torch.tensor(heading - frame.yaw, dtype=torch.float32)
        rails.edge_length[e] = length
        rails.edge_speed_limit[e] = edge.speed_limit_mps or 0.0
        rails.edge_curvature[e] = abs(edge.baseline_path.get_curvature_at_arc_length(0.0))
        outgoing = [index.get(out.id, -1) for out in edge.outgoing_edges]
        if len(outgoing) > EDGE_OUTGOING:
            raise ValueError(f"Edge {identity} has {len(outgoing)} outgoing edges")
        rails.edge_outgoing[e, : len(outgoing)] = torch.tensor(outgoing, dtype=torch.int64)
        end = edge.baseline_path.discrete_path[-1]
        for out in edge.outgoing_edges:
            if out.id in index:
                start = out.baseline_path.discrete_path[0]
                gaps.append(math.hypot(end.x - start.x, end.y - start.y))
        rails.edge_has_lights[e] = edge.has_traffic_lights()
        rails.edge_signal[e] = signal_index.get(str(identity), -1)
        rails.edge_on_route[e] = edge.get_roadblock_id() in route
        rails.edge_mask[e] = True
        if rails.edge_has_lights[e]:
            for slot, polygon in enumerate(_stop_polygons(edge, frame)):
                rails.stop_points[e, slot, : len(polygon)] = torch.tensor(
                    polygon, dtype=torch.float32
                )
                rails.stop_mask[e, slot, : len(polygon)] = True
    for f, lights in enumerate(statuses):
        for light in lights:
            slot = signal_index[str(light.lane_connector_id)]
            rails.signal_green[f, slot] |= light.status == TrafficLightStatusType.GREEN
            rails.signal_red[f, slot] |= light.status == TrafficLightStatusType.RED

    slots = {token: slot for slot, token in enumerate(track_tokens)}

    def set_rail(slot, edge, progress, pose, speed, target, size):
        rails.rail_valid[slot] = True
        rails.rail_edge[slot] = index[edge.id]
        rails.rail_progress[slot] = float(progress)
        xy = frame.points([[pose.x, pose.y]])[0]
        rails.rail_pose[slot] = torch.tensor([xy[0], xy[1], frame.heading(pose.heading)])
        rails.rail_speed[slot] = float(speed)
        rails.rail_target[slot] = float(target)
        rails.rail_size[slot] = torch.tensor([size[0], size[1]], dtype=torch.float32)

    for slot, category in enumerate(categories):
        rails.role[slot] = (
            ReactiveRole.EGO
            if category == "ego"
            else ReactiveRole.OPEN_LOOP
            if category in OPEN_LOOP_CATEGORIES
            else ReactiveRole.ABSENT
        )
    missing = 0
    for token, agent in agents.items():
        slot = slots.get(token)
        if slot is None:
            missing += 1  # dropped by the scene's actor capacity
            continue
        box = agent._initial_state.box
        set_rail(
            slot,
            agent._route[0],
            agent._state.progress,
            box.center,
            agent._state.velocity,
            agent._policy.target_velocity,
            (box.length, box.width),
        )
        rails.role[slot] = ReactiveRole.IDM
    if ego_edge is not None:
        ego = slots["ego"]
        pose = ego_edge.baseline_path.get_nearest_pose_from_position(ego_state.agent.center)
        set_rail(
            ego,
            ego_edge,
            ego_progress,
            pose,
            ego_state.dynamic_car_state.speed,
            ego_edge.speed_limit_mps or IDM_TARGET_VELOCITY,
            (ego_state.car_footprint.length, ego_state.car_footprint.width),
        )
        rails.speed_limit.fill_(ego_edge.speed_limit_mps or IDM_TARGET_VELOCITY)

    # Map segments and chunks touching a signalized connector polygon.
    segment_signal = signal_links(map_api, frame, signal_ids, list(scene.map.segments.numpy()))
    chunk_signal = signal_links(map_api, frame, signal_ids, chunk_pieces)
    info = {
        "idm_agents": len(agents),
        "idm_agents_beyond_capacity": missing,
        "ego_rail": ego_edge is not None,
        "edges": len(edges),
        "signals": len(signal_ids),
        "max_edge_length_m": float(rails.edge_length[rails.edge_mask].max()),
        "max_connection_gap_m": max(gaps, default=0.0),
        "roles": {role.name: int((rails.role == role).sum()) for role in ReactiveRole},
    }
    return rails, segment_signal, chunk_signal, info
