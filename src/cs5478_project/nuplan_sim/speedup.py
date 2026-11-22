"""Faster database access for nuPlan devkit simulations, with identical results.

The devkit opens a new SQLite connection for every query and reads each iteration's
traffic lights with a full scan of ``traffic_light_status`` (the table has no index on
the lidar frame): two or three scans per simulation step, about 90% of a simulation's
time. ``install`` (called when the planner module is imported, so in every simulation
process) patches the devkit to

- reuse one read-only connection per database file and thread,
- read a scenario's traffic lights and iteration times once, in one query each, and
- answer ``is_in_layer`` (the time-to-collision metric asks it for every tracked object
  at every step) from a spatial index instead of testing every polygon of the layer.
"""

import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

MAX_CONNECTIONS = 16
_local = threading.local()
_installed = False


def _connection(db_file):
    cache = getattr(_local, "connections", None)
    if cache is None or _local.pid != os.getpid():  # never share connections across a fork
        cache = _local.connections = OrderedDict()
        _local.pid = os.getpid()
    connection = cache.get(db_file)
    if connection is None:
        uri = Path(db_file).resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        cache[db_file] = connection
        if len(cache) > MAX_CONNECTIONS:
            cache.popitem(last=False)[1].close()
    else:
        cache.move_to_end(db_file)
    return connection


def execute_many(query_text, query_parameters, db_file):
    cursor = _connection(db_file).cursor()
    try:
        cursor.execute(query_text, query_parameters)
        yield from cursor
    finally:
        cursor.close()


def execute_one(query_text, query_parameters, db_file):
    cursor = _connection(db_file).cursor()
    try:
        cursor.execute(query_text, query_parameters)
        result = cursor.fetchone()
        if result is not None and cursor.fetchone() is not None:
            raise RuntimeError("execute_one query returned multiple rows.")
        return result
    finally:
        cursor.close()


def _by_token(scenario, query):
    """Rows of ``query`` (with one ``IN ({})`` over lidar tokens) grouped by token hex."""
    tokens = scenario._lidarpc_tokens
    rows = {token: [] for token in tokens}
    for start in range(0, len(tokens), 500):
        chunk = tokens[start : start + 500]
        text = query.format(",".join("?" for _ in chunk))
        parameters = [bytearray.fromhex(token) for token in chunk]
        for row in execute_many(text, parameters, scenario._log_file):
            rows[bytes(row["token"]).hex()].append(row)
    return rows


def _traffic_lights(scenario):
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType

    cache = scenario.__dict__.get("_cs5478_traffic_lights")
    if cache is None:
        # The devkit's query per token, for all of the scenario's tokens in one scan.
        query = """
            SELECT  tl.lidar_pc_token AS token,
                    CASE WHEN tl.status == "green" THEN 0
                         WHEN tl.status == "yellow" THEN 1
                         WHEN tl.status == "red" THEN 2
                         ELSE 3
                    END AS status,
                    tl.lane_connector_id,
                    lp.timestamp AS timestamp
            FROM traffic_light_status AS tl
            INNER JOIN lidar_pc AS lp
                ON lp.token = tl.lidar_pc_token
            WHERE tl.lidar_pc_token IN ({})
            ORDER BY tl.rowid
        """
        cache = {
            token: [
                TrafficLightStatusData(
                    status=TrafficLightStatusType(row["status"]),
                    lane_connector_id=row["lane_connector_id"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]
            for token, rows in _by_token(scenario, query).items()
        }
        scenario.__dict__["_cs5478_traffic_lights"] = cache
    return cache


def _time_points(scenario):
    cache = scenario.__dict__.get("_cs5478_time_points")
    if cache is None:
        query = "SELECT token, timestamp FROM lidar_pc WHERE token IN ({})"
        cache = {
            token: int(rows[0]["timestamp"]) for token, rows in _by_token(scenario, query).items()
        }
        scenario.__dict__["_cs5478_time_points"] = cache
    return cache


def install():
    """Patch the devkit once per process."""
    global _installed
    if _installed:
        return
    _installed = True
    import nuplan.database.nuplan_db.query_session as query_session
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.database.nuplan_db import nuplan_scenario_queries
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

    for module in (query_session, nuplan_scenario_queries):
        module.execute_many = execute_many
        module.execute_one = execute_one

    def get_traffic_light_status_at_iteration(self, iteration):
        return iter(_traffic_lights(self)[self._lidarpc_tokens[iteration]])

    def get_time_point(self, iteration):
        return TimePoint(time_us=_time_points(self)[self._lidarpc_tokens[iteration]])

    NuPlanScenario.get_traffic_light_status_at_iteration = get_traffic_light_status_at_iteration
    NuPlanScenario.get_time_point = get_time_point

    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap

    original = NuPlanMap.is_in_layer

    def is_in_layer(self, point, layer):
        if layer == SemanticMapLayer.TURN_STOP:
            return original(self, point, layer)
        trees = self.__dict__.setdefault("_cs5478_trees", {})
        tree = trees.get(layer)
        if tree is None:
            geometries = self._get_vector_map_layer(layer).geometry.values
            tree = trees[layer] = shapely.STRtree(geometries)
        # The devkit's test: any polygon of the layer contains the point.
        return len(tree.query(shapely.Point(point.x, point.y), predicate="within")) > 0

    NuPlanMap.is_in_layer = is_in_layer
