"""Build the policy's scene from what a nuPlan planner is given: no log and no files.

``build_scene`` runs once, at the first planning step, from the initial ego state and
observation, the map API, and the route roadblocks. It uses the exporter's own map,
route, chunk, and signal code, so the policy sees the features it was trained on. What
changes during the episode (tracked objects, static objects, traffic lights) is written
into the batch by the planner at every step.

Where the exporter used the log, a planner cannot:

- Map crop: around ego, the nearby objects, and the route ahead, not around every
  logged position of the episode. ``crop_map`` cuts it again around ego's current
  position when ego comes near its edge (the planner calls it).
- Signal table: every signalized lane connector in the crop, not those with a status
  later in the log.
"""

import math
from dataclasses import replace

import numpy as np
import torch

from cs5478_project.data.map_chunks import polyline_chunks
from cs5478_project.data.map_simplification import simplify_map
from cs5478_project.data.nuplan.map_geometry import build_map
from cs5478_project.data.nuplan.rails import IDM_TARGET_VELOCITY, _Frame, signal_links
from cs5478_project.data.nuplan.routes import ego_map_route
from cs5478_project.data.nuplan.tracks import EGO_REAR_TO_CENTER, EGO_SIZE, EGO_WHEELBASE
from cs5478_project.datatypes.map import MAX_MAP_CHUNKS
from cs5478_project.datatypes.rails import SIGNALS, ReactiveRole, empty_rails
from cs5478_project.datatypes.route import ROUTE_POINTS, RouteSource, empty_routes
from cs5478_project.datatypes.scene import ActorType, Scene
from cs5478_project.datatypes.state import WorldState
from cs5478_project.datatypes.static import empty_static
from cs5478_project.defaults import DEFAULT_DT, EPISODE_FRAMES

# (route length ahead, radius of first-frame objects) the crop covers [m]; later entries
# are fallbacks for maps too dense for the chunk capacity. Exported scenes cover about
# 100 m behind and 140 m ahead of ego (medians).
CROPS = ((150.0, 100.0), (100.0, 60.0), (60.0, 30.0), (30.0, 0.0))
# The crop always reaches this far around ego (before build_map's 25 m margin), and is
# cut again once ego is within RECROP_M of its edge.
AROUND_EGO_M = 30.0
RECROP_M = 40.0
FALLBACK_ROUTE_M = 100.0


def _route(map_api, route_roadblock_ids, origin):
    """Ego's route [ROUTE_POINTS, 2] in scene frame, and whether the map gave one."""
    points = None
    try:
        points, _ = ego_map_route(map_api, list(route_roadblock_ids), *origin)
    except Exception:  # a broken route must not end the simulation
        points = None
    if points is not None:
        return np.asarray(points, dtype=np.float32), True
    # Without usable route baselines: straight ahead.
    straight = np.zeros((ROUTE_POINTS, 2), np.float32)
    straight[:, 0] = np.linspace(0.0, FALLBACK_ROUTE_M, ROUTE_POINTS)
    return straight, False


def _signalized_connectors(map_api, frame, bounds, ego):
    """Ids of the lane connectors with traffic lights in the crop (nearest to ``ego``
    [2] first beyond the signal capacity), sorted."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon, box

    lower, upper = np.array(bounds[:2]), np.array(bounds[2:])
    center = (lower + upper) / 2
    world = center @ frame.rotation.T + frame.origin
    radius = float(np.linalg.norm(upper - lower) / 2)
    crop = box(*lower, *upper)
    layer = SemanticMapLayer.LANE_CONNECTOR
    found = []
    for connector in map_api.get_proximal_map_objects(Point2D(*world), radius, [layer])[layer]:
        if not connector.has_traffic_lights():
            continue
        polygon = Polygon(frame.points(np.asarray(connector.polygon.exterior.coords)))
        if polygon.intersects(crop):
            found.append((polygon.distance(ShapelyPoint(*ego)), str(connector.id)))
    return sorted(identity for _, identity in sorted(found)[:SIGNALS])


def crop_map(scene, map_api, origin, objects, center=(0.0, 0.0)):
    """``scene`` with the map cut around ``center`` (ego's position), the route ahead of
    it, and the nearby ``objects`` [N, 2], with its signal links; and {lane connector id:
    signal slot}. ``origin`` is the scene frame (x, y, yaw) in world coordinates;
    ``objects`` and ``center`` are in scene frame. ``scene`` needs its routes."""
    frame = _Frame(origin)
    center = np.asarray(center, dtype=np.float64)
    route_points = scene.routes.points[scene.ego_index].numpy().astype(np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route_points, axis=0), axis=1))]
    start = arc[np.linalg.norm(route_points - center, axis=1).argmin()]
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 2)
    distance = np.linalg.norm(objects - center, axis=1)
    corners = center + AROUND_EGO_M * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    for ahead, around in CROPS:
        points = np.concatenate(
            (
                corners,
                route_points[(arc >= start) & (arc <= start + ahead)],
                objects[distance <= around],
            )
        )
        map_data, _ = build_map(scene, map_api, origin, points=points, focus=center[None])
        # Chunks come from the unsimplified geometry; simplification only thins segments.
        chunk_points, chunk_type, chunk_pieces = polyline_chunks(map_data)
        if len(chunk_type) <= MAX_MAP_CHUNKS:
            break
    else:
        raise ValueError(f"The map around ego needs {len(chunk_type)} chunks")
    map_data = simplify_map(replace(map_data, chunk_points=chunk_points, chunk_type=chunk_type))
    signal_ids = _signalized_connectors(map_api, frame, map_data.bounds, center)
    map_data = replace(
        map_data,
        segment_signal=signal_links(map_api, frame, signal_ids, list(map_data.segments.numpy())),
        chunk_signal=signal_links(map_api, frame, signal_ids, chunk_pieces),
    )
    scene = replace(scene, map=map_data)
    scene.validate()
    return scene, {identity: slot for slot, identity in enumerate(signal_ids)}


def build_scene(ego_state, observation, map_api, route_roadblock_ids):
    """(Scene with ego only, {lane connector id: signal slot}, whether the route is from
    the map). The scene frame is ego's initial rear-axle pose."""
    from nuplan.planning.simulation.observation.idm.idm_agents_builder import (
        get_starting_segment,
    )

    rear = ego_state.rear_axle
    origin = (float(rear.x), float(rear.y), float(rear.heading))
    frame = _Frame(origin)
    dynamic = ego_state.dynamic_car_state
    velocity = dynamic.rear_axle_velocity_2d
    acceleration = float(dynamic.rear_axle_acceleration_2d.x)
    initial = WorldState(
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.tensor([[velocity.x, velocity.y]], dtype=torch.float32),
        torch.tensor([acceleration], dtype=torch.float32),
        torch.tensor([acceleration], dtype=torch.float32),
        torch.tensor([ego_state.tire_steering_angle], dtype=torch.float32),
        torch.tensor([dynamic.angular_velocity], dtype=torch.float32),
    )
    frames = EPISODE_FRAMES
    size = torch.tensor([EGO_SIZE], dtype=torch.float32)
    route_points, from_map = _route(map_api, route_roadblock_ids, origin)
    routes = empty_routes((1,))
    routes.points[0] = torch.tensor(route_points)
    routes.mask[0] = True
    routes.source[0] = RouteSource.MAP
    rails = empty_rails((), frames, 1)
    rails.role[0] = ReactiveRole.EGO
    edge, _ = get_starting_segment(ego_state.agent, map_api)
    if edge is not None:
        rails.speed_limit.fill_(edge.speed_limit_mps or IDM_TARGET_VELOCITY)
    scene = Scene(
        scenario_id="nuplan-online",
        dt=DEFAULT_DT,
        actor_type=torch.tensor([ActorType.VEHICLE]),
        actor_mask=torch.ones(1, dtype=torch.bool),
        size=size,
        wheelbase=torch.tensor([EGO_WHEELBASE]),
        reference_to_center=torch.tensor([EGO_REAR_TO_CENTER]),
        initial_state=initial,
        reference_state=WorldState(
            *(value.expand(frames, *value.shape).clone() for value in initial)
        ),
        reference_valid=torch.ones(frames, 1, dtype=torch.bool),
        ego_index=0,
        routes=routes,
        actor_ids=torch.zeros(1, dtype=torch.int64),
        static=empty_static((0,), frames),
        reference_size=size.expand(frames, 1, 2).clone(),
        rails=rails,
    )
    objects = frame.points(
        [[obj.center.x, obj.center.y] for obj in observation.tracked_objects.tracked_objects]
        or np.zeros((0, 2))
    )
    scene, signals = crop_map(scene, map_api, origin, objects)
    return scene, signals, from_map


def to_scene(origin, x, y):
    ox, oy, yaw = origin
    dx, dy = x - ox, y - oy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy
