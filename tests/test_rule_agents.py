"""Numeric references, interaction behavior, lifecycle, and compiled rule execution."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from cs5478_project.agents.config import IDMConfig, InteractionConfig, PedestrianConfig
from cs5478_project.agents.idm import idm_acceleration
from cs5478_project.agents.route import cache_routes, project_route, sample_route
from cs5478_project.data import SceneDataLoader, SceneDataset
from cs5478_project.data.preparation import prepare_scene
from cs5478_project.data.rails import with_route_rails
from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.route import RouteSource, empty_routes
from cs5478_project.datatypes.scene import ActorType, Scene
from cs5478_project.datatypes.state import WorldState
from cs5478_project.dynamics.bicycle import (
    BicycleConfig,
    KinematicBicycleModel,
    adapt_action,
    heading_vector,
    no_reverse_command,
)
from cs5478_project.env import ReplayEnv, RuleBasedEnv


def traffic_scene(
    positions=((0.0, 0.0),),
    speeds=(4.0,),
    types=(ActorType.VEHICLE,),
    *,
    headings=None,
    frames=31,
    visibility=None,
) -> Scene:
    count = len(types)
    heading = torch.tensor(headings or [0.0] * count, dtype=torch.float32)
    speed = torch.tensor(speeds, dtype=torch.float32)
    velocity = speed[:, None] * torch.stack((heading.cos(), heading.sin()), dim=-1)
    position = torch.tensor(positions, dtype=torch.float32)
    time = torch.arange(frames, dtype=torch.float32) * 0.1
    zeros = torch.zeros(frames, count)
    state = WorldState(
        position[None] + velocity[None] * time[:, None, None],
        heading[None].expand(frames, -1).clone(),
        velocity[None].expand(frames, -1, -1).clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
    )
    actor_type = torch.tensor(types, dtype=torch.int64)
    routes = empty_routes((count,))
    for slot, kind in enumerate(types):
        if kind == ActorType.STATIC:
            continue
        distance = 120.0 if kind == ActorType.VEHICLE else 30.0
        direction = torch.stack((heading[slot].cos(), heading[slot].sin()))
        routes.points[slot] = position[slot] + torch.linspace(0, distance, 100)[:, None] * direction
        routes.mask[slot] = True
        routes.source[slot] = RouteSource.GT
        routes.horizon_s[slot] = 30.0
    return with_route_rails(
        prepare_scene(
            Scene(
                scenario_id="traffic",
                dt=0.1,
                actor_type=actor_type,
                actor_mask=torch.ones(count, dtype=torch.bool),
                size=torch.tensor([[4.0, 2.0], [0.6, 0.6], [1.0, 1.0]])[actor_type],
                wheelbase=torch.full((count,), 3.0),
                reference_to_center=(actor_type == ActorType.VEHICLE).float(),
                initial_state=WorldState(*(value[0].clone() for value in state)),
                reference_state=state,
                reference_valid=(
                    torch.ones(frames, count, dtype=torch.bool)
                    if visibility is None
                    else visibility
                ),
                routes=routes,
            )
        )
    )


def batch_of(*scenes, capacity=4, static_capacity=2):
    return next(
        iter(
            SceneDataLoader(
                SceneDataset(list(scenes)),
                batch_size=len(scenes),
                max_actors=capacity,
                max_static_objects=static_capacity,
                max_frames=max(scene.num_frames for scene in scenes),
                max_map_segments=1,
            )
        )
    ).tensors


def assert_state_close(actual, expected, **kwargs):
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, **kwargs)


def test_idm_acceleration_matches_nuplan_forward_euler():
    from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy

    config = IDMConfig()
    speeds = torch.tensor([0.0, 2.0, 10.0, 15.0, 5.0, 1.0], dtype=torch.float64)
    gaps = torch.tensor([1000.0, 10.0, 30.0, 100.0, 0.0, 1.0], dtype=torch.float64)
    leads = torch.tensor([0.0, 0.0, 8.0, -2.0, 0.0, 10.0], dtype=torch.float64)
    policy = IDMPolicy(10.0, 1.0, 1.5, 1.0, 2.0)
    expected = [
        np.clip(
            policy.idm_model([], [0, float(v)], [float(gap), float(lead), 0], policy.idm_params)[1],
            -2.0,
            1.0,
        )
        for v, gap, lead in zip(speeds, gaps, leads, strict=True)
    ]
    torch.testing.assert_close(
        idm_acceleration(speeds, gaps, leads, config), torch.tensor(expected)
    )


def test_bicycle_matches_nuplan_actuator_filter_and_forward_euler():
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel as NuPlanBicycle,
    )

    parameters = get_pacifica_parameters()
    heading, speed, steering, acceleration = 0.6, 4.0, 0.1, 0.2
    ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(2.0, 3.0, heading),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
        tire_steering_angle=steering,
        time_point=TimePoint(0),
        vehicle_parameters=parameters,
    )
    ideal = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=parameters.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.4, 0.0),
        tire_steering_rate=0.1,
    )
    expected = NuPlanBicycle(parameters).propagate_state(ego, ideal, TimePoint(100000))

    def scalar(value):
        return torch.tensor([[value]], dtype=torch.float64)

    state = WorldState(
        torch.tensor([[[2.0, 3.0]]], dtype=torch.float64),
        scalar(heading),
        torch.tensor(
            [[[speed * math.cos(heading), speed * math.sin(heading)]]], dtype=torch.float64
        ),
        scalar(acceleration),
        scalar(0.3),
        scalar(steering),
        scalar(speed * math.tan(steering) / parameters.wheel_base),
    )
    result = KinematicBicycleModel.step(
        state, VehicleAction(scalar(1.0), scalar(0.1)), scalar(parameters.wheel_base), scalar(0.1)
    )
    np.testing.assert_allclose(
        result.position.numpy().ravel(), [expected.rear_axle.x, expected.rear_axle.y]
    )
    assert float(result.heading[0, 0]) == pytest.approx(expected.rear_axle.heading)
    assert float(result.steering[0, 0]) == pytest.approx(expected.tire_steering_angle)
    assert float(result.acceleration[0, 0]) == pytest.approx(
        expected.dynamic_car_state.rear_axle_acceleration_2d.x
    )
    assert float(torch.linalg.vector_norm(result.velocity)) == pytest.approx(
        expected.dynamic_car_state.rear_axle_velocity_2d.x
    )


def test_pure_bicycle_matches_nuplan_even_when_it_would_reverse():
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel as NuPlanBicycle,
    )

    from cs5478_project.dynamics.bicycle import nuplan_bicycle_step

    parameters = get_pacifica_parameters()
    heading, speed, steering, acceleration = -0.4, 0.3, -0.2, -2.5
    ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(1.0, -2.0, heading),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
        tire_steering_angle=steering,
        time_point=TimePoint(0),
        vehicle_parameters=parameters,
    )
    ideal = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=parameters.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(-6.0, 0.0),
        tire_steering_rate=-0.3,
    )
    expected = NuPlanBicycle(parameters).propagate_state(ego, ideal, TimePoint(100000))
    assert expected.dynamic_car_state.rear_axle_velocity_2d.x < 0  # nuPlan's model reverses

    def scalar(value):
        return torch.tensor([[value]], dtype=torch.float64)

    direction = torch.tensor([[[math.cos(heading), math.sin(heading)]]], dtype=torch.float64)
    state = WorldState(
        torch.tensor([[[1.0, -2.0]]], dtype=torch.float64),
        scalar(heading),
        speed * direction,
        scalar(acceleration),
        scalar(0.0),
        scalar(steering),
        scalar(speed * math.tan(steering) / parameters.wheel_base),
    )
    config = BicycleConfig()
    result = nuplan_bicycle_step(
        state,
        scalar(-6.0),
        scalar(-0.3),
        scalar(parameters.wheel_base),
        scalar(0.1),
        acceleration_time_constant=config.acceleration_time_constant,
        steering_time_constant=config.steering_time_constant,
        max_steering=config.max_steering,
    )
    np.testing.assert_allclose(
        result.position.numpy().ravel(), [expected.rear_axle.x, expected.rear_axle.y]
    )
    assert float(result.heading[0, 0]) == pytest.approx(expected.rear_axle.heading)
    assert float(result.steering[0, 0]) == pytest.approx(expected.tire_steering_angle)
    signed = float((result.velocity * heading_vector(result.heading)).sum())
    assert signed == pytest.approx(expected.dynamic_car_state.rear_axle_velocity_2d.x)
    # The action layer floors the same command so the next speed is exactly zero.
    floored = no_reverse_command(
        state, scalar(-6.0), scalar(0.1), config.acceleration_time_constant
    )
    stopped = nuplan_bicycle_step(
        state,
        floored,
        scalar(-0.3),
        scalar(parameters.wheel_base),
        scalar(0.1),
        acceleration_time_constant=config.acceleration_time_constant,
        steering_time_constant=config.steering_time_constant,
        max_steering=config.max_steering,
    )
    assert float((stopped.velocity * heading_vector(stopped.heading)).sum()) == pytest.approx(
        0.0, abs=1e-12
    )


def test_action_adapter_limits_jerk_and_prevents_reverse():
    scene = batch_of(traffic_scene(speeds=(0.0,)))
    state = scene.initial_state
    action = VehicleAction(torch.full_like(state.heading, -100.0), torch.ones_like(state.heading))
    config = BicycleConfig()
    control = adapt_action(state, action, scene.dt[:, None], config)
    assert control.acceleration[0, 0] == pytest.approx(-0.5)
    assert control.steering_rate[0, 0] == pytest.approx(0.6)
    result = KinematicBicycleModel.step(state, action, scene.wheelbase, scene.dt[:, None])
    assert torch.equal(result.velocity, torch.zeros_like(result.velocity))
    assert torch.equal(result.acceleration, torch.zeros_like(result.acceleration))


def test_cached_polyline_projection_sampling_and_degenerate_routes():
    routes = empty_routes((1, 2))
    routes.points[0, 0, 1] = torch.tensor([10.0, 0.0])
    routes.points[0, 0, 2:] = torch.tensor([10.0, 10.0])
    routes.mask[0, 0, :3] = True
    cache = cache_routes(routes)
    points = torch.tensor([[[[5.0, 2.0], [12.0, 5.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    projected = project_route(cache, points)
    torch.testing.assert_close(projected.progress[0, 0], torch.tensor([5.0, 15.0]))
    torch.testing.assert_close(projected.distance[0, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        sample_route(cache, torch.tensor([[15.0, 0.0]]))[0, 0], torch.tensor([10.0, 5.0])
    )
    for value in project_route(cache, points):
        assert torch.isfinite(value).all()


def test_vehicle_reacts_to_stopped_lead_and_does_not_read_future_gt():
    clear = traffic_scene(speeds=(6.0,))
    blocked = traffic_scene(((0.0, 0.0), (15.0, 0.0)), (6.0, 0.0), (0, 0))
    batch = batch_of(clear, blocked)
    env = RuleBasedEnv()
    env.reset(batch)
    for _ in range(15):
        result = env.step()
    assert result.state.velocity[1, 0, 0] < result.state.velocity[0, 0, 0] - 1
    changed = batch_of(clear, blocked)
    changed.reference_state.position[:, 1:] += 10000
    changed.reference_state.velocity[:, 1:] *= 100
    other = RuleBasedEnv()
    other.reset(changed)
    for _ in range(15):
        other.step()
    assert_state_close(other.state, env.state, rtol=0, atol=0)


def test_pedestrian_oracle_controls_speed_without_copying_positions():
    scene = traffic_scene(((0.0, 0.0),), (0.0,), (1,), frames=10)
    scene.reference_state.velocity[4:, 0, 0] = 2.0
    batch = batch_of(scene)
    oracle = RuleBasedEnv()
    oracle.reset(batch)
    for _ in range(3):
        oracle.step()
    assert oracle.state.position[0, 0, 0] == 0
    for _ in range(5):
        oracle.step()
    assert oracle.state.position[0, 0, 0] > 0
    assert batch.reference_state.position[0, 8, 0, 0] == 0


def test_pedestrian_predicts_conflict_and_bounds_displacement():
    clear = traffic_scene(((0.0, 0.0),), (1.4,), (1,))
    conflict = traffic_scene(((0.0, 0.0), (3.0, 0.0)), (1.4, 2.0), (1, 0), headings=[0.0, math.pi])
    batch = batch_of(clear, conflict)
    batch.initial_state.position[:, 0, 1] = 0.2
    batch.reference_state.position[:, 0, 0, 1] = 0.2
    env = RuleBasedEnv()
    env.reset(batch)
    result = env.step()
    speeds = torch.linalg.vector_norm(result.state.velocity[:, 0], dim=-1)
    assert speeds[1] < speeds[0]
    assert speeds[1] == pytest.approx(1.1, abs=1e-4)
    movement = torch.linalg.vector_norm(
        result.state.position[:, 0] - batch.initial_state.position[:, 0], dim=-1
    )
    assert bool((movement <= speeds * batch.dt + 1e-5).all())
    assert bool((movement <= 0.16).all())


def test_pedestrian_ignores_clutter_and_keeps_vehicle_candidates():
    clear = traffic_scene(((0.0, 0.0),), (1.4,), (1,))
    clutter_positions = tuple((0.5 + index * 0.02, 0.0) for index in range(17))
    positions = ((0.0, 0.0),) + clutter_positions + clutter_positions
    speeds = (1.4,) + (0.0,) * 34
    types = (1,) + (1,) * 17 + (2,) * 17
    clutter = traffic_scene(positions, speeds, types)
    conflict = traffic_scene(
        positions + ((3.0, 0.0),),
        speeds + (2.0,),
        types + (0,),
        headings=[0.0] * len(positions) + [math.pi],
    )
    batch = batch_of(clear, clutter, conflict, capacity=32, static_capacity=32)
    env = RuleBasedEnv()
    env.reset(batch)
    result = env.step()
    speed = torch.linalg.vector_norm(result.state.velocity[:, 0], dim=-1)
    torch.testing.assert_close(speed[:2], torch.tensor([1.4, 1.4]))
    assert speed[2] == pytest.approx(1.1, abs=1e-4)


def test_pedestrian_stops_at_route_end_and_stationary_route_is_finite():
    scene = traffic_scene(((0.0, 0.0), (5.0, 5.0)), (0.0, 0.0), (1, 1), frames=101)
    scene.routes.points[0, :, 0] = torch.linspace(0, 1, 100)
    scene.routes.points[1] = torch.tensor([5.0, 5.0])
    scene.routes.mask[1, 1:] = False
    scene.reference_state.velocity[1:, 0, 0] = 1.4
    env = RuleBasedEnv()
    env.reset(batch_of(scene))
    for _ in range(100):
        result = env.step()
    assert result.state.position[0, 0, 0] == pytest.approx(1.0, abs=1e-4)
    assert torch.linalg.vector_norm(result.state.velocity[0, 0]) < 1e-3
    torch.testing.assert_close(result.state.position[0, 1], torch.tensor([5.0, 5.0]))
    assert all(torch.isfinite(value).all() for value in result.state)


def test_controlled_actor_spawns_once_and_persists_across_missing_logs():
    visibility = torch.ones(9, 2, dtype=torch.bool)
    visibility[:2, 1] = False
    visibility[5, 1] = False
    scene = traffic_scene(
        ((0.0, 0.0), (20.0, 4.0)), (4.0, 1.0), (0, 1), frames=9, visibility=visibility
    )
    scene.routes.start_time_s[1] = 0.2
    scene.routes.horizon_s[1] = 0.2
    scene.reference_state.position[6:, 1, 0] += 50
    env = RuleBasedEnv()
    batch = batch_of(scene)
    env.reset(batch)
    assert not env.step().actor_mask[0, 1]
    result = env.step()
    torch.testing.assert_close(result.state.position[0, 1], scene.reference_state.position[2, 1])
    previous = result.state.position[0, 1].clone()
    for _ in range(6):
        result = env.step()
        assert result.actor_mask[0, 1]
        assert (result.state.position[0, 1] - previous).norm() <= 0.4 + 1e-5
        previous = result.state.position[0, 1].clone()
    assert result.state.position[0, 1, 0] < 25


def test_missing_routes_and_rails_and_explicit_control_mask_match_log_playback():
    scene = traffic_scene(((0.0, 0.0), (30.0, 3.0)), (4.0, 1.0), (0, 1), frames=5)
    # IDM vehicles need rails and pedestrians need routes; without them both replay logs.
    batch = batch_of(replace(scene, routes=None, rails=None))
    rule, replay = RuleBasedEnv(), ReplayEnv()
    rule.reset(batch)
    replay.reset(batch)
    for _ in range(6):
        assert_state_close(rule.step().state, replay.step().state, rtol=0, atol=0)
    batch = batch_of(scene)
    rule.reset(batch, control_types=torch.zeros_like(batch.actor_type))
    replay.reset(batch)
    for _ in range(4):
        assert_state_close(rule.step().state, replay.step().state, rtol=0, atol=0)


def test_partial_reset_terminal_freeze_and_log_ego():
    batch = batch_of(traffic_scene(frames=4), traffic_scene(frames=8))
    env = RuleBasedEnv(control_ego=False)
    env.reset(batch)
    for _ in range(4):
        env.step()
    state = WorldState(*(value.clone() for value in env.state))
    progress = env.progress.clone()
    env.reset(batch, torch.tensor([True, False]))
    for actual, previous in zip(env.state, state, strict=True):
        torch.testing.assert_close(actual[1], previous[1], rtol=0, atol=0)
    torch.testing.assert_close(env.progress[1], progress[1], rtol=0, atol=0)
    result = env.step()
    assert result.episode.step.tolist() == [1, 5]
    torch.testing.assert_close(result.state.position[0, 0], batch.reference_state.position[0, 1, 0])
    for _ in range(8):
        result = env.step()
    frozen = WorldState(*(value.clone() for value in result.state))
    result = env.step()
    assert not result.advanced.any()
    assert not result.just_finished.any()
    assert_state_close(result.state, frozen, rtol=0, atol=0)


def test_fullgraph_uses_one_graph_per_controller_set_across_lifecycle_masks():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    visibility = torch.ones(8, 3, dtype=torch.bool)
    visibility[:2, 1] = False
    visibility[4:6, 2] = False
    scene = traffic_scene(
        ((0.0, 0.0), (10.0, 4.0), (8.0, 0.0)),
        (3.0, 1.0, 0.0),
        (0, 1, 2),
        frames=8,
        visibility=visibility,
    )
    batch = batch_of(scene)
    eager, compiled = RuleBasedEnv(), RuleBasedEnv()
    eager_kernel, kernels = compiled._kernel, {}

    def kernel(**flags):
        key = tuple(flags.items())
        if key not in kernels:
            kernels[key] = torch.compile(
                eager_kernel(**flags), backend=backend, fullgraph=True, dynamic=False
            )
        return kernels[key]

    compiled._kernel = kernel
    for enabled in (True, False):
        control_types = torch.where(batch.actor_mask, batch.actor_type + 1, ControlType.LOG)
        if not enabled:
            control_types.zero_()
        eager.reset(batch, control_types=control_types)
        compiled.reset(batch, control_types=control_types)
        for _ in range(10):
            expected, actual = eager.step(), compiled.step()
            assert_state_close(actual.state, expected.state, rtol=0, atol=0)
            assert torch.equal(actual.actor_mask, expected.actor_mask)
            assert torch.equal(actual.static_mask, expected.static_mask)
            assert torch.equal(actual.episode.step, expected.episode.step)
            torch.testing.assert_close(compiled.progress, eager.progress, rtol=0, atol=0)
    # One graph with IDM and pedestrians in use, one for the log-only variant.
    assert len(graphs) == 2


@pytest.mark.parametrize("config", [IDMConfig, PedestrianConfig, InteractionConfig, BicycleConfig])
def test_rule_parameters_reject_nan(config):
    name = next(iter(config.__dataclass_fields__))
    with pytest.raises(ValueError):
        config(**{name: float("nan")})
