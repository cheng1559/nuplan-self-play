"""nuPlan reactive IDM agents on rails: routing, signals, stop lines, leads, lifecycle."""

import math

import pytest
import torch
from shapely.geometry import LineString, Polygon
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.agents.geometry import box_corners
from cs5478_project.agents.idm import _pieces_distance, _rectangles
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.rails import EDGE_POINTS, ReactiveRole
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.env import RuleBasedEnv
from cs5478_project.server.services.controls import reactive_controls


def set_edge(rails, edge, start, end, *, curvature=0.0, speed_limit=0.0, outgoing=()):
    start, end = torch.tensor(start), torch.tensor(end)
    t = torch.linspace(0, 1, EDGE_POINTS)[:, None]
    rails.edge_points[edge] = start + t * (end - start)
    delta = end - start
    rails.edge_heading[edge] = math.atan2(float(delta[1]), float(delta[0]))
    rails.edge_length[edge] = float(delta.norm())
    rails.edge_curvature[edge] = curvature
    rails.edge_speed_limit[edge] = speed_limit
    rails.edge_outgoing[edge] = -1
    rails.edge_outgoing[edge, : len(outgoing)] = torch.tensor(outgoing)
    rails.edge_mask[edge] = True


def junction_scene(*, frames=60, speed=8.0, lights=None):
    """Ego parked off the road; one vehicle on a 20 m lane entering a junction with two exits.

    Edge 1 heads straight (curvature 0) and edge 2 turns left (curvature 0.1).
    ``lights`` maps an exit edge to (has_lights, green, red) for every frame.
    """
    scene = traffic_scene(((10.0, -30.0), (0.0, 0.0)), (0.0, speed), (0, 0), frames=frames)
    rails = scene.rails
    for edge in range(rails.edge_mask.shape[0]):
        rails.edge_mask[edge] = False
    set_edge(rails, 0, (0.0, 0.0), (20.0, 0.0), outgoing=(2, 1))
    set_edge(rails, 1, (20.0, 0.0), (60.0, 0.0))
    set_edge(rails, 2, (20.0, 0.0), (20.0, 40.0), curvature=0.1)
    for slot, (edge, has_lights, green, red) in enumerate(
        (edge, *state) for edge, state in (lights or {}).items()
    ):
        rails.edge_has_lights[edge] = has_lights
        rails.edge_signal[edge] = slot
        rails.signal_green[:, slot] = green
        rails.signal_red[:, slot] = red
    rails.rail_valid[0] = False
    rails.rail_valid[1] = True
    rails.rail_edge[1] = 0
    rails.rail_progress[1] = 0.0
    rails.rail_pose[1] = torch.tensor([0.0, 0.0, 0.0])
    rails.rail_speed[1] = speed
    rails.rail_size[1] = scene.size[1]
    rails.role[1] = ReactiveRole.IDM
    return scene


def run(scene, steps, config_controls=None):
    batch = batch_of(scene)
    env = RuleBasedEnv()
    controls = torch.zeros_like(batch.actor_type)
    controls[0, 1] = ControlType.IDM
    if config_controls is not None:
        controls = config_controls(batch)
    result = env.reset(batch, control_types=controls)
    trace = [env.state.position[0, 1].clone()]
    for _ in range(steps):
        result = env.step()
        trace.append(env.state.position[0, 1].clone())
    return env, result, torch.stack(trace)


def test_plan_route_takes_the_straightest_allowed_exit():
    _, _, trace = run(junction_scene(), 40)
    assert trace[-1, 0] > 30 and abs(float(trace[-1, 1])) < 1e-3


def test_signalized_exit_needs_green_and_unsignalized_exit_must_not_be_red():
    # Straight connector is signalized but not green: the agent turns left.
    lights = {1: (True, False, False)}
    _, _, trace = run(junction_scene(lights=lights), 40)
    assert trace[-1, 1] > 5 and abs(float(trace[-1, 0]) - 20) < 0.5
    # Both exits blocked: the path ends at the junction and the agent stops before it.
    lights = {1: (True, False, True), 2: (False, False, True)}
    env, _, trace = run(junction_scene(lights=lights, frames=120), 110)
    center = trace[-1, 0] + float(env.scene.reference_to_center[0, 1])
    assert 15 < center <= 20 and env.rails.velocity[0, 1] < 0.05


def test_red_connector_stop_line_stops_the_agent():
    scene = junction_scene(lights={1: (True, True, False)}, frames=120)
    rails = scene.rails
    # The straight exit is in the route: green while planning, red afterwards.
    rails.signal_green[3:, 0] = False
    rails.signal_red[3:, 0] = True
    rails.stop_points[1, 0, :4] = torch.tensor(
        [[21.0, -2.0], [21.5, -2.0], [21.5, 2.0], [21.0, 2.0]]
    )
    rails.stop_mask[1, 0, :4] = True
    env, _, trace = run(scene, 110)
    front = trace[-1, 0] + float(env.scene.reference_to_center[0, 1]) + scene.size[1, 0] / 2
    assert front < 21.0 and env.rails.velocity[0, 1] < 0.05


def test_speed_limit_of_the_end_segment_becomes_the_target():
    scene = junction_scene(speed=2.0, frames=110)
    scene.rails.edge_speed_limit[1] = 4.0
    env, _, _ = run(scene, 100)
    assert env.rails.target[0, 1] == pytest.approx(4.0)
    assert 3.5 < float(env.rails.velocity[0, 1]) <= 4.0


def test_agents_beyond_the_radius_are_removed_for_good():
    scene = junction_scene(frames=30)
    env, result, _ = run(scene, 1)
    assert bool(result.actor_mask[0, 1])
    env.scene.reference_state.position[0, :, 0] = torch.tensor([500.0, 0.0])
    env._state = env.state._replace(position=env.state.position.clone())
    env._state.position[0, 0] = torch.tensor([500.0, 0.0])
    result = env.step()
    assert not bool(result.actor_mask[0, 1]) and bool(env.rails.removed[0, 1])
    env._state.position[0, 0] = torch.tensor([0.0, 0.0])
    assert not bool(env.step().actor_mask[0, 1])


def test_reset_snaps_idm_agents_onto_their_rails():
    scene = junction_scene()
    scene.rails.rail_pose[1] = torch.tensor([3.0, 0.0, 0.0])
    scene.rails.rail_progress[1] = 3.0
    batch = batch_of(scene)
    env = RuleBasedEnv()
    controls = torch.zeros_like(batch.actor_type)
    controls[0, 1] = ControlType.IDM
    env.reset(batch, control_types=controls)
    center = env.state.position[0, 1] + batch.reference_to_center[0, 1] * torch.tensor([1.0, 0.0])
    torch.testing.assert_close(center, torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(env.state.velocity[0, 1], torch.tensor([8.0, 0.0]))


def test_flat_capped_buffers_match_shapely():
    points = torch.tensor([[0.0, 0.0], [4.0, 0.0], [8.0, 1.0]], dtype=torch.float64)
    rects, valid = _rectangles(points, torch.tensor(1.0, dtype=torch.float64))
    reference = LineString(points.numpy()).buffer(1.0, cap_style="flat", join_style="mitre")
    for center, expected_touch in (
        ((9.5, 1.2), False),
        ((8.8, 1.0), False),
        ((8.6, 1.0), True),
        ((-0.5, 0.0), True),
    ):
        box = box_corners(
            torch.tensor(center, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
        distance = _pieces_distance(rects, valid, box[None], torch.ones(1, dtype=torch.bool))
        shapely_distance = reference.distance(Polygon(box.numpy()))
        assert float(distance) == pytest.approx(shapely_distance, abs=1e-6)
        assert (float(distance) <= 1e-6) == expected_touch


def test_reactive_roles_map_to_controls():
    role = torch.tensor([[ReactiveRole.EGO, ReactiveRole.IDM, ReactiveRole.OPEN_LOOP, 0]])
    mask = torch.tensor([[True, True, True, True]])
    controls = reactive_controls(torch.zeros_like(role), role, mask)
    assert controls.tolist() == [
        [ControlType.LOG, ControlType.IDM, ControlType.LOG, ControlType.ABSENT]
    ]
    ego = reactive_controls(torch.zeros_like(role), role, mask, torch.tensor([0]), ControlType.IDM)
    assert ego[0, 0] == ControlType.IDM


def test_absent_actors_never_appear_and_blind_agents_ignore_objects():
    scene = traffic_scene(((0.0, 0.0), (12.0, 0.0)), (4.0, 0.0), (0, 0), frames=30)
    batch = batch_of(scene)
    controls = torch.zeros_like(batch.actor_type)
    controls[0, 0] = ControlType.IDM
    controls[0, 1] = ControlType.ABSENT
    env = RuleBasedEnv()
    result = env.reset(batch, control_types=controls)
    assert not bool(result.actor_mask[0, 1])
    for _ in range(25):
        result = env.step()
        assert not bool(result.actor_mask[0, 1])
    # With the stopped vehicle replayed instead, a blind ego drives through it.
    controls[0, 1] = ControlType.LOG
    blind = torch.zeros_like(batch.actor_mask)
    blind[0, 0] = True
    seeing, ignoring = RuleBasedEnv(), RuleBasedEnv()
    seeing.reset(batch, control_types=controls)
    ignoring.reset(batch, control_types=controls, blind=blind)
    for _ in range(25):
        seeing.step()
        ignoring.step()
    assert ignoring.state.position[0, 0, 0] > seeing.state.position[0, 0, 0] + 3
    assert batch.actor_type[0, 1] == ActorType.VEHICLE


def test_default_controls_give_idm_only_to_railed_vehicles():
    # The second vehicle appears after frame zero, so nuPlan's builder never selects it.
    visible = torch.ones(10, 3, dtype=torch.bool)
    visible[:3, 1] = False
    scene = traffic_scene(
        ((0.0, 0.0), (30.0, 0.0), (5.0, 5.0)),
        (4.0, 4.0, 1.0),
        (ActorType.VEHICLE, ActorType.VEHICLE, ActorType.PEDESTRIAN),
        frames=10,
        visibility=visible,
    )
    batch = batch_of(scene)
    env = RuleBasedEnv()
    env.reset(batch)
    assert batch.rails.rail_valid[0, :3].tolist() == [True, False, False]
    assert env.control_types[0, :3].tolist() == [
        ControlType.IDM,
        ControlType.LOG,
        ControlType.PEDESTRIAN,
    ]


def test_rectangle_touch_matches_the_polygon_distance_test():
    from cs5478_project.agents.geometry import box_corners
    from cs5478_project.agents.idm import _pieces_distance, _rectangles_touch

    generator = torch.Generator().manual_seed(0)

    def rectangles(*shape):
        center = torch.rand(*shape, 2, generator=generator) * 20
        heading = torch.rand(*shape, generator=generator) * 6.3
        size = 0.5 + torch.rand(*shape, 2, generator=generator) * 5
        return box_corners(center, heading, size)

    first, second = rectangles(500, 3), rectangles(500, 4)
    first_valid = torch.rand(500, 3, generator=generator) > 0.2
    second_valid = torch.rand(500, 4, generator=generator) > 0.2
    expected = _pieces_distance(first, first_valid, second, second_valid) <= 1e-6
    touch = _rectangles_touch(first, first_valid, second, second_valid)
    assert expected.any() and (~expected).any()
    assert torch.equal(touch, expected)
