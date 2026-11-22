"""nuPlan-style metrics and training rewards on small log-replayed scenes."""

import math

import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.datatypes.map import MapElementType
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.env import RuleBasedEnv
from cs5478_project.evaluation import DrivingEvaluator, EvaluationConfig
from cs5478_project.evaluation.evaluator import comfort_signals, within_comfort
from cs5478_project.evaluation.savgol import savgol


@pytest.mark.parametrize(
    ("window", "order", "derivative"), [(8, 2, 0), (15, 2, 1), (5, 2, 1), (5, 3, 2)]
)
def test_savgol_matches_scipy_for_each_valid_length(window, order, derivative):
    values = np.cumsum(np.random.default_rng(0).normal(size=(3, 60)), 1)
    lengths = [60, 41, window]
    ours = savgol(torch.tensor(values), torch.tensor(lengths), window, order, derivative)
    for row, length in enumerate(lengths):
        expected = savgol_filter(values[row, :length], window, order, deriv=derivative)
        np.testing.assert_allclose(ours[row, :length].numpy(), expected, atol=1e-10)


def run(batch, evaluated, config=EvaluationConfig()):
    """Replay the log for every actor; return per-step rewards and episode metrics."""
    env = RuleBasedEnv()
    result = env.reset(batch, control_types=torch.zeros_like(batch.actor_type))
    evaluator = DrivingEvaluator(config)
    evaluator.reset(
        env.scene, env.state, env.episode.step, result.actor_mask, result.static_mask,
        env.control_types, evaluated,
    )  # fmt: skip
    rewards = []
    for _ in range(int(batch.num_frames.max()) - 1):
        result = env.step()
        rewards.append(
            evaluator.step(
                env.state,
                env.episode.step,
                result.actor_mask,
                result.static_mask,
                env.control_types,
                result.advanced,
            )  # fmt: skip
        )
    return rewards, evaluator.metrics()


def rear_end():
    # A drives at 10 m/s toward B, which waits 8 m ahead: their boxes touch after 0.4 s.
    return batch_of(
        traffic_scene(((0.0, 0.0), (8.0, 0.0)), (10.0, 0.0), (ActorType.VEHICLE,) * 2, frames=11)
    )


def test_rear_end_is_at_fault_for_the_follower_only():
    batch = rear_end()
    evaluated = batch.actor_mask.clone()
    rewards, metrics = run(batch, evaluated)
    assert metrics.no_ego_at_fault_collisions.tolist() == [0.0, 1.0]
    # The stopped leader never sees a short TTC; the follower does.
    assert metrics.time_to_collision_within_bound.tolist() == [0.0, 1.0]
    assert metrics.score.tolist()[0] == 0.0
    done = torch.stack([reward.done[0, 0] for reward in rewards])
    hard = torch.stack([reward.hard[0, 0] for reward in rewards])
    # Overlap begins at the 0.4 s frame (touching counts), which ends the follower's episode.
    # The leader is stopped, so the collision reward does not penalize it either.
    assert done.nonzero().flatten().tolist() == [3]
    assert hard[3] == -1.0 and rewards[3].reward[0, 0] == -1.0
    assert all(reward.reward[0, 0] == 0 for reward in rewards[4:])
    assert rewards[-1].terminated[0].tolist()[:2] == [True, False]


def test_any_collision_while_moving_is_penalized():
    # As above, but the leader crawls at 1 m/s: nuPlan blames only the follower, while the
    # reward penalizes and ends both episodes.
    batch = batch_of(
        traffic_scene(((0.0, 0.0), (8.0, 0.0)), (10.0, 1.0), (ActorType.VEHICLE,) * 2, frames=11)
    )
    rewards, metrics = run(batch, batch.actor_mask.clone())
    assert metrics.no_ego_at_fault_collisions.tolist() == [0.0, 1.0]
    first = [int(torch.stack([r.done[0, slot] for r in rewards]).nonzero()[0]) for slot in (0, 1)]
    assert first[0] == first[1]
    assert rewards[first[0]].hard[0, :2].tolist() == [-1.0, -1.0]


def with_lane(batch, start, end, half_width=1.75):
    """Add one straight lane centerline from start to end with boundaries on both sides."""
    start, end = torch.tensor(start), torch.tensor(end)
    direction = (end - start) / (end - start).norm()
    side = torch.stack((-direction[1], direction[0])) * half_width
    t = torch.linspace(0, 1, batch.map.chunk_points.shape[2])[:, None]
    line = start + t * (end - start)
    chunk_points, chunk_type, chunk_mask = (
        batch.map.chunk_points.clone(), batch.map.chunk_type.clone(), batch.map.chunk_mask.clone()
    )  # fmt: skip
    kinds = (
        MapElementType.LANE_CENTERLINE,
        MapElementType.LANE_BOUNDARY,
        MapElementType.LANE_BOUNDARY,
    )
    for index, (points, kind) in enumerate(zip((line, line + side, line - side), kinds)):
        chunk_points[:, index] = points
        chunk_type[:, index] = kind
        chunk_mask[:, index] = True
    return batch._replace(
        map=batch.map._replace(
            chunk_points=chunk_points, chunk_type=chunk_type, chunk_mask=chunk_mask
        )
    )


def test_wrong_way_driving_is_scored_then_ends_the_episode():
    batch = batch_of(traffic_scene(((0.0, 0.0),), (10.0,), (ActorType.VEHICLE,), frames=31))
    # The lane runs toward -x, so driving +x at 10 m/s goes 1 m backward per step.
    rewards, _ = run(with_lane(batch, (80.0, 0.0), (-20.0, 0.0)), batch.actor_mask.clone())
    direction = torch.stack([reward.direction[0, 0] for reward in rewards[:6]])
    # Backward progress over the last second: 1, 2, 3 m, ... (> 2 m scores 0).
    assert direction.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    done = torch.stack([reward.done[0, 0] for reward in rewards])
    assert done.nonzero().flatten().tolist() == [6]  # 7 m > 6 m
    assert rewards[6].hard[0, 0] == -1.0


def test_centerline_scores_the_lateral_room_used():
    batch = batch_of(traffic_scene(((0.0, 0.5),), (5.0,), (ActorType.VEHICLE,), frames=11))
    rewards, _ = run(with_lane(batch, (-20.0, 0.0), (80.0, 0.0)), batch.actor_mask.clone())
    # 0.5 m off center in a 3.5 m lane with a 2 m wide car: 0.5 / 0.75 of the room used.
    expected = 1 - 0.25 * 0.5 / 0.75
    assert rewards[0].centerline[0, 0] == pytest.approx(expected)
    assert rewards[0].direction[0, 0] == 1.0


def test_a_jerk_violation_fails_comfort_for_the_rest_of_the_episode():
    scene = traffic_scene(((0.0, 0.0),), (5.0,), (ActorType.VEHICLE,), frames=61)
    # Accelerate, then brake: the smoothed jerk reaches 5 m/s^3 (bound 4.13) near 2.5 s.
    acceleration = scene.reference_state.acceleration
    acceleration[10:25] = 2.0
    acceleration[25:40] = -3.0
    rewards, metrics = run(batch_of(scene), batch_of(scene).actor_mask.clone())
    comfort = torch.stack([reward.comfort[0, 0] for reward in rewards])
    assert comfort[:5].tolist() == [1.0] * 5
    # Acceleration and jerk are calm again by then, but the jerk history keeps failing
    # (without it these frames would score 0.75, the repeat factor).
    assert comfort[-10:].tolist() == [0.0] * 10
    assert metrics.ego_is_comfortable.tolist() == [0.0]


def test_ttc_scores_the_closing_gap():
    batch = rear_end()
    rewards, _ = run(batch, batch.actor_mask.clone())
    # After the first step the gap is 3 m at 10 m/s: TTC 0.3 s is below 0.95 s.
    assert rewards[0].ttc[0, 0] == 0.0
    assert rewards[0].ttc[0, 1] == 1.0


def test_free_drive_reaches_its_goal_once_with_full_scores():
    batch = batch_of(traffic_scene(((0.0, 0.0),), (5.0,), (ActorType.VEHICLE,), frames=31))
    rewards, metrics = run(batch, batch.actor_mask.clone())
    goal = torch.stack([reward.goal[0, 0] for reward in rewards])
    # The log ends at the goal, so it is paid exactly once, on a late step.
    assert (goal > 0).sum() == 1 and goal.sum() == 0.5
    assert int(goal.nonzero()[0]) >= goal.shape[0] - 5
    assert metrics.score.tolist() == [1.0]
    assert metrics.ego_progress_along_expert_route.tolist() == [1.0]
    # Soft rewards sum to one over the episode when every soft score is perfect.
    total = sum(float(reward.reward[0, 0]) for reward in rewards)
    assert total == pytest.approx(1.5)


def test_leaving_the_drivable_area_ends_the_episode():
    batch = batch_of(traffic_scene(((0.0, 0.0),), (10.0,), (ActorType.VEHICLE,), frames=31))
    # Drivable area: x < 20 m. The grid stores the distance outside it.
    grid = batch.map.drivable_distance
    columns = 120
    x = -10 + torch.arange(columns) * 0.5
    grid[0, :40, :columns] = (x - 20).clamp_min(0).to(grid.dtype)
    # The reward uses the road boundary polyline instead: a curb at x = 20 m.
    batch = batch._replace(
        map=batch.map._replace(
            drivable_distance=grid,
            drivable_origin=torch.tensor([[-10.0, -10.0]]),
            drivable_size=torch.tensor([[40, columns]]),
            segments=torch.tensor([[[[20.0, -10.0], [20.0, 10.0]]]]),
            segment_type=torch.tensor([[MapElementType.ROAD_BOUNDARY]]),
            segment_mask=torch.tensor([[True]]),
        )
    )
    rewards, metrics = run(batch, batch.actor_mask.clone())
    done = torch.stack([reward.done[0, 0] for reward in rewards])
    # The reward ends the episode when the front edge (center + 2 m) touches the curb at
    # x = 20 m, with -1 scaled by 1 + 5 tanh(2 v / 40); the metric allows nuPlan's 0.3 m.
    first = int(done.nonzero()[0])
    assert 1.70 <= (first + 1) * 0.1 <= 1.80
    assert rewards[first].hard[0, 0] == pytest.approx(-(1 + 5 * math.tanh(0.5)))
    assert metrics.drivable_area_compliance.tolist() == [0.0]


def test_comfort_bounds_follow_nuplan_signals():
    dt = torch.tensor([0.1])
    t = torch.arange(60) * 0.1
    gentle = torch.stack((torch.full_like(t, -2.0), torch.full_like(t, 2.0), 0.1 * t), -1)
    harsh = gentle.clone()
    harsh[:, 0] = -6.0  # below nuPlan's -4.05 m/s^2
    swerve = gentle.clone()
    swerve[:, 2] = 0.3 * torch.sin(2 * t)  # yaw rate up to 0.6 rad/s, yaw accel 1.2 rad/s^2
    length = torch.tensor([60])
    for series, expected in ((gentle, True), (harsh, False), (swerve, True)):
        signals = comfort_signals(series[None], length, dt)
        assert bool(within_comfort(signals, EvaluationConfig()).all()) is expected


def test_boundary_crossing_sees_edges_and_corner_paths():
    from types import SimpleNamespace

    from cs5478_project.agents.geometry import box_corners
    from cs5478_project.evaluation.geometry import boundary_crossing

    curb = MapElementType.ROAD_BOUNDARY
    map_batch = SimpleNamespace(
        segments=torch.tensor([[[[5.0, -5.0], [5.0, 5.0]], [[0.0, 3.0], [9.0, 3.0]]]]),
        segment_type=torch.tensor([[curb, MapElementType.LANE_BOUNDARY]]),
        segment_mask=torch.tensor([[True, True]]),
    )
    size = torch.tensor([[4.0, 2.0]])

    def box(x):
        return box_corners(torch.tensor([[x, 0.0]]), torch.zeros(1), size)

    world = torch.zeros(1, dtype=torch.int64)
    # Clear of the curb; the lane line at y = 3 does not count.
    assert not boundary_crossing(map_batch, world, box(0.0), box(0.0), curb).item()
    # The front edge touches it.
    assert boundary_crossing(map_batch, world, box(3.0), box(3.0), curb).item()
    # Jumping across it within one step: only the corner paths cross.
    assert boundary_crossing(map_batch, world, box(10.0), box(0.0), curb).item()
