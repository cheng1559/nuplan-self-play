"""Compare our closed-loop metrics with nuPlan's metric classes on the same driving.

Our environment drives ego (random SingleFrameNet by default, or the log with
``--policy log``) and ``DrivingEvaluator`` scores it. nuPlan then replays the recorded
ego commands in its own loop, builds a SimulationHistory, and
runs its closed-loop metrics. Both scores are printed per scene.

    uv run --env-file .env python scripts/validate_metrics.py data/scenes/*.pt
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("NUPLAN_DATA_STORE", "local")
warnings.filterwarnings("ignore")

import torch  # noqa: E402

import cs5478_project.dynamics.action_adapter as action_adapter  # noqa: E402
from cs5478_project.data import SceneDataLoader, SceneDataset, load_scene  # noqa: E402
from cs5478_project.data.nuplan.scenario import build_scenario, read_metadata  # noqa: E402
from cs5478_project.env.learned import LearnedEnv  # noqa: E402
from cs5478_project.evaluation import DrivingEvaluator  # noqa: E402
from cs5478_project.models.checkpoint import random_single_frame  # noqa: E402

NAMES = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_progress_along_expert_route",
    "ego_is_comfortable",
)


def run_ours(path, loaded, policy):
    """Our NR rollout: metrics for ego and the ideal commands applied to it (if driven)."""
    batch = next(iter(SceneDataLoader(SceneDataset([path]), batch_size=1, drop_last=False))).tensors
    ego = int(batch.ego_index[0])
    env = LearnedEnv(loaded)
    controls = torch.zeros_like(batch.actor_type)
    if policy != "log":
        controls[0, ego] = env.learned_type
    commands = []
    original = action_adapter.nuplan_bicycle_step

    def spy(state, ideal, steering_rate, *args, **kwargs):
        commands.append((float(ideal[0, ego]), float(steering_rate[0, ego])))
        return original(state, ideal, steering_rate, *args, **kwargs)

    action_adapter.nuplan_bicycle_step = spy
    try:
        result = env.reset(batch, control_types=controls)
        evaluated = torch.zeros_like(batch.actor_mask)
        evaluated[0, ego] = True
        evaluator = DrivingEvaluator()
        evaluator.reset(
            env.scene, env.state, env.episode.step, result.actor_mask, result.static_mask,
            env.control_types, evaluated,
        )  # fmt: skip
        for _ in range(int(batch.num_frames[0]) - 1):
            result = env.step()
            evaluator.step(
                env.state, env.episode.step, result.actor_mask, result.static_mask,
                env.control_types, result.advanced,
            )  # fmt: skip
    finally:
        action_adapter.nuplan_bicycle_step = original
    metrics = evaluator.metrics()
    ours = {name: float(getattr(metrics, name)[0]) for name in NAMES}
    ours["score"] = float(metrics.score[0])
    return ours, commands


def nuplan_history(scenario, commands, frames):
    """nuPlan's NR loop over our ``frames``, replaying ego commands (or the log ego)."""
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.state_representation import StateVector2D
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel,
    )
    from nuplan.planning.simulation.history.simulation_history import (
        SimulationHistory,
        SimulationHistorySample,
    )
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory,
    )

    class History:
        current_state = None

    parameters = get_pacifica_parameters()
    model = KinematicBicycleModel(parameters)
    observation = TracksObservation(scenario)
    history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    buffer = History()
    log = list(scenario.get_expert_ego_trajectory())
    ego = scenario.initial_ego_state
    for i in range(frames):
        current = SimulationIteration(scenario.get_time_point(i), i)
        if commands is None:
            next_ego = log[min(i + 1, frames - 1)]
        elif i + 1 < frames:
            acceleration, steering_rate = commands[i]
            ideal = DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=parameters.rear_axle_to_center,
                rear_axle_velocity_2d=ego.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
                tire_steering_rate=steering_rate,
            )
            following = scenario.get_time_point(i + 1)
            next_ego = model.propagate_state(ego, ideal, following - current.time_point)
        else:
            next_ego = ego
        trajectory = InterpolatedTrajectory([ego, next_ego]) if next_ego is not ego else None
        history.add_sample(
            SimulationHistorySample(
                current,
                ego,
                trajectory or InterpolatedTrajectory([ego, log[-1]]),
                observation.get_observation(),
                list(scenario.get_traffic_light_status_at_iteration(i)),
            )
        )
        if i + 1 < frames:
            following = SimulationIteration(scenario.get_time_point(i + 1), i + 1)
            buffer.current_state = (ego, None)
            observation.update_observation(current, following, buffer)
            ego = next_ego
    return history


def nuplan_metrics(history, scenario):
    """nuPlan's closed-loop metric scores and the weighted closed-loop score."""
    from nuplan.planning.metrics.evaluation_metrics.common import (
        drivable_area_compliance as drivable,
    )
    from nuplan.planning.metrics.evaluation_metrics.common import ego_is_comfortable as comfort
    from nuplan.planning.metrics.evaluation_metrics.common import ego_is_making_progress as making
    from nuplan.planning.metrics.evaluation_metrics.common import (
        ego_jerk,
        ego_lane_change,
        ego_lon_jerk,
    )
    from nuplan.planning.metrics.evaluation_metrics.common import ego_lat_acceleration as lat
    from nuplan.planning.metrics.evaluation_metrics.common import ego_lon_acceleration as lon
    from nuplan.planning.metrics.evaluation_metrics.common import (
        ego_progress_along_expert_route as progress,
    )
    from nuplan.planning.metrics.evaluation_metrics.common import ego_yaw_acceleration as yaw_accel
    from nuplan.planning.metrics.evaluation_metrics.common import ego_yaw_rate as yaw_rate
    from nuplan.planning.metrics.evaluation_metrics.common import (
        no_ego_at_fault_collisions as collisions,
    )
    from nuplan.planning.metrics.evaluation_metrics.common import (
        speed_limit_compliance as speed,
    )
    from nuplan.planning.metrics.evaluation_metrics.common import (
        time_to_collision_within_bound as ttc,
    )

    lane = ego_lane_change.EgoLaneChangeStatistics("ego_lane_change", "Planning", 0.3)
    jerk = ego_jerk.EgoJerkStatistics("ego_jerk", "Dynamics", max_abs_mag_jerk=8.37)
    lat_accel = lat.EgoLatAccelerationStatistics("ego_lat_acceleration", "Dynamics", 4.89)
    lon_accel = lon.EgoLonAccelerationStatistics("ego_lon_acceleration", "Dynamics", -4.05, 2.40)
    lon_jerk = ego_lon_jerk.EgoLonJerkStatistics("ego_lon_jerk", "Dynamics", 4.13)
    yaw_a = yaw_accel.EgoYawAccelerationStatistics("ego_yaw_acceleration", "Dynamics", 1.93)
    yaw_r = yaw_rate.EgoYawRateStatistics("ego_yaw_rate", "Dynamics", 0.95)
    route = progress.EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route", "Planning", score_progress_threshold=2
    )
    for metric in (lane, jerk, lat_accel, lon_accel, lon_jerk, yaw_a, yaw_r, route):
        metric.compute(history, scenario)
    collision = collisions.EgoAtFaultCollisionStatistics(
        "no_ego_at_fault_collisions", "Dynamics", lane
    )
    high = {
        "drivable_area_compliance": drivable.DrivableAreaComplianceStatistics(
            "drivable_area_compliance", "Planning", lane, 0.3
        ),
        "no_ego_at_fault_collisions": collision,
        "time_to_collision_within_bound": ttc.TimeToCollisionStatistics(
            "time_to_collision_within_bound", "Planning", lane, collision, 0.1, 3.0, 0.95
        ),
        "speed_limit_compliance": speed.SpeedLimitComplianceStatistics(
            "speed_limit_compliance", "Violations", lane, 1.0, 2.23
        ),
        "ego_is_comfortable": comfort.EgoIsComfortableStatistics(
            "ego_is_comfortable", "Violations", jerk, lat_accel, lon_accel, lon_jerk, yaw_a, yaw_r
        ),
        "ego_is_making_progress": making.EgoIsMakingProgressStatistics(
            "ego_is_making_progress", "Planning", route, 0.2
        ),
    }
    scores = {
        "ego_progress_along_expert_route": float(route.compute(history, scenario)[0].metric_score)
    }
    for name, metric in high.items():
        scores[name] = float(metric.compute(history, scenario)[0].metric_score)
    weighted = (
        5 * scores["time_to_collision_within_bound"]
        + 5 * scores["ego_progress_along_expert_route"]
        + 4 * scores["speed_limit_compliance"]
        + 2 * scores["ego_is_comfortable"]
    ) / 16
    scores["score"] = (
        scores["no_ego_at_fault_collisions"]
        * scores["drivable_area_compliance"]
        * scores["ego_is_making_progress"]
        * weighted
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--policy", choices=["random", "log"], default="random")
    parser.add_argument("--seed", type=int, default=1, help="Random SingleFrameNet seed.")
    args = parser.parse_args()
    loaded = random_single_frame(seed=args.seed)
    for path in args.scenes:
        ours, commands = run_ours(path, loaded, args.policy)
        frames = load_scene(path).num_frames
        # nuPlan compares against the expert over the simulated span, so the scenario must
        # be as long as our episode (the exporter's scenarios are 45 s).
        scenario = build_scenario(read_metadata(path), duration=(frames - 1) * 0.1)
        frames = min(frames, scenario.get_number_of_iterations())
        history = nuplan_history(scenario, None if args.policy == "log" else commands, frames)
        theirs = nuplan_metrics(history, scenario)
        row = {"scene": path.stem, "policy": args.policy}
        for name in (*NAMES, "score"):
            row[name] = [round(ours[name], 3), round(theirs[name], 3)]
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
