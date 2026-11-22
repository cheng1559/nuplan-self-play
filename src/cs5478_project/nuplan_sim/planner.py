"""SingleFrameNet as a nuPlan planner: one command per simulation step, closed loop.

Everything the policy observes comes from the planner's inputs: the map API and route
roadblocks at initialization, and ego's state, the tracked objects, and the traffic
lights at every step (no log and no exported scene). ``online.build_scene`` turns the
first step's inputs into the policy's scene; our environment then serves as a mirror of
the devkit's simulation: each step its state is overwritten with the devkit's, the
policy decides on it, and the resulting ideal acceleration and steering rate go to
``OneStageController``.
"""

import dataclasses
import math
import threading

import torch
from nuplan.common.actor_state.state_representation import TimeDuration
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
    KinematicBicycleModel,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

import cs5478_project.dynamics.action_adapter as action_adapter
from cs5478_project.data import SceneDataLoader, SceneDataset
from cs5478_project.data.nuplan.tracks import EGO_REAR_TO_CENTER, EGO_SIZE, EGO_WHEELBASE
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.datatypes.state import WorldState
from cs5478_project.models.checkpoint import load_single_frame
from cs5478_project.nuplan_sim.controller import CommandTrajectory, propagate
from cs5478_project.nuplan_sim.online import RECROP_M, build_scene, crop_map, to_scene
from cs5478_project.nuplan_sim.speedup import install
from cs5478_project.training.randomization import conditions_from
from cs5478_project.training.rollout import PPOEnv, ego_controls

# The command of the latest bicycle step in this thread (ideal acceleration and steering
# rate [B, A]): what our environment applied to ego is what the devkit must apply.
_latest = threading.local()
_bicycle_step = action_adapter.nuplan_bicycle_step


def _recording_step(state, ideal, steering_rate, *args, **kwargs):
    _latest.command = (ideal, steering_rate)
    return _bicycle_step(state, ideal, steering_rate, *args, **kwargs)


action_adapter.nuplan_bicycle_step = _recording_step
install()

_models = {}
# The exporter's actor types: vehicles; pedestrians and bicycles; the rest are static.
_DYNAMIC = {
    TrackedObjectType.VEHICLE: ActorType.VEHICLE,
    TrackedObjectType.PEDESTRIAN: ActorType.PEDESTRIAN,
    TrackedObjectType.BICYCLE: ActorType.PEDESTRIAN,
}


def _load(checkpoint, device, goal_horizon_s):
    key = (str(checkpoint), str(device), goal_horizon_s)
    if key not in _models:
        loaded = load_single_frame(checkpoint, device=device)
        if goal_horizon_s is not None:
            config = dataclasses.replace(loaded.config, goal_horizon_s=goal_horizon_s)
            loaded = dataclasses.replace(loaded, config=config)
        _models[key] = loaded
    return _models[key]


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class SingleFramePlanner(AbstractPlanner):
    """``action`` is the Beta ``mode`` or ``mean``; ``goal_horizon_s`` overrides the
    checkpoint's goal lookahead."""

    requires_scenario = False

    def __init__(
        self, checkpoint, action="mode", goal_horizon_s=None, device="cpu", reward_weights=None
    ):
        self._checkpoint = checkpoint
        # Reward weights told to a reward-conditioned policy ({name: value}; None: defaults).
        self._reward_weights = dict(reward_weights or {})
        self._action = action
        self._goal_horizon_s = goal_horizon_s
        self._device = torch.device(device)
        self._model = KinematicBicycleModel(get_pacifica_parameters())
        self._env = None
        self._command = (0.0, 0.0)
        # Diagnostics: the route came from the map; objects beyond the slot capacities;
        # times the map was cut again around ego.
        self.map_route = True
        self.dropped = 0
        self.recrops = 0

    def name(self) -> str:
        return "single_frame_planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        if self._device.type == "cpu":
            torch.set_num_threads(1)
        self._map_api = initialization.map_api
        self._route_roadblock_ids = list(initialization.route_roadblock_ids)
        self._env = None

    def _batch(self, scene):
        loader = SceneDataLoader(
            SceneDataset([scene], validate=False),
            batch_size=1,
            drop_last=False,
            device=self._device,
            idm_fields=False,
        )
        return next(iter(loader)).tensors

    def _start(self, ego_state, observation):
        self._scene, self._signals, self.map_route = build_scene(
            ego_state, observation, self._map_api, self._route_roadblock_ids
        )
        self._origin = self._scene.map.world_from_scene
        batch = self._batch(self._scene)
        loaded = _load(self._checkpoint, self._device, self._goal_horizon_s)
        env = PPOEnv(
            loaded,
            stochastic=False,
            deterministic=self._action,
            amp=self._device.type == "cuda",
        )
        env.reset(batch, control_types=ego_controls(batch))
        if self._reward_weights:
            env.conditions = conditions_from(self._reward_weights, self._device)[None]
        self._env, self._ego = env, int(batch.ego_index[0])
        self._last_step = int(batch.num_frames[0]) - 1
        self._slots, self._statics = {}, {}

    def _recrop(self, ego_state, observation):
        """Cut the map again around ego once it is near the edge of the current crop."""
        x, y = to_scene(self._origin, ego_state.rear_axle.x, ego_state.rear_axle.y)
        lower_x, lower_y, upper_x, upper_y = self._scene.map.bounds
        if min(x - lower_x, upper_x - x, y - lower_y, upper_y - y) >= RECROP_M:
            return
        objects = [
            to_scene(self._origin, obj.center.x, obj.center.y)
            for obj in observation.tracked_objects.tracked_objects
        ]
        self._scene, self._signals = crop_map(
            self._scene, self._map_api, self._origin, objects, (x, y)
        )
        # Only the map changes; the signal rows are rewritten every step.
        self._env._scene = self._env.scene._replace(map=self._batch(self._scene).map)
        self.recrops += 1

    def _sync(self, ego_state, observation, traffic_lights, step):
        """Overwrite the mirror with the devkit's ego, objects, and traffic lights."""
        env, ego = self._env, self._ego
        scene = env.scene
        state = WorldState(*(value.clone() for value in env.state))
        yaw0 = self._origin[2]
        capacity = scene.actor_mask.shape[1]

        rear = ego_state.rear_axle
        dynamic = ego_state.dynamic_car_state
        x, y = to_scene(self._origin, rear.x, rear.y)
        heading = float(state.heading[0, ego])
        heading += _wrap(rear.heading - yaw0 - heading)
        vx, vy = dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y
        c, s = math.cos(heading), math.sin(heading)
        state.position[0, ego] = torch.tensor([x, y])
        state.heading[0, ego] = heading
        state.velocity[0, ego] = torch.tensor([c * vx - s * vy, s * vx + c * vy])
        state.acceleration[0, ego] = dynamic.rear_axle_acceleration_2d.x
        state.steering[0, ego] = ego_state.tire_steering_angle
        state.yaw_rate[0, ego] = dynamic.angular_velocity

        present = torch.zeros_like(scene.actor_mask[0])
        present[ego] = True
        static = scene.static
        static.valid[0, step] = False
        cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
        for obj in observation.tracked_objects.tracked_objects:
            token = obj.track_token
            x, y = to_scene(self._origin, obj.center.x, obj.center.y)
            size = torch.tensor([max(0.1, obj.box.length), max(0.1, obj.box.width)])
            kind = _DYNAMIC.get(obj.tracked_object_type)
            if kind is None:
                # Static objects keep the pose and size of their first observation.
                slot = self._statics.get(token)
                if slot is None:
                    if len(self._statics) >= static.mask.shape[1]:
                        self.dropped += 1
                        continue
                    slot = self._statics[token] = len(self._statics)
                    static.ids[0, slot] = capacity + slot
                    static.position[0, slot] = torch.tensor([x, y])
                    static.heading[0, slot] = _wrap(obj.center.heading - yaw0)
                    static.size[0, slot] = size
                    static.mask[0, slot] = True
                static.valid[0, step, slot] = True
                continue
            slot = self._slots.get(token)
            if slot is None:
                if len(self._slots) + 1 >= capacity:
                    self.dropped += 1
                    continue
                slot = self._slots[token] = len(self._slots) + 1  # slot 0 is ego
                vehicle = kind == ActorType.VEHICLE
                scale = float(size[0]) / EGO_SIZE[0]
                scene.actor_type[0, slot] = int(kind)
                scene.actor_mask[0, slot] = True
                scene.size[0, slot] = size
                scene.wheelbase[0, slot] = EGO_WHEELBASE * scale if vehicle else 1.0
                scene.reference_to_center[0, slot] = EGO_REAR_TO_CENTER * scale if vehicle else 0.0
                state.heading[0, slot] = _wrap(obj.center.heading - yaw0)
            heading = float(state.heading[0, slot])
            heading += _wrap(obj.center.heading - yaw0 - heading)
            offset = float(scene.reference_to_center[0, slot])
            velocity = getattr(obj, "velocity", None)
            if velocity is None:
                vx = vy = 0.0
            else:
                vx = cos0 * velocity.x + sin0 * velocity.y
                vy = -sin0 * velocity.x + cos0 * velocity.y
            # Our reference point is behind the box center; with a zero yaw rate the
            # observation's center velocity is this velocity.
            state.position[0, slot] = torch.tensor(
                [x - offset * math.cos(heading), y - offset * math.sin(heading)]
            )
            state.heading[0, slot] = heading
            state.velocity[0, slot] = torch.tensor([vx, vy])
            state.yaw_rate[0, slot] = 0.0
            scene.reference_size[0, step, slot] = size
            present[slot] = True

        red = scene.rails.signal_red
        red[0, step] = False
        for light in traffic_lights:
            slot = self._signals.get(str(light.lane_connector_id))
            if slot is not None and light.status == TrafficLightStatusType.RED:
                red[0, step, slot] = True

        env._state = state
        env._presence[0, step] = present
        env._goals = env._goal_kernel(env.state, env.scene, env._cache, env.policy.config)

    def compute_planner_trajectory(self, current_input):
        ego_state, observation = current_input.history.current_state
        step = current_input.iteration.index
        if self._env is None:
            self._start(ego_state, observation)
        env = self._env
        if step < self._last_step:
            if int(env.episode.step[0]) != step:
                raise RuntimeError(
                    f"Simulation iteration {step}, mirror at step {int(env.episode.step[0])}"
                )
            self._recrop(ego_state, observation)
            self._sync(ego_state, observation, current_input.traffic_light_data or [], step)
            env.step()
            ideal, steering_rate = _latest.command
            self._command = (float(ideal[0, self._ego]), float(steering_rate[0, self._ego]))
        acceleration, steering_rate = self._command
        dt = TimeDuration.from_s(float(env.scene.dt[0]))
        following = propagate(self._model, ego_state, acceleration, steering_rate, dt)
        return CommandTrajectory([ego_state, following], acceleration, steering_rate)
