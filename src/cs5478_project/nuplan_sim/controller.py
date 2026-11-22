"""One-stage ego controller: the planner's command drives nuPlan's bicycle directly."""

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
    KinematicBicycleModel,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class CommandTrajectory(InterpolatedTrajectory):
    """The current and predicted next ego state, with the command that links them."""

    def __init__(self, states, acceleration: float, steering_rate: float):
        super().__init__(states)
        self.acceleration = acceleration  # ideal rear-axle acceleration [m/s^2]
        self.steering_rate = steering_rate  # tire steering rate [rad/s]


def propagate(model, ego_state, acceleration, steering_rate, sampling_time):
    """nuPlan's bicycle step from an ideal acceleration and steering rate."""
    ideal = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=ego_state.car_footprint.rear_axle_to_center_dist,
        rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
        tire_steering_rate=steering_rate,
    )
    return model.propagate_state(ego_state, ideal, sampling_time)


class OneStageController(AbstractEgoController):
    """No tracker: apply a ``CommandTrajectory``'s command with ``KinematicBicycleModel``
    (the motion model of nuPlan's two-stage controller, with its default constants)."""

    def __init__(self, scenario):
        self._scenario = scenario
        self._model = KinematicBicycleModel(
            scenario.initial_ego_state.car_footprint.vehicle_parameters
        )
        self._state = None

    def reset(self) -> None:
        self._state = None

    def get_state(self):
        if self._state is None:
            self._state = self._scenario.initial_ego_state
        return self._state

    def update_state(self, current_iteration, next_iteration, ego_state, trajectory) -> None:
        if not isinstance(trajectory, CommandTrajectory):
            raise TypeError("OneStageController needs a CommandTrajectory")
        self._state = propagate(
            self._model,
            ego_state,
            trajectory.acceleration,
            trajectory.steering_rate,
            next_iteration.time_point - current_iteration.time_point,
        )
