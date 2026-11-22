"""Physical vehicle action and required pedestrian oracle inputs."""

from typing import NamedTuple

from torch import Tensor


class VehicleAction(NamedTuple):
    """Unnormalized jerk [m/s^3] and steering rate [rad/s], both [B, A]."""

    jerk: Tensor
    steering_rate: Tensor


class PedestrianOracle(NamedTuple):
    """Next-frame GT speed [m/s] and availability, both [B, A]."""

    preferred_speed: Tensor
    valid: Tensor
