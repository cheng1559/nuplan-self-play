"""Host-validated parameters shared by eager and compiled rule policies."""

import math
from dataclasses import dataclass, fields


def _positive_parameters(config) -> None:
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field.name} must be a finite positive number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field.name} must be a finite positive number")


@dataclass(frozen=True)
class IDMConfig:
    """nuPlan reactive agents (idm_agents_observation.yaml) plus fixed GPU capacities."""

    target_velocity: float = 10.0
    min_gap: float = 1.0
    headway_time: float = 1.5
    accel_max: float = 1.0
    decel_max: float = 2.0
    minimum_path_length: float = 20.0
    radius: float = 100.0
    # Edges kept in each agent's route window; nuPlan's stop-line memory is its last five.
    route_edges: int = 8
    route_memory: int = 5
    # plan_route is a while loop; this bounds lane extensions per step.
    extensions_per_step: int = 3
    # Path-to-go and projected-footprint polylines, and exact lead candidates per agent.
    corridor_samples: int = 16
    footprint_samples: int = 4
    candidates: int = 16
    # Red stop lines checked per agent (of route_memory edges x STOP_LINES slots).
    red_stop_lines: int = 4
    # IDM agents per world that run the rail geometry (nuPlan scenes have at most ~40).
    max_agents: int = 64
    # nuPlan agents take the straightest exit; an IDM ego prefers its mission route first.
    ego_follows_route: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "ego_follows_route":
                if not isinstance(value, bool):
                    raise ValueError("ego_follows_route must be a bool")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a finite positive number")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be a finite positive number")
        for name in (
            "route_edges",
            "route_memory",
            "extensions_per_step",
            "corridor_samples",
            "footprint_samples",
            "candidates",
            "max_agents",
        ):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.route_memory > self.route_edges:
            raise ValueError("route_memory cannot exceed route_edges")


@dataclass(frozen=True)
class PedestrianConfig:
    """Route following with bounded speed changes and predictive yielding."""

    desired_speed: float = 1.4
    max_speed: float = 4.0
    accel_max: float = 1.5
    decel_max: float = 3.0
    prediction_horizon: float = 2.0
    clearance: float = 0.3
    slowing_distance: float = 1.5

    def __post_init__(self) -> None:
        _positive_parameters(self)
        if self.desired_speed > self.max_speed:
            raise ValueError("desired_speed must not exceed max_speed")


@dataclass(frozen=True)
class InteractionConfig:
    """Bound expensive route queries after a dense center-distance broad phase."""

    neighbors: int = 16
    radius: float = 50.0
    route_search_distance: float = 8.0

    def __post_init__(self) -> None:
        _positive_parameters(self)
        if not isinstance(self.neighbors, int):
            raise ValueError("neighbors must be an integer")
