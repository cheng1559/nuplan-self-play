"""Configuration for the single-frame Beta policy, its observation, and its action semantics."""

import math
from dataclasses import asdict, dataclass, fields

LEGACY_OPTIONS = {
    "enable_lane_attention": False,
    "num_lane_attention_layers": 1,
    "value_components": 1,
    "route_input": False,
    "route_lane_distance": 4.0,
    "goal_count": 1,
    "goal_reached_input": False,
}


@dataclass(frozen=True)
class SingleFrameConfig:
    # Network (the report's runs: width 256, one agent-attention layer, 4.1 M parameters).
    embed_dim: int = 256
    num_heads: int = 4
    num_agent_attention_layers: int = 1
    # Observe the episode's reward weights (domain randomization), appended to the
    # kinematics; at evaluation they are the defaults.
    reward_conditions: bool = False
    # Observe the previous action (jerk, steering rate) and the acceleration command.
    last_action_input: bool = True
    # Observation capacities and windows, in meters.
    max_agents: int = 96
    min_vehicle_agents: int = 32
    min_pedestrian_agents: int = 32
    max_lanes: int = 256
    agent_filter_radius: float = 200.0
    lane_forward_range: float = 200.0
    lane_backward_range: float = 200.0
    # Fixed MLP input scales. Sinusoidal embeddings use raw meters.
    position_scale: float = 50.0
    velocity_scale: float = 10.0
    size_scale: float = 5.0
    # Action semantics: jerk is integrated from the measured acceleration
    # once per policy tick, and the target is held between ticks.
    min_jerk: float = -8.0
    max_jerk: float = 5.0
    max_steering_rate: float = 0.8
    # One decision per simulation step (10 Hz), like nuPlan's per-iteration planner.
    policy_interval_s: float = 0.1
    max_acceleration: float = 4.0
    max_steering_angle: float = math.pi / 3
    positive_jerk_limit: float = 1.0
    acceleration_time_constant: float = 0.2
    steering_time_constant: float = 0.05
    # Route-sampled goals.
    goal_horizon_s: float = 2.0
    goal_min_speed: float = 5.0

    def __post_init__(self):
        if self.embed_dim <= 0 or self.num_heads <= 0 or self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if self.embed_dim % 4:
            raise ValueError("embed_dim must be divisible by 4 for sinusoidal embeddings")
        if self.num_agent_attention_layers < 1:
            raise ValueError("Attention layer counts must be at least one")
        if self.max_agents < 1 or self.max_lanes < 1:
            raise ValueError("max_agents and max_lanes must be positive")
        if min(self.min_vehicle_agents, self.min_pedestrian_agents) < 0:
            raise ValueError("Reserved agent counts must be nonnegative")
        positive = (
            "agent_filter_radius",
            "lane_forward_range",
            "lane_backward_range",
            "position_scale",
            "velocity_scale",
            "size_scale",
            "max_jerk",
            "max_steering_rate",
            "policy_interval_s",
            "max_acceleration",
            "max_steering_angle",
            "goal_horizon_s",
            "goal_min_speed",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.min_jerk) or self.min_jerk >= 0:
            raise ValueError("min_jerk must be negative and finite")
        for name in ("positive_jerk_limit", "acceleration_time_constant", "steering_time_constant"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "SingleFrameConfig":
        known = {field.name for field in fields(cls)}
        values = dict(values)
        # Options of earlier development checkpoints that the released model does not
        # use; they are dropped when they hold their inactive value.
        for name, inactive in LEGACY_OPTIONS.items():
            if name in values and values[name] == inactive:
                del values[name]
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown SingleFrameConfig fields: {sorted(unknown)}")
        return cls(**values)
