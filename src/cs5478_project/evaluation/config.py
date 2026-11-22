"""Thresholds for nuPlan-style metrics and the per-frame training reward."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    # Candidates per evaluated actor for collision and TTC checks (nearest box centers).
    neighbors: int = 16
    neighbor_radius: float = 60.0

    # no_ego_at_fault_collisions and time_to_collision_within_bound (nuPlan yaml values).
    stopped_speed: float = 5e-2
    ttc_stopped_speed: float = 5e-3
    ahead_angle_deg: float = 30.0
    behind_angle_deg: float = 150.0
    ttc_step: float = 0.1
    ttc_horizon: float = 3.0
    least_min_ttc: float = 0.95
    # nuPlan treats lateral collisions as at fault, and widens TTC to lateral tracks, when
    # ego's corners are not in common or connected lanes. We call a query "straddling"
    # when a corner is outside every lane, or its center is this far from the nearest
    # lane centerline (about where a corner of a 2.3 m wide car enters the next lane).
    lane_straddle_offset: float = 1.0

    # drivable_area_compliance.
    drivable_violation: float = 0.3

    # speed_limit_compliance.
    max_overspeed: float = 2.23

    # ego_is_comfortable bounds (nuPlan yaml values).
    min_lon_accel: float = -4.05
    max_lon_accel: float = 2.40
    max_abs_lon_jerk: float = 4.13
    max_abs_mag_jerk: float = 8.37
    max_abs_yaw_rate: float = 0.95
    max_abs_yaw_accel: float = 1.93

    # ego_progress_along_expert_route and ego_is_making_progress.
    progress_threshold: float = 2.0
    min_progress_ratio: float = 0.2

    # Closed-loop score: multipliers times the weighted mean of these four.
    weight_ttc: float = 5.0
    weight_progress: float = 5.0
    weight_speed: float = 4.0
    weight_comfort: float = 2.0

    # Reward: hard + (1 - done) * (goal + prod(soft) / steps), see docs and the report.
    # Any new overlap while ego moves faster than stopped_speed, regardless of fault.
    collision_reward: float = -1.0
    # The box edges or corner paths touch a road boundary (the drivable-area and car-park
    # contour), scaled by 1 + gain * tanh(2 v / speed_norm).
    offroad_reward: float = -1.0
    offroad_speed_gain: float = 5.0
    offroad_speed_norm: float = 40.0
    # Overspeed score: 1 - overspeed / tolerance.
    overspeed_tolerance: float = 1.5
    # Centerline score: 1 - (1 - centerline_score) * (share of lateral room used), where
    # a lane boundary line lies within centerline_band of the box center.
    centerline_score: float = 0.75
    centerline_band: float = 4.0
    # Driving direction: backward progress along the nearest centerline (within the band)
    # over the last wrong_way_horizon seconds. Past wrong_way_threshold the soft score is
    # wrong_way_score; past wrong_way_severe the episode ends with wrong_way_reward.
    wrong_way_horizon: float = 1.0
    wrong_way_threshold: float = 2.0
    wrong_way_severe: float = 6.0
    wrong_way_score: float = 0.0
    wrong_way_reward: float = -1.0
    goal_reward: float = 0.5
    goal_threshold: float = 1.5
    # Soft score when a comfort bound fails, and the factor applied to every later frame
    # once any bound has failed.
    comfort_violation_score: float = 0.0
    comfort_repeat_factor: float = 0.75
    ttc_repeat_factor: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
