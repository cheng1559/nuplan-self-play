"""Fixed-shape, query-centric single-frame policy observation."""

from typing import NamedTuple

from torch import Tensor

from cs5478_project.datatypes.map import CHUNK_POINTS

# Per-agent features: center x/y, velocity x/y, cos/sin heading, length/width,
# four corners x/y, and one-hot vehicle/pedestrian/static.
AGENT_FEATURES = 19
# Query kinematics: local velocity x/y, length/width, steering, acceleration, yaw rate.
KINEMATIC_FEATURES = 7
# Reward-weight conditions appended to kinematics for domain-randomized policies
# (training/randomization.py), at their defaults: collision, off-road, TTC, comfort,
# goal bonus, goal radius, speed.
REWARD_CONDITIONS = 7
# With last-action input: previous jerk and steering rate (each in [-1, 1]) and the
# acceleration command, appended to kinematics before any reward conditions.
LAST_ACTION_FEATURES = 3
DEFAULT_REWARD_CONDITIONS = (0.2, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0)
# Map chunks: CHUNK_POINTS ordered x/y points flattened, MapElementType one-hot (6),
# speed limit, red light.
LANE_FEATURES = CHUNK_POINTS * 2 + 8


class Observation(NamedTuple):
    """Each row is one query actor in its rear-axle frame, x forward, meters.

    Index 0 of the agent axis is always the query itself and is always valid.
    Padding rows are zero-filled and must be ignored through their masks.
    """

    agents: Tensor  # [Q, K, AGENT_FEATURES]
    agent_mask: Tensor  # [Q, K] bool
    goal: Tensor  # [Q, 2]
    kinematics: Tensor  # [Q, KINEMATIC_FEATURES (+ REWARD_CONDITIONS)]
    lanes: Tensor  # [Q, M, LANE_FEATURES]
    lane_mask: Tensor  # [Q, M] bool
