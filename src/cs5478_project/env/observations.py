"""Single-frame, query-centric policy observations built from the current world state only."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from cs5478_project.agents.geometry import box_corners, rotate
from cs5478_project.agents.signals import chunk_red
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.map import CHUNK_POINTS, MapElementType
from cs5478_project.datatypes.observation import DEFAULT_REWARD_CONDITIONS, Observation
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.dynamics.bicycle import heading_vector, wrap_angle


class ObservationSelection(NamedTuple):
    """What each query kept: slots over [dynamic actors, static objects] and map segments."""

    agent_index: Tensor  # [Q, K] slot in the concatenated dynamic + static axis
    agent_mask: Tensor  # [Q, K]
    lane_index: Tensor  # [Q, M] map chunk index
    lane_mask: Tensor  # [Q, M]


def select_neighbors(distance, types, candidate, self_index, count, vehicles, pedestrians):
    """Return [Q, count] slots: self first, then reserved nearest vehicles and pedestrians,
    then the nearest remaining dynamic actors, with static objects last.

    Columns after the first may point at invalid slots when too few candidates exist;
    the caller masks them with ``candidate``.
    """
    count = min(count, distance.shape[1])
    if count <= 1:
        return self_index[:, None]
    distance = distance.masked_fill(~candidate, torch.inf)
    order = distance.argsort(-1)
    vehicle = (candidate & (types == ActorType.VEHICLE)).gather(1, order)
    pedestrian = (candidate & (types == ActorType.PEDESTRIAN)).gather(1, order)
    reserve_v = torch.zeros_like(vehicle).scatter(
        1, order, vehicle & (vehicle.cumsum(1) <= vehicles)
    )
    reserve_p = torch.zeros_like(pedestrian).scatter(
        1, order, pedestrian & (pedestrian.cumsum(1) <= pedestrians)
    )
    static = candidate & (types == ActorType.STATIC)
    finite_max = torch.where(distance.isfinite(), distance, 0).amax(1, keepdim=True)
    score = distance + static * (finite_max + 1)
    score = torch.where(reserve_v, -2.0, score)
    score = torch.where(reserve_p, -1.0, score)
    others = score.topk(count - 1, largest=False).indices
    return torch.cat((self_index[:, None], others), 1)


# Chunk points used to rank chunks: both ends and the middle, at most 10 m apart.
PROBE_POINTS = (0, CHUNK_POINTS // 2, CHUNK_POINTS - 1)


def select_chunks(points, mask, count, config):
    """Select [Q, count] chunks by forward-weighted distance.

    ``points`` is [Q, C, P, 2] probe points in each query's frame. A chunk's distance is
    its nearest probe inside the window with x halved; chunks reaching ahead count half.
    """
    x, y = points.unbind(-1)
    window = (
        (x >= -config.lane_backward_range)
        & (x <= config.lane_forward_range)
        & (y.abs() <= config.agent_filter_radius)
    )
    distance = torch.stack((x * 0.5, y), -1).norm(dim=-1).masked_fill(~window, torch.inf).amin(-1)
    valid = mask & distance.isfinite()
    score = torch.where((window & (x > 0)).any(-1), distance * 0.5, distance).masked_fill(
        ~valid, torch.inf
    )
    index = score.topk(min(count, points.shape[1]), largest=False).indices
    return index, valid.gather(1, index)


def build_observation(
    state,
    scene,
    step,
    actor_mask,
    control_types,
    goals,
    query_world,
    query_slot,
    config,
    blind=None,
    return_selection: bool = False,
    conditions=None,
    last_action=None,
):
    """Observe the current frame from each query actor's rear axle; no history or GT future.

    ``actor_mask`` is the runtime [B, A] membership at ``step``. ``goals`` is
    [B, A, G, 2] and the last goal is used. Blind queries see only themselves and the map.
    With ``config.reward_conditions``, ``conditions`` [Q, REWARD_CONDITIONS] (the
    episode's reward weights; defaults when None) are appended to the kinematics; with
    ``config.last_action_input``, so are the previous action (``last_action``, a
    VehicleAction of [B, A] tensors; zero when None) and the acceleration command.
    """
    world = torch.arange(step.shape[0], device=step.device)
    static = scene.static

    # Box centers and center velocities of all dynamic actors, then static objects.
    direction = heading_vector(state.heading)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    offset = scene.reference_to_center
    center = state.position + offset[..., None] * direction
    center_velocity = state.velocity + (state.yaw_rate * offset)[..., None] * lateral
    # Logged actors use their per-frame annotated size; simulated actors keep a fixed size.
    sizes = torch.where(
        (control_types == ControlType.LOG)[..., None], scene.reference_size[world, step], scene.size
    )
    position = torch.cat((center, static.position), 1)
    velocity = torch.cat((center_velocity, torch.zeros_like(static.position)), 1)
    heading = torch.cat((state.heading, static.heading), 1)
    size = torch.cat((sizes, static.size), 1)
    valid = torch.cat((actor_mask, static.valid[world, step] & static.mask), 1)
    types = torch.cat(
        (scene.actor_type, torch.full_like(static.heading, ActorType.STATIC, dtype=torch.int64)),
        1,
    )
    slots = position.shape[1]

    q_position = state.position[query_world, query_slot]
    q_heading = state.heading[query_world, query_slot]
    local = rotate(position[query_world] - q_position[:, None], q_heading[:, None])
    distance = local.norm(dim=-1)
    is_self = torch.arange(slots, device=step.device)[None] == query_slot[:, None]
    candidate = valid[query_world] & (distance <= config.agent_filter_radius) & ~is_self
    if blind is not None:
        candidate = candidate & ~blind[query_world, query_slot, None]
    index = select_neighbors(
        distance,
        types[query_world],
        candidate,
        query_slot,
        config.max_agents,
        config.min_vehicle_agents,
        config.min_pedestrian_agents,
    )
    agent_mask = torch.cat(
        (torch.ones_like(index[:, :1], dtype=torch.bool), candidate.gather(1, index[:, 1:])), 1
    )
    flat = query_world[:, None] * slots + index

    def take(value):
        return value.flatten(0, 1)[flat]

    agent_position = local.gather(1, index[..., None].expand(-1, -1, 2))
    agent_velocity = rotate(take(velocity), q_heading[:, None])
    agent_heading = wrap_angle(take(heading) - q_heading[:, None])
    agent_size = take(size)
    corners = box_corners(agent_position, agent_heading, agent_size).flatten(-2)
    agents = (
        torch.cat(
            (
                agent_position,
                agent_velocity,
                agent_heading.cos()[..., None],
                agent_heading.sin()[..., None],
                agent_size,
                corners,
                F.one_hot(take(types), len(ActorType)).to(agent_position.dtype),
            ),
            -1,
        )
        * agent_mask[..., None]
    )

    kinematics = torch.cat(
        (
            rotate(state.velocity[query_world, query_slot], q_heading),
            sizes[query_world, query_slot],
            state.steering[query_world, query_slot, None],
            state.acceleration[query_world, query_slot, None],
            state.yaw_rate[query_world, query_slot, None],
        ),
        -1,
    )
    if getattr(config, "last_action_input", False):
        if last_action is None:
            jerk = steering_rate = torch.zeros_like(kinematics[:, 0])
        else:
            jerk = last_action.jerk[query_world, query_slot]
            steering_rate = last_action.steering_rate[query_world, query_slot]
        span = config.max_jerk - config.min_jerk
        previous = torch.stack(
            (
                2 * (jerk - config.min_jerk) / span - 1,
                steering_rate / config.max_steering_rate,
                state.acceleration_command[query_world, query_slot],
            ),
            -1,
        )
        kinematics = torch.cat((kinematics, previous.to(kinematics.dtype)), -1)
    if getattr(config, "reward_conditions", False):
        if conditions is None:
            conditions = kinematics.new_tensor(DEFAULT_REWARD_CONDITIONS).expand(
                kinematics.shape[0], -1
            )
        kinematics = torch.cat((kinematics, conditions.to(kinematics.dtype)), -1)
    goal = rotate(goals[query_world, query_slot] - q_position[:, None], q_heading[:, None])
    goal = goal[:, -1]  # [Q, 2]

    # Rank every chunk by a few probe points, then transform only the kept chunks.
    probes = scene.map.chunk_points[:, :, list(PROBE_POINTS)][query_world]  # [Q, C, 3, 2]
    probes = rotate(probes - q_position[:, None, None], q_heading[:, None, None])
    lane_index, lane_mask = select_chunks(
        probes, scene.map.chunk_mask[query_world], config.max_lanes, config
    )
    points = rotate(
        scene.map.chunk_points[query_world[:, None], lane_index] - q_position[:, None, None],
        q_heading[:, None, None],
    )  # [Q, M, P, 2]
    red = chunk_red(scene, step)
    chunk_type = scene.map.chunk_type[query_world].gather(1, lane_index)
    features = [
        points.flatten(-2),
        F.one_hot(chunk_type, len(MapElementType)).to(points.dtype),
        scene.rails.speed_limit[query_world, None, None].expand(-1, lane_index.shape[1], 1),
        red[query_world].gather(1, lane_index)[..., None].to(points.dtype),
    ]
    lanes = torch.cat(features, -1) * lane_mask[..., None]
    observation = Observation(agents, agent_mask, goal, kinematics, lanes, lane_mask)
    if return_selection:
        return observation, ObservationSelection(index, agent_mask, lane_index, lane_mask)
    return observation
