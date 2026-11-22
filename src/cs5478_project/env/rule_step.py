"""Synchronous fixed-shape vehicle/pedestrian transitions with logged spawn events."""

import torch
from torch import Tensor

from cs5478_project.agents.config import IDMConfig, InteractionConfig, PedestrianConfig
from cs5478_project.agents.idm import RailPose, RailState, reactive_idm_step
from cs5478_project.agents.neighbors import query_neighbors
from cs5478_project.agents.pedestrian import RuleBasedPedestrianAgent
from cs5478_project.agents.replay import LogAgent
from cs5478_project.agents.route import RouteCache, project_route
from cs5478_project.datatypes.action import PedestrianOracle, VehicleAction
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.scene import SceneBatch
from cs5478_project.datatypes.state import EpisodeState, WorldState, select_state
from cs5478_project.datatypes.transition import StepResult
from cs5478_project.dynamics.action_adapter import step_jerk_bicycle
from cs5478_project.dynamics.bicycle import (
    BicycleConfig,
    KinematicBicycleModel,
    heading_vector,
    wrap_angle,
)


def rail_world_state(scene, pose: RailPose, previous: WorldState, previous_speed, dt) -> WorldState:
    """Express rail box-center poses as rear-axle WorldState, with finite-difference rates."""
    direction = heading_vector(pose.heading)
    yaw_rate = wrap_angle(pose.heading - previous.heading) / dt
    acceleration = (pose.velocity - previous_speed) / dt
    moving = pose.velocity > 0.5
    steering = torch.where(
        moving,
        torch.atan(yaw_rate * scene.wheelbase / torch.where(moving, pose.velocity, 1.0)),
        0.0,
    )
    return WorldState(
        pose.center - scene.reference_to_center[..., None] * direction,
        pose.heading,
        pose.velocity[..., None] * direction,
        acceleration,
        acceleration,
        steering,
        yaw_rate,
    )


def rule_step_kernel(
    state: WorldState,
    episode: EpisodeState,
    scene: SceneBatch,
    cache: RouteCache,
    previous_progress: Tensor,
    rail_state: RailState,
    control_types: Tensor,
    idm: IDMConfig,
    pedestrian: PedestrianConfig,
    interaction: InteractionConfig,
    bicycle: BicycleConfig,
    external_action: VehicleAction | None = None,
    external_config=None,
    external_policy_forward: Tensor | None = None,
    actor_presence: Tensor | None = None,
    blind: Tensor | None = None,
    *,
    reactive: bool = True,
    pedestrians: bool = True,
) -> tuple[StepResult, Tensor, RailState]:
    """Compute all reactions from the same current world, then commit together.

    Logs supply first-arrival states. Controlled actors persist across missing
    annotations; IDM actors without exported rails and other controlled actors
    without routes explicitly fall back to log playback. ``reactive`` and
    ``pedestrians`` are compile-time switches: the environment turns them off when no
    actor uses IDM or pedestrian control, which skips that work entirely.
    """
    world = torch.arange(episode.step.shape[0], device=episode.step.device)
    advanced = ~episode.done
    next_step = torch.minimum(episode.step + advanced.to(torch.int64), scene.num_frames - 1)
    presence = scene.reference_valid if actor_presence is None else actor_presence
    current_valid = presence[world, episode.step] & ~rail_state.removed
    current_static = scene.static.valid[world, episode.step]
    dt = scene.dt[:, None]
    is_idm = control_types == ControlType.IDM
    has_input = torch.where(is_idm, scene.rails.rail_valid, scene.routes.mask.any(dim=-1))
    rule_mask = (
        (control_types != ControlType.LOG)
        & (control_types != ControlType.ABSENT)
        & current_valid
        & presence[world, next_step]
        & has_input
        & advanced[:, None]
    )
    learned_mask = control_types == ControlType.SINGLE_FRAME
    projection = project_route(
        cache,
        state.position,
        minimum=(previous_progress - 1.0).clamp_min(0),
        maximum=previous_progress + interaction.route_search_distance,
    )
    progress = torch.maximum(previous_progress, projection.progress)
    if reactive:
        pose, next_rails = reactive_idm_step(
            rail_state,
            state,
            scene,
            next_step,
            control_types,
            current_valid,
            advanced,
            scene.dt,
            idm,
            blind=blind,
        )
        vehicle_state = rail_world_state(scene, pose, state, rail_state.velocity, dt)
    else:  # no actor uses IDM, so this state is never selected
        next_rails, vehicle_state = rail_state, state
    if pedestrians:
        pedestrian_queries = rule_mask & (control_types == ControlType.PEDESTRIAN)
        neighbors = query_neighbors(
            state,
            scene,
            current_valid,
            current_static,
            pedestrian_queries if blind is None else pedestrian_queries & ~blind,
            interaction,
            vehicle_only_queries=control_types == ControlType.PEDESTRIAN,
        )
        oracle = PedestrianOracle(
            torch.linalg.vector_norm(scene.reference_state.velocity[world, next_step], dim=-1),
            scene.reference_valid[world, next_step],
        )
        pedestrian_state = RuleBasedPedestrianAgent.step(
            state,
            cache,
            progress,
            neighbors,
            scene.size,
            dt,
            oracle,
            pedestrian,
            reference_to_center=scene.reference_to_center,
        )
    else:  # no actor uses pedestrian control, so this state is never selected
        pedestrian_state = state
    simulated = select_state(is_idm, vehicle_state, pedestrian_state)
    if external_action is not None:
        learned = (
            step_jerk_bicycle(
                state,
                external_action,
                scene.wheelbase,
                dt,
                external_config,
                external_policy_forward,
            )
            if external_config is not None
            else KinematicBicycleModel.step(state, external_action, scene.wheelbase, dt, bicycle)
        )
        simulated = select_state(learned_mask, learned, simulated)
    recorded = LogAgent.get_state(scene.reference_state, next_step)
    # Radius-removed IDM agents leave the world at the next frame.
    rule_mask = rule_mask & ~next_rails.removed
    candidate = select_state(rule_mask, simulated, recorded)
    next_state = select_state(advanced[:, None] & scene.actor_mask, candidate, state)
    next_projection = project_route(
        cache,
        next_state.position,
        minimum=(progress - 1.0).clamp_min(0),
        maximum=progress + interaction.route_search_distance,
    )
    # Spawned/logged actors need a global projection, rather than stale progress.
    logged_projection = project_route(cache, recorded.position)
    next_progress = torch.where(
        rule_mask,
        torch.maximum(progress, next_projection.progress),
        logged_projection.progress,
    )
    next_progress = torch.where(advanced[:, None], next_progress, previous_progress)
    done = episode.done | (next_step >= scene.num_frames - 1)
    return (
        StepResult(
            next_state,
            EpisodeState(next_step, done),
            advanced,
            done & ~episode.done,
            presence[world, next_step] & ~next_rails.removed,
            scene.static.valid[world, next_step],
        ),
        next_progress,
        next_rails,
    )
