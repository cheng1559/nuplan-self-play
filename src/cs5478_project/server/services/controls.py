"""Resolve preset and per-source actor overrides before device execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from cs5478_project.datatypes.control import ControlType
from cs5478_project.server.schemas import AgentControl, ControlPreset

# The reactive SingleFrameNet preset's background is set after loading (mixed_controls).
EGO_PRESETS = {
    "ego_idm": "idm",
    "ego_single_frame": "single_frame",
    "ego_single_frame_reactive": "single_frame",
}
SELFPLAY_PRESETS = {"single_frame_selfplay": "single_frame"}

if TYPE_CHECKING:
    from torch import Tensor

    from cs5478_project.datatypes.scene import Scene


def resolve_controls(
    scenes: list[Scene],
    source_ids: list[str],
    preset: ControlPreset,
    overrides: list[AgentControl],
    capacity: int,
) -> tuple[Tensor, list[list[str]]]:
    """Resolve identities and controller assignments, preserving fixed capacity."""
    import torch

    from cs5478_project.datatypes.scene import ActorType

    if preset not in get_args(ControlPreset):
        raise ValueError(f"Unknown control preset: {preset}")
    codes = {
        "log": ControlType.LOG,
        "idm": ControlType.IDM,
        "pedestrian": ControlType.PEDESTRIAN,
        "single_frame": ControlType.SINGLE_FRAME,
        "absent": ControlType.ABSENT,
    }
    tensors = torch.zeros((len(scenes), capacity), dtype=torch.int64)
    modes = []
    rails_by_world = []
    identities = {}
    for world, (scene, source_id) in enumerate(zip(scenes, source_ids, strict=True)):
        selected = ["log"] * scene.num_actors
        railed = (
            scene.rails.rail_valid.tolist()
            if scene.rails is not None
            else [False] * scene.num_actors
        )
        rails_by_world.append(railed)
        for slot in scene.actor_mask.nonzero().flatten().tolist():
            identities[(source_id, f"actor-{int(scene.actor_ids[slot])}")] = (world, slot)
            if preset == "idm_pedestrian":
                # Only vehicles nuPlan's builder selected (with rails) run IDM.
                vehicle = "idm" if railed[slot] else "log"
                selected[slot] = (
                    vehicle if scene.actor_type[slot] == ActorType.VEHICLE else "pedestrian"
                )
            if preset in SELFPLAY_PRESETS:
                selected[slot] = (
                    SELFPLAY_PRESETS[preset]
                    if scene.actor_type[slot] == ActorType.VEHICLE
                    else "pedestrian"
                )
        if preset == "nuplan_reactive" and scene.rails is not None:
            # nuPlan R background: IDM agents on rails, open-loop logs, absent the rest.
            roles = ("absent", "idm", "log", "log")
            selected = [roles[int(role)] for role in scene.rails.role]
            selected[scene.ego_index] = "log"
        if preset in EGO_PRESETS and (EGO_PRESETS[preset] != "idm" or railed[scene.ego_index]):
            selected[scene.ego_index] = EGO_PRESETS[preset]
        modes.append(selected)
    seen = set()
    for override in overrides:
        key = (override.source_id, override.actor_id)
        if key in seen:
            raise ValueError("Duplicate source/actor control override")
        seen.add(key)
        if key not in identities:
            raise ValueError(f"Unknown dynamic actor: {override.source_id}/{override.actor_id}")
        world, slot = identities[key]
        if override.control_type == "idm" and not rails_by_world[world][slot]:
            raise ValueError(
                f"{override.source_id}/{override.actor_id} has no exported rail, so it cannot "
                "use IDM; nuPlan's reactive builder did not select it"
            )
        modes[world][slot] = override.control_type
    for world, selected in enumerate(modes):
        tensors[world, : len(selected)] = torch.tensor([codes[mode] for mode in selected])
    return tensors, modes


def resolve_blind(scenes, source_ids, overrides, capacity):
    """Resolve independent perception flags after controller identities are validated."""
    import torch

    flags = {(item.source_id, item.actor_id): item.blind for item in overrides}
    result = torch.zeros((len(scenes), capacity), dtype=torch.bool)
    for world, (scene, source_id) in enumerate(zip(scenes, source_ids, strict=True)):
        for slot in scene.actor_mask.nonzero().flatten().tolist():
            result[world, slot] = flags.get(
                (source_id, f"actor-{int(scene.actor_ids[slot])}"), False
            )
    return result


def resolve_goals(scenes, source_ids, overrides, capacity):
    """Fixed goals of the overrides that set one: a mask [B, A] and scene points [B, A, 2]."""
    import torch

    chosen = {
        (item.source_id, item.actor_id): item.goal for item in overrides if item.goal is not None
    }
    mask = torch.zeros((len(scenes), capacity), dtype=torch.bool)
    points = torch.zeros((len(scenes), capacity, 2))
    for world, (scene, source_id) in enumerate(zip(scenes, source_ids, strict=True)):
        for slot in scene.actor_mask.nonzero().flatten().tolist():
            goal = chosen.get((source_id, f"actor-{int(scene.actor_ids[slot])}"))
            if goal is not None:
                mask[world, slot] = True
                points[world, slot] = torch.tensor(goal)
    return mask, points


def reactive_controls(actor_type, role, actor_mask, ego_index=None, ego_control=None):
    """nuPlan reactive (R) background: IDM agents, open-loop log replay, absent the rest.

    Ego keeps ``ego_control`` (LOG by default), like the planner under test in nuPlan.
    """
    import torch

    from cs5478_project.datatypes.rails import ReactiveRole

    controls = torch.full_like(role, ControlType.LOG)
    controls = torch.where(role == ReactiveRole.IDM, ControlType.IDM, controls)
    controls = torch.where(role == ReactiveRole.ABSENT, ControlType.ABSENT, controls)
    if ego_index is None:
        ego = role == ReactiveRole.EGO
    else:
        ego = torch.arange(role.shape[-1], device=role.device) == ego_index[..., None]
    controls = torch.where(ego, ControlType.LOG if ego_control is None else ego_control, controls)
    return torch.where(actor_mask, controls, ControlType.LOG)
