"""Separate static geometry and select bounded actor sets on the CPU."""

from dataclasses import replace

import torch

from cs5478_project.datatypes.rails import ACTOR_FIELDS
from cs5478_project.datatypes.scene import ActorType, Scene
from cs5478_project.datatypes.state import WorldState
from cs5478_project.datatypes.static import StaticObjects, empty_static


def prepare_scene(
    scene: Scene,
    *,
    max_dynamic_actors: int | None = None,
    max_static_objects: int | None = None,
) -> Scene:
    """Freeze static objects at first observation and optionally select capacities.

    All episode tracks, including future arrivals, are candidates. Dynamic
    overflow keeps ego, then vehicles, then pedestrians ordered by minimum
    simultaneous distance to ego. Static overflow uses the minimum distance
    to any retained vehicle while both tracks are observed. Ties use source IDs.
    No category is removed merely because its velocity is zero.
    """
    scene.validate()
    for limit in (max_dynamic_actors, max_static_objects):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("actor capacities must be positive integers")
    ids = scene.actor_ids if scene.actor_ids is not None else torch.arange(scene.num_actors)
    dynamic = (scene.actor_mask & (scene.actor_type != ActorType.STATIC)).nonzero().flatten()
    static_slots = (scene.actor_mask & (scene.actor_type == ActorType.STATIC)).nonzero().flatten()
    objects = scene.static or empty_static((0,), scene.num_frames)
    if len(static_slots):
        first = scene.reference_valid[:, static_slots].to(torch.int64).argmax(dim=0)
        additions = StaticObjects(
            ids[static_slots],
            scene.reference_state.position[first, static_slots],
            scene.reference_state.heading[first, static_slots],
            scene.size[static_slots],
            torch.ones(len(static_slots), dtype=torch.bool),
            scene.reference_valid[:, static_slots],
        )
        objects = StaticObjects(
            *(
                torch.cat((old, new), dim=1 if name == "valid" else 0)
                for name, old, new in zip(StaticObjects._fields, objects, additions, strict=True)
            )
        )
    if max_dynamic_actors is not None and len(dynamic) > max_dynamic_actors:
        heading = scene.reference_state.heading
        centers = scene.reference_state.position + scene.reference_to_center[
            None, :, None
        ] * torch.stack((heading.cos(), heading.sin()), dim=-1)
        distances = torch.linalg.vector_norm(centers - centers[:, scene.ego_index, None], dim=-1)
        distances.masked_fill_(~scene.reference_valid, float("inf"))
        distances = distances.amin(dim=0)
        ranked = sorted(
            dynamic.tolist(),
            key=lambda slot: (
                slot != scene.ego_index,
                int(scene.actor_type[slot]) != ActorType.VEHICLE,
                float(distances[slot]),
                int(ids[slot]),
            ),
        )
        dynamic = torch.tensor(sorted(ranked[:max_dynamic_actors]), dtype=torch.int64)
    static_indices = objects.mask.nonzero().flatten()
    if max_static_objects is not None and len(static_indices) > max_static_objects:
        vehicles = dynamic[scene.actor_type[dynamic] == ActorType.VEHICLE]
        heading = scene.reference_state.heading[:, vehicles]
        positions = scene.reference_state.position[:, vehicles] + scene.reference_to_center[
            vehicles
        ][None, :, None] * torch.stack((heading.cos(), heading.sin()), dim=-1)
        distances = torch.cdist(
            objects.position[static_indices][None].expand(scene.num_frames, -1, -1), positions
        )
        joint = objects.valid[:, static_indices, None] & scene.reference_valid[:, None, vehicles]
        distances.masked_fill_(~joint, float("inf"))
        scores = distances.amin(dim=(0, 2))
        ranked = sorted(
            range(len(static_indices)),
            key=lambda index: (float(scores[index]), int(objects.ids[static_indices[index]])),
        )
        static_indices = static_indices[torch.tensor(sorted(ranked[:max_static_objects]))]
    objects = StaticObjects(
        *(
            value[:, static_indices] if name == "valid" else value[static_indices]
            for name, value in zip(StaticObjects._fields, objects, strict=True)
        )
    )
    ego_index = int((dynamic == scene.ego_index).nonzero().item())
    prepared = replace(
        scene,
        actor_ids=ids[dynamic],
        actor_type=scene.actor_type[dynamic],
        actor_mask=scene.actor_mask[dynamic],
        size=scene.size[dynamic],
        wheelbase=scene.wheelbase[dynamic],
        reference_to_center=scene.reference_to_center[dynamic],
        initial_state=WorldState(*(value[dynamic] for value in scene.initial_state)),
        reference_state=WorldState(*(value[:, dynamic] for value in scene.reference_state)),
        reference_valid=scene.reference_valid[:, dynamic],
        reference_size=scene.reference_size[:, dynamic]
        if scene.reference_size is not None
        else None,
        routes=type(scene.routes)(*(value[dynamic] for value in scene.routes))
        if scene.routes is not None
        else None,
        rails=scene.rails._replace(
            **{name: getattr(scene.rails, name)[dynamic] for name in ACTOR_FIELDS}
        )
        if scene.rails is not None
        else None,
        ego_index=ego_index,
        static=objects,
    )
    prepared.validate()
    return prepared
