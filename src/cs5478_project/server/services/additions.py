"""Add stationary actors to a loaded scene before it is batched (visualizer scene edits)."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs5478_project.datatypes.scene import Scene
    from cs5478_project.server.schemas import AddedActor

# nuPlan's ego proportions, as the exporter scales them to other vehicles.
EGO_LENGTH, EGO_WHEELBASE, EGO_REAR_TO_CENTER = 5.176, 3.089, 1.461


def add_actors(scene: Scene, items: list[AddedActor]) -> tuple[Scene, dict[int, int]]:
    """``scene`` plus one stationary actor per item, and each new actor id's serial.

    Vehicles and pedestrians become dynamic slots that hold their pose for the whole
    episode (log control; a policy can drive a vehicle once it is given a goal); static
    objects join the static table. An id is the scene's largest id + 1 + the item's
    serial, so it stays the same when other items are added or removed.
    """
    import torch

    from cs5478_project.datatypes.rails import ACTOR_FIELDS, ReactiveRole, empty_rails
    from cs5478_project.datatypes.route import ROUTE_POINTS, RouteData, RouteSource
    from cs5478_project.datatypes.scene import ActorType
    from cs5478_project.datatypes.state import WorldState
    from cs5478_project.datatypes.static import StaticObjects, empty_static

    if not items:
        return scene, {}
    frames, n = scene.num_frames, scene.num_actors
    actor_ids = scene.actor_ids if scene.actor_ids is not None else torch.arange(n)
    static = scene.static if scene.static is not None else empty_static((0,), frames)
    base = int(torch.cat((actor_ids[scene.actor_mask], static.ids[static.mask])).max()) + 1
    moving = [item for item in items if item.kind != "static"]
    fixed = [item for item in items if item.kind == "static"]
    changes: dict = {}

    def pose(group):
        center = torch.tensor([item.position for item in group], dtype=torch.float32)
        heading = torch.tensor(
            [math.atan2(math.sin(item.heading), math.cos(item.heading)) for item in group],
            dtype=torch.float32,
        )
        size = torch.tensor([(item.length, item.width) for item in group], dtype=torch.float32)
        ids = torch.tensor([base + item.serial for item in group], dtype=torch.int64)
        return center, heading, size, ids

    if moving:
        k = len(moving)
        center, heading, size, ids = pose(moving)
        vehicle = torch.tensor([item.kind == "vehicle" for item in moving])
        scale = size[:, 0] / EGO_LENGTH
        wheelbase = torch.where(vehicle, EGO_WHEELBASE * scale, 1.0)
        offset = torch.where(vehicle, EGO_REAR_TO_CENTER * scale, 0.0)
        # Vehicle states refer to the rear axle.
        direction = torch.stack((heading.cos(), heading.sin()), -1)
        position = center - direction * offset[:, None]
        still = WorldState(
            position,
            heading,
            torch.zeros((k, 2)),
            *(torch.zeros(k) for _ in range(4)),
        )

        def append(old, new, axis=0):
            return torch.cat((old, new.to(old.dtype)), axis)

        changes.update(
            actor_type=append(
                scene.actor_type, torch.where(vehicle, ActorType.VEHICLE, ActorType.PEDESTRIAN)
            ),
            actor_mask=append(scene.actor_mask, torch.ones(k, dtype=torch.bool)),
            size=append(scene.size, size),
            wheelbase=append(scene.wheelbase, wheelbase),
            reference_to_center=append(scene.reference_to_center, offset),
            initial_state=WorldState(
                *(append(old, new) for old, new in zip(scene.initial_state, still, strict=True))
            ),
            reference_state=WorldState(
                *(
                    append(old, new[None].expand(frames, *new.shape), 1)
                    for old, new in zip(scene.reference_state, still, strict=True)
                )
            ),
            reference_valid=append(scene.reference_valid, torch.ones((frames, k)), 1),
            actor_ids=append(actor_ids, ids),
        )
        if scene.reference_size is not None:
            changes["reference_size"] = append(
                scene.reference_size, size[None].expand(frames, k, 2), 1
            )
        if scene.routes is not None:
            # A stationary actor's route is its one point (padding repeats it).
            first = torch.zeros((k, ROUTE_POINTS), dtype=torch.bool)
            first[:, 0] = True
            new = RouteData(
                position[:, None].expand(k, ROUTE_POINTS, 2),
                first,
                torch.full((k,), RouteSource.GT),
                torch.zeros(k),
                torch.full((k,), (frames - 1) * scene.dt),
            )
            changes["routes"] = RouteData(
                *(append(old, value) for old, value in zip(scene.routes, new, strict=True))
            )
        if scene.rails is not None:
            # No rail (never an IDM agent); nuPlan R replays it like an open-loop object.
            blank = empty_rails((), frames, k)._replace(
                rail_pose=torch.cat((center, heading[:, None]), -1),
                rail_size=size,
                role=torch.full((k,), ReactiveRole.OPEN_LOOP),
            )
            changes["rails"] = scene.rails._replace(
                **{
                    name: append(getattr(scene.rails, name), getattr(blank, name))
                    for name in ACTOR_FIELDS
                }
            )
    if fixed:
        k = len(fixed)
        center, heading, size, ids = pose(fixed)
        changes["static"] = StaticObjects(
            ids=torch.cat((static.ids, ids)),
            position=torch.cat((static.position, center)),
            heading=torch.cat((static.heading, heading)),
            size=torch.cat((static.size, size)),
            mask=torch.cat((static.mask, torch.ones(k, dtype=torch.bool))),
            valid=torch.cat((static.valid, torch.ones((frames, k), dtype=torch.bool)), 1),
        )
    return dataclasses.replace(scene, **changes), {
        base + item.serial: item.serial for item in items
    }
