"""Resample a nuPlan log window into a 10 Hz scene with later arrivals and GT routes."""

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from cs5478_project.data.nuplan.source import FrameRow, load_boxes
from cs5478_project.data.preparation import prepare_scene
from cs5478_project.data.routes import build_gt_routes
from cs5478_project.datatypes.scene import ActorType, Scene
from cs5478_project.datatypes.state import WorldState
from cs5478_project.defaults import (
    DEFAULT_DT,
    EPISODE_FRAMES,
    MAX_DYNAMIC_ACTORS,
    MAX_STATIC_OBJECTS,
)

SOURCE_FUTURE_S = 45.0
EGO_SIZE = (5.176, 2.297)
EGO_WHEELBASE = 3.089
EGO_REAR_TO_CENTER = 1.461
PEDESTRIAN_CATEGORIES = ("pedestrian", "bicycle")


def resample(times, values, target):
    """Linear interpolation inside contiguous annotations; hold across gaps, marked invalid."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(target, times, flat[:, i]) for i in range(flat.shape[1])], -1)
    right = np.searchsorted(times, target).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    distance = np.minimum(abs(target - times[left]), abs(target - times[right]))
    valid = (distance <= 0.051) & ((times[right] - times[left] <= 0.076) | (distance < 0.001))
    result = np.where(valid[:, None], result, flat[left])
    return result.reshape((len(target),) + values.shape[1:]).astype(np.float32), valid


def build_tracks(
    conn, frames: list[FrameRow], index: int, scenario_id: str, min_future_s=SOURCE_FUTURE_S
):
    """Return (scene, track_tokens, categories, info) for the window starting at frames[index].

    Tracks first seen within the episode are actor candidates; 45 s of source future
    feeds their GT routes. A log that ends sooner (at least ``min_future_s`` ahead) leaves
    the remaining frames invalid, so routes are shorter. Vehicle geometry scales Pacifica
    parameters by length.
    """
    anchor = frames[index]
    raw = [f for f in frames[index:] if f.timestamp_us <= anchor.timestamp_us + 45_100_000]
    if (raw[-1].timestamp_us - anchor.timestamp_us) / 1e6 < min_future_s:
        raise ValueError(f"Source has less than {min_future_s:.0f} s of future")
    times = np.array([(f.timestamp_us - anchor.timestamp_us) / 1e6 for f in raw])
    boxes = load_boxes(conn, [f.token for f in raw])
    by_track = defaultdict(list)
    for frame, time in zip(raw, times, strict=True):
        for obj in boxes.get(frame.token, []):
            by_track[obj.track_token_hex].append((time, obj))
    episode_s = (EPISODE_FRAMES - 1) * DEFAULT_DT
    selected = sorted(
        (token for token, rows in by_track.items() if rows[0][0] <= episode_s + 0.051),
        key=lambda token: (by_track[token][0][0], token),
    )
    n = len(selected) + 1
    total = round(SOURCE_FUTURE_S / DEFAULT_DT) + 1
    target = np.arange(total) * DEFAULT_DT
    position = np.zeros((total, n, 2), np.float32)
    velocity = position.copy()
    sizes = position.copy()
    heading = np.zeros((total, n), np.float32)
    acceleration, yaw_rate = heading.copy(), heading.copy()
    valid = np.zeros((total, n), bool)
    actor_type = np.full(n, ActorType.STATIC, np.int64)
    actor_type[0] = ActorType.VEHICLE
    wheelbase = np.ones(n, np.float32)
    offset = np.zeros(n, np.float32)
    wheelbase[0], offset[0] = EGO_WHEELBASE, EGO_REAR_TO_CENTER
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    rotation = np.array([[c, -s], [s, c]])

    def fill(slot, t, xy, v, yaw, acc, rate, dimensions):
        position[:, slot], valid[:, slot] = resample(t, xy, target)
        velocity[:, slot], _ = resample(t, v, target)
        heading[:, slot], _ = resample(t, np.unwrap(yaw), target)
        acceleration[:, slot], _ = resample(t, acc, target)
        yaw_rate[:, slot], _ = resample(t, rate, target)
        sizes[:, slot], _ = resample(t, dimensions, target)

    ego_yaw = np.unwrap([f.yaw - anchor.yaw for f in raw])
    ego_xy = (np.array([[f.x, f.y] for f in raw]) - [anchor.x, anchor.y]) @ rotation
    ego_v = np.array(
        [
            [f.vx * math.cos(y) - f.vy * math.sin(y), f.vx * math.sin(y) + f.vy * math.cos(y)]
            for f, y in zip(raw, ego_yaw, strict=True)
        ]
    )
    fill(
        0,
        times,
        ego_xy,
        ego_v,
        ego_yaw,
        np.array([f.ax for f in raw]),
        np.array([f.yaw_rate for f in raw]),
        np.tile(EGO_SIZE, (len(raw), 1)),
    )
    tokens, categories = ["ego"], ["ego"]
    for slot, token in enumerate(selected, 1):
        rows = by_track[token]
        t = np.array([row[0] for row in rows])
        objects = [row[1] for row in rows]
        category = objects[0].category
        actor_type[slot] = (
            ActorType.VEHICLE
            if category == "vehicle"
            else ActorType.PEDESTRIAN
            if category in PEDESTRIAN_CATEGORIES
            else ActorType.STATIC
        )
        length = max(0.1, objects[0].length)
        if actor_type[slot] == ActorType.VEHICLE:
            wheelbase[slot] = EGO_WHEELBASE * length / EGO_SIZE[0]
            offset[slot] = EGO_REAR_TO_CENTER * length / EGO_SIZE[0]
        yaw = np.unwrap([obj.yaw - anchor.yaw for obj in objects])
        rate, acc = np.zeros(len(objects)), np.zeros(len(objects))
        v = np.array([[obj.vx, obj.vy] for obj in objects]) @ rotation
        center = (np.array([[obj.x, obj.y] for obj in objects]) - [anchor.x, anchor.y]) @ rotation
        splits = np.r_[0, np.flatnonzero(np.diff(t) > 0.076) + 1, len(t)]
        for start, stop in zip(splits[:-1], splits[1:], strict=True):
            if stop - start >= 2:
                rate[start:stop] = np.gradient(yaw[start:stop], t[start:stop])
                dv = np.gradient(v[start:stop], t[start:stop], axis=0)
                acc[start:stop] = dv[:, 0] * np.cos(yaw[start:stop]) + dv[:, 1] * np.sin(
                    yaw[start:stop]
                )
        direction = np.stack((np.cos(yaw), np.sin(yaw)), -1)
        lateral = np.stack((-np.sin(yaw), np.cos(yaw)), -1)
        xy = center - offset[slot] * direction
        v = v - (rate * offset[slot])[:, None] * lateral
        dims = np.array([[max(0.1, o.length), max(0.1, o.width)] for o in objects])
        fill(slot, t, xy, v, yaw, acc, rate, dims)
        tokens.append(token)
        categories.append(category)
    speed = (velocity * np.stack((np.cos(heading), np.sin(heading)), -1)).sum(-1)
    moving = abs(speed) > 0.5
    steering = np.where(
        moving, np.arctan(yaw_rate * wheelbase[None] / np.where(moving, speed, 1)), 0
    ).astype(np.float32)
    # nuPlan builds every logged ego state with a zero tire steering angle; keep its yaw
    # rate and (vx, vy) velocity, so the bicycle starts exactly as nuPlan's does.
    steering[:, 0] = 0.0
    f = EPISODE_FRAMES
    state = WorldState(
        *(
            torch.tensor(value[:f], dtype=torch.float32)
            for value in (
                position,
                heading,
                velocity,
                acceleration,
                acceleration,
                steering,
                yaw_rate,
            )
        )
    )
    scene = Scene(
        scenario_id=scenario_id,
        dt=DEFAULT_DT,
        actor_type=torch.tensor(actor_type),
        actor_mask=torch.tensor(valid[:f].any(0)),
        size=torch.tensor(sizes[valid.argmax(0), np.arange(n)]),
        wheelbase=torch.tensor(wheelbase),
        reference_to_center=torch.tensor(offset),
        initial_state=WorldState(*(value[0].clone() for value in state)),
        reference_state=state,
        reference_valid=torch.tensor(valid[:f]),
        actor_ids=torch.arange(n),
        reference_size=torch.tensor(sizes[:f]),
    )
    # Tracks first seen just after the episode are not eligible actor slots.
    keep = scene.actor_mask.nonzero().flatten()
    if len(keep) != n:
        scene = replace(
            scene,
            actor_type=scene.actor_type[keep],
            actor_mask=scene.actor_mask[keep],
            size=scene.size[keep],
            wheelbase=scene.wheelbase[keep],
            reference_to_center=scene.reference_to_center[keep],
            initial_state=WorldState(*(value[keep] for value in scene.initial_state)),
            reference_state=WorldState(*(value[:, keep] for value in scene.reference_state)),
            reference_valid=scene.reference_valid[:, keep],
            actor_ids=scene.actor_ids[keep],
            reference_size=scene.reference_size[:, keep],
        )
        position, valid = position[:, keep], valid[:, keep]
    scene = replace(
        scene,
        routes=build_gt_routes(
            scene, future_positions=torch.tensor(position), future_valid=torch.tensor(valid)
        ),
    )
    scene = prepare_scene(
        scene, max_dynamic_actors=MAX_DYNAMIC_ACTORS, max_static_objects=MAX_STATIC_OBJECTS
    )
    slot_tokens = [tokens[int(identity)] for identity in scene.actor_ids]
    slot_categories = [categories[int(identity)] for identity in scene.actor_ids]
    info = {
        "source_tracks": n,
        "dynamic": scene.num_actors,
        "static": len(scene.static.ids),
        "future_arrivals": int((scene.actor_mask & ~scene.reference_valid[0]).sum()),
    }
    return scene, slot_tokens, slot_categories, info
