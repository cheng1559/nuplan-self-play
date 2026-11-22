"""Read immutable replay records without loading a model or using a GPU."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_FIELDS = (
    "position",
    "heading",
    "velocity",
    "acceleration",
    "acceleration_command",
    "steering",
    "yaw_rate",
)
ACTOR_TYPES = ("vehicle", "pedestrian", "static")


@dataclass(frozen=True)
class RecordedCase:
    """One world's unpadded recorded horizon, with original actor slot identities."""

    scenario_id: str
    dt: float
    actor_type: np.ndarray
    actor_mask: np.ndarray
    size: np.ndarray
    reference_to_center: np.ndarray
    state: dict[str, np.ndarray]
    valid: np.ndarray | None = None
    actor_ids: np.ndarray | None = None
    static: dict[str, np.ndarray] | None = None

    @property
    def total_actors(self) -> int:
        return int(self.actor_mask.sum()) + (
            int(self.static["mask"].sum()) if self.static is not None else 0
        )

    def actor_id(self, slot: int) -> str:
        return f"actor-{slot if self.actor_ids is None else self.actor_ids[slot]}"

    @property
    def num_frames(self) -> int:
        return self.state["heading"].shape[0]

    @property
    def slots(self) -> np.ndarray:
        return np.flatnonzero(self.actor_mask)

    def active_slots(self, index: int) -> np.ndarray:
        return (
            self.slots
            if self.valid is None
            else np.flatnonzero(self.valid[index] & self.actor_mask)
        )

    def centers(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        heading = self.state["heading"][start:stop]
        direction = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        return (
            self.state["position"][start:stop] + direction * self.reference_to_center[None, :, None]
        )

    def bounds(self) -> tuple[float, float, float, float]:
        centers = self.centers()
        active = (
            np.broadcast_to(self.actor_mask, centers.shape[:2])
            if self.valid is None
            else self.valid & self.actor_mask[None]
        )
        radius = np.linalg.norm(self.size, axis=-1) / 2
        minimum = (centers - radius[None, :, None])[active].min(axis=0)
        maximum = (centers + radius[None, :, None])[active].max(axis=0)
        if self.static is not None and self.static["mask"].any():
            mask = self.static["mask"]
            radius = np.linalg.norm(self.static["size"][mask], axis=-1) / 2
            position = self.static["position"][mask]
            minimum = np.minimum(minimum, (position - radius[:, None]).min(axis=0))
            maximum = np.maximum(maximum, (position + radius[:, None]).max(axis=0))
        return (*minimum.tolist(), *maximum.tolist())

    def validate(self) -> None:
        """Validate external artifact shapes and finite JSON-compatible values."""
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a nonempty string")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.actor_mask.ndim != 1 or not self.actor_mask.any():
            raise ValueError("actor_mask must be [N] with at least one valid actor")
        n = self.actor_mask.shape[0]
        if self.actor_type.shape != (n,) or not np.isin(self.actor_type, (0, 1, 2)).all():
            raise ValueError("actor_type must contain known physical types in [N]")
        if self.size.shape != (n, 2) or not np.isfinite(self.size).all():
            raise ValueError("size must contain finite [N, 2] values")
        if not (self.size > 0).all():
            raise ValueError("actor dimensions must be positive")
        if (
            self.reference_to_center.shape != (n,)
            or not np.isfinite(self.reference_to_center).all()
        ):
            raise ValueError("reference_to_center must contain finite [N] values")
        heading = self.state["heading"]
        if heading.ndim != 2 or heading.shape[0] < 1:
            raise ValueError("heading must have shape [F, N] with at least one frame")
        f = heading.shape[0]
        if self.actor_ids is not None and self.actor_ids.shape != (n,):
            raise ValueError("actor IDs must match dynamic actor slots")
        if self.static is not None:
            objects = self.static
            s = objects["mask"].shape[0]
            for name, shape in (
                ("ids", (s,)),
                ("position", (s, 2)),
                ("heading", (s,)),
                ("size", (s, 2)),
                ("mask", (s,)),
                ("valid", (f, s)),
            ):
                value = objects[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"static.{name} must contain finite values of shape {shape}")
            if not (objects["size"] > 0).all() or not np.array_equal(
                objects["valid"].any(axis=0), objects["mask"]
            ):
                raise ValueError("invalid static size or observation membership")
        if self.valid is not None and (
            self.valid.shape != (f, n)
            or self.valid.dtype != np.bool_
            or (self.valid & ~self.actor_mask[None]).any()
        ):
            raise ValueError("record validity must match frames and actor membership")
        for name in STATE_FIELDS:
            value = self.state[name]
            expected = (f, n, 2) if name in ("position", "velocity") else (f, n)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"state.{name} must contain finite values of shape {expected}")


def load_replay_json(path: Path) -> list[RecordedCase]:
    """Split replay_debug version 1 batches into their individual world records.

    An unrelated JSON document is not a replay artifact and returns no cases.
    A matching format with a malformed or unsupported payload raises an error.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("format") != "replay_debug":
        return []
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported replay schema version in {path}")
    cases = []
    try:
        for batch in payload["batches"]:
            batch_state = {
                name: np.asarray(batch["state"][name], dtype=np.float64) for name in STATE_FIELDS
            }
            for world, scenario_id in enumerate(batch["scenario_ids"]):
                count = batch["num_frames"][world]
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("num_frames must contain positive integers")
                state = {name: value[:count, world] for name, value in batch_state.items()}
                if state["heading"].shape[0] != count:
                    raise ValueError("recorded states are shorter than num_frames")
                case = RecordedCase(
                    scenario_id=scenario_id,
                    dt=float(batch["dt"][world]),
                    actor_type=np.asarray(batch["actor_type"][world], dtype=np.int64),
                    actor_mask=np.asarray(batch["actor_mask"][world], dtype=bool),
                    valid=np.asarray(
                        batch.get("actor_valid", batch.get("reference_valid"))[world][:count],
                        dtype=bool,
                    )
                    if "actor_valid" in batch or "reference_valid" in batch
                    else None,
                    size=np.asarray(batch["size"][world], dtype=np.float64),
                    reference_to_center=np.asarray(
                        batch["reference_to_center"][world], dtype=np.float64
                    ),
                    state=state,
                    actor_ids=np.asarray(batch["actor_ids"][world], dtype=np.int64)
                    if "actor_ids" in batch
                    else None,
                    static={
                        name: np.asarray(
                            value[world][:count] if name == "valid" else value[world],
                            dtype=np.bool_
                            if name in ("mask", "valid")
                            else np.int64
                            if name == "ids"
                            else np.float32,
                        )
                        for name, value in batch["static"].items()
                    }
                    if "static" in batch
                    else None,
                )
                case.validate()
                cases.append(case)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid replay artifact {path}: {error}") from error
    if not cases:
        raise ValueError(f"Replay artifact contains no scenarios: {path}")
    return cases


def load_scene_record(path: Path) -> RecordedCase:
    """Read a normalized scenario on CPU; import PyTorch only for this file type."""
    from cs5478_project.data.dataset import load_scene

    scene = load_scene(path)
    return RecordedCase(
        scenario_id=scene.scenario_id,
        dt=scene.dt,
        actor_type=scene.actor_type.numpy(),
        actor_mask=scene.actor_mask.numpy(),
        actor_ids=scene.actor_ids.numpy(),
        static={name: value.numpy() for name, value in scene.static._asdict().items()},
        valid=scene.reference_valid.numpy(),
        size=scene.size.numpy(),
        reference_to_center=scene.reference_to_center.numpy(),
        state={
            name: value.numpy()
            for name, value in zip(STATE_FIELDS, scene.reference_state, strict=True)
        },
    )
