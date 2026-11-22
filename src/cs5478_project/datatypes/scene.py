"""Validated scene input and the tensor-only fixed-capacity batch contract."""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.datatypes.map import MapBatch, MapData, validate_map_batch
from cs5478_project.datatypes.rails import RailData, validate_rails
from cs5478_project.datatypes.route import RouteData, validate_routes
from cs5478_project.datatypes.state import WorldState
from cs5478_project.datatypes.static import StaticObjects, validate_static


class ActorType(IntEnum):
    """Physical actor categories, independent of their future control mode."""

    VEHICLE = 0
    PEDESTRIAN = 1
    STATIC = 2


@dataclass(frozen=True)
class Scene:
    """One CPU scenario sampled at a constant dt, including initial frame zero.

    Dynamic slots cover selected tracks appearing during the episode. Static
    objects store geometry once and retain only temporal visibility. Reference
    states use [F, N, ...] and initial states use [N, ...]. Source identifiers
    remain outside the compiled simulation batch.
    """

    scenario_id: str
    dt: float
    actor_type: Tensor
    actor_mask: Tensor
    size: Tensor
    wheelbase: Tensor
    reference_to_center: Tensor
    initial_state: WorldState
    reference_state: WorldState
    reference_valid: Tensor
    map: MapData | None = None
    ego_index: int = 0
    routes: RouteData | None = None
    actor_ids: Tensor | None = None
    static: StaticObjects | None = None
    reference_size: Tensor | None = None
    rails: RailData | None = None

    @property
    def num_frames(self) -> int:
        """Number of recorded states, including the reset state."""
        return self.reference_state.heading.shape[0]

    @property
    def num_actors(self) -> int:
        """Number of actor slots supplied by the producer, including padding."""
        return self.actor_mask.shape[0]

    def validate(self) -> None:
        """Reject invalid input on the host, before batching or compilation."""
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a nonempty string")
        if isinstance(self.dt, bool) or not isinstance(self.dt, (int, float)):
            raise ValueError("dt must be a finite positive number")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a finite positive number")
        if not isinstance(self.actor_mask, Tensor) or self.actor_mask.ndim != 1:
            raise ValueError("actor_mask must have shape [N]")
        n = self.actor_mask.shape[0]
        if n < 1:
            raise ValueError("a scene must contain at least one actor slot")
        device = torch.device("cpu")
        _validate_tensor("actor_mask", self.actor_mask, (n,), torch.bool, device)
        if not bool(self.actor_mask.any()):
            raise ValueError("a scene must contain at least one valid actor")
        if not isinstance(self.reference_state, WorldState):
            raise ValueError("reference_state must be a WorldState")
        heading = self.reference_state.heading
        if not isinstance(heading, Tensor) or heading.ndim != 2:
            raise ValueError("reference_state.heading must have shape [F, N]")
        f = heading.shape[0]
        if f < 1:
            raise ValueError("a scene must contain at least one recorded frame")
        _validate_actor_properties(self, (n,), device)
        _validate_state("initial_state", self.initial_state, (n,), device)
        _validate_state("reference_state", self.reference_state, (f, n), device)
        _validate_tensor("reference_valid", self.reference_valid, (f, n), torch.bool, device)
        if bool((self.reference_valid & ~self.actor_mask[None]).any()) or not torch.equal(
            self.reference_valid.any(dim=0), self.actor_mask
        ):
            raise ValueError("reference_valid must match actor membership")
        _validate_initial_frame(
            self.initial_state, self.reference_state, self.actor_mask, batched=False
        )
        if (
            isinstance(self.ego_index, bool)
            or not isinstance(self.ego_index, int)
            or not 0 <= self.ego_index < n
            or not bool(self.actor_mask[self.ego_index])
        ):
            raise ValueError("ego_index must identify a valid actor")
        if not bool(self.reference_valid[:, self.ego_index].all()):
            raise ValueError("ego must be present throughout the episode")
        if self.routes is not None:
            validate_routes(self.routes, self.actor_mask)
        if self.map is not None:
            if not isinstance(self.map, MapData):
                raise ValueError("scene.map must be MapData or None")
            self.map.validate()
        if self.actor_ids is not None:
            _validate_tensor("actor_ids", self.actor_ids, (n,), torch.int64, device)
            if bool((self.actor_ids[self.actor_mask] < 0).any()):
                raise ValueError("actor IDs must be nonnegative")
        if self.static is not None:
            validate_static(self.static, tuple(self.static.mask.shape), f, device)
            dynamic_ids = self.actor_ids if self.actor_ids is not None else torch.arange(n)
            ids = torch.cat((dynamic_ids[self.actor_mask], self.static.ids[self.static.mask]))
            if ids.unique().numel() != ids.numel():
                raise ValueError("dynamic and static object IDs must be unique")
        if self.reference_size is not None:
            _validate_tensor(
                "reference_size", self.reference_size, (f, n, 2), torch.float32, device
            )
            if not bool((self.reference_size > 0).all()):
                raise ValueError("reference_size must be positive")
        if self.rails is not None:
            validate_rails(self.rails, (), f, n, device)


class SceneBatch(NamedTuple):
    """Fixed dynamic [B, D, F] and static [B, S] geometry with [B, F, S] validity."""

    dt: Tensor
    num_frames: Tensor
    actor_type: Tensor
    actor_mask: Tensor
    size: Tensor
    wheelbase: Tensor
    reference_to_center: Tensor
    initial_state: WorldState
    reference_state: WorldState
    reference_valid: Tensor
    map: MapBatch
    ego_index: Tensor
    routes: RouteData
    actor_ids: Tensor
    static: StaticObjects
    reference_size: Tensor
    rails: RailData


def validate_scene_batch(batch: SceneBatch) -> None:
    """Validate an external batch outside the compiled step, on CPU or GPU."""
    if not isinstance(batch, SceneBatch):
        raise ValueError("batch must be a SceneBatch")
    if not isinstance(batch.actor_mask, Tensor) or batch.actor_mask.ndim != 2:
        raise ValueError("actor_mask must have shape [B, N]")
    b, n = batch.actor_mask.shape
    if b < 1 or n < 1:
        raise ValueError("batch and actor capacities must be positive")
    device = batch.actor_mask.device
    _validate_tensor("actor_mask", batch.actor_mask, (b, n), torch.bool, device)
    if not bool(batch.actor_mask.any(dim=1).all()):
        raise ValueError("every world must contain at least one valid actor")
    _validate_tensor("dt", batch.dt, (b,), torch.float32, device)
    if not bool((batch.dt > 0).all()):
        raise ValueError("dt must be positive")
    _validate_tensor("num_frames", batch.num_frames, (b,), torch.int64, device)
    if not isinstance(batch.reference_state, WorldState):
        raise ValueError("reference_state must be a WorldState")
    heading = batch.reference_state.heading
    if not isinstance(heading, Tensor) or heading.ndim != 3:
        raise ValueError("reference_state.heading must have shape [B, F, N]")
    f = heading.shape[1]
    if f < 1 or not bool(((batch.num_frames >= 1) & (batch.num_frames <= f)).all()):
        raise ValueError("num_frames must be between one and the reference frame capacity")
    _validate_actor_properties(batch, (b, n), device)
    _validate_state("initial_state", batch.initial_state, (b, n), device)
    _validate_state("reference_state", batch.reference_state, (b, f, n), device)
    _validate_tensor("reference_valid", batch.reference_valid, (b, f, n), torch.bool, device)
    _validate_tensor("reference_size", batch.reference_size, (b, f, n, 2), torch.float32, device)
    if not bool((batch.reference_size > 0).all()):
        raise ValueError("reference_size must be positive")
    frames = torch.arange(f, device=device)[None, :, None]
    expected_valid = (frames < batch.num_frames[:, None, None]) & batch.actor_mask[:, None, :]
    if bool((batch.reference_valid & ~expected_valid).any()) or not torch.equal(
        batch.reference_valid.any(dim=1), batch.actor_mask
    ):
        raise ValueError("reference_valid must match frame counts and actor membership")
    _validate_initial_frame(
        batch.initial_state, batch.reference_state, batch.actor_mask, batched=True
    )
    validate_map_batch(batch.map, b, device)
    validate_rails(batch.rails, (b,), f, n, device)
    _validate_tensor("ego_index", batch.ego_index, (b,), torch.int64, device)
    if not bool(((batch.ego_index >= 0) & (batch.ego_index < n)).all()) or not bool(
        batch.actor_mask.gather(1, batch.ego_index[:, None]).all()
    ):
        raise ValueError("ego_index must identify a valid actor in every world")
    ego_valid = batch.reference_valid.gather(
        2, batch.ego_index[:, None, None].expand(b, f, 1)
    ).squeeze(-1)
    if not torch.equal(ego_valid, frames.squeeze(-1) < batch.num_frames[:, None]):
        raise ValueError("ego must be present throughout actual episode frames")
    validate_routes(batch.routes, batch.actor_mask)
    _validate_tensor("actor_ids", batch.actor_ids, (b, n), torch.int64, device)
    if bool(((batch.actor_type == ActorType.STATIC) & batch.actor_mask).any()):
        raise ValueError("batch actors must be dynamic; use the static buffer")
    validate_static(batch.static, tuple(batch.static.mask.shape), f, device)
    if batch.static.mask.ndim != 2 or batch.static.mask.shape[0] != b:
        raise ValueError("static buffer must share the batch dimension")
    if bool((batch.static.valid & (frames >= batch.num_frames[:, None, None])).any()):
        raise ValueError("static validity cannot extend beyond the episode")


def _validate_tensor(
    name: str, value: Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values, including padding")


def _validate_state(
    name: str, state: WorldState, leading_shape: tuple[int, ...], device: torch.device
) -> None:
    if not isinstance(state, WorldState):
        raise ValueError(f"{name} must be a WorldState")
    for field_name, value in zip(WorldState._fields, state, strict=True):
        shape = leading_shape + (2,) if field_name in ("position", "velocity") else leading_shape
        _validate_tensor(f"{name}.{field_name}", value, shape, torch.float32, device)


def _validate_actor_properties(scene: Scene | SceneBatch, shape: tuple[int, ...], device) -> None:
    _validate_tensor("actor_type", scene.actor_type, shape, torch.int64, device)
    _validate_tensor("size", scene.size, shape + (2,), torch.float32, device)
    _validate_tensor("wheelbase", scene.wheelbase, shape, torch.float32, device)
    _validate_tensor("reference_to_center", scene.reference_to_center, shape, torch.float32, device)
    if not bool(
        ((scene.actor_type >= ActorType.VEHICLE) & (scene.actor_type <= ActorType.STATIC)).all()
    ):
        raise ValueError("actor_type contains an unknown physical actor type")
    if not bool((scene.size > 0).all()):
        raise ValueError("size must be positive, including safe padding values")
    if not bool((scene.wheelbase > 0).all()):
        raise ValueError("wheelbase must be positive, including safe padding values")


def _validate_initial_frame(
    initial: WorldState, reference: WorldState, actor_mask: Tensor, *, batched: bool
) -> None:
    for field_name, init_value, ref_value in zip(
        WorldState._fields, initial, reference, strict=True
    ):
        first_value = ref_value[:, 0] if batched else ref_value[0]
        if not torch.equal(init_value[actor_mask], first_value[actor_mask]):
            raise ValueError(f"initial_state.{field_name} must equal reference frame zero")
