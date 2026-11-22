"""Explicit reset and fixed-capacity batched scenario playback."""

import torch

from cs5478_project.datatypes.scene import SceneBatch, validate_scene_batch
from cs5478_project.datatypes.state import EpisodeState, WorldState, clone_state, select_state
from cs5478_project.datatypes.transition import StepResult
from cs5478_project.env.step import step_kernel


class ReplayEnv:
    """Own playback state while keeping I/O and validation outside the step graph.

    Scene tensors are immutable inputs. Returned state belongs to the environment;
    clone tensors before modifying or retaining them for mutable downstream use.
    """

    def __init__(self, *, compile_step: bool = False) -> None:
        self._step_kernel = (
            torch.compile(step_kernel, dynamic=False, fullgraph=True)
            if compile_step
            else step_kernel
        )
        self._scene: SceneBatch | None = None
        self._state: WorldState | None = None
        self._episode: EpisodeState | None = None

    @property
    def scene(self) -> SceneBatch:
        """Return the active scene batch after the first reset."""
        if self._scene is None:
            raise RuntimeError("Call reset(scene) before accessing the scene.")
        return self._scene

    @property
    def state(self) -> WorldState:
        """Return the latest physical state."""
        if self._state is None:
            raise RuntimeError("Call reset(scene) before accessing state.")
        return self._state

    @property
    def episode(self) -> EpisodeState:
        """Return each world's frame index and sticky completion flag."""
        if self._episode is None:
            raise RuntimeError("Call reset(scene) before accessing episode state.")
        return self._episode

    @torch.no_grad()
    def reset(self, scene: SceneBatch, mask: torch.Tensor | None = None) -> StepResult:
        """Load frame zero, optionally replacing selected world slots.

        Later resets must preserve batch size, actor/frame capacities, dtypes, and
        device. Partial resets merge both scene data and runtime state. The first
        reset must supply the complete batch without a mask.
        """
        validate_scene_batch(scene)
        if self._scene is not None:
            _check_layout(self._scene, scene)
        batch_size = scene.actor_mask.shape[0]
        device = scene.actor_mask.device
        if mask is not None:
            if self._scene is None:
                raise ValueError("The first reset must load a full batch without a mask.")
            if mask.shape != (batch_size,) or mask.dtype != torch.bool or mask.device != device:
                raise ValueError("Reset mask must be bool [B] on the scene device.")
            scene = _select_scene(mask, scene, self.scene)
            self._state = select_state(mask, scene.initial_state, self.state)
            self._episode = EpisodeState(
                step=torch.where(mask, torch.zeros_like(self.episode.step), self.episode.step),
                done=torch.where(mask, scene.num_frames == 1, self.episode.done),
            )
        else:
            self._state = clone_state(scene.initial_state)
            self._episode = EpisodeState(
                step=torch.zeros(batch_size, dtype=torch.int64, device=device),
                done=scene.num_frames == 1,
            )
        self._scene = scene
        flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        active = scene.reference_valid[torch.arange(batch_size, device=device), self.episode.step]
        static_active = scene.static.valid[
            torch.arange(batch_size, device=device), self.episode.step
        ]
        return StepResult(self.state, self.episode, flags, flags.clone(), active, static_active)

    @torch.no_grad()
    def step(self) -> StepResult:
        """Expose the final frame before reporting completion, without automatic reset."""
        result = self._step_kernel(self.state, self.episode, self.scene)
        self._state, self._episode = result.state, result.episode
        return result


def _check_layout(previous: SceneBatch, incoming: SceneBatch) -> None:
    """Reject capacity/device changes before they reach a compiled kernel."""
    for name in SceneBatch._fields:
        old_value, new_value = getattr(previous, name), getattr(incoming, name)
        old_fields = old_value if isinstance(old_value, tuple) else (old_value,)
        new_fields = new_value if isinstance(new_value, tuple) else (new_value,)
        for old_tensor, new_tensor in zip(old_fields, new_fields, strict=True):
            if (
                old_tensor.shape != new_tensor.shape
                or old_tensor.dtype != new_tensor.dtype
                or old_tensor.device != new_tensor.device
            ):
                raise ValueError(f"Reset must preserve the shape, dtype, and device of {name}.")


def _select_scene(mask: torch.Tensor, incoming: SceneBatch, previous: SceneBatch) -> SceneBatch:
    """Replace world slots consistently across scene tensors and nested physical state."""
    fields = []
    for name in SceneBatch._fields:
        new_value, old_value = getattr(incoming, name), getattr(previous, name)
        if isinstance(new_value, tuple):
            fields.append(
                type(new_value)(
                    *(
                        torch.where(
                            mask.reshape(mask.shape + (1,) * (new_tensor.ndim - 1)),
                            new_tensor,
                            old_tensor,
                        )
                        for new_tensor, old_tensor in zip(new_value, old_value, strict=True)
                    )
                )
            )
        else:
            expanded = mask.reshape(mask.shape + (1,) * (new_value.ndim - mask.ndim))
            fields.append(torch.where(expanded, new_value, old_value))
    return SceneBatch(*fields)
