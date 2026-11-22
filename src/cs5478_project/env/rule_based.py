"""State ownership for tensor IDM vehicles and route-following pedestrians."""

from functools import partial

import torch
from torch import Tensor

from cs5478_project.agents.config import IDMConfig, InteractionConfig, PedestrianConfig
from cs5478_project.agents.idm import RailState, initial_rail_state, rail_pose
from cs5478_project.agents.route import cache_routes, project_route
from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.scene import ActorType, SceneBatch
from cs5478_project.datatypes.state import WorldState, select_state
from cs5478_project.datatypes.transition import StepResult
from cs5478_project.dynamics.bicycle import BicycleConfig
from cs5478_project.env.env import ReplayEnv
from cs5478_project.env.lifecycle import cache_actor_presence
from cs5478_project.env.rule_step import rail_world_state, rule_step_kernel


class RuleBasedEnv(ReplayEnv):
    """Simulate controlled actors from logged arrival through episode completion.

    Controllers are int64 [B, A] inputs independent of physical actor categories.
    Pedestrians always receive oracle speed; control_ego=False always logs ego.
    """

    def __init__(
        self,
        *,
        compile_step: bool = False,
        control_ego: bool = True,
        idm: IDMConfig = IDMConfig(),
        pedestrian: PedestrianConfig = PedestrianConfig(),
        interaction: InteractionConfig = InteractionConfig(),
        bicycle: BicycleConfig = BicycleConfig(),
    ) -> None:
        super().__init__()
        self._compile_step = compile_step
        self._kernel_configs = dict(
            idm=idm, pedestrian=pedestrian, interaction=interaction, bicycle=bicycle
        )
        self._kernels = {}
        self._rule_step_kernel = self._kernel(reactive=True, pedestrians=True)
        self._control_ego = control_ego
        self._idm = idm
        self._rails: RailState | None = None
        self._progress: Tensor | None = None
        self._control_types: Tensor | None = None
        self._presence: Tensor | None = None
        self._blind: Tensor | None = None

    def _kernel(self, *, reactive: bool, pedestrians: bool):
        """The step kernel specialized to which rule controllers are in use (cached)."""
        key = (reactive, pedestrians)
        if key not in self._kernels:
            kernel = partial(
                rule_step_kernel, **self._kernel_configs, reactive=reactive, pedestrians=pedestrians
            )
            self._kernels[key] = (
                torch.compile(kernel, dynamic=False, fullgraph=True)
                if self._compile_step
                else kernel
            )
        return self._kernels[key]

    @property
    def blind(self) -> Tensor:
        """Per-query perception flags [B, A]; other actors can still observe blind actors."""
        if self._blind is None:
            raise RuntimeError("Call reset(scene) before accessing blind flags.")
        return self._blind

    @property
    def actor_presence(self) -> Tensor:
        """Cached [B, F, A] membership for each actor's applied controller."""
        if self._presence is None:
            raise RuntimeError("Call reset(scene) before accessing actor presence.")
        return self._presence

    @property
    def actor_mask(self) -> Tensor:
        """Runtime membership at each world's current step, without radius-removed agents."""
        world = torch.arange(self.scene.num_frames.shape[0], device=self.scene.dt.device)
        return self.actor_presence[world, self.episode.step] & ~self.rails.removed

    @property
    def rails(self) -> RailState:
        """IDM agents' route windows, progress, speeds, and removal flags."""
        if self._rails is None:
            raise RuntimeError("Call reset(scene) before accessing rails.")
        return self._rails

    @property
    def control_types(self) -> Tensor:
        """Current per-actor controller assignments, with LOG in padded slots."""
        if self._control_types is None:
            raise RuntimeError("Call reset(scene) before accessing control_types.")
        return self._control_types

    @property
    def progress(self) -> Tensor:
        """Latest monotonic route progress [B, A], initialized on actor arrival."""
        if self._progress is None:
            raise RuntimeError("Call reset(scene) before accessing progress.")
        return self._progress

    @torch.no_grad()
    def reset(
        self,
        scene: SceneBatch,
        mask: Tensor | None = None,
        *,
        control_types: Tensor | None = None,
        blind: Tensor | None = None,
    ) -> StepResult:
        """Reset full/selected worlds, including cached routes and controller state."""
        if control_types is None:
            # IDM goes to the vehicles nuPlan's builder selected (those with rails);
            # other vehicles replay their logs.
            vehicle = torch.where(scene.rails.rail_valid, ControlType.IDM, ControlType.LOG)
            control_types = torch.where(
                scene.actor_type == ActorType.VEHICLE, vehicle, ControlType.PEDESTRIAN
            )
            control_types = torch.where(scene.actor_mask, control_types, ControlType.LOG)
        if (
            control_types.shape != scene.actor_mask.shape
            or control_types.dtype != torch.int64
            or control_types.device != scene.actor_mask.device
        ):
            raise ValueError("control_types must be int64 [B, A] on the scene device")
        known = (control_types >= ControlType.LOG) & (control_types <= ControlType.PEDESTRIAN)
        if not bool((known | (control_types == ControlType.ABSENT)).all()):
            raise ValueError("Unknown control type")
        if bool(((control_types != ControlType.LOG) & ~scene.actor_mask).any()):
            raise ValueError("Padding actors must use LOG control")
        if bool(((control_types == ControlType.IDM) & ~scene.rails.rail_valid).any()):
            raise ValueError("IDM control needs an exported rail; this actor has none")
        incoming_control_types = control_types.clone()
        if blind is None:
            blind = torch.zeros_like(scene.actor_mask)
        if (
            blind.shape != scene.actor_mask.shape
            or blind.dtype != torch.bool
            or blind.device != scene.actor_mask.device
        ):
            raise ValueError("blind must be bool [B, A] on the scene device")
        if bool((blind & ~scene.actor_mask).any()):
            raise ValueError("Padding actors cannot be blind")
        if not self._control_ego:
            incoming_control_types.scatter_(1, scene.ego_index[:, None], ControlType.LOG)
        result = super().reset(scene, mask)
        self._cache = cache_routes(self.scene.routes)
        progress = project_route(self._cache, result.state.position).progress
        if mask is None:
            self._progress = progress
            self._control_types = incoming_control_types
            self._blind = blind.clone()
        else:
            self._progress = torch.where(mask[:, None], progress, self.progress)
            self._control_types = torch.where(
                mask[:, None], incoming_control_types, self.control_types
            )
            self._blind = torch.where(mask[:, None], blind, self.blind)
        self._presence = cache_actor_presence(self.scene, self.control_types)
        self._rule_step_kernel = self._kernel(
            reactive=bool((self.control_types == ControlType.IDM).any()),
            pedestrians=bool((self.control_types == ControlType.PEDESTRIAN).any()),
        )
        rails = initial_rail_state(self.scene, self.control_types, self._idm)
        if mask is None or self._rails is None:
            self._rails = rails
        else:
            self._rails = RailState(
                *(
                    torch.where(mask.reshape(-1, *([1] * (new.ndim - 1))), new, old)
                    for new, old in zip(rails, self._rails, strict=True)
                )
            )
        # nuPlan reports IDM agents snapped onto their lanes from the first frame.
        snapped = (
            (self.control_types == ControlType.IDM)
            & self.scene.rails.rail_valid
            & (mask[:, None] if mask is not None else True)
        )
        pose = rail_pose(self.scene, self._rails)
        zeros = torch.zeros_like(pose.velocity)
        on_rail = rail_world_state(
            self.scene, pose, self.state._replace(heading=pose.heading), pose.velocity, 1.0
        )
        on_rail = WorldState(
            on_rail.position, on_rail.heading, on_rail.velocity, zeros, zeros, zeros, zeros
        )
        self._state = select_state(snapped, on_rail, self.state)
        return result._replace(state=self.state, actor_mask=self.actor_mask)

    @torch.no_grad()
    def step(self) -> StepResult:
        """Advance every unfinished world without device-to-host synchronization."""
        result, self._progress, self._rails = self._rule_step_kernel(
            self.state,
            self.episode,
            self.scene,
            self._cache,
            self.progress,
            self.rails,
            self.control_types,
            actor_presence=self.actor_presence,
            blind=self.blind,
        )
        self._state, self._episode = result.state, result.episode
        return result
