"""Learned-policy control with rule/log backgrounds and shared lifecycle.

SingleFrameNet drives the selected actors from the current frame through the jerk
action adapter; every other actor follows its log or rule controller.
"""

import torch

from cs5478_project.agents.goals import sample_goals
from cs5478_project.agents.single_frame import SingleFramePolicy
from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.datatypes.control import ControlType
from cs5478_project.env.lifecycle import cache_actor_presence
from cs5478_project.env.rule_based import RuleBasedEnv
from cs5478_project.models.checkpoint import LoadedSingleFrame


class LearnedEnv(RuleBasedEnv):
    """Run the same network for each selected actor in its own local coordinate frame."""

    def __init__(self, loaded, *, compile_step=False, sampling_method="argmax", fixed_goals=False):
        """``fixed_goals`` samples each actor's route goal once at reset and keeps it for the
        episode (training); otherwise the goal moves along the route every step."""
        super().__init__(compile_step=compile_step)
        self._fixed_goals = fixed_goals
        if not isinstance(loaded, LoadedSingleFrame):
            raise TypeError("LearnedEnv runs a loaded SingleFrameNet")
        self.policy = SingleFramePolicy(loaded, sampling_method=sampling_method)
        self.learned_type = ControlType.SINGLE_FRAME
        self._policy_kernel = (
            torch.compile(self.policy, fullgraph=True, dynamic=False)
            if compile_step
            else self.policy
        )
        self._goal_kernel = (
            torch.compile(sample_goals, fullgraph=True, dynamic=False)
            if compile_step
            else sample_goals
        )
        self._goals = None
        # Fixed goals that replace route goals: (mask [B, A], scene points [B, A, 2]).
        self.goal_overrides = None

    @property
    def goals(self):
        if self._goals is None:
            raise RuntimeError("Call reset before accessing goals")
        return self._goals

    @property
    def goal_mask(self):
        has_goal = self.scene.routes.mask.any(-1)
        if self.goal_overrides is not None:
            has_goal = has_goal | self.goal_overrides[0]
        return (self.control_types == self.learned_type) & has_goal & self.actor_mask

    def _route_goals(self):
        goals = self._goal_kernel(self.state, self.scene, self._cache, self.policy.config)
        if self.goal_overrides is not None:
            pinned, points = self.goal_overrides
            goals = torch.where(pinned[..., None, None], points[:, :, None].to(goals), goals)
        return goals

    @torch.no_grad()
    def reset(self, scene, mask=None, *, control_types=None, blind=None):
        if control_types is None:
            control_types = torch.zeros_like(scene.actor_type)
            control_types.scatter_(1, scene.ego_index[:, None], self.learned_type)
        if (
            control_types.shape != scene.actor_mask.shape
            or control_types.dtype != torch.int64
            or control_types.device != scene.dt.device
        ):
            raise ValueError("control_types must be int64 [B, A] on the scene device")
        if bool(((control_types < 0) | (control_types > max(ControlType))).any()):
            raise ValueError("Unknown control type")
        if bool(((control_types != 0) & ~scene.actor_mask).any()):
            raise ValueError("Padding actors must use LOG control")
        policy_ratios = self.policy.config.policy_interval_s / scene.dt
        policy_gaps = policy_ratios.round().long()
        if bool(((policy_ratios - policy_gaps).abs() > 1e-4).any()) or bool(
            (policy_gaps < 1).any()
        ):
            raise ValueError("Policy interval must be an integer multiple of each world's dt")
        self._policy_gap = policy_gaps
        result = super().reset(
            scene,
            mask,
            control_types=torch.where(control_types == self.learned_type, 0, control_types),
            blind=blind,
        )
        self._control_types = (
            control_types.clone()
            if mask is None
            else torch.where(mask[:, None], control_types, self.control_types)
        )
        self._presence = cache_actor_presence(self.scene, self.control_types)
        indices = (self.control_types == self.learned_type).nonzero()
        self._query_world, self._query_slot = indices[:, 0], indices[:, 1]
        zeros = torch.zeros_like(self.state.heading)
        if mask is None or not hasattr(self, "_last_action"):
            self._last_action = VehicleAction(zeros, zeros.clone())
        else:
            self._last_action = VehicleAction(
                *(torch.where(mask[:, None], zeros, value) for value in self._last_action)
            )
        goals = self._route_goals()
        if self._fixed_goals and mask is not None and self._goals is not None:
            goals = torch.where(mask[:, None, None, None], goals, self._goals)
        self._goals = goals
        return result._replace(actor_mask=self.actor_mask)

    def _act(self):
        return self._policy_kernel(
            self.state,
            self.scene,
            self.episode.step,
            self.actor_mask,
            self.control_types,
            self.goals,
            self._query_world,
            self._query_slot,
            self.blind,
            last_action=self._last_action,
        )

    @torch.no_grad()
    def step(self):
        world = torch.arange(self.scene.dt.shape[0], device=self.scene.dt.device)
        current_valid = self.actor_presence[world, self.episode.step]
        previous_valid = self.actor_presence[world, (self.episode.step - 1).clamp_min(0)]
        policy_forward = (self.episode.step % self._policy_gap == 0)[:, None] | (
            current_valid & ~previous_valid
        )
        if self._query_slot.numel():
            action, self.values = self._act()
        else:
            action = VehicleAction(
                torch.zeros_like(self.state.heading), torch.zeros_like(self.state.heading)
            )
        action = VehicleAction(
            *(
                torch.where(policy_forward, new, old)
                for new, old in zip(action, self._last_action, strict=True)
            )
        )
        self._last_action = action
        result, self._progress, self._rails = self._rule_step_kernel(
            self.state,
            self.episode,
            self.scene,
            self._cache,
            self.progress,
            self.rails,
            self.control_types,
            external_action=action,
            external_config=self.policy.config,
            external_policy_forward=policy_forward,
            actor_presence=self.actor_presence,
            blind=self.blind,
        )
        self._state, self._episode = result.state, result.episode
        if not self._fixed_goals:
            self._goals = self._route_goals()
        return result
