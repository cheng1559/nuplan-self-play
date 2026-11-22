"""Deterministic SingleFrameNet control: current-frame observation to dense vehicle actions."""

import torch
from torch import nn

from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.env.observations import build_observation
from cs5478_project.models.distributions import beta_mean, beta_mode, decode_unit


class SingleFramePolicy(nn.Module):
    """``argmax`` uses the Beta mode and ``expectation`` uses its mean."""

    def __init__(self, loaded, *, sampling_method="argmax"):
        super().__init__()
        if sampling_method not in ("argmax", "expectation"):
            raise ValueError("SingleFrameNet sampling_method must be argmax or expectation")
        self.model = loaded.model
        self.config = loaded.config
        self.checkpoint = loaded.checkpoint
        self.sampling_method = sampling_method

    def forward(
        self,
        state,
        scene,
        step,
        actor_mask,
        control_types,
        goals,
        query_world,
        query_slot,
        blind=None,
        last_action=None,
    ):
        observation = build_observation(
            state,
            scene,
            step,
            actor_mask,
            control_types,
            goals,
            query_world,
            query_slot,
            self.config,
            blind=blind,
            last_action=last_action,
        )
        output = self.model(observation)
        select = beta_mean if self.sampling_method == "expectation" else beta_mode
        jerk, steering_rate = decode_unit(select(output.alpha, output.beta), self.config)
        shape = scene.actor_mask.shape
        # Each query is a unique dynamic actor; assignments are validated at reset.
        index = query_world * shape[1] + query_slot
        dense = jerk.new_zeros((shape[0] * shape[1], 2)).index_copy(
            0, index, torch.stack((jerk, steering_rate), -1)
        )
        dense = dense.reshape(*shape, 2)
        return VehicleAction(dense[..., 0], dense[..., 1]), output.value
