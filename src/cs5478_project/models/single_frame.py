"""Single-frame actor-critic with a Beta policy over normalized jerk and steering rate.

Per-entity MLP embeddings, query-to-agent cross-attention, a max-pooled map token, a
sinusoidal goal embedding, a shared combiner, and actor/value heads. It takes one frame
and returns per-query outputs.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cs5478_project.datatypes.map import CHUNK_POINTS, MapElementType
from cs5478_project.datatypes.observation import (
    AGENT_FEATURES,
    KINEMATIC_FEATURES,
    LANE_FEATURES,
    LAST_ACTION_FEATURES,
    REWARD_CONDITIONS,
    Observation,
)
from cs5478_project.models.config import SingleFrameConfig
from cs5478_project.models.layers import gen_sineembed_for_position, mha_sdpa

ACTION_DIM = 2
# softplus(0.54) + 1 ~= 2: the initial Beta(2, 2) is a symmetric bell at the range midpoint.
BETA_INIT_BIAS = 0.54


class PolicyOutput(NamedTuple):
    alpha: Tensor  # [Q, 2], order: jerk, steering rate; always > 1
    beta: Tensor  # [Q, 2]
    value: Tensor  # [Q], the sum of value_parts
    value_parts: Tensor  # [Q, 1], the value as a single reward component


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        layer_init(nn.Linear(inputs, hidden)), nn.ReLU(), layer_init(nn.Linear(hidden, outputs))
    )


class CrossAttentionBlock(nn.Module):
    """Pre-norm single-query cross-attention followed by a residual feed-forward block."""

    def __init__(self, dim, heads, *, separate_kv_norm):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        # The agent query is itself one of the keys, so it shares their normalization.
        self.norm_kv = nn.LayerNorm(dim) if separate_kv_norm else None
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = mlp(dim, dim * 4, dim)

    def forward(self, query, key_value, ignore):
        norm_kv = self.norm_q if self.norm_kv is None else self.norm_kv
        query = query + mha_sdpa(
            self.attention,
            query=self.norm_q(query),
            key_value=norm_kv(key_value),
            num_heads=self.heads,
            key_padding_mask=ignore,
        )
        return query + self.ffn(self.norm_ffn(query))


def _scales(config):
    p, v, s = 1 / config.position_scale, 1 / config.velocity_scale, 1 / config.size_scale
    agent = [p, p, v, v, 1, 1, s, s] + [p] * 8 + [1, 1, 1]
    kinematics = [v, v, s, s, 1, 1 / config.max_acceleration, 1]
    action = [1, 1, 1 / config.max_acceleration] if config.last_action_input else []
    conditions = [1] * REWARD_CONDITIONS if config.reward_conditions else []
    lane = [p] * (CHUNK_POINTS * 2) + [1] * len(MapElementType) + [v, 1]
    assert (len(agent), len(kinematics), len(lane)) == (
        AGENT_FEATURES,
        KINEMATIC_FEATURES,
        LANE_FEATURES,
    )
    kinematics = kinematics + action + conditions
    return (torch.tensor(values) for values in (agent, kinematics, lane))


class SingleFrameNet(nn.Module):
    def __init__(self, config: SingleFrameConfig = SingleFrameConfig()):
        super().__init__()
        self.config = config
        dim, heads = config.embed_dim, config.num_heads
        agent_scale, kinematic_scale, lane_scale = _scales(config)
        # Derived from the config, so they are not stored in checkpoints.
        self.register_buffer("agent_scale", agent_scale, persistent=False)
        self.register_buffer("kinematic_scale", kinematic_scale, persistent=False)
        self.register_buffer("lane_scale", lane_scale, persistent=False)

        self.agent_mlp = mlp(AGENT_FEATURES, dim, dim)
        conditions = REWARD_CONDITIONS if config.reward_conditions else 0
        action = LAST_ACTION_FEATURES if config.last_action_input else 0
        self.kinematics_mlp = mlp(KINEMATIC_FEATURES + action + conditions, dim, dim)
        # One flattened chunk of ordered points per token; no per-point encoder needed.
        self.lane_mlp = mlp(LANE_FEATURES, dim // 2, dim)
        self.agent_layers = nn.ModuleList(
            CrossAttentionBlock(dim, heads, separate_kv_norm=False)
            for _ in range(config.num_agent_attention_layers)
        )
        # Map token for queries whose window contains no valid segment.
        self.default_lane = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.combiner = nn.Sequential(layer_init(nn.Linear(dim * 4, dim * 4)), nn.ReLU())
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(dim * 4, dim * 4)),
            nn.ReLU(),
            layer_init(nn.Linear(dim * 4, ACTION_DIM * 2), std=0.01, bias=BETA_INIT_BIAS),
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(dim * 4, dim * 4)),
            nn.ReLU(),
            layer_init(nn.Linear(dim * 4, 1), std=1.0),
        )

    def forward(self, observation: Observation) -> PolicyOutput:
        dim = self.config.embed_dim
        agents = self.agent_mlp(observation.agents * self.agent_scale)  # [Q, K, D]
        query = agents[:, :1]
        # The query slot is always valid, so no attention row is fully masked.
        ignore_agents = ~observation.agent_mask
        for layer in self.agent_layers:
            query = layer(query, agents, ignore_agents)

        lanes = observation.lanes
        lane_features = self.lane_mlp(lanes * self.lane_scale)
        # Sinusoids need fp32 phases (meters * 2pi reach ~1e3 rad), but their values fit
        # bf16; casting keeps autocast activations in bf16 instead of promoting to fp32.
        last = 2 * (CHUNK_POINTS - 1)
        lane_position = gen_sineembed_for_position(
            lanes[..., 0:2], dim
        ) + gen_sineembed_for_position(lanes[..., last : last + 2], dim)
        lane_embedding = lane_features + lane_position.to(lane_features.dtype)  # [Q, M, D]
        lane_mask = observation.lane_mask
        has_lanes = lane_mask.any(-1, keepdim=True)  # [Q, 1]
        lane_token = lane_embedding.masked_fill(~lane_mask[..., None], -torch.inf).amax(1)
        lane_token = torch.where(has_lanes, lane_token, self.default_lane.to(lane_token.dtype))

        kinematics = self.kinematics_mlp(observation.kinematics * self.kinematic_scale)
        goal = gen_sineembed_for_position(observation.goal, dim).to(kinematics.dtype)  # [Q, D]
        query, lane_token = query.to(kinematics.dtype), lane_token.to(kinematics.dtype)
        features = self.combiner(torch.cat((query[:, 0], goal, kinematics, lane_token), -1))
        # Heads run in fp32 even under autocast: Beta parameters and values need more than
        # bf16's three significant digits, and these layers are small.
        with torch.autocast(features.device.type, enabled=False):
            features = features.float()
            raw_alpha, raw_beta = self.actor_head(features).chunk(2, -1)
            parts = self.value_head(features)
        return PolicyOutput(
            F.softplus(raw_alpha) + 1.0, F.softplus(raw_beta) + 1.0, parts.sum(-1), parts
        )
