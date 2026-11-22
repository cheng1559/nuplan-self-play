"""Independent Beta distributions over normalized actions in (0, 1), summed over action dims.

PPO stores the normalized sample ``unit`` and its log-probability, never the decoded
physical action.
"""

import torch
from torch import Tensor

UNIT_EPS = 1e-6


def beta_log_prob(alpha: Tensor, beta: Tensor, unit: Tensor) -> Tensor:
    """Joint log-density [...] of ``unit`` [..., D], clamped away from the open bounds."""
    unit = unit.clamp(UNIT_EPS, 1 - UNIT_EPS)
    log_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    return ((alpha - 1) * unit.log() + (beta - 1) * (-unit).log1p() + log_norm).sum(-1)


def beta_entropy(alpha: Tensor, beta: Tensor) -> Tensor:
    total = alpha + beta
    log_b = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(total)
    return (
        log_b
        - (alpha - 1) * torch.digamma(alpha)
        - (beta - 1) * torch.digamma(beta)
        + (total - 2) * torch.digamma(total)
    ).sum(-1)


def beta_mean(alpha: Tensor, beta: Tensor) -> Tensor:
    return alpha / (alpha + beta)


def beta_mode(alpha: Tensor, beta: Tensor) -> Tensor:
    """Interior mode, falling back to the mean where a parameter rounds to one."""
    mode = (alpha - 1) / (alpha + beta - 2).clamp_min(UNIT_EPS)
    return torch.where((alpha > 1) & (beta > 1), mode, beta_mean(alpha, beta))


def sample_beta(alpha: Tensor, beta: Tensor) -> Tensor:
    """Sample with the global RNG, kept strictly inside (0, 1)."""
    return torch.distributions.Beta(alpha, beta).sample().clamp(UNIT_EPS, 1 - UNIT_EPS)


def decode_unit(unit: Tensor, config) -> tuple[Tensor, Tensor]:
    """Map unit [..., 2] to physical jerk [m/s^3] and steering rate [rad/s]."""
    unit = unit.clamp(UNIT_EPS, 1 - UNIT_EPS)
    jerk = config.min_jerk + unit[..., 0] * (config.max_jerk - config.min_jerk)
    steering_rate = (2 * unit[..., 1] - 1) * config.max_steering_rate
    return jerk, steering_rate
