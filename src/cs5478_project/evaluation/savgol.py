"""Savitzky-Golay smoothing and derivatives as nuPlan's metrics compute them.

nuPlan calls scipy's ``savgol_filter`` (mode "interp"): interior samples use a centered
window, and the first and last ``window // 2`` samples use the polynomial fitted to the
first and last full window. This module reproduces that for batched rows of different
lengths, including scipy's even-window convention: interior sample i uses samples
``i - (window - 1) // 2 ... i + window // 2`` evaluated at the window center.
"""

import functools

import torch
from torch import Tensor


@functools.cache
def _weights(window: int, order: int, derivative: int) -> Tensor:
    """[window + 1, window]: row j < window evaluates the fitted polynomial's derivative
    at sample j; the last row evaluates it at the window center (window - 1) / 2."""
    x = torch.arange(window, dtype=torch.float64)
    design = torch.stack([x**power for power in range(order + 1)], 1)  # [W, P]
    fit = torch.linalg.pinv(design)  # [P, W]
    at = torch.cat((x, x.new_tensor([(window - 1) / 2])))
    basis = torch.zeros(window + 1, order + 1, dtype=torch.float64)
    for power in range(derivative, order + 1):
        coefficient = 1.0
        for step in range(derivative):
            coefficient *= power - step
        basis[:, power] = coefficient * at ** (power - derivative)
    return basis @ fit


def savgol(values: Tensor, length: Tensor, window: int, order: int, derivative: int = 0) -> Tensor:
    """Filter rows [N, T] whose first ``length`` [N] samples are valid.

    Returns [N, T] in units per sample (divide by dt**derivative). Rows shorter than the
    window return zeros: they are too short to evaluate.
    """
    steps = values.shape[-1]
    window = min(window, steps)
    weights = _weights(window, order, derivative).to(values)
    half = (window - 1) // 2
    index = torch.arange(steps, device=values.device)
    start = (index - half).clamp_min(0)
    start = torch.minimum(start[None], (length[:, None] - window).clamp_min(0))  # [N, T]
    gather = (start[..., None] + torch.arange(window, device=values.device)).clamp_max(steps - 1)
    windows = values.gather(1, gather.flatten(1)).view(*gather.shape)  # [N, T, W]
    position = (index[None] - start).clamp(0, window - 1)
    if window % 2 == 0:
        # Interior samples of even windows sit between two samples (scipy's convention).
        edge = window // 2
        interior = (
            (position == half) & (index[None] >= edge) & (index[None] < length[:, None] - edge)
        )
        position = torch.where(interior, window, position)
    output = (weights[position] * windows).sum(-1)
    return torch.where(length[:, None] >= window, output, 0.0)
