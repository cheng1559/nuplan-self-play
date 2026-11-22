"""Generalized advantage estimation and the clipped PPO update for SingleFrameNet.

One update can span several rollouts. Each
rollout is reduced to its valid samples with their advantages; observations, which
dominate memory, are kept in half precision on ``store`` (the host by default for CUDA
training) and moved to the device one micro-batch at a time. Gradients accumulate over
the micro-batches of each minibatch, so the update matches one large minibatch.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.datatypes.observation import Observation
from cs5478_project.models.distributions import beta_entropy, beta_log_prob
from cs5478_project.training.config import PPOConfig
from cs5478_project.training.distributed import Distributed
from cs5478_project.training.rollout import Rollout


def advantages(rollout: Rollout, gamma: float, lam: float) -> tuple[Tensor, Tensor]:
    """GAE per reward component over [T, Q, C]; returns the summed advantage [T, Q] and
    the per-component targets [T, Q, C]. ``done`` stops bootstrapping; samples after it
    are masked later. All components share ``done``, so the summed advantage equals GAE
    on the summed reward and value."""
    steps = rollout.reward.shape[0]
    advantage = torch.zeros_like(rollout.reward)
    running = torch.zeros_like(rollout.last_value)
    for t in reversed(range(steps)):
        next_value = rollout.last_value if t == steps - 1 else rollout.value[t + 1]
        alive = (~rollout.done[t]).float()[:, None]
        delta = rollout.reward[t] + gamma * next_value * alive - rollout.value[t]
        running = delta + gamma * lam * alive * running
        advantage[t] = running
    return advantage.sum(-1), advantage + rollout.value


class Samples(NamedTuple):
    """Valid samples of one or more rollouts, [N] each (value and target [N, C];
    observation on ``store``)."""

    observation: Observation
    unit: Tensor
    log_prob: Tensor
    value: Tensor
    advantage: Tensor
    target: Tensor


def flatten(rollout: Rollout, gamma: float, lam: float, store=None) -> Samples:
    """Advantages of one rollout and its valid samples; observations go to ``store``."""
    advantage, target = advantages(rollout, gamma, lam)
    index = rollout.valid.flatten().nonzero().squeeze(1)

    def flat(x):
        return x.flatten(0, 1)[index]

    def stored(x):
        x = flat(x)
        return (x.half() if x.is_floating_point() else x).to(store or x.device)

    return Samples(
        Observation(*(stored(x) for x in rollout.observation)),
        flat(rollout.unit),
        flat(rollout.log_prob),
        flat(rollout.value),
        flat(advantage),
        flat(target),
    )


def merge(parts: list[Samples]) -> Samples:
    return Samples(
        Observation(*(torch.cat(fields) for fields in zip(*(p.observation for p in parts)))),
        *(torch.cat(fields) for fields in zip(*(p[1:] for p in parts))),
    )


class PPO:
    """AdamW, cosine learning rate, and the PPO losses with value clipping."""

    def __init__(self, model, config: PPOConfig, total_updates: int, distributed=None):
        self.model = model
        self.config = config
        # Data-parallel ranks: shared advantage statistics, minibatch count, and averaged
        # gradients (training/distributed.py); one process by default.
        self.distributed = distributed or Distributed()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )
        floor = config.min_learning_rate_fraction

        def schedule(update):
            progress = min(update / max(total_updates, 1), 1.0)
            cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return floor + (1 - floor) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self._forward = torch.compile(model, dynamic=False) if config.compile else model

    def _loss(self, observation, unit, old_log_prob, old_value, advantage, target):
        config = self.config
        device = unit.device.type
        with torch.autocast(device, torch.bfloat16, enabled=config.amp):
            output = self._forward(observation)
        log_prob = beta_log_prob(output.alpha, output.beta, unit)
        entropy = beta_entropy(output.alpha, output.beta)
        ratio = (log_prob - old_log_prob).exp()
        clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        value = output.value_parts
        value_clipped = old_value + (value - old_value).clamp(
            -config.value_clip_epsilon, config.value_clip_epsilon
        )
        # Clipped value loss.
        squared = torch.maximum((value - target).square(), (value_clipped - target).square())
        value_loss = 0.5 * squared.mean(0).sum()
        loss = policy_loss + config.value_coeff * value_loss - config.entropy_coeff * entropy.mean()
        with torch.no_grad():
            stats = torch.stack(
                (
                    policy_loss,
                    value_loss,
                    entropy.mean(),
                    ((ratio - 1) - (log_prob - old_log_prob)).mean(),  # approx KL
                    ((ratio - 1).abs() > config.clip_epsilon).float().mean(),
                )
            )
        return loss, stats

    def update(self, rollouts) -> dict:
        """One PPO update on a Rollout, a list of them, or already flattened Samples."""
        config = self.config
        if isinstance(rollouts, Samples):
            data = rollouts
        else:
            rollouts = rollouts if isinstance(rollouts, list) else [rollouts]
            data = merge([flatten(r, config.gamma, config.gae_lambda) for r in rollouts])
        device = data.unit.device
        observation = data.observation
        unit, old_log_prob, old_value, target = data.unit, data.log_prob, data.value, data.target
        advantage = data.advantage
        distributed = self.distributed
        if config.normalize_advantages:
            moments = torch.stack(
                (advantage.sum(), advantage.square().sum(), advantage.new_tensor(len(advantage)))
            )
            total, square, count = distributed.all_sum(moments.double()).tolist()
            mean = total / count
            std = max((square / count - mean * mean) * count / max(count - 1, 1), 0.0) ** 0.5
            advantage = (advantage - mean) / max(std, 1e-8)
        samples = unit.shape[0]
        batch = min(config.minibatch_size, distributed.all_min(samples, device))
        micro = min(config.microbatch_size, batch)
        minibatches = distributed.all_min(samples // batch, device)
        totals, count = torch.zeros(5, device=device), 0
        grad_norm = torch.zeros((), device=device)
        self.model.train()
        epochs_run = 0
        for _ in range(config.optimization_epochs):
            epochs_run += 1
            order = torch.randperm(samples)
            for start in range(0, minibatches * batch, batch):
                self.optimizer.zero_grad(set_to_none=True)
                rows = order[start : start + batch]
                for first in range(0, batch - micro + 1, micro):
                    host = rows[first : first + micro]
                    part = host.to(device)
                    inputs = Observation(
                        *(x[host.to(x.device)].to(device, non_blocking=True) for x in observation)
                    )
                    inputs = Observation(
                        *(x.float() if x.is_floating_point() else x for x in inputs)
                    )
                    loss, stats = self._loss(
                        inputs, unit[part], old_log_prob[part], old_value[part],
                        advantage[part], target[part],
                    )  # fmt: skip
                    (loss * (micro / (batch - batch % micro))).backward()
                    totals += stats
                    count += 1
                distributed.average_gradients(self.model.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_grad_norm
                )
                self.optimizer.step()
        self.scheduler.step()
        self.model.eval()
        names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        values = (totals / max(count, 1)).tolist()
        total = target.sum(-1)
        residual = (total - old_value.sum(-1)).var() / total.var().clamp_min(1e-8)
        return {
            **dict(zip(names, values, strict=True)),
            "explained_variance": float(1 - residual),
            "grad_norm": float(grad_norm),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "samples": int(distributed.all_sum(torch.tensor(samples, device=device))),
            "minibatches": count * micro // max(batch, 1),
            "epochs": epochs_run,
        }
