"""Data-parallel PPO across processes (``torchrun --nproc_per_node N sim-train ...``).

Every rank simulates its own worlds from a disjoint shard of the training scenes and
computes gradients on its own samples; gradients are averaged before each optimizer
step, so all ranks keep identical weights. Rank 0 logs, evaluates, and saves.
"""

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Distributed:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    def all_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """In place: the mean over ranks."""
        if self.enabled:
            dist.all_reduce(tensor)
            tensor /= self.world_size
        return tensor

    def all_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor)
        return tensor

    def all_min(self, value: int, device) -> int:
        if not self.enabled:
            return value
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor)

    def average_gradients(self, parameters) -> None:
        """Average ``.grad`` over ranks with one flattened all-reduce."""
        if not self.enabled:
            return
        parameters = [p for p in parameters if p.requires_grad]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
        flat = torch.cat([g.reshape(-1) for g in grads])
        self.all_mean(flat)
        for p, g in zip(parameters, flat.split([g.numel() for g in grads]), strict=True):
            p.grad = g.view_as(p).clone() if p.grad is None else p.grad.copy_(g.view_as(p))

    def broadcast_parameters(self, module: torch.nn.Module) -> None:
        if self.enabled:
            for tensor in [*module.parameters(), *module.buffers()]:
                dist.broadcast(tensor.data, 0)

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


def setup(device: str = "cuda") -> Distributed:
    """Join the torchrun process group when launched with more than one process.

    CUDA training uses NCCL (gloo for host tensors) with one GPU per local rank; a CPU
    run uses gloo only and never touches the GPUs.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return Distributed()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.device(device).type == "cuda"
    if cuda:
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"Local rank {local_rank} has no GPU of its own")
        torch.cuda.set_device(local_rank)
    backend = "cpu:gloo,cuda:nccl" if cuda else "gloo"
    # Evaluation runs on rank 0 while the others wait: allow it time.
    dist.init_process_group(backend, timeout=datetime.timedelta(hours=2))
    return Distributed(rank, world_size, local_rank)
