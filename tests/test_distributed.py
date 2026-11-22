"""Data-parallel PPO and sim-train across two CPU processes (gloo)."""

import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_rule_agents import batch_of, traffic_scene
from test_training import SMALL

from cs5478_project.data import save_scene
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.evaluation import DrivingEvaluator
from cs5478_project.models.checkpoint import random_single_frame
from cs5478_project.training.config import PPOConfig
from cs5478_project.training.distributed import Distributed
from cs5478_project.training.ppo import PPO
from cs5478_project.training.rollout import PPOEnv, collect


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ppo_rank(rank, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.manual_seed(rank)  # different episodes per rank
    speeds = (5.0, 2.0) if rank == 0 else (8.0, 0.0)
    scene = traffic_scene(((0.0, 0.0), (20.0, 0.0)), speeds, (ActorType.VEHICLE,) * 2, frames=12)
    loaded = random_single_frame(SMALL)  # same seed: same initial weights
    rollout, _ = collect(PPOEnv(loaded, amp=False), DrivingEvaluator(), batch_of(scene, scene))
    config = PPOConfig(worlds=2, minibatch_size=8, microbatch_size=4, compile=False, amp=False)
    PPO(loaded.model, config, 1, Distributed(rank, 2, 0)).update(rollout)
    results[rank] = [p.detach().clone() for p in loaded.model.parameters()]
    dist.destroy_process_group()


def test_data_parallel_updates_keep_ranks_identical(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CPU-only children
    results = mp.Manager().dict()
    mp.spawn(_ppo_rank, args=(_free_port(), results), nprocs=2, join=True)
    first, second = results[0], results[1]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    initial = list(random_single_frame(SMALL).model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(first, initial, strict=True))


def _train_rank(rank, port, data, output):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2"
    )
    from cs5478_project.training.train import main

    main(
        ["--train", str(data), "--eval", f"set={data}", "--output", str(output),
         "--device", "cpu", "--worlds", "2", "--epochs", "1", "--minibatch-size", "8",
         "--microbatch-size", "8", "--eval-worlds", "2", "--no-compile", "--no-amp",
         "--no-cache-scenes", "--loader-workers", "0", "--no-prefetch",
         "--model", "embed_dim=16", "--model", "num_heads=2", "--model", "max_agents=4",
         "--model", "max_lanes=4"]
    )  # fmt: skip


@pytest.mark.timeout(600)
def test_sim_train_runs_on_two_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CPU-only children
    data = tmp_path / "scenes"
    data.mkdir()
    for index in range(4):
        scene = traffic_scene(
            ((0.0, 0.0), (20.0 + index, 0.0)), (5.0, 2.0), (ActorType.VEHICLE,) * 2, frames=12
        )
        save_scene(scene, data / f"scene{index}.pt")
    output = tmp_path / "run"
    mp.spawn(_train_rank, args=(_free_port(), data, output), nprocs=2, join=True)
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["kind"] == "train"]
    # Each rank has one batch of two scenes: one update over four scenes.
    assert len(train) == 1 and train[0]["scenes"] == 4
    assert {r["epoch"] for r in records if r["kind"] == "eval"} == {0, 1}
    assert (output / "checkpoints" / "epoch_001.pt").exists()
