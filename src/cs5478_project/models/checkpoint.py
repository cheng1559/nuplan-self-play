"""Self-describing single-frame checkpoints: weights plus the exact config, loaded strictly."""

from dataclasses import dataclass
from pathlib import Path

import torch

from cs5478_project.models.config import SingleFrameConfig
from cs5478_project.models.single_frame import SingleFrameNet

CHECKPOINT_FORMAT = "cs5478.single_frame"
# 2: map tokens are flattened polyline chunks instead of two-point segments.
CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class LoadedSingleFrame:
    model: SingleFrameNet
    config: SingleFrameConfig
    checkpoint: str
    update: int | None = None


def save_single_frame(path: str | Path, model: SingleFrameNet, *, update: int | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "update": update,
        },
        path,
    )


def load_single_frame(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedSingleFrame:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a SingleFrameNet checkpoint: {checkpoint}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported SingleFrameNet checkpoint version: {payload.get('version')}")
    config = SingleFrameConfig.from_dict(payload["config"])
    model = SingleFrameNet(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return LoadedSingleFrame(
        model.to(device).eval(), config, str(checkpoint), payload.get("update")
    )


def random_single_frame(
    config: SingleFrameConfig = SingleFrameConfig(),
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> LoadedSingleFrame:
    """Freshly initialized weights, reproducible from ``seed`` without touching global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SingleFrameNet(config)
    return LoadedSingleFrame(model.to(device).eval(), config, f"random-init:{seed}")
