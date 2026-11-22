"""Lightweight discovery of local policy checkpoints, without importing torch."""

from pathlib import Path

# Pseudo-checkpoint path for freshly initialized SingleFrameNet weights, e.g. "random-init:0".
RANDOM_INIT_PREFIX = "random-init:"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


RELEASE_NAMES = {
    "selfplay": "Self-play (report)",
    "selfplay_dr": "Self-play + DR (report)",
    "replay": "Replayed traffic (report)",
}


def _entry(name, path):
    return {"name": name, "path": str(path), "config_path": "", "kind": "single_frame"}


def discover_single_frame_checkpoints(
    release: Path = PROJECT_ROOT / "checkpoints",
    training: Path = PROJECT_ROOT / "outputs" / "train",
) -> list[dict[str, str]]:
    """The report's checkpoints (``checkpoints/*.pt``), then those of local training runs
    (``outputs/train/<run>/checkpoints/*.pt``, newest first), then random init. Configs
    are embedded in the files."""
    found = []
    if release.is_dir():
        paths = sorted(release.glob("*.pt"), key=lambda path: list(RELEASE_NAMES).index(path.stem)
                       if path.stem in RELEASE_NAMES else len(RELEASE_NAMES))  # fmt: skip
        found += [_entry(RELEASE_NAMES.get(p.stem, p.stem), p.resolve()) for p in paths]
    if training.is_dir():
        paths = sorted(
            training.glob("*/checkpoints/*.pt"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        found += [_entry(f"{p.parent.parent.name} · {p.stem}", p.resolve()) for p in paths]
    found.append(_entry("SingleFrameNet random init", f"{RANDOM_INIT_PREFIX}0"))
    return found
