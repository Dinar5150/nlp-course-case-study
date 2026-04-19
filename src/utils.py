"""Small utility helpers used across the repo."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def get_repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the repo root when needed."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_output_dir(cfg: object) -> Path:
    """Return the main output directory for one run."""
    return ensure_dir(cfg.paths.output_dir)


def get_stage_dir(cfg: object, stage_name: str) -> Path:
    """Return a directory for an intermediate training stage."""
    return ensure_dir(get_output_dir(cfg) / stage_name)
