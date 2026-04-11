"""Small utility helpers used across the repo."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected at runtime
    torch = None


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
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_json(path_like: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_jsonl(path_like: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_dataframe(path_like: str | Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame as CSV."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def write_text(path_like: str | Path, text: str) -> None:
    """Write a UTF-8 text file."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into dot-separated keys."""
    items: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_dict(nested_value, nested_prefix))
        return items

    if isinstance(value, list):
        items[prefix] = json.dumps(value)
        return items

    items[prefix] = value
    return items

