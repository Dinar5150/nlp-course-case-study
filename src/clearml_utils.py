"""ClearML integration with graceful local fallback."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class ClearMLTracker:
    """Very small ClearML wrapper that degrades to a no-op tracker."""

    task: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.task is not None

    def log_config(self, cfg: object) -> None:
        if not self.enabled:
            return
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        self.task.connect(resolved_cfg, name="hydra_config")

    def log_metrics(self, title: str, metrics: dict[str, Any], iteration: int = 0) -> None:
        if not self.enabled:
            return
        logger = self.task.get_logger()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.report_scalar(title=title, series=name, value=float(value), iteration=iteration)

    def log_text(self, text: str) -> None:
        if self.enabled:
            self.task.get_logger().report_text(text)

    def log_artifact_paths(self, artifact_paths: dict[str, str]) -> None:
        if not self.enabled:
            return
        for name, path_str in artifact_paths.items():
            path = Path(path_str)
            self.log_text(f"{name}: {path}")
            if not path.exists():
                continue
            try:
                self.task.upload_artifact(name=name, artifact_object=path)
            except Exception as exc:  # pragma: no cover - depends on ClearML backend state
                self.log_text(f"Could not upload artifact {name}: {exc}")

    def close(self) -> None:
        if self.enabled:
            self.task.close()


def init_clearml_task(cfg: object) -> ClearMLTracker:
    """Create a ClearML task when possible, otherwise return a no-op tracker."""
    if not bool(cfg.clearml.enabled):
        return ClearMLTracker(task=None)

    try:
        from clearml import Task

        offline_mode = bool(cfg.clearml.offline_mode) or os.getenv("CLEARML_OFFLINE_MODE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if offline_mode:
            Task.set_offline(offline_mode=True)

        task = Task.init(
            project_name=str(cfg.clearml.project_name),
            task_name=str(cfg.clearml.task_name),
            tags=list(cfg.clearml.tags),
            reuse_last_task_id=False,
        )
        tracker = ClearMLTracker(task=task)
        tracker.log_config(cfg)
        return tracker
    except Exception as exc:
        if bool(cfg.clearml.allow_no_credentials):
            warnings.warn(
                f"ClearML initialization failed, continuing with local-only outputs. Reason: {exc}",
                stacklevel=2,
            )
            return ClearMLTracker(task=None)
        raise

