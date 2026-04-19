"""Minimal ClearML helpers for experiment logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from clearml import Task
from omegaconf import OmegaConf


@dataclass
class ClearMLTracker:
    """Small wrapper around a ClearML task."""

    task: Any | None

    @property
    def enabled(self) -> bool:
        return self.task is not None

    @property
    def logger(self) -> Any | None:
        return self.task.get_logger() if self.task is not None else None

    def upload_artifact(self, name: str, artifact: Any) -> None:
        """Upload a Python object as a ClearML artifact."""
        if self.task is None:
            return
        self.task.upload_artifact(name=name, artifact_object=artifact)

    def log_metrics(self, title: str, metrics: dict[str, Any]) -> None:
        """Log scalar metrics."""
        if self.logger is None:
            return
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue
            self.logger.report_scalar(title=title, series=str(key), value=float(value), iteration=0)

    def log_table(self, title: str, table: pd.DataFrame) -> None:
        """Log a dataframe as a ClearML table."""
        if self.logger is None or table.empty:
            return
        self.logger.report_table(title=title, series="data", iteration=0, table_plot=table)

    def close(self) -> None:
        """Close the task when logging is complete."""
        if self.task is not None:
            self.task.close()


def init_clearml_task(cfg: Any) -> ClearMLTracker:
    """Create a ClearML task and connect the resolved config."""
    Task.set_offline(bool(cfg.clearml.offline_mode))
    task = Task.init(
        project_name=str(cfg.clearml.project_name),
        task_name=str(cfg.clearml.task_name),
        reuse_last_task_id=False,
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True))
    return ClearMLTracker(task=task)
