"""Hydra entry point for small practical Optuna searches."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.clearml_utils import ClearMLTracker, init_clearml_task
from src.train import run_experiment
from src.utils import ensure_dir, save_json


def rewrite_output_paths(cfg: DictConfig, output_dir: str) -> None:
    """Rewrite all path fields for an Optuna trial output directory."""
    cfg.paths.output_dir = output_dir
    cfg.paths.train_stage_dir = f"{output_dir}/train"
    cfg.paths.selftrain_stage_dir = f"{output_dir}/selftrain"
    cfg.paths.sampled_ids_path = f"{output_dir}/sampled_example_ids.json"
    cfg.paths.pseudo_labels_path = f"{output_dir}/pseudo_labels.jsonl"
    cfg.paths.predictions_path = f"{output_dir}/predictions.jsonl"
    cfg.paths.metrics_path = f"{output_dir}/metrics.json"
    cfg.paths.per_label_metrics_path = f"{output_dir}/per_label_metrics.json"
    cfg.paths.error_analysis_path = f"{output_dir}/error_analysis_sample.csv"
    cfg.paths.results_summary_path = f"{output_dir}/results_summary.csv"
    cfg.paths.best_checkpoint_path = f"{output_dir}/best_checkpoint.txt"
    cfg.paths.results_comparison_path = f"{output_dir}/results_comparison.csv"
    cfg.paths.error_type_reduction_path = f"{output_dir}/error_type_reduction.csv"
    cfg.paths.case_study_summary_path = f"{output_dir}/case_study_summary.md"


def build_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    """Create the requested Optuna sampler."""
    sampler_name = str(cfg.optuna.sampler).lower()
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=int(cfg.seed))
    return optuna.samplers.TPESampler(seed=int(cfg.seed))


def build_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    """Create the requested Optuna pruner."""
    pruner_name = str(cfg.optuna.pruner).lower()
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_warmup_steps=2)


def sample_trial_params(trial: optuna.Trial, cfg: DictConfig) -> dict[str, Any]:
    """Sample a small, practical hyperparameter set."""
    search_space = cfg.optuna.search_space
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(search_space.learning_rate.low),
            float(search_space.learning_rate.high),
            log=bool(search_space.learning_rate.log),
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            float(search_space.weight_decay.low),
            float(search_space.weight_decay.high),
        ),
        "num_train_epochs": trial.suggest_int(
            "num_train_epochs",
            int(search_space.num_train_epochs.low),
            int(search_space.num_train_epochs.high),
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            float(search_space.warmup_ratio.low),
            float(search_space.warmup_ratio.high),
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            list(search_space.per_device_train_batch_size.choices),
        ),
        "max_length": trial.suggest_categorical(
            "max_length",
            list(search_space.max_length.choices),
        ),
        "pseudo_label_threshold": trial.suggest_float(
            "pseudo_label_threshold",
            float(search_space.pseudo_label_threshold.low),
            float(search_space.pseudo_label_threshold.high),
        ),
    }


def build_trial_cfg(base_cfg: DictConfig, trial: optuna.Trial, params: dict[str, Any]) -> DictConfig:
    """Create an isolated config for one Optuna trial."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)

    cfg.clearml.enabled = False
    cfg.optuna.enabled = False

    cfg.trainer.learning_rate = params["learning_rate"]
    cfg.trainer.weight_decay = params["weight_decay"]
    cfg.trainer.num_train_epochs = params["num_train_epochs"]
    cfg.trainer.warmup_ratio = params["warmup_ratio"]
    cfg.trainer.per_device_train_batch_size = params["per_device_train_batch_size"]
    cfg.trainer.max_length = params["max_length"]
    cfg.experiment.pseudo_label_threshold = params["pseudo_label_threshold"]

    trial_output_dir = f"{cfg.paths.optuna_dir}/trial_{trial.number}"
    rewrite_output_paths(cfg, trial_output_dir)
    return cfg


def save_optimization_history(study: optuna.Study, output_path: Path) -> None:
    """Save a simple optimization history plot without extra heavy deps."""
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        return

    xs = list(range(len(completed_trials)))
    ys = [float(trial.value) for trial in completed_trials]
    best_so_far: list[float] = []
    current_best = float("-inf")
    for value in ys:
        current_best = max(current_best, value)
        best_so_far.append(current_best)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", label="trial value")
    plt.plot(xs, best_so_far, linestyle="--", label="best so far")
    plt.xlabel("Trial")
    plt.ylabel("Validation F1")
    plt.title("Optuna Optimization History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for Optuna search."""
    ensure_dir(cfg.paths.optuna_dir)

    tracker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(tracker_cfg, False)
    tracker_cfg.clearml.task_name = f"optuna_{cfg.experiment.name}_seed_{cfg.seed}"
    tracker = init_clearml_task(tracker_cfg)

    sampler = build_sampler(cfg)
    pruner = build_pruner(cfg)
    study = optuna.create_study(
        direction=str(cfg.optuna.direction),
        sampler=sampler,
        pruner=pruner,
        study_name=str(cfg.optuna.study_name),
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial, cfg)
        trial_cfg = build_trial_cfg(cfg, trial, params)
        result = run_experiment(trial_cfg, init_tracking=False, generate_reports=False)
        validation_f1 = float(result["validation_f1"])
        trial.set_user_attr("test_f1", float(result["test_f1"]))
        trial.set_user_attr("output_dir", str(trial_cfg.paths.output_dir))
        trial.report(validation_f1, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return validation_f1

    study.optimize(objective, n_trials=int(cfg.optuna.n_trials), timeout=int(cfg.optuna.timeout))

    optuna_dir = Path(str(cfg.paths.optuna_dir))
    trials_frame = study.trials_dataframe()
    trials_path = optuna_dir / "optuna_trials.csv"
    trials_frame.to_csv(trials_path, index=False)

    best_params_path = optuna_dir / "best_params.json"
    best_score_path = optuna_dir / "best_score.json"
    save_json(best_params_path, study.best_params)
    save_json(best_score_path, {"best_validation_f1": study.best_value})

    history_path = optuna_dir / "optimization_history.png"
    save_optimization_history(study, history_path)

    tracker.log_metrics("optuna", {"best_validation_f1": float(study.best_value), "n_trials": int(len(study.trials))})
    tracker.log_text(f"Best params: {study.best_params}")
    tracker.log_artifact_paths(
        {
            "best_params_path": str(best_params_path),
            "best_score_path": str(best_score_path),
            "optuna_trials_path": str(trials_path),
            "optimization_history_path": str(history_path),
        }
    )
    tracker.close()


if __name__ == "__main__":
    main()

