"""Hydra entry point for a small Optuna search."""

from __future__ import annotations

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback

from src.clearml_utils import init_clearml_task
from src.train import run_experiment
from src.utils import get_optuna_dir


class OptunaPruningCallback(TrainerCallback):
    """Report evaluation F1 to Optuna and prune weak trials early."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self.step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if metrics is None or "eval_f1" not in metrics:
            return control
        self.trial.report(float(metrics["eval_f1"]), step=self.step)
        self.step += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at eval step {self.step} with eval_f1={metrics['eval_f1']:.4f}")
        return control


def build_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    """Create the configured Optuna pruner."""
    pruner_cfg = cfg.optuna.pruner
    if str(pruner_cfg.type) == "hyperband":
        max_resource = pruner_cfg.max_resource
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruner_cfg.min_resource),
            max_resource="auto" if str(max_resource) == "auto" else int(max_resource),
            reduction_factor=int(pruner_cfg.reduction_factor),
        )
    if str(pruner_cfg.type) == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported Optuna pruner: {pruner_cfg.type}")


def sample_trial_params(trial: optuna.Trial, cfg: DictConfig) -> dict[str, float]:
    """Sample a small but useful search space for this project."""
    space = cfg.optuna.search_space
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(space.learning_rate.low),
            float(space.learning_rate.high),
            log=bool(space.learning_rate.log),
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            float(space.weight_decay.low),
            float(space.weight_decay.high),
        ),
        "num_train_epochs": trial.suggest_int(
            "num_train_epochs",
            int(space.num_train_epochs.low),
            int(space.num_train_epochs.high),
        ),
        "max_length": trial.suggest_categorical(
            "max_length",
            list(space.max_length.choices),
        ),
    }


def build_trial_cfg(base_cfg: DictConfig, trial_number: int, params: dict[str, float]) -> DictConfig:
    """Create one isolated config for an Optuna trial."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.optuna.enabled = False
    cfg.trainer.learning_rate = params["learning_rate"]
    cfg.trainer.weight_decay = params["weight_decay"]
    cfg.trainer.num_train_epochs = params["num_train_epochs"]
    cfg.trainer.max_length = params["max_length"]
    cfg.paths.output_dir = str(get_optuna_dir(cfg) / f"trial_{trial_number}")
    return cfg

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for Optuna search."""
    tracker = init_clearml_task(cfg)
    study = optuna.create_study(
        direction=str(cfg.optuna.direction),
        sampler=optuna.samplers.TPESampler(seed=int(cfg.seed)),
        pruner=build_pruner(cfg),
        study_name=str(cfg.optuna.study_name),
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial, cfg)
        trial_cfg = build_trial_cfg(cfg, trial.number, params)
        result = run_experiment(trial_cfg, callbacks=[OptunaPruningCallback(trial)])
        trial.set_user_attr("test_f1", float(result["test_f1"]))
        return float(result["validation_f1"])

    study.optimize(objective, n_trials=int(cfg.optuna.n_trials), timeout=int(cfg.optuna.timeout))
    trials_frame = study.trials_dataframe()
    tracker.upload_artifact("optuna_best_params", study.best_params)
    tracker.upload_artifact("optuna_best_score", {"best_validation_f1": study.best_value})
    tracker.log_table("optuna_trials", trials_frame)
    tracker.close()


if __name__ == "__main__":
    main()
