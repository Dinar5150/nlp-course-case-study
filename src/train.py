"""Main Hydra entry point for the NER adaptation pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
import hydra
import torch
from omegaconf import DictConfig
from transformers import DataCollatorForTokenClassification, Trainer, TrainerCallback, TrainingArguments

from src.clearml_utils import init_clearml_task
from src.data import prepare_datasets, tokenize_dataset
from src.metrics import build_compute_metrics, build_error_analysis_dataframe, build_prediction_records, compute_seqeval_metrics
from src.model_utils import ID2LABEL, load_adapted_model, load_tokenizer
from src.pseudo_label import generate_pseudo_labels, merge_gold_and_pseudo
from src.utils import get_output_dir, get_stage_dir, set_global_seed


class ClearMLEpochCallback(TrainerCallback):
    """Log only the most useful per-epoch training curves to ClearML."""

    def __init__(self, tracker: Any, stage_name: str) -> None:
        self.tracker = tracker
        self.stage_name = stage_name

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or state.epoch is None:
            return control
        epoch = int(round(float(state.epoch)))
        if "loss" in logs:
            self.tracker.log_scalar("train_epoch", f"{self.stage_name}_loss", float(logs["loss"]), iteration=epoch)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not metrics or state.epoch is None:
            return control
        epoch = int(round(float(state.epoch)))
        if "eval_f1" in metrics:
            self.tracker.log_scalar("eval_epoch", f"{self.stage_name}_f1", float(metrics["eval_f1"]), iteration=epoch)
        if "eval_loss" in metrics:
            self.tracker.log_scalar("eval_epoch", f"{self.stage_name}_loss", float(metrics["eval_loss"]), iteration=epoch)
        return control


def build_training_arguments(cfg: DictConfig, output_dir: str, do_train: bool, num_train_epochs: int | None = None) -> TrainingArguments:
    """Create a compact set of Trainer arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        num_train_epochs=float(num_train_epochs or cfg.trainer.num_train_epochs),
        per_device_train_batch_size=int(cfg.trainer.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.trainer.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        eval_strategy="epoch" if do_train else "no",
        save_strategy="epoch" if do_train else "no",
        load_best_model_at_end=do_train,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        fp16=bool(cfg.trainer.fp16) and torch.cuda.is_available(),
        logging_strategy="epoch" if do_train else "no",
        save_total_limit=2,
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
        report_to=[],
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    output_dir: str,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
    num_train_epochs: int | None = None,
    callbacks: list[TrainerCallback] | None = None,
) -> Trainer:
    """Create a token-classification Trainer."""
    return Trainer(
        model=model,
        args=build_training_arguments(cfg, output_dir, do_train=train_dataset is not None, num_train_epochs=num_train_epochs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(ID2LABEL),
        callbacks=callbacks,
    )


def evaluate_dataset(model: Any, tokenizer: Any, cfg: DictConfig, dataset: Any, output_dir: str) -> tuple[Any, dict[str, float], dict[str, dict[str, float]]]:
    """Run evaluation and decode metrics."""
    trainer = build_trainer(model, tokenizer, cfg, output_dir, eval_dataset=dataset)
    prediction_output = trainer.predict(dataset)
    metrics, per_label_metrics, _, _ = compute_seqeval_metrics(
        prediction_output.predictions,
        prediction_output.label_ids,
        ID2LABEL,
    )
    return prediction_output, metrics, per_label_metrics


def run_experiment(cfg: DictConfig, callbacks: list[TrainerCallback] | None = None) -> dict[str, Any]:
    """Run one experiment and send experiment outputs to ClearML."""
    set_global_seed(int(cfg.seed))
    tracker = init_clearml_task(cfg)
    try:
        output_dir = get_output_dir(cfg)
        tokenizer = load_tokenizer(cfg.model)
        model = load_adapted_model(cfg.model)
        prepared = prepare_datasets(cfg, tokenizer)
        tracker.upload_artifact(
            "sampled_example_ids",
            {
                "seed": int(cfg.seed),
                "shot_count": int(cfg.experiment.shot_count),
                "example_ids": prepared.sampled_example_ids,
            },
        )

        training_metrics: dict[str, float] = {}
        best_checkpoint = str(cfg.model.model_name_or_path)
        pseudo_examples = 0

        if bool(cfg.experiment.do_finetune):
            if prepared.train_tokenized is None or len(prepared.train_raw) == 0:
                raise ValueError("The training set is empty.")
            trainer = build_trainer(
                model,
                tokenizer,
                cfg,
                str(get_stage_dir(cfg, "train")),
                train_dataset=prepared.train_tokenized,
                eval_dataset=prepared.validation_tokenized,
                num_train_epochs=int(cfg.trainer.num_train_epochs),
                callbacks=[ClearMLEpochCallback(tracker, "train"), *(callbacks or [])],
            )
            training_metrics = trainer.train().metrics
            model = trainer.model
            best_checkpoint = trainer.state.best_model_checkpoint or str(get_stage_dir(cfg, "train"))

        if bool(cfg.experiment.do_self_training):
            pseudo_result = generate_pseudo_labels(
                raw_dataset=prepared.unlabeled_raw,
                model=model,
                tokenizer=tokenizer,
                max_length=int(cfg.trainer.max_length),
                confidence_threshold=float(cfg.experiment.pseudo_label_threshold),
            )
            pseudo_examples = len(pseudo_result.dataset)
            tracker.upload_artifact("pseudo_labels", pseudo_result.records)
            merged_raw = merge_gold_and_pseudo(prepared.train_raw, pseudo_result.dataset)
            if len(merged_raw) > 0:
                merged_tokenized = tokenize_dataset(merged_raw, tokenizer, max_length=int(cfg.trainer.max_length))
                trainer = build_trainer(
                    model,
                    tokenizer,
                    cfg,
                    str(get_stage_dir(cfg, "selftrain")),
                    train_dataset=merged_tokenized,
                    eval_dataset=prepared.validation_tokenized,
                    num_train_epochs=int(cfg.experiment.self_training_epochs),
                    callbacks=[ClearMLEpochCallback(tracker, "selftrain"), *(callbacks or [])],
                )
                training_metrics.update({f"selftrain_{key}": value for key, value in trainer.train().metrics.items()})
                model = trainer.model
                best_checkpoint = trainer.state.best_model_checkpoint or str(get_stage_dir(cfg, "selftrain"))

        validation_prediction, validation_metrics, validation_per_label = evaluate_dataset(
            model,
            tokenizer,
            cfg,
            prepared.validation_tokenized,
            str(output_dir / "validation_eval"),
        )
        test_prediction, test_metrics, test_per_label = evaluate_dataset(
            model,
            tokenizer,
            cfg,
            prepared.test_tokenized,
            str(output_dir / "test_eval"),
        )

        predictions = build_prediction_records(prepared.test_raw, test_prediction.predictions, test_prediction.label_ids, ID2LABEL)
        error_analysis = build_error_analysis_dataframe(
            raw_dataset=prepared.test_raw,
            predictions=test_prediction.predictions,
            labels=test_prediction.label_ids,
            id2label=ID2LABEL,
            token_counts=prepared.token_counts,
            rare_token_threshold=int(cfg.error_analysis.rare_token_threshold),
            max_rows=int(cfg.error_analysis.max_error_rows),
        )
        error_counts = {
            "boundary": int(error_analysis["boundary_error"].fillna(False).astype(bool).sum()) if not error_analysis.empty else 0,
            "label_confusion": int(error_analysis["label_confusion"].fillna(False).astype(bool).sum()) if not error_analysis.empty else 0,
            "noisy_token": int(error_analysis["noisy_token_issue"].fillna(False).astype(bool).sum()) if not error_analysis.empty else 0,
            "oov_rare": int(error_analysis["oov_rare_issue"].fillna(False).astype(bool).sum()) if not error_analysis.empty else 0,
        }

        tracker.log_scalar("final", "validation_f1", float(validation_metrics["f1"]))
        tracker.log_scalar("final", "test_f1", float(test_metrics["f1"]))
        tracker.log_scalar("data", "train_examples", int(len(prepared.train_raw)))
        tracker.log_scalar("data", "pseudo_examples", int(pseudo_examples))
        tracker.log_metrics("error_types", error_counts)
        tracker.upload_artifact(
            "run_summary",
            {
                "validation": validation_metrics,
                "test": test_metrics,
                "training": training_metrics,
                "error_counts": error_counts,
                "validation_per_label": validation_per_label,
                "test_per_label": test_per_label,
                "train_examples": int(len(prepared.train_raw)),
                "pseudo_examples": int(pseudo_examples),
                "best_checkpoint": best_checkpoint,
            },
        )
        tracker.upload_artifact("predictions", predictions)
        tracker.log_table("error_analysis_sample", error_analysis)
        tracker.log_table(
            "per_label_metrics",
            pd.DataFrame(
                [
                    {"split": split_name, "label": label_name, **label_metrics}
                    for split_name, split_metrics in {"validation": validation_per_label, "test": test_per_label}.items()
                    for label_name, label_metrics in split_metrics.items()
                ]
            ),
        )
        return {
            "validation_f1": float(validation_metrics["f1"]),
            "test_f1": float(test_metrics["f1"]),
            "pseudo_examples": int(pseudo_examples),
            "train_examples": int(len(prepared.train_raw)),
            "best_checkpoint": best_checkpoint,
        }
    finally:
        tracker.close()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
