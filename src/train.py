"""Main Hydra entry point for the NER-only adaptation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from src.clearml_utils import ClearMLTracker, init_clearml_task
from src.data import prepare_datasets, tokenize_dataset
from src.metrics import build_compute_metrics, build_error_analysis_dataframe, build_prediction_records, compute_seqeval_metrics
from src.model_utils import ID2LABEL, load_adapted_model, load_tokenizer
from src.pseudo_label import generate_pseudo_labels, merge_gold_and_pseudo
from src.reporting import build_run_summary_row, generate_case_study_artifacts
from src.utils import ensure_dir, save_dataframe, save_json, save_jsonl, set_global_seed, write_text


def build_training_arguments(
    cfg: DictConfig,
    output_dir: str,
    *,
    for_training: bool,
    num_train_epochs: int | None = None,
) -> TrainingArguments:
    """Create Trainer arguments from the Hydra config."""
    use_fp16 = bool(cfg.trainer.fp16) and torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(ensure_dir(output_dir)),
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        num_train_epochs=float(num_train_epochs or cfg.trainer.num_train_epochs),
        warmup_ratio=float(cfg.trainer.warmup_ratio),
        per_device_train_batch_size=int(cfg.trainer.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.trainer.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        eval_strategy=str(cfg.trainer.eval_strategy) if for_training else "no",
        save_strategy=str(cfg.trainer.save_strategy) if for_training else "no",
        logging_strategy="steps",
        logging_steps=int(cfg.trainer.logging_steps),
        save_total_limit=int(cfg.trainer.save_total_limit),
        load_best_model_at_end=bool(cfg.trainer.load_best_model_at_end) if for_training else False,
        metric_for_best_model=str(cfg.trainer.metric_for_best_model),
        greater_is_better=bool(cfg.trainer.greater_is_better),
        label_smoothing_factor=float(cfg.trainer.label_smoothing_factor),
        gradient_checkpointing=bool(cfg.trainer.gradient_checkpointing),
        fp16=use_fp16,
        dataloader_num_workers=int(cfg.trainer.dataloader_num_workers),
        report_to=[],
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    output_dir: str,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
    for_training: bool,
    num_train_epochs: int | None = None,
) -> Trainer:
    """Create a Hugging Face Trainer."""
    if bool(cfg.trainer.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return Trainer(
        model=model,
        args=build_training_arguments(
            cfg,
            output_dir=output_dir,
            for_training=for_training,
            num_train_epochs=num_train_epochs,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(ID2LABEL),
    )


def train_stage(
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str,
    *,
    num_train_epochs: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> tuple[Trainer, dict[str, float]]:
    """Run a single training stage and return the trainer plus train metrics."""
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        for_training=True,
        num_train_epochs=num_train_epochs,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer, train_result.metrics


def evaluate_split(
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    tokenized_dataset: Any,
    output_dir: str,
) -> tuple[Any, dict[str, float], dict[str, dict[str, float]]]:
    """Run prediction on a split and compute decoded metrics."""
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=output_dir,
        eval_dataset=tokenized_dataset,
        for_training=False,
    )
    prediction_output = trainer.predict(tokenized_dataset)
    metrics, per_label_metrics, _, _ = compute_seqeval_metrics(
        prediction_output.predictions,
        prediction_output.label_ids,
        ID2LABEL,
    )
    return prediction_output, metrics, per_label_metrics


def maybe_log_clearml_summary(tracker: ClearMLTracker, cfg: DictConfig, metrics: dict[str, Any]) -> None:
    """Log compact experiment metadata to ClearML when enabled."""
    if not tracker.enabled:
        return

    tracker.log_text(
        "\n".join(
            [
                f"model_name={cfg.model.model_name_or_path}",
                f"dataset={cfg.data.dataset_name}",
                f"experiment={cfg.experiment.name}",
                f"shot_count={int(cfg.experiment.shot_count)}",
                f"seed={int(cfg.seed)}",
                f"normalization={bool(cfg.experiment.do_normalization)}",
                f"self_training={bool(cfg.experiment.do_self_training)}",
                f"pseudo_label_threshold={float(cfg.experiment.pseudo_label_threshold)}",
            ]
        )
    )
    tracker.log_metrics("summary", metrics)


def run_experiment(
    cfg: DictConfig,
    *,
    init_tracking: bool = True,
    generate_reports: bool = True,
) -> dict[str, Any]:
    """Run one full experiment and write local artifacts."""
    ensure_dir(cfg.paths.output_dir)
    ensure_dir(cfg.paths.train_stage_dir)
    ensure_dir(cfg.paths.selftrain_stage_dir)
    set_global_seed(int(cfg.seed))

    tracker = init_clearml_task(cfg) if init_tracking else ClearMLTracker(task=None)
    tokenizer = load_tokenizer(cfg.model)
    model = load_adapted_model(cfg.model)
    prepared = prepare_datasets(cfg, tokenizer)

    pseudo_examples = 0
    training_metrics: dict[str, float] = {}
    best_checkpoint = str(cfg.model.model_name_or_path)

    if bool(cfg.experiment.do_finetune):
        if prepared.train_tokenized is None or len(prepared.train_raw) == 0:
            raise ValueError("do_finetune is enabled, but the few-shot training set is empty.")
        train_trainer, training_metrics = train_stage(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            train_dataset=prepared.train_tokenized,
            eval_dataset=prepared.validation_tokenized,
            output_dir=cfg.paths.train_stage_dir,
            num_train_epochs=int(cfg.trainer.num_train_epochs),
            resume_from_checkpoint=cfg.trainer.resume_from_checkpoint,
        )
        model = train_trainer.model
        best_checkpoint = train_trainer.state.best_model_checkpoint or str(Path(cfg.paths.train_stage_dir))

    if bool(cfg.experiment.do_self_training):
        pseudo_result = generate_pseudo_labels(
            raw_dataset=prepared.unlabeled_raw,
            model=model,
            tokenizer=tokenizer,
            max_length=int(cfg.trainer.max_length),
            confidence_threshold=float(cfg.experiment.pseudo_label_threshold),
            output_path=cfg.paths.pseudo_labels_path,
        )
        pseudo_examples = len(pseudo_result.dataset)
        merged_raw = merge_gold_and_pseudo(prepared.train_raw, pseudo_result.dataset)

        if len(merged_raw) > 0:
            merged_tokenized = tokenize_dataset(merged_raw, tokenizer, max_length=int(cfg.trainer.max_length))
            selftrain_epochs = int(cfg.experiment.self_training_epochs)
            selftrain_trainer, selftrain_metrics = train_stage(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                train_dataset=merged_tokenized,
                eval_dataset=prepared.validation_tokenized,
                output_dir=cfg.paths.selftrain_stage_dir,
                num_train_epochs=selftrain_epochs if selftrain_epochs > 0 else int(cfg.trainer.num_train_epochs),
                resume_from_checkpoint=None,
            )
            training_metrics.update({f"selftrain_{key}": value for key, value in selftrain_metrics.items()})
            model = selftrain_trainer.model
            best_checkpoint = selftrain_trainer.state.best_model_checkpoint or str(Path(cfg.paths.selftrain_stage_dir))
    else:
        save_jsonl(cfg.paths.pseudo_labels_path, [])

    validation_prediction, validation_metrics, validation_per_label = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        tokenized_dataset=prepared.validation_tokenized,
        output_dir=str(Path(cfg.paths.output_dir) / "validation_eval"),
    )
    test_prediction, test_metrics, test_per_label = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        tokenized_dataset=prepared.test_tokenized,
        output_dir=str(Path(cfg.paths.output_dir) / "test_eval"),
    )

    prediction_records = build_prediction_records(
        raw_dataset=prepared.test_raw,
        predictions=test_prediction.predictions,
        labels=test_prediction.label_ids,
        id2label=ID2LABEL,
    )
    error_frame = build_error_analysis_dataframe(
        raw_dataset=prepared.test_raw,
        predictions=test_prediction.predictions,
        labels=test_prediction.label_ids,
        id2label=ID2LABEL,
        token_counts=prepared.token_counts,
        rare_token_threshold=int(cfg.reporting.rare_token_threshold),
        max_rows=int(cfg.reporting.max_error_rows),
    )

    metrics_payload = {
        "validation": validation_metrics,
        "test": test_metrics,
        "training": training_metrics,
        "train_examples": int(len(prepared.train_raw)),
        "pseudo_examples": int(pseudo_examples),
        "best_checkpoint": best_checkpoint,
    }
    per_label_payload = {
        "validation": validation_per_label,
        "test": test_per_label,
    }

    save_jsonl(cfg.paths.predictions_path, prediction_records)
    save_json(cfg.paths.metrics_path, metrics_payload)
    save_json(cfg.paths.per_label_metrics_path, per_label_payload)
    save_dataframe(cfg.paths.error_analysis_path, error_frame)
    write_text(cfg.paths.best_checkpoint_path, best_checkpoint)

    summary_frame = build_run_summary_row(
        cfg=cfg,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_checkpoint=best_checkpoint,
        train_examples=len(prepared.train_raw),
        pseudo_examples=pseudo_examples,
    )
    save_dataframe(cfg.paths.results_summary_path, summary_frame)

    report_summary: dict[str, Any] = {}
    if generate_reports:
        report_summary = generate_case_study_artifacts(cfg)

    tracker.log_metrics("validation", validation_metrics)
    tracker.log_metrics("test", test_metrics)
    maybe_log_clearml_summary(
        tracker,
        cfg,
        {
            "train_examples": len(prepared.train_raw),
            "pseudo_examples": pseudo_examples,
            "validation_f1": validation_metrics["f1"],
            "test_f1": test_metrics["f1"],
        },
    )
    tracker.log_artifact_paths(
        {
            "predictions_path": str(Path(cfg.paths.predictions_path)),
            "metrics_path": str(Path(cfg.paths.metrics_path)),
            "per_label_metrics_path": str(Path(cfg.paths.per_label_metrics_path)),
            "error_analysis_path": str(Path(cfg.paths.error_analysis_path)),
            "results_summary_path": str(Path(cfg.paths.results_summary_path)),
            "case_study_summary_path": str(Path(cfg.paths.case_study_summary_path)),
        }
    )
    tracker.close()

    return {
        "validation_f1": float(validation_metrics["f1"]),
        "test_f1": float(test_metrics["f1"]),
        "pseudo_examples": int(pseudo_examples),
        "train_examples": int(len(prepared.train_raw)),
        "best_checkpoint": best_checkpoint,
        **report_summary,
    }


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
