"""Metrics, decoding, and lightweight error analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
import pandas as pd
from seqeval.scheme import IOB2

from src.normalize import is_social_noise_token

_SEQEVAL_METRIC = None


@dataclass
class ErrorFlags:
    """Simple boolean error-type flags for a sentence-level mismatch."""

    boundary_error: bool
    label_confusion: bool
    noisy_token_issue: bool
    oov_rare_issue: bool


def get_seqeval_metric() -> Any:
    """Load the Hugging Face Evaluate seqeval wrapper once."""
    global _SEQEVAL_METRIC
    if _SEQEVAL_METRIC is None:
        _SEQEVAL_METRIC = evaluate.load("seqeval")
    return _SEQEVAL_METRIC


def decode_prediction_sequences(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Decode logits or label IDs into filtered word-level label sequences."""
    if predictions.ndim == 3:
        predicted_ids = predictions.argmax(axis=2)
    else:
        predicted_ids = predictions

    decoded_predictions: list[list[str]] = []
    decoded_labels: list[list[str]] = []

    for prediction_row, label_row in zip(predicted_ids, labels):
        sentence_predictions: list[str] = []
        sentence_labels: list[str] = []
        for predicted_id, gold_id in zip(prediction_row, label_row):
            if int(gold_id) == -100:
                continue
            sentence_predictions.append(id2label[int(predicted_id)])
            sentence_labels.append(id2label[int(gold_id)])
        decoded_predictions.append(sentence_predictions)
        decoded_labels.append(sentence_labels)

    return decoded_predictions, decoded_labels


def compute_seqeval_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[list[str]], list[list[str]]]:
    """Compute strict entity-level metrics and per-label metrics."""
    decoded_predictions, decoded_labels = decode_prediction_sequences(predictions, labels, id2label)
    metric = get_seqeval_metric()

    results = metric.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        mode="strict",
        scheme=IOB2,
    )
    overall = {
        "precision": float(results["overall_precision"]),
        "recall": float(results["overall_recall"]),
        "f1": float(results["overall_f1"]),
        "accuracy": float(results["overall_accuracy"]),
    }
    per_label = {
        label: {
            "precision": float(results.get(label, {}).get("precision", 0.0)),
            "recall": float(results.get(label, {}).get("recall", 0.0)),
            "f1": float(results.get(label, {}).get("f1", 0.0)),
            "number": int(results.get(label, {}).get("number", 0)),
        }
        for label in ("PER", "LOC", "ORG")
    }
    return overall, per_label, decoded_predictions, decoded_labels


def build_compute_metrics(id2label: dict[int, str]) -> Any:
    """Create a Trainer-compatible compute_metrics callback."""

    def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_prediction
        metrics, _, _, _ = compute_seqeval_metrics(predictions, labels, id2label)
        return metrics

    return compute_metrics


def extract_entities(labels: list[str]) -> list[dict[str, Any]]:
    """Extract simple entity spans from an IOB2 sequence."""
    entities: list[dict[str, Any]] = []
    current_type: str | None = None
    current_start: int | None = None

    for index, tag in enumerate(labels + ["O"]):
        if tag == "O":
            if current_type is not None and current_start is not None:
                entities.append({"start": current_start, "end": index - 1, "label": current_type})
                current_type = None
                current_start = None
            continue

        prefix, entity_type = tag.split("-", maxsplit=1)

        if prefix == "B" or current_type != entity_type:
            if current_type is not None and current_start is not None:
                entities.append({"start": current_start, "end": index - 1, "label": current_type})
            current_type = entity_type
            current_start = index

    return entities


def spans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Check whether two spans overlap."""
    return left["start"] <= right["end"] and right["start"] <= left["end"]


def flag_sentence_errors(
    tokens: list[str],
    gold_labels: list[str],
    pred_labels: list[str],
    token_counts: dict[str, int],
    rare_token_threshold: int,
) -> ErrorFlags:
    """Assign simple sentence-level error flags."""
    gold_entities = extract_entities(gold_labels)
    pred_entities = extract_entities(pred_labels)
    mismatched_tokens = [token for token, gold, pred in zip(tokens, gold_labels, pred_labels) if gold != pred]

    boundary_error = any(
        gold_entity["label"] == pred_entity["label"]
        and spans_overlap(gold_entity, pred_entity)
        and (gold_entity["start"], gold_entity["end"]) != (pred_entity["start"], pred_entity["end"])
        for gold_entity in gold_entities
        for pred_entity in pred_entities
    )

    label_confusion = any(
        (gold_entity["start"], gold_entity["end"]) == (pred_entity["start"], pred_entity["end"])
        and gold_entity["label"] != pred_entity["label"]
        for gold_entity in gold_entities
        for pred_entity in pred_entities
    ) or any(
        gold_entity["label"] != pred_entity["label"] and spans_overlap(gold_entity, pred_entity)
        for gold_entity in gold_entities
        for pred_entity in pred_entities
    )

    noisy_token_issue = any(is_social_noise_token(token) for token in mismatched_tokens)
    oov_rare_issue = any(token_counts.get(token.lower(), 0) <= rare_token_threshold for token in mismatched_tokens)

    return ErrorFlags(
        boundary_error=boundary_error,
        label_confusion=label_confusion,
        noisy_token_issue=noisy_token_issue,
        oov_rare_issue=oov_rare_issue,
    )


def choose_primary_error_type(flags: ErrorFlags) -> str:
    """Choose a single readable error label for CSV export."""
    if flags.boundary_error:
        return "boundary"
    if flags.label_confusion:
        return "label_confusion"
    if flags.noisy_token_issue:
        return "noisy_token"
    if flags.oov_rare_issue:
        return "oov_rare"
    return "other"


def build_prediction_records(
    raw_dataset: Any,
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Export decoded predictions for auditability."""
    decoded_predictions, decoded_labels = decode_prediction_sequences(predictions, labels, id2label)
    records: list[dict[str, Any]] = []

    for index, (predicted_labels, gold_labels) in enumerate(zip(decoded_predictions, decoded_labels)):
        row = raw_dataset[index]
        records.append(
            {
                "example_id": row["example_id"],
                "tokens": row["original_tokens"],
                "model_tokens": row["tokens"],
                "gold_labels": gold_labels,
                "predicted_labels": predicted_labels,
            }
        )

    return records


def build_error_analysis_dataframe(
    raw_dataset: Any,
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
    token_counts: dict[str, int],
    rare_token_threshold: int,
    max_rows: int,
) -> pd.DataFrame:
    """Create a compact sentence-level error analysis table."""
    columns = [
        "example_id",
        "token_sequence",
        "model_token_sequence",
        "gold_labels",
        "predicted_labels",
        "mismatch_tokens",
        "mismatch_count",
        "boundary_error",
        "label_confusion",
        "noisy_token_issue",
        "oov_rare_issue",
        "primary_error_type",
    ]
    decoded_predictions, decoded_labels = decode_prediction_sequences(predictions, labels, id2label)
    rows: list[dict[str, Any]] = []

    for index, (predicted_labels, gold_labels) in enumerate(zip(decoded_predictions, decoded_labels)):
        if predicted_labels == gold_labels:
            continue

        raw_row = raw_dataset[index]
        model_tokens = list(raw_row["tokens"])
        original_tokens = list(raw_row["original_tokens"])
        mismatch_tokens = [token for token, gold, pred in zip(model_tokens, gold_labels, predicted_labels) if gold != pred]
        flags = flag_sentence_errors(
            tokens=model_tokens,
            gold_labels=gold_labels,
            pred_labels=predicted_labels,
            token_counts=token_counts,
            rare_token_threshold=rare_token_threshold,
        )
        rows.append(
            {
                "example_id": raw_row["example_id"],
                "token_sequence": " ".join(original_tokens),
                "model_token_sequence": " ".join(model_tokens),
                "gold_labels": " ".join(gold_labels),
                "predicted_labels": " ".join(predicted_labels),
                "mismatch_tokens": " ".join(mismatch_tokens),
                "mismatch_count": sum(gold != pred for gold, pred in zip(gold_labels, predicted_labels)),
                "boundary_error": flags.boundary_error,
                "label_confusion": flags.label_confusion,
                "noisy_token_issue": flags.noisy_token_issue,
                "oov_rare_issue": flags.oov_rare_issue,
                "primary_error_type": choose_primary_error_type(flags),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows[:max_rows], columns=columns)
