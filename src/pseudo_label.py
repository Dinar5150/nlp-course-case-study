"""Pseudo-label generation and merging for self-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets

from src.model_utils import ID2LABEL
from src.utils import save_jsonl


@dataclass
class PseudoLabelResult:
    """Pseudo-labeling outputs."""

    dataset: Dataset
    records: list[dict[str, Any]]
    skipped_truncated: int


def _empty_pseudo_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "example_id": [],
            "tokens": [],
            "original_tokens": [],
            "target_ner_tags": [],
            "pseudo_confidence": [],
            "is_pseudo": [],
        }
    )


def generate_pseudo_labels(
    raw_dataset: Dataset,
    model: Any,
    tokenizer: Any,
    max_length: int,
    confidence_threshold: float,
    output_path: str,
) -> PseudoLabelResult:
    """Generate sentence-level pseudo labels from a model."""
    if len(raw_dataset) == 0:
        save_jsonl(output_path, [])
        return PseudoLabelResult(dataset=_empty_pseudo_dataset(), records=[], skipped_truncated=0)

    device = next(model.parameters()).device
    model.eval()

    kept_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    skipped_truncated = 0

    for row in raw_dataset:
        encoded = tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        if word_ids:
            max_word_id = max((word_id for word_id in word_ids if word_id is not None), default=-1)
            if max_word_id + 1 < len(row["tokens"]):
                skipped_truncated += 1
                continue

        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        predicted_tags: list[int] = []
        token_confidences: list[float] = []
        seen_word_ids: set[int] = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            label_id = int(np.argmax(probabilities[token_index]))
            predicted_tags.append(label_id)
            token_confidences.append(float(probabilities[token_index][label_id]))

        if len(predicted_tags) != len(row["tokens"]):
            skipped_truncated += 1
            continue

        sentence_confidence = float(np.mean(token_confidences)) if token_confidences else 0.0
        audit_record = {
            "example_id": row["example_id"],
            "tokens": row["original_tokens"],
            "model_tokens": row["tokens"],
            "predicted_labels": [ID2LABEL[label_id] for label_id in predicted_tags],
            "confidence": sentence_confidence,
        }

        if sentence_confidence >= confidence_threshold:
            kept_rows.append(
                {
                    "example_id": f"pseudo-{row['example_id']}",
                    "tokens": row["tokens"],
                    "original_tokens": row["original_tokens"],
                    "target_ner_tags": predicted_tags,
                    "pseudo_confidence": sentence_confidence,
                    "is_pseudo": True,
                }
            )
            audit_record["kept"] = True
        else:
            audit_record["kept"] = False

        audit_rows.append(audit_record)

    save_jsonl(output_path, audit_rows)

    if not kept_rows:
        return PseudoLabelResult(dataset=_empty_pseudo_dataset(), records=audit_rows, skipped_truncated=skipped_truncated)

    pseudo_dataset = Dataset.from_dict(
        {
            "example_id": [row["example_id"] for row in kept_rows],
            "tokens": [row["tokens"] for row in kept_rows],
            "original_tokens": [row["original_tokens"] for row in kept_rows],
            "target_ner_tags": [row["target_ner_tags"] for row in kept_rows],
            "pseudo_confidence": [row["pseudo_confidence"] for row in kept_rows],
            "is_pseudo": [row["is_pseudo"] for row in kept_rows],
        }
    )
    return PseudoLabelResult(dataset=pseudo_dataset, records=audit_rows, skipped_truncated=skipped_truncated)


def attach_gold_dataset_metadata(gold_dataset: Dataset) -> Dataset:
    """Ensure the gold dataset has the same columns as a pseudo-labeled dataset."""
    if len(gold_dataset) == 0:
        return Dataset.from_dict(
            {
                "example_id": [],
                "tokens": [],
                "original_tokens": [],
                "target_ner_tags": [],
                "pseudo_confidence": [],
                "is_pseudo": [],
            }
        )

    return gold_dataset.map(
        lambda _: {"pseudo_confidence": -1.0, "is_pseudo": False},
        desc="Marking gold training rows",
    )


def merge_gold_and_pseudo(gold_dataset: Dataset, pseudo_dataset: Dataset) -> Dataset:
    """Merge gold and pseudo-labeled data."""
    gold_with_metadata = attach_gold_dataset_metadata(gold_dataset)
    if len(pseudo_dataset) == 0:
        return gold_with_metadata
    if len(gold_with_metadata) == 0:
        return pseudo_dataset
    return concatenate_datasets([gold_with_metadata, pseudo_dataset])

