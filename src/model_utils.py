"""Model helpers for loading an adapted token-classification model."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

TARGET_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

SOURCE_TO_TARGET_LABEL = {
    "O": "O",
    "B-PER": "B-PER",
    "I-PER": "I-PER",
    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
}


def get_classifier_layer(model: Any) -> torch.nn.Module:
    """Return the token-classification output layer."""
    for candidate in ("classifier", "score"):
        layer = getattr(model, candidate, None)
        if layer is not None:
            return layer
    raise AttributeError("Could not find a classifier layer on the model.")


def load_tokenizer(model_cfg: object) -> Any:
    """Load the tokenizer for the source checkpoint."""
    return AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        use_fast=bool(model_cfg.use_fast),
    )


def load_adapted_model(model_cfg: object) -> Any:
    """Load the source checkpoint and adapt its head to the reduced label set."""
    target_config = AutoConfig.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        num_labels=len(TARGET_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        finetuning_task="ner",
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        config=target_config,
        ignore_mismatched_sizes=True,
    )

    source_model = AutoModelForTokenClassification.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
    )

    source_classifier = get_classifier_layer(source_model)
    target_classifier = get_classifier_layer(model)

    with torch.no_grad():
        source_label2id = source_model.config.label2id
        for source_label, target_label in SOURCE_TO_TARGET_LABEL.items():
            if source_label not in source_label2id:
                continue
            source_idx = int(source_label2id[source_label])
            target_idx = LABEL2ID[target_label]
            target_classifier.weight[target_idx].copy_(source_classifier.weight[source_idx])
            target_classifier.bias[target_idx].copy_(source_classifier.bias[source_idx])

    return model

