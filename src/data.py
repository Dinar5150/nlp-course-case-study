"""Dataset loading, label mapping, sampling, and tokenization."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.model_utils import LABEL2ID
from src.normalize import normalize_tokens

WNUT_TO_TARGET = {
    "B-person": "B-PER",
    "I-person": "I-PER",
    "B-location": "B-LOC",
    "I-location": "I-LOC",
    "B-corporation": "B-ORG",
    "I-corporation": "I-ORG",
}

SENTENCE_TYPE_PRIORITY = ("ORG", "PER", "LOC")
ENTITY_TYPE_BY_ID = {
    label_id: label_name.split("-", maxsplit=1)[1]
    for label_name, label_id in LABEL2ID.items()
    if label_name != "O"
}


@dataclass
class PreparedDatasets:
    """Prepared raw and tokenized dataset views for an experiment."""

    train_raw: Dataset
    unlabeled_raw: Dataset
    train_full_raw: Dataset
    validation_raw: Dataset
    test_raw: Dataset
    train_tokenized: Dataset | None
    validation_tokenized: Dataset
    test_tokenized: Dataset
    token_counts: Counter[str]
    sampled_example_ids: list[str]


def map_label_name_to_target_id(label_name: str) -> int:
    """Map a WNUT17 label name into the reduced target label space."""
    if label_name == "O":
        return LABEL2ID["O"]
    target_name = WNUT_TO_TARGET.get(label_name)
    if target_name is None:
        return -100
    return LABEL2ID[target_name]


def convert_split_to_reduced_labels(split_dataset: Dataset, split_name: str) -> Dataset:
    """Convert a WNUT17 split into a compact, standardized schema."""
    label_names = split_dataset.features["ner_tags"].feature.names

    def _convert(example: dict[str, Any], index: int) -> dict[str, Any]:
        reduced_tags = [map_label_name_to_target_id(label_names[tag]) for tag in example["ner_tags"]]
        tokens = list(example["tokens"])
        return {
            "example_id": f"{split_name}-{index}",
            "tokens": tokens,
            "original_tokens": tokens,
            "target_ner_tags": reduced_tags,
        }

    return split_dataset.map(
        _convert,
        with_indices=True,
        remove_columns=split_dataset.column_names,
        desc=f"Mapping {split_name} labels",
    )


def maybe_normalize_dataset(split_dataset: Dataset, cfg: object, enabled: bool) -> Dataset:
    """Apply optional normalization to a split."""
    if not enabled:
        return split_dataset

    return split_dataset.map(
        lambda example: {"tokens": normalize_tokens(example["tokens"], cfg)},
        desc="Normalizing tokens",
    )


def get_sentence_entity_types(target_ner_tags: list[int]) -> set[str]:
    """Return the entity types present in a sentence."""
    entity_types: set[str] = set()
    for tag_id in target_ner_tags:
        if tag_id <= 0:
            continue
        entity_types.add(ENTITY_TYPE_BY_ID[int(tag_id)])
    return entity_types


def count_sentence_entities(target_ner_tags: list[int]) -> tuple[int, int]:
    """Count total entities and ORG entities in a sentence."""
    entity_count = 0
    org_entity_count = 0
    previous_tag = 0

    for tag_id in target_ner_tags:
        if tag_id <= 0:
            previous_tag = 0
            continue
        is_entity_start = tag_id % 2 == 1 or tag_id != previous_tag + 1
        if is_entity_start:
            entity_count += 1
            if ENTITY_TYPE_BY_ID[int(tag_id)] == "ORG":
                org_entity_count += 1
        previous_tag = tag_id

    return entity_count, org_entity_count


def score_sentence(
    entity_types: set[str],
    entity_count: int,
    org_entity_count: int,
) -> tuple[int, int, int, int]:
    """Score candidate few-shot sentences, prioritizing ORG-rich entity spans."""
    return (
        org_entity_count,
        int("ORG" in entity_types),
        entity_count,
        len(entity_types),
    )


def sample_fewshot_sentences(train_dataset: Dataset, shot_count: int, seed: int) -> tuple[Dataset, Dataset]:
    """Deterministically sample a sentence-level few-shot subset."""
    if shot_count < 0:
        raise ValueError("shot_count must be non-negative.")
    if shot_count == 0:
        return train_dataset.select([]), train_dataset
    if shot_count > len(train_dataset):
        raise ValueError(f"Requested {shot_count} examples, but only {len(train_dataset)} are available.")

    rng = random.Random(seed)
    shuffled_indices = list(range(len(train_dataset)))
    rng.shuffle(shuffled_indices)

    sentence_types = {
        index: get_sentence_entity_types(train_dataset[index]["target_ner_tags"])
        for index in shuffled_indices
    }
    sentence_entity_counts = {
        index: count_sentence_entities(train_dataset[index]["target_ner_tags"])
        for index in shuffled_indices
    }
    selected: list[int] = []
    selected_set: set[int] = set()

    for entity_type in SENTENCE_TYPE_PRIORITY:
        for index in shuffled_indices:
            if index in selected_set:
                continue
            if entity_type in sentence_types[index]:
                selected.append(index)
                selected_set.add(index)
                break
        if len(selected) >= shot_count:
            break

    remaining_entity_indices = [
        index for index in shuffled_indices if index not in selected_set and sentence_types[index]
    ]
    remaining_entity_indices.sort(
        key=lambda index: score_sentence(
            sentence_types[index],
            entity_count=sentence_entity_counts[index][0],
            org_entity_count=sentence_entity_counts[index][1],
        ),
        reverse=True,
    )
    for index in remaining_entity_indices:
        if len(selected) >= shot_count:
            break
        selected.append(index)
        selected_set.add(index)

    for index in shuffled_indices:
        if len(selected) >= shot_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    selected = sorted(selected)
    remaining = sorted(index for index in range(len(train_dataset)) if index not in selected_set)
    return train_dataset.select(selected), train_dataset.select(remaining)


def tokenize_and_align_labels(examples: dict[str, list[Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Tokenize word-level tokens and align labels to the first sub-token."""
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels: list[list[int]] = []
    for batch_index, word_labels in enumerate(examples["target_ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        labels: list[int] = []

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(int(word_labels[word_id]))
            else:
                labels.append(-100)
            previous_word_id = word_id

        aligned_labels.append(labels)

    tokenized["labels"] = aligned_labels
    return tokenized


def tokenize_dataset(split_dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    """Tokenize a standardized split for token classification."""
    return split_dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=split_dataset.column_names,
        desc="Tokenizing split",
    )


def count_tokens(split_dataset: Dataset) -> Counter[str]:
    """Count lowercase tokens for rare-token heuristics."""
    counter: Counter[str] = Counter()
    for tokens in split_dataset["tokens"]:
        for token in tokens:
            counter[token.lower()] += 1
    return counter


def prepare_datasets(cfg: object, tokenizer: Any) -> PreparedDatasets:
    """Load, map, sample, normalize, and tokenize WNUT17."""
    raw_datasets: DatasetDict = load_dataset(cfg.data.dataset_name, cache_dir=cfg.data.cache_dir)

    processed_splits = {
        split_name: convert_split_to_reduced_labels(raw_datasets[split_name], split_name)
        for split_name in [cfg.data.train_split, cfg.data.validation_split, cfg.data.test_split]
    }

    if bool(cfg.experiment.do_normalization):
        processed_splits = {
            split_name: maybe_normalize_dataset(split_dataset, cfg.normalization, enabled=True)
            for split_name, split_dataset in processed_splits.items()
        }

    train_full_raw = processed_splits[cfg.data.train_split]
    validation_raw = processed_splits[cfg.data.validation_split]
    test_raw = processed_splits[cfg.data.test_split]

    train_raw, unlabeled_raw = sample_fewshot_sentences(
        train_full_raw,
        shot_count=int(cfg.experiment.shot_count),
        seed=int(cfg.seed),
    )

    train_tokenized = None
    if len(train_raw) > 0:
        train_tokenized = tokenize_dataset(train_raw, tokenizer, max_length=int(cfg.trainer.max_length))

    validation_tokenized = tokenize_dataset(validation_raw, tokenizer, max_length=int(cfg.trainer.max_length))
    test_tokenized = tokenize_dataset(test_raw, tokenizer, max_length=int(cfg.trainer.max_length))

    return PreparedDatasets(
        train_raw=train_raw,
        unlabeled_raw=unlabeled_raw,
        train_full_raw=train_full_raw,
        validation_raw=validation_raw,
        test_raw=test_raw,
        train_tokenized=train_tokenized,
        validation_tokenized=validation_tokenized,
        test_tokenized=test_tokenized,
        token_counts=count_tokens(train_full_raw),
        sampled_example_ids=list(train_raw["example_id"]) if len(train_raw) > 0 else [],
    )
