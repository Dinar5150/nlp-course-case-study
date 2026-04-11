# NER-Only Domain Adaptation Case Study

This repository is a compact, reproducible research repo for a **news-to-social domain-adaptation case study in named entity recognition (NER)**. It is the **NER-only implementation** of a broader POS/NER course brief, intentionally scoped down to keep the project feasible in roughly 4 to 5 weeks for a student project.

The setup transfers a news-trained NER model,
`philschmid/distilroberta-base-ner-conll2003`,
to noisy social-media text from **WNUT17**. The goal is not to build a large framework. The goal is to make a clean baseline, implement a few practical adaptation recipes, and generate enough artifacts to support a short case-study report.

## What the repo covers

- Source-only baseline: evaluate the news-trained model on WNUT17 without any target-domain updates.
- Few-shot adaptation: continue fine-tuning on 50, 100, or 250 WNUT17 train sentences.
- Normalization: apply lightweight, transparent text normalization before tokenization.
- Self-training: pseudo-label unlabeled target-domain sentences and merge high-confidence examples with the few-shot gold set.
- Combined recipe: few-shot + normalization + self-training.
- Reporting: compare runs, compute in-domain gains over the source-only baseline, and summarize which error types decrease.

## NER-only scope

This repo does **not** implement POS tagging. It focuses only on **NER**, because the coursework goal here is a manageable domain-adaptation case study with clear, reproducible experiments.

## Task framing

- Source domain: news text, represented by a CoNLL03-trained checkpoint.
- Target domain: WNUT17 social-media text.
- Overlapping label space only:
  - `person -> PER`
  - `location -> LOC`
  - `corporation -> ORG`
- Ignored WNUT17 labels:
  - `creative-work`
  - `group`
  - `product`

Unsupported WNUT labels are set to `-100` during training and excluded from evaluation.

Implementation note: the current `datasets` package no longer supports the legacy scripted `wnut_17` loader. The repo therefore uses the parquet-backed Hugging Face mirror `flaitenberger/wnut_17`, which preserves the standard WNUT17 train, validation, and test splits.

## Repo tree

```text
repo/
  README.md
  requirements.txt
  .gitignore
  conf/
    config.yaml
    data/
      wnut17.yaml
    model/
      distilroberta_ner.yaml
    trainer/
      default.yaml
      low_vram.yaml
    experiment/
      source_only.yaml
      fewshot_50.yaml
      fewshot_100.yaml
      fewshot_250.yaml
      selftrain.yaml
      normalize.yaml
      combined.yaml
    optuna/
      default.yaml
  src/
    __init__.py
    train.py
    data.py
    metrics.py
    model_utils.py
    normalize.py
    pseudo_label.py
    optuna_search.py
    clearml_utils.py
    reporting.py
    utils.py
  scripts/
    run_kaggle.sh
    run_source_only.sh
    run_fewshot.sh
    run_selftrain.sh
    run_optuna.sh
  notebooks/
    01_error_analysis.ipynb
```

## Installation

Python 3.10+ is assumed.

If you are using the requested local environment:

```bash
conda activate machine-learning
pip install -r requirements.txt
```

If you prefer not to activate the environment directly:

```bash
conda run -n machine-learning python -m pip install -r requirements.txt
```

## Kaggle workflow

This repo is designed to run from a Kaggle notebook shell or a notebook cell with `%%bash`.

### 1. Clone and install

```bash
git clone <your-repo-url>
cd nlp-course-case-study
pip install -r requirements.txt
```

### 2. Optional ClearML environment variables

```bash
export CLEARML_API_ACCESS_KEY="..."
export CLEARML_API_SECRET_KEY="..."
export CLEARML_API_HOST="https://api.clear.ml"
export CLEARML_WEB_HOST="https://app.clear.ml"
export CLEARML_FILES_HOST="https://files.clear.ml"
```

If these are missing and local continuation is allowed by config, the code falls back to local-only execution without crashing.

### 3. Run a baseline

```bash
bash scripts/run_kaggle.sh
```

### 4. Run follow-up experiments

```bash
python -m src.train experiment=fewshot_100 seed=1
python -m src.train experiment=selftrain seed=1 trainer=low_vram
python -m src.optuna_search experiment=fewshot_100 optuna.n_trials=10 trainer=low_vram
```

## Hydra workflow

The project is CLI-first. The main entry point is:

```bash
python -m src.train experiment=source_only seed=1
```

Common overrides:

```bash
python -m src.train experiment=fewshot_100 seed=2
python -m src.train experiment=fewshot_100 seed=2 trainer.fp16=true
python -m src.train experiment=fewshot_100 seed=2 trainer.max_length=128
python -m src.train experiment=selftrain seed=1 experiment.do_self_training=true
python -m src.train experiment=normalize seed=1 experiment.do_normalization=true
```

The Hydra run directory is readable and stable:

```text
outputs/<experiment_name>/seed_<seed>
```

## Core commands

```bash
python -m src.train experiment=source_only seed=1
python -m src.train experiment=fewshot_50 seed=1
python -m src.train experiment=fewshot_100 seed=1 trainer=low_vram
python -m src.train experiment=selftrain seed=1 experiment.pseudo_label_threshold=0.95
python -m src.train experiment=normalize seed=1
python -m src.train experiment=combined seed=1
python -m src.optuna_search experiment=fewshot_100 optuna.n_trials=10 trainer=low_vram
```

## How the source-only baseline is defined

The source-only baseline:

- loads `philschmid/distilroberta-base-ner-conll2003`
- adapts the classifier head to the reduced label set `O, PER, LOC, ORG`
- copies overlapping classifier weights from the source checkpoint
- performs **no target-domain parameter updates**
- evaluates directly on WNUT17 validation and test sets

This gives the zero-shot cross-domain baseline that every adapted system should beat.

## How label mapping works

WNUT17 labels are reduced to:

- `O`
- `B-PER`, `I-PER`
- `B-LOC`, `I-LOC`
- `B-ORG`, `I-ORG`

Mappings:

- `B-person -> B-PER`
- `I-person -> I-PER`
- `B-location -> B-LOC`
- `I-location -> I-LOC`
- `B-corporation -> B-ORG`
- `I-corporation -> I-ORG`

Ignored:

- `creative-work`
- `group`
- `product`

Ignored labels are converted to `-100`, which removes them from loss computation and evaluation.

## How few-shot sampling works

Few-shot sampling is done by **sentence count** on the WNUT17 train split.

- `fewshot_50` samples 50 train sentences
- `fewshot_100` samples 100 train sentences
- `fewshot_250` samples 250 train sentences

Sampling is deterministic for a given seed. Every run saves the selected example IDs to disk so the exact sample can be reused or inspected later.

## Normalization recipe

Normalization is deliberately lightweight and transparent:

- usernames like `@john123` become `@USER`
- URLs become `HTTPURL`
- repeated characters are collapsed conservatively, for example `coooool -> cool`
- optional hashtag normalization can map `#Topic` to `Topic`

Normalization preserves token boundaries as much as possible because the goal is practical preprocessing, not aggressive text rewriting.

## Self-training recipe

Self-training works like this:

1. Fine-tune on a small gold WNUT17 subset.
2. Predict labels for the remaining WNUT17 train examples.
3. Keep only pseudo-labeled sentences above a configurable confidence threshold.
4. Merge pseudo-labeled sentences with the gold few-shot set.
5. Continue training on the merged set.

Pseudo-labeled examples are saved to disk for auditability.

## ClearML setup

ClearML is optional at runtime.

If enabled and credentials are available, the code logs:

- resolved Hydra config
- key metrics
- artifact paths
- model name
- dataset setup
- shot count
- seed
- pseudo-label threshold
- normalization flag
- final comparison metrics

If ClearML is unavailable or credentials are missing:

- the run prints a warning
- local outputs still get written
- execution continues if `clearml.allow_no_credentials=true`

Offline mode is supported by config or `CLEARML_OFFLINE_MODE=1`.

## Offline mode behavior

Offline mode only affects ClearML tracking. It does **not** make model or dataset downloads fully offline by itself. If the Hugging Face model and WNUT17 dataset are not already cached, the first run still needs network access.

## Optuna workflow

Optuna is kept intentionally small and practical. The default search space covers:

- `learning_rate`
- `weight_decay`
- `num_train_epochs`
- `warmup_ratio`
- `per_device_train_batch_size`
- `max_length`
- `pseudo_label_threshold`

The default objective is validation F1. Search artifacts are saved under the Optuna output directory:

- `best_params.json`
- `best_score.json`
- `optuna_trials.csv`
- `optimization_history.png` when plotting works

Example:

```bash
python -m src.optuna_search experiment=fewshot_100 optuna.n_trials=10 trainer=low_vram
```

## Expected outputs

Each run saves the following under `outputs/<experiment>/seed_<seed>/`:

- `predictions.jsonl`
- `metrics.json`
- `per_label_metrics.json`
- `error_analysis_sample.csv`
- `results_summary.csv`
- `best_checkpoint.txt`
- `sampled_example_ids.json`
- `pseudo_labels.jsonl` when self-training is enabled
- `results_comparison.csv`
- `error_type_reduction.csv`
- `case_study_summary.md`

## How in-domain gains are computed

For every adapted experiment, the reporting code compares test F1 to the **source-only run with the same seed**.

- absolute gain = `experiment_test_f1 - source_only_test_f1`
- relative gain percent = `100 * absolute_gain / max(source_only_test_f1, 1e-8)`

## How error-type reduction is computed

Each run exports sentence-level error rows with a simple heuristic error type:

- boundary error
- label confusion
- noisy-token issue
- OOV or rare-token issue

For each experiment, reduction is computed relative to the matching-seed source-only run:

- absolute reduction = `source_only_count - experiment_count`
- relative reduction percent = `100 * absolute_reduction / max(source_only_count, 1)`

The heuristics are intentionally simple and documented in code. This is enough for a compact course case study without turning the repo into a full annotation-analysis project.

## Resume and inspect experiments

You can resume training from a checkpoint by overriding:

```bash
python -m src.train experiment=fewshot_100 trainer.resume_from_checkpoint=/path/to/checkpoint
```

Useful files to inspect after a run:

- `metrics.json` for the main scores
- `per_label_metrics.json` for PER/LOC/ORG breakdowns
- `predictions.jsonl` for full decoded predictions
- `error_analysis_sample.csv` for qualitative analysis
- `case_study_summary.md` for the current across-run comparison

## Practical recommendation for the case study

The intended coursework story is:

- start from the source-only baseline
- test simple normalization
- test few-shot adaptation with 50, 100, and 250 examples
- test self-training
- optionally test a combined recipe
- report which recipes give the best in-domain gains and which error types go down

This is realistic for 4 to 5 weeks because the code stays small, the experiments are bounded, and every run writes audit-friendly artifacts.
