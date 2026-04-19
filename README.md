# NER Adaptation Experiments

This repository is a small `Hydra + ClearML` experiment runner for a news-to-social NER adaptation case study. It transfers `philschmid/distilroberta-base-ner-conll2003` to WNUT17 and keeps only the overlapping labels `PER`, `LOC`, and `ORG`.

The repo does one thing: run experiments and log them to ClearML.

## Stack

- Hydra for config composition and CLI overrides
- ClearML for experiment tracking
- Hugging Face Transformers and Datasets for training and evaluation

## Config Shape

There are only two experiment configs:

- `experiment=source_only`
- `experiment=adapt`

`adapt` is the single target-domain adaptation pipeline. You control the method with flags:

- `experiment.shot_count`
- `experiment.do_normalization`
- `experiment.do_self_training`

## Install

```bash
conda activate machine-learning
pip install -r requirements.txt
```

Set ClearML credentials before running:

```bash
set CLEARML_API_ACCESS_KEY=...
set CLEARML_API_SECRET_KEY=...
set CLEARML_API_HOST=https://api.clear.ml
```

If you want to test without a server:

```bash
python -m src.train experiment=source_only clearml.offline_mode=true
```

## Main Commands

Baseline:

```bash
python -m src.train experiment=source_only seed=1
```

Few-shot adaptation:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=50
python -m src.train experiment=adapt seed=1 experiment.shot_count=100
```

Normalization:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_normalization=true
```

Self-training:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_self_training=true
```

Combined adaptation:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_normalization=true experiment.do_self_training=true
```

## What Gets Logged

Each run logs to ClearML:

- resolved config
- validation, test, and training metrics
- per-label metrics table
- error-analysis table
- sampled example IDs
- pseudo-label audit records when self-training is enabled
- decoded predictions

The only local output that remains is the Hugging Face training directory under `outputs/`, because `Trainer` needs a filesystem location for checkpoints.
