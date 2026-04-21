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

Default training settings are intentionally simple:

- `learning_rate=3e-5`
- `num_train_epochs=5`
- `max_length=128`
- `early_stopping_patience=2`

The main few-shot setting is `250` sentences. The sampler is deterministic and label-aware:

- it tries to cover `ORG`, `PER`, and `LOC` early
- it prefers entity-bearing sentences over all-`O` sentences
- it uses the run seed only for deterministic tie-breaking

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
python -m src.train experiment=adapt seed=1 experiment.shot_count=100
python -m src.train experiment=adapt seed=1 experiment.shot_count=250
```

Normalization ablation:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_normalization=true
python -m src.train experiment=adapt seed=1 experiment.shot_count=250 experiment.do_normalization=true
```

Self-training:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_self_training=true
python -m src.train experiment=adapt seed=1 experiment.shot_count=250 experiment.do_self_training=true
```

Combined adaptation:

```bash
python -m src.train experiment=adapt seed=1 experiment.shot_count=100 experiment.do_normalization=true experiment.do_self_training=true
python -m src.train experiment=adapt seed=1 experiment.shot_count=250 experiment.do_normalization=true experiment.do_self_training=true
```

Recommended three-seed matrix:

```bash
python -m src.train experiment=source_only seed=1
python -m src.train experiment=source_only seed=2
python -m src.train experiment=source_only seed=3

python -m src.train experiment=adapt seed=1 experiment.shot_count=100
python -m src.train experiment=adapt seed=2 experiment.shot_count=100
python -m src.train experiment=adapt seed=3 experiment.shot_count=100

python -m src.train experiment=adapt seed=1 experiment.shot_count=250
python -m src.train experiment=adapt seed=2 experiment.shot_count=250
python -m src.train experiment=adapt seed=3 experiment.shot_count=250

python -m src.train experiment=adapt seed=1 experiment.shot_count=250 experiment.do_self_training=true
python -m src.train experiment=adapt seed=2 experiment.shot_count=250 experiment.do_self_training=true
python -m src.train experiment=adapt seed=3 experiment.shot_count=250 experiment.do_self_training=true

python -m src.train experiment=adapt seed=1 experiment.shot_count=250 experiment.do_normalization=true
python -m src.train experiment=adapt seed=2 experiment.shot_count=250 experiment.do_normalization=true
python -m src.train experiment=adapt seed=3 experiment.shot_count=250 experiment.do_normalization=true
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

Self-training stays simple:

- pseudo-labeled sentences must pass the confidence threshold
- pseudo-labeled sentences must contain at least one predicted non-`O` entity
- pseudo-labeled additions are capped to `2x` the gold few-shot size, ranked by confidence

The only local output that remains is the Hugging Face training directory under `outputs/`, because `Trainer` needs a filesystem location for checkpoints.
