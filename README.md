# News-to-Social NER Case Study

This repository contains a simple course project for domain adaptation in named entity recognition.

The main file is:

- `01_domain_adaptation_experiments.ipynb`

The notebook studies how a news-trained NER model transfers to noisy social media text from WNUT17.

## What is in the repository

- `01_domain_adaptation_experiments.ipynb`  
  A self-contained notebook with all experiments.
- `poster.pdf`  
  The final poster for the case study.
- `requirements.txt`  
  Minimal Python packages needed for the notebook.
- `.gitignore`  
  Git ignore rules for local files.

## What the notebook does

The notebook focuses only on NER.

It uses:

- source model: `andi611/roberta-base-ner-conll2003`
- target dataset: WNUT17

The target task is reduced to the overlapping entity types:

- `PER`
- `LOC`
- `ORG`

The notebook compares these settings:

- source-only baseline
- few-shot adaptation on 250 target sentences
- few-shot adaptation with simple text normalization
- few-shot adaptation with self-training
- few-shot adaptation with normalization and self-training

It also compares these runs with a stronger reference model:

- `emilys/twitter-roberta-base-WNUT`

## How to run

Create or activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook
```

Then run `01_domain_adaptation_experiments.ipynb` from top to bottom.

## Notes

- The notebook is self-contained and does not depend on any `src/` package.
- Training is done directly inside the notebook with fixed parameters.
- The project is intentionally simple and aimed at a student case study, not at building a full training framework.
