#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Installing Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[2/3] ClearML environment summary"
if [[ -n "${CLEARML_API_ACCESS_KEY:-}" && -n "${CLEARML_API_SECRET_KEY:-}" ]]; then
  echo "ClearML credentials detected in environment variables."
else
  echo "ClearML credentials not detected. The run will continue locally if allowed by config."
fi

echo "[3/3] Running default source-only baseline"
python -m src.train experiment=source_only seed="${SEED:-1}"

echo
echo "Example follow-up commands:"
echo "python -m src.train experiment=fewshot_50 seed=1"
echo "python -m src.train experiment=fewshot_100 seed=1 trainer=low_vram"
echo "python -m src.train experiment=normalize seed=1"
echo "python -m src.train experiment=selftrain seed=1 experiment.pseudo_label_threshold=0.95"
echo "python -m src.train experiment=combined seed=1"
echo "python -m src.optuna_search experiment=fewshot_100 optuna.n_trials=10 trainer=low_vram"

