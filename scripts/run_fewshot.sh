#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:-fewshot_100}"
python -m src.train "experiment=${EXPERIMENT_NAME}" "$@"

