#!/usr/bin/env bash
set -euo pipefail

python -m src.optuna_search "$@"

