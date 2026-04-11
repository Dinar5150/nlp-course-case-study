#!/usr/bin/env bash
set -euo pipefail

python -m src.train experiment=source_only "$@"

