#!/usr/bin/env bash
# scripts/torchrun_train.sh
set -euo pipefail
NPROC=${1:-2}; shift || true
torchrun --nproc_per_node ${NPROC} train.py --distributed 1 "$@"
