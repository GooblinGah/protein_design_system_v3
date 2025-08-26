#!/usr/bin/env bash
set -euo pipefail
pip install -r requirements.txt
python scripts/smoke_data.py
if command -v nvidia-smi >/dev/null 2>&1 && python - <<'PY' >/dev/null 2>&1; then
import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[final] GPU detected. Training with AMP."
  python train.py --epochs 2 --amp 1 --seed 1337 --use_wandb 0
else
  echo "[final] No GPU detected. Training on CPU."
  python train.py --epochs 1 --amp 0 --seed 1337 --use_wandb 0
fi
python eval.py --candidates_fasta data_pipeline/data/processed/external2024.fa || true
python generate.py --prompt "alpha beta hydrolase with motif GXSXG, length 260..320, secreted" --checkpoint checkpoints/model.pt --resample_max 1
echo "[final] Done."
