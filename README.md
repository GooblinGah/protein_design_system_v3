
# Protein Design System

End-to-end, proposal-aligned system for prompt-conditioned protein design with:

- **Planner + Validator** (ontology keywords, negative regex, refusal logger)
- **DSL → FSA** motif constraints (min-occurrences, overlapping)
- **Segmental-HMM style controller** (duration MLP, z-tier: normal/stretched/sparse, boundary anneal, shock detector, monotone consensus index)
- **Pointer/Copy head** with **provenance** (top‑k sources & weights)
- **Dynamic copy control** (monitors → dynamic `gate_reg_weight` scaling and gate bias)
- **Differentiable identity penalty** during training (+ hard caps at gen)
- **Tamper-evident safety ledger** (hash-chain + verifier)
- **Homology guards**: sequence identity (MMseqs2 optional path), structure TM-score (TM-align), **active-site RMSD**
- **Liability screens** (homopolymers, hydrophobic runs, low complexity)
- **Retrieval + alignment** (portable k‑mer index + profile consensus/conservation)
- **Baselines B1/B2/B3 + prompt suite + eval scaffolds**
- **Visualizer** (Streamlit app) + static HTML exporter
- **Training pragmatics**: seeds, resume, AMP, grad accumulation, DDP, torch.compile, W&B
- **Data pipeline**: pre-split clustering at 30–40% identity, 20% family holdout, external 2024 test constructor

## Quickstart (CPU smoke)

```bash
pip install -r requirements.txt
python scripts/smoke_data.py
python train.py --epochs 1 --amp 0 --use_wandb 0
python3 generate.py --prompt "alpha beta hydrolase with motif GXSXG, length 260..320, secreted" \
  --checkpoint checkpoints/model.pt --resample_max 1
```

## Multi-GPU DDP

```bash
./scripts/torchrun_train.sh 2 --epochs 5 --amp 1 --seed 42
```

## One-shot

```bash
./final_run.sh
```

See README sections inside for details and flags.


### Decoding
- **FSA beam search** enforces motif constraints with small beams (default 4) and applies **positional priors** from the controller (length window + z-tier hysteresis).
- If exemplar MSAs are provided, a **PSSM/profile path** is available for computing column features post-hoc; a lightweight real-time prior hook is scaffolded for integration with the gate.


## Docker (reproducible env)
```bash
docker build -t pds:v1 .
docker run --rm -it -v $PWD:/app pds:v1 bash -lc "python scripts/smoke_data.py && python train.py --epochs 1 --amp 0 --use_wandb 0"
```

## CI
A GitHub Actions workflow (`.github/workflows/ci.yml`) runs unit tests, a 1‑epoch smoke train, constrained generation, ledger verification, and diagnostics.


### Controller learning
Train the duration model from your processed data:
```bash
python controller/train_duration.py --data_dir data_pipeline/data/processed --epochs 10
```
`generate.py` logs **planned vs realized** segment durations and the **monotone consensus index**.

### Retrieval→MSA (optional)
Provide a FASTA database for k-NN retrieval and MSA construction:
```bash
python generate.py --retrieval_db ./retrieval/db.fasta --retrieval_topk 16 ...
```
If MUSCLE is available (`MUSCLE_CMD`), the MSA uses real alignment; otherwise a simple padded stack is used. Ledger records include **retrieval IDs and similarities**.

### Diagnostics (expanded)
Run:
```bash
python eval/diagnostics.py --ledger data_pipeline/data/safety_ledger_generated.jsonl --exemplar_fasta exemplars.fa
```
Generates extra plots: motif-window gate histogram, tier occupancy over time, and per-segment duration error.


### Duration-MLP prior in decoding
If you train `controller/duration_mlp.pt` with `controller/train_duration.py`, `generate.py` will **load it automatically** and construct per-segment **duration targets** (segments = motif_min_occurrences+1). Decoding applies a **duration prior** that gently encourages each segment to reach its target and penalizes overruns, in addition to the global length z-tier prior.
