#!/usr/bin/env python3
import argparse, os, random, importlib
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from datasets import SeqGenDataset, collate_batch
from models.transformer_lm import TransformerCopyLM
from trainers.trainer import Trainer
from utils import VOCAB_SIZE

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--save", default="checkpoints/model.pt")
    ap.add_argument("--use_wandb", type=int, default=0)
    ap.add_argument("--wandb_project", type=str, default="protein_design")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--identity_tau", type=float, default=0.7)
    ap.add_argument("--identity_weight", type=float, default=0.0)
    ap.add_argument("--identity_warmup_epochs", type=int, default=0)
    ap.add_argument("--distributed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # tiny synthetic data
    items = []
    aa = "ACDEFGHIKLMNPQRSTVWY"
    def synth_seq(L=180):
        import random
        seq = [random.choice(aa) for _ in range(L)]
        pos = random.randint(10, L-10-5)
        seq[pos:pos+5] = list("GASAG")
        return "".join(seq)
    for i in range(10):
        items.append({"prompt": "Design alpha beta hydrolase with motif GXSXG, length 260..320, secreted", "sequence": synth_seq(), "exemplar": synth_seq(120)})
    train_items = items[:7]; val_items = items[7:9]

    from torch.utils.data import DataLoader
    train_loader = DataLoader(SeqGenDataset(train_items, use_exemplar=True), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(SeqGenDataset(val_items, use_exemplar=True), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCopyLM(vocab_size=VOCAB_SIZE).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs*max(1,len(train_loader)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    wb = None
    if args.use_wandb:
        if importlib.util.find_spec("wandb") is not None:
            import wandb as _wb; wb=_wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            try: wb.watch(model, log="gradients", log_freq=200)
            except Exception: pass
        else:
            print("wandb not installed; proceeding without W&B.")

    trainer = Trainer(model, optim, sched, loss_fn, device, args.save,
                      amp=bool(args.amp), gate_reg_weight=0.1, curriculum={"phase_epochs":[999], "gate_reg":[0.1]},
                      identity_tau=args.identity_tau, identity_weight=args.identity_weight, identity_warmup_epochs=args.identity_warmup_epochs,
                      use_wandb=bool(args.use_wandb), wandb=wb)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
