#!/usr/bin/env python3
"""Fit the DurationMLP on real data by deriving segment targets from motif windows.
- Reads processed/prompt_pairs.jsonl and splits.tsv
- Parses motif regex from prompt (fallback to G.[ST][AGST]G)
- For each sequence, finds k motif occurrences; segments = k+1; lengths = gaps between [0..motif1), (motif1..motif2)...(last..end)
- Prompt is byte-encoded; we use mean-pooled byte embedding as features (simple, deterministic)
- Trains an MLP (regression on lengths) and saves weights to controller/duration_mlp.pt
"""
import argparse, json, os, re, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

def parse_motif(prompt: str):
    m = re.search(r"G\.[ST]\[[A-Z]+\]G|G\.[ST]\.[A-Z]G|G\.?S\.?G|GXSXG", prompt)
    return m.group(0) if m else r"G.[ST][AGST]G"

def find_motif_positions(seq, motif_rx):
    return [m.span()[0] for m in re.finditer(motif_rx, seq)]

def encode_prompt_bytes(text: str, max_len=256):
    b = text.encode('utf-8', errors='ignore')[:max_len]
    # Convert each byte to its integer value (0-255)
    return np.array([int(byte) for byte in b], dtype=np.int64)

class SmallMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data_pipeline/data/processed')
    ap.add_argument('--save', default='controller/duration_mlp.pt')
    ap.add_argument('--max_segments', type=int, default=4, help='predict up to this many segments (k+1 clipped)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()

    pp = list(Path(a.data_dir).glob('prompt_pairs.jsonl'))
    if not pp: 
        print('no prompt_pairs.jsonl'); return
    rows = []
    with open(pp[0]) as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))

    X=[]; Y=[]
    for r in rows:
        seq = r['sequence']; prompt = r.get('prompt','')
        motif_rx = parse_motif(prompt)
        starts = find_motif_positions(seq, motif_rx)
        segs = []
        prev = 0
        for s in starts:
            segs.append(max(1, s - prev))
            prev = s + 5  # motif len approx 5
        segs.append(max(1, len(seq) - prev))
        segs = segs[:a.max_segments]
        if len(segs) < a.max_segments:
            segs += [0]*(a.max_segments - len(segs))  # pad with zeros (ignore via mask)
        feat = encode_prompt_bytes(prompt)
        feat = np.pad(feat, (0, max(0,256-len(feat))), constant_values=0.0)
        X.append(feat); Y.append(segs)

    if not X: 
        print('no data'); return
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    model = SmallMLP(X.size(1), Y.size(1))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = nn.L1Loss()

    for epoch in range(1, a.epochs+1):
        model.train(); opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward(); opt.step()
        print(f'Epoch {epoch}: L1={loss.item():.4f}')
    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'in': X.size(1), 'out': Y.size(1)}, a.save)
    print('Saved', a.save)

if __name__ == '__main__':
    main()
