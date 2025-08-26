
from torch.utils.data import Dataset
from typing import List, Dict, Any
from utils import encode_prompt_bytes, encode_seq, PAD, BOS, EOS, SEP, EXB, EXE

class SeqGenDataset(Dataset):
    def __init__(self, items, max_prompt_len=200, max_seq_len=256, use_exemplar=False):
        self.items = items
        self.max_prompt_len = max_prompt_len
        self.max_seq_len = max_seq_len
        self.use_exemplar = use_exemplar

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        prompt_toks = encode_prompt_bytes(ex["prompt"], self.max_prompt_len)
        seq_toks = encode_seq(ex["sequence"], self.max_seq_len)
        ex_toks = []
        if self.use_exemplar and ex.get("exemplar"):
            ex_toks = [EXB] + encode_seq(ex["exemplar"], self.max_seq_len//2) + [EXE]

        x = [BOS] + prompt_toks + [SEP] + ex_toks + [SEP] + seq_toks[:-1]
        y = [ -100 ] * (1 + len(prompt_toks) + 1 + len(ex_toks) + 1) + seq_toks[1:]
        return {"input": x, "labels": y, "exemplar": ex.get("exemplar","")}

def collate_batch(batch):
    import torch
    maxlen = max(len(b["input"]) for b in batch)
    X, Y, attn, exemplars = [], [], [], []
    for b in batch:
        x, y = b["input"], b["labels"]
        X.append(x + [PAD]*(maxlen-len(x)))
        Y.append(y + [-100]*(maxlen-len(y)))
        attn.append([1]*len(x) + [0]*(maxlen-len(x)))
        exemplars.append(b["exemplar"])
    return (torch.tensor(X, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long),
            torch.tensor(attn, dtype=torch.float32),
            exemplars)
