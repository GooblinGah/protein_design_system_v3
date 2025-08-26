
from typing import List, Dict, Tuple
import re, json
PAD, BOS, EOS, SEP, EXB, EXE = 0,1,2,3,4,5
AA_START = 6
PROMPT_START = 26
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {c: AA_START + i for i, c in enumerate(AA_VOCAB)}
ID_TO_AA = {v:k for k,v in AA_TO_ID.items()}
VOCAB_SIZE = PROMPT_START + 256

def encode_prompt_bytes(text: str, max_len: int) -> List[int]:
    b = text.encode("utf-8", errors="ignore")[:max_len]
    return [PROMPT_START + byte for byte in b]

def encode_seq(seq: str, max_len: int) -> List[int]:
    toks = [AA_TO_ID.get(c) for c in seq if c in AA_TO_ID]
    return [t for t in toks if t is not None][:max_len]

def detok_seq(toks: List[int]) -> str:
    return "".join(ID_TO_AA.get(t, "") for t in toks if t in ID_TO_AA)

def load_fasta(path: str):
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name = line[1:]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)

def save_fasta(path: str, recs):
    with open(path, "w") as f:
        for name, seq in recs:
            f.write(f">{name}\n{seq}\n")

def max_identity_vs_refs(seq: str, refs):
    def pid(a,b):
        L = min(len(a), len(b))
        if L == 0: return 0.0
        same = sum(1 for i in range(L) if a[i]==b[i])
        return same/float(L)
    best=0.0
    for _,r in refs:
        best=max(best,pid(seq,r))
    return best
