
# retrieval/profile.py
import numpy as np
def consensus_and_conservation(seqs, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    if not seqs: return "", []
    L = max(len(s) for s in seqs); A = alphabet; idx={a:i for i,a in enumerate(A)}
    counts = np.zeros((L, len(A)), dtype=float)
    for s in seqs:
        for i,c in enumerate(s):
            if c in idx: counts[i, idx[c]] += 1.0
    total = counts.sum(axis=1, keepdims=True).clip(min=1.0)
    freqs = counts/total
    consensus = "".join(A[i] for i in freqs.argmax(axis=1))
    cons = freqs.max(axis=1).tolist()
    return consensus, cons
