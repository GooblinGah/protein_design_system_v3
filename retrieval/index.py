
# retrieval/index.py
import numpy as np
V = "ACDEFGHIKLMNPQRSTVWY"; IDX = {a:i for i,a in enumerate(V)}
def _kmer_embed(seq: str, k=3):
    if len(seq)<k: return np.zeros(len(V)**k, dtype=float)
    vec = np.zeros(len(V)**k, dtype=float)
    for i in range(len(seq)-k+1):
        h=0; ok=True
        for j in range(k):
            a = seq[i+j]; 
            if a not in IDX: ok=False; break
            h = h*len(V) + IDX[a]
        if ok: vec[h]+=1.0
    if vec.sum()>0: vec/=np.linalg.norm(vec)
    return vec
class KNNRetriever:
    def __init__(self, refs):
        self.names=[n for n,_ in refs]; self.seqs=[s for _,s in refs]
        self.X = np.stack([_kmer_embed(s) for s in self.seqs], axis=0) if refs else np.zeros((0,1))
    def query(self, seq_or_embed, topk=5):
        q = _kmer_embed(seq_or_embed) if isinstance(seq_or_embed,str) else np.asarray(seq_or_embed,float)
        sims = self.X @ q if self.X.size else np.zeros((0,))
        idx = np.argsort(-sims)[:topk]
        return [(self.names[i], self.seqs[i], float(sims[i])) for i in idx]
