# retrieval/faiss_index.py
import numpy as np
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    faiss = None
    HAVE_FAISS = False

AA = "ACDEFGHIKLMNPQRSTVWY"; IDX={a:i for i,a in enumerate(AA)}

def kmer_embed(seq: str, k=3):
    if len(seq) < k: return np.zeros(len(AA)**k, dtype=np.float32)
    v = np.zeros(len(AA)**k, dtype=np.float32)
    for i in range(len(seq)-k+1):
        h=0; ok=True
        for j in range(k):
            c = seq[i+j]
            if c not in IDX: ok=False; break
            h = h*len(AA) + IDX[c]
        if ok: v[h]+=1.0
    n = np.linalg.norm(v); 
    return v/n if n>0 else v

class Retriever:
    def __init__(self, names, seqs, use_faiss=True):
        self.names = names; self.seqs = seqs
        self.X = np.stack([kmer_embed(s) for s in seqs]).astype('float32')
        self.idx = None
        if use_faiss and HAVE_FAISS:
            self.idx = faiss.IndexFlatIP(self.X.shape[1])
            self.idx.add(self.X)
    def query(self, seq_or_embed, topk=16):
        q = kmer_embed(seq_or_embed) if isinstance(seq_or_embed,str) else np.asarray(seq_or_embed, dtype='float32')
        if self.idx is not None:
            D,I = self.idx.search(q.reshape(1,-1).astype('float32'), topk)
            sims = D[0].tolist(); idxs=I[0].tolist()
        else:
            sims = (self.X @ q).tolist()
            idxs = np.argsort(-np.array(sims))[:topk].tolist()
            sims = [sims[i] for i in idxs]
        return [(self.names[i], self.seqs[i], float(sims[k])) for k,i in enumerate(idxs)]
