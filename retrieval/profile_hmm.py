
# retrieval/profile_hmm.py

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a:i for i,a in enumerate(ALPHABET)}

def build_pssm(msa_seqs, pseudocount=1.0):
    """Return (PSSM logits [L,20], consensus [L], conservation [L]) from a list of ungapped sequences aligned by padding with '-'."""
    if not msa_seqs:
        return np.zeros((0,20), dtype=float), "", []
    L = max(len(s) for s in msa_seqs)
    counts = np.full((L, len(ALPHABET)), pseudocount, dtype=float)
    for s in msa_seqs:
        for i, c in enumerate(s):
            if c == '-' or c not in IDX: 
                continue
            counts[i, IDX[c]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    pssm = np.log(np.clip(probs, 1e-6, 1.0))
    cons_idx = probs.argmax(axis=1)
    consensus = "".join(ALPHABET[i] for i in cons_idx)
    conservation = probs.max(axis=1).tolist()
    return pssm, consensus, conservation

def viterbi_profile(pssm_logits, seq, ins_pen=-2.0, del_pen=-2.0):
    """
    Simple profile alignment: dynamic programming with states M(I) and deletions.
    Returns aligned index mapping: for each seq pos j -> profile column i (or -1 if gap), and log-score.
    """
    Lp = pssm_logits.shape[0]; Ls = len(seq)
    if Lp == 0 or Ls == 0:
        return [-1]*Ls, -1e9

    # DP matrices
    M = np.full((Lp+1, Ls+1), -1e12, dtype=float)
    I = np.full((Lp+1, Ls+1), -1e12, dtype=float)
    D = np.full((Lp+1, Ls+1), -1e12, dtype=float)
    back = np.zeros((3, Lp+1, Ls+1), dtype=np.int16)  # 0:M,1:I,2:D

    M[0,0] = 0.0
    for i in range(1, Lp+1):
        D[i,0] = D[i-1,0] + del_pen
        back[2,i,0] = 2
    for j in range(1, Ls+1):
        I[0,j] = I[0,j-1] + ins_pen
        back[1,0,j] = 1

    for i in range(1, Lp+1):
        for j in range(1, Ls+1):
            aa = seq[j-1]
            emit = pssm_logits[i-1, IDX.get(aa, 0)]
            # Match state
            vals = np.array([M[i-1,j-1], I[i-1,j-1], D[i-1,j-1]]) + emit
            k = int(np.argmax(vals))
            M[i,j] = vals[k]; back[0,i,j] = k
            # Insert in sequence (stay at profile i, advance seq j)
            valsI = np.array([M[i,j-1], I[i,j-1]]) + ins_pen
            kI = int(np.argmax(valsI)); I[i,j] = valsI[kI]; back[1,i,j] = [0,1][kI]
            # Delete in sequence (advance profile i, stay seq j)
            valsD = np.array([M[i-1,j], D[i-1,j]]) + del_pen
            kD = int(np.argmax(valsD)); D[i,j] = valsD[kD]; back[2,i,j] = [0,2][kD]

    # Termination
    end_scores = np.array([M[Lp,Ls], I[Lp,Ls], D[Lp,Ls]])
    state = int(np.argmax(end_scores))
    i, j = Lp, Ls
    mapping = [-1]*Ls
    while i>0 or j>0:
        if state == 0:  # M
            mapping[j-1] = i-1
            state = int(back[0,i,j]); i -= 1; j -= 1
        elif state == 1:  # I
            state = int(back[1,i,j]); j -= 1
        else:  # D
            state = int(back[2,i,j]); i -= 1
    return mapping, float(np.max(end_scores))

def column_features(msa_seqs, seq):
    """Build PSSM and align seq; return per-position (cons_prob, match_flag, cons_aa_idx) lists aligned to seq positions."""
    pssm, consensus, conservation = build_pssm(msa_seqs)
    if pssm.shape[0] == 0:
        L = len(seq)
        return [0.0]*L, [0]*L, [-1]*L, pssm, consensus, conservation
    mapping, score = viterbi_profile(pssm, seq)
    cons_probs = []
    match_flags = []
    cons_idx = []
    for pos, col in enumerate(mapping):
        if col < 0:
            cons_probs.append(0.0); match_flags.append(0); cons_idx.append(-1)
        else:
            cons_probs.append(float(np.exp(pssm[col].max())))
            match_flags.append(1 if seq[pos] == consensus[col] else 0)
            cons_idx.append(int(np.argmax(pssm[col])))
    return cons_probs, match_flags, cons_idx, pssm, consensus, conservation
