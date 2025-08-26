# data_pipeline/scripts/cluster_split.py
import argparse, random
from pathlib import Path
from utils import load_fasta, max_identity_vs_refs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_fasta", required=True)
    ap.add_argument("--identity", type=float, default=0.35)
    ap.add_argument("--out_dir", default="data_pipeline/data/processed")
    a = ap.parse_args()
    recs = list(load_fasta(a.in_fasta)); random.shuffle(recs)
    reps=[]; clusters=[]
    for name, seq in recs:
        placed=False
        for i,(rname,rseq) in enumerate(reps):
            if max_identity_vs_refs(seq, [(rname,rseq)]) >= a.identity:
                clusters[i].append((name,seq)); placed=True; break
        if not placed: reps.append((name,seq)); clusters.append([(name,seq)])
    n=len(clusters); n_test=max(1,int(0.2*n)); n_val=max(1, int(0.1*(n-n_test)))
    test_idx=set(range(0,n_test)); val_idx=set(range(n_test,n_test+n_val))
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out/"splits.tsv","w") as f:
        f.write("accession\tsplit\n")
        for i,cl in enumerate(clusters):
            split="train"
            if i in test_idx: split="test"
            elif i in val_idx: split="val"
            for name,_ in cl: f.write(f"{name}\t{split}\n")
    print("Wrote", out/"splits.tsv")
if __name__ == "__main__": main()
