#!/usr/bin/env python3
# scripts/smoke_data.py
import json, random
from pathlib import Path
def synth_seq(L=280):
    aa = "ACDEFGHIKLMNPQRSTVWY"; seq = [random.choice(aa) for _ in range(L)]
    pos = random.randint(20, 40); motif=list("GASAG"); seq[pos:pos+5]=motif
    return "".join(seq)
def main(out_root="data_pipeline/data"):
    pr = Path(out_root) / "processed"; ir = Path(out_root) / "interim"
    pr.mkdir(parents=True, exist_ok=True); ir.mkdir(parents=True, exist_ok=True)
    with open(pr/"splits.tsv","w") as f:
        f.write("accession\tsplit\n"); 
        for i in range(10): f.write(f"ACC{i}\t{'train' if i<7 else ('val' if i<9 else 'test')}\n")
    with open(pr/"prompt_pairs.jsonl","w") as f:
        for i in range(10):
            f.write(json.dumps({"accession": f"ACC{i}", "prompt": "Design alpha beta hydrolase with motif GXSXG, length 260..320, secreted", "sequence": synth_seq()})+"\n")
    with open(ir/"filtered.jsonl","w") as f:
        for i in range(10): f.write(json.dumps({"accession": f"ACC{i}", "sequence": synth_seq(150)})+"\n")
    print("Wrote synthetic dataset under", out_root)
if __name__ == "__main__": main()
