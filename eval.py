#!/usr/bin/env python3
# eval.py
import json, argparse
from utils import load_fasta, max_identity_vs_refs
from safety.liability import has_homopolymer, hydrophobic_run, low_complexity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_fasta", default=None)
    ap.add_argument("--ref_fasta", default=None)
    a = ap.parse_args()

    if not a.candidates_fasta:
        print("Provide --candidates_fasta to evaluate sequences."); return

    refs = list(load_fasta(a.ref_fasta)) if a.ref_fasta else []
    rows = []
    for name, seq in load_fasta(a.candidates_fasta):
        ident = max_identity_vs_refs(seq, refs) if refs else None
        liab = {"hpoly": has_homopolymer(seq), "hydro_run": hydrophobic_run(seq), "lowcomp": low_complexity(seq)}
        rows.append({"name": name, "len": len(seq), "identity": ident, **liab})
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
