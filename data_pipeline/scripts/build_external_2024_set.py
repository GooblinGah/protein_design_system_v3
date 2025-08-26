# data_pipeline/scripts/build_external_2024_set.py
import argparse
from utils import load_fasta
from pathlib import Path
import json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_fasta", required=True)
    ap.add_argument("--out_jsonl", default="data_pipeline/data/processed/external2024.jsonl")
    a = ap.parse_args()
    Path(a.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_jsonl,"w") as f:
        for name,seq in load_fasta(a.input_fasta):
            f.write(json.dumps({"accession":name,"prompt":"external2024","sequence":seq})+"\n")
    print("Wrote", a.out_jsonl)
if __name__ == "__main__": main()
