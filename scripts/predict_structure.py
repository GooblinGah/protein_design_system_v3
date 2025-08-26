#!/usr/bin/env python3
# scripts/predict_structure.py
import argparse, os, sys, subprocess
def run_shell(cmd):
    try: subprocess.check_call(cmd, shell=True); return 0
    except Exception as e: sys.stderr.write(f"[predict] shell failed: {e}\n"); return 1
def run_esm_api(fasta, out_pdb):
    try:
        import torch, esm
        model = esm.pretrained.esmfold_v1()
        model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
        name, seq=None,[]
        with open(fasta) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                if line.startswith(">"):
                    if name is None: name=line[1:]
                    else: break
                else: seq.append(line)
        seq="".join(seq).upper()
        with torch.no_grad(): pdb = model.infer_pdb(seq)
        open(out_pdb,"w").write(pdb); return 0 if os.path.exists(out_pdb) else 2
    except Exception as e:
        sys.stderr.write(f"[predict] ESM API failed: {e}\n"); return 2
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True); ap.add_argument("--out_pdb", required=True)
    ap.add_argument("--cmd", default=os.environ.get("PREDICT_CMD"))
    a = ap.parse_args()
    if run_esm_api(a.fasta, a.out_pdb)==0: print("[predict] wrote", a.out_pdb); return
    if a.cmd and run_shell(a.cmd.format(fasta=a.fasta, out_pdb=a.out_pdb))==0 and os.path.exists(a.out_pdb):
        print("[predict] wrote", a.out_pdb); return
    sys.stderr.write("[predict] no predictor succeeded\n"); sys.exit(1)
if __name__ == "__main__": main()
