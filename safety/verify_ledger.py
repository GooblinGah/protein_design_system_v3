#!/usr/bin/env python3
# safety/verify_ledger.py
import json, hashlib, sys
def verify(log_path, chain_path):
    prev = "0"*64
    with open(log_path) as L, open(chain_path) as C:
        for rec_line, ch_line in zip(L, C):
            rec = json.loads(rec_line)
            rh = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            expected = hashlib.sha256((prev+rh).encode()).hexdigest()
            ch = ch_line.strip()
            if expected != ch:
                print("FAIL at record:", rec.get("ts"), "expected", expected, "got", ch); return 1
            prev = ch
    print("OK: chain verified"); return 0
if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv)>1 else "data_pipeline/data/safety_ledger_generated.jsonl"
    ch = sys.argv[2] if len(sys.argv)>2 else log+".chain"
    raise SystemExit(verify(log, ch))
