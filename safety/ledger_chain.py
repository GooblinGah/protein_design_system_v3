
# safety/ledger_chain.py
import os, json, hashlib, time
class HashChainLedger:
    def __init__(self, path="data_pipeline/data/safety_ledger_generated.jsonl", chain_path=None):
        self.path = path
        self.chain_path = chain_path or (path + ".chain")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.chain_path):
            with open(self.chain_path, "w") as f: f.write("")
    def _prev_hash(self):
        if not os.path.exists(self.chain_path): return "0"*64
        h = "0"*64
        with open(self.chain_path, "r") as f:
            for line in f: h = line.strip() or h
        return h or "0"*64
    def append(self, record: dict):
        record["ts"] = int(time.time())
        payload = json.dumps(record, sort_keys=True).encode("utf-8")
        rh = hashlib.sha256(payload).hexdigest()
        prev = self._prev_hash()
        ch = hashlib.sha256((prev + rh).encode("utf-8")).hexdigest()
        with open(self.path, "a") as f: f.write(json.dumps(record)+"\n")
        with open(self.chain_path, "a") as f: f.write(ch+"\n")
        return ch
