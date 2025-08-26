# viz/prov_to_html.py
import json, argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data_pipeline/data/safety_ledger_generated.jsonl")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default="provenance.html")
    a = ap.parse_args()
    rec = list(open(a.ledger))[a.idx]; rec=json.loads(rec)
    seq=rec.get("sequence",""); prov=rec.get("provenance", [])
    out = ["<html><body><pre style='font-family:monospace'>"]
    for i,c in enumerate(seq):
        src = prov[i].get("source","vocab") if i < len(prov) else "vocab"
        color = "#ffd54f" if src=="copy" else "#e0e0e0"
        out.append(f"<span title='{src}' style='background:{color}'>" + c + "</span>")
    out.append("</pre></body></html>")
    open(a.out,"w").write("".join(out))
    print("Wrote", a.out)
if __name__ == "__main__": main()
