# Write a single, self-contained dataset builder that:
# - pulls UniProt sequences via REST (or uses an existing FASTA)
# - filters & formats to your repo schema
# - creates splits.tsv (length-stratified; optional approximate clustering)
# - writes interim/filtered.jsonl
# - builds retrieval/db.fasta
# - optionally computes retrieval_topk.jsonl with a lightweight 3-mer embedding
# The script is standalone and can be dropped into scripts/ in the repo.

import os, textwrap, pathlib, json

script = r'''#!/usr/bin/env python3
"""
build_dataset.py
----------------
Pull UniProt sequences and prepare all files expected by the repo:

Creates:
  data_pipeline/data/
    raw/uniprot.fasta
    processed/prompt_pairs.jsonl
    processed/splits.tsv
    processed/retrieval_topk.jsonl            (optional, --write_topk 1)
    processed/refs_train.fasta
    processed/refs_val.fasta
    processed/refs_test.fasta
    interim/filtered.jsonl
  retrieval/db.fasta

Design choices:
- Fetch via UniProt REST (search + stream) OR consume a local FASTA (use --in_fasta).
- Filter to 20-AA alphabet and length bounds (default 60..1024).
- Prompts: family/motif/length window (+ "secreted" if --secreted 1).
- Splits: length-stratified 80/10/10; optional approximate clustering (--approx_cluster 1).
- retrieval_topk.jsonl: lightweight 3-mer hashed embedding + cosine for top-k (best for <=~200k sequences).

WARNING: Pulling *all* UniProtKB is enormous (hundreds of GB). Use filters (reviewed:true, organism_id, ec, keyword).
"""

import argparse, os, sys, re, json, time, hashlib, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import urllib.parse
import requests

BASE = "https://rest.uniprot.org"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------------- REST helpers ----------------

class Backoff:
    def __init__(self, base=1.0, cap=60.0, factor=2.0):
        self.base=base; self.cap=cap; self.factor=factor; self.n=0
    def next(self, retry_after: Optional[float]=None):
        if retry_after is not None:
            try: return min(float(retry_after), self.cap)
            except: return self.base
        t = min(self.base * (self.factor ** self.n), self.cap); self.n+=1; return t
    def reset(self): self.n=0

def session(user_agent="pds-build-dataset/1.0 (+https://example.org)"):
    s = requests.Session(); s.headers.update({"User-Agent": user_agent}); return s

def robust_get(s: requests.Session, url: str, params: Dict[str,str], stream=False, tries=8):
    b=Backoff()
    for _ in range(tries):
        try:
            r = s.get(url, params=params, stream=stream, timeout=60)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                time.sleep(b.next(float(ra) if ra and ra.isdigit() else None)); continue
            r.raise_for_status(); b.reset(); return r
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else None
            if sc and 500 <= sc < 600:
                time.sleep(b.next()); continue
            raise
        except requests.RequestException:
            time.sleep(b.next())
    raise RuntimeError(f"GET failed: {url}")

def search_accessions(s: requests.Session, query: str, size: int, include_isoform: bool, cursor: Optional[str]=None):
    url = f"{BASE}/uniprotkb/search"
    p = {"query": query, "fields":"accession", "format":"json", "size":str(size), "includeIsoform": "true" if include_isoform else "false"}
    if cursor: p["cursor"]=cursor
    r = robust_get(s, url, p)
    data = r.json()
    accs = [row.get("primaryAccession") for row in data.get("results", []) if row.get("primaryAccession")]
    # cursor parsing
    nxt = None
    if "next" in data: 
        nxt = data["next"].split("cursor=")[-1] if "cursor=" in data["next"] else None
    if not nxt and "links" in data and "next" in data["links"]:
        nxturl = data["links"]["next"]["href"] if isinstance(data["links"]["next"], dict) else data["links"]["next"]
        if "cursor=" in nxturl: nxt = nxturl.split("cursor=")[-1]
    if not nxt and "next" in r.links and "url" in r.links["next"]:
        nxturl = r.links["next"]["url"]; nxt = nxturl.split("cursor=")[-1] if "cursor=" in nxturl else None
    return accs, nxt

def stream_fasta_for_accs(s: requests.Session, accs: List[str], include_isoform: bool) -> bytes:
    url = f"{BASE}/uniprotkb/stream"
    q = " OR ".join(f"accession:{a}" for a in accs)
    p = {"format":"fasta", "query": q, "includeIsoform": "true" if include_isoform else "false"}
    r = robust_get(s, url, p, stream=True)
    return b"".join(chunk for chunk in r.iter_content(chunk_size=262144) if chunk)

# ---------------- FASTA parsing ----------------

def fasta_iter(path: Path):
    name=None; seq=[]
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            if line[0]==">":
                if name: yield name, "".join(seq)
                name=line[1:].split()[0]; seq=[]
            else:
                seq.append(line.strip())
    if name: yield name, "".join(seq)

# ---------------- Processing ----------------

def filter_seq(seq: str, min_len: int, max_len: int) -> bool:
    return min_len <= len(seq) <= max_len and set(seq) <= AA20

def length_window(seq_len: int, pad: int=10, min_floor: int=60):
    lo = max(min_floor, seq_len - pad); hi = seq_len + pad; return lo, hi

def make_prompt(family: str, motif: str, lo: int, hi: int, secreted: bool):
    parts=[family]
    if motif: parts.append(f"with motif {motif}")
    parts.append(f"length {lo}..{hi}")
    if secreted: parts.append("secreted")
    return " ".join(parts)

def bucket_len(L: int):
    if L<150: return "060-149"
    if L<300: return "150-299"
    if L<600: return "300-599"
    return "600-1024"

def stratified_split(records: List[Dict], seed: int = 1337) -> Dict[str,str]:
    # 80/10/10 across length buckets
    rnd = random.Random(seed)
    by_bucket = {}
    for r in records:
        by_bucket.setdefault(r["len_bucket"], []).append(r["accession"])
    mapping = {}
    for b, ids in by_bucket.items():
        rnd.shuffle(ids)
        n=len(ids); n_tr=int(0.8*n); n_v=int(0.1*n)
        for acc in ids[:n_tr]: mapping[acc]="train"
        for acc in ids[n_tr:n_tr+n_v]: mapping[acc]="val"
        for acc in ids[n_tr+n_v:]: mapping[acc]="test"
    return mapping

# Optional: extremely simple approximate clustering by minhash-like signatures to make split family-aware.
def approx_cluster_ids(records: List[Dict], k=3, n_hash=4, mod=100003):
    # Build signature per seq: n_hash min-hash over k-mers.
    sigs=[]
    for r in records:
        s=r["sequence"]; kmers=set(s[i:i+k] for i in range(len(s)-k+1))
        sig=[min((hash((h,km)) % mod) for km in kmers)) if kmers else 0 for h in range(n_hash)]
        sigs.append((r["accession"], tuple(sig)))
    # Group by signature tuple
    groups={}
    for acc,sig in sigs:
        groups.setdefault(sig, []).append(acc)
    # Map accession->cluster id
    acc2clu={}
    for i,(sig, ids) in enumerate(groups.items()):
        for acc in ids: acc2clu[acc]=i
    return acc2clu

def stratified_cluster_split(records: List[Dict], seed: int = 1337) -> Dict[str,str]:
    rnd = random.Random(seed)
    # cluster ids
    acc2clu = approx_cluster_ids(records)
    clu2ids={}
    for r in records:
        clu2ids.setdefault(acc2clu[r["accession"]], []).append(r["accession"])
    # shuffle clusters
    clus=list(clu2ids.keys()); rnd.shuffle(clus)
    # greedily assign clusters to train/val/test to hit 80/10/10 by count
    total=len(records); target = {"train":0.8*total, "val":0.1*total, "test":0.1*total}
    mapping={}; counts={"train":0,"val":0,"test":0}
    order=["train","val","test"]
    i=0
    for clu in clus:
        ids = clu2ids[clu]; # choose the split with max remaining capacity
        rem = {sp: target[sp]-counts[sp] for sp in order}
        split = max(rem, key=lambda sp: rem[sp])
        for acc in ids: mapping[acc]=split
        counts[split]+=len(ids)
        i+=1
    return mapping

# 3-mer hashed embedding for retrieval top-k
def kmer_embed(seq: str, k=3, dim=2048, seed=17):
    vec=[0]*dim
    for i in range(len(seq)-k+1):
        km=seq[i:i+k]
        h=hash((seed,km)) % dim
        vec[h]+=1
    # L2 normalize
    s=sum(v*v for v in vec)**0.5 or 1.0
    return [v/s for v in vec]

def topk_neighbors(records: List[Dict], topk=8, dim=2048):
    # Build embeddings
    embs=[]
    for r in records:
        embs.append((r["accession"], kmer_embed(r["sequence"], dim=dim)))
    # Cosine brute-force (OK for <=200k; else sample)
    import heapq
    acc2idx={acc:i for i,(acc,_) in enumerate(embs)}
    results={}
    for i,(acc_i,vi) in enumerate(embs):
        heap=[]
        for j,(acc_j,vj) in enumerate(embs):
            if i==j: continue
            # cosine = dot since normalized
            dot=sum(a*b for a,b in zip(vi,vj))
            if len(heap)<topk:
                heapq.heappush(heap, (dot, acc_j))
            else:
                if dot>heap[0][0]:
                    heapq.heapreplace(heap, (dot, acc_j))
        heap.sort(reverse=True)
        results[acc_i]=[{"acc": b, "sim": float(a)} for a,b in heap]
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data_pipeline/data", help="Root for data/ dirs")
    ap.add_argument("--in_fasta", default=None, help="Use an existing FASTA instead of fetching")
    ap.add_argument("--query", default="reviewed:true", help='UniProtKB query (e.g., "reviewed:true AND keyword:Hydrolase")')
    ap.add_argument("--include_isoform", type=int, default=0, help="1 to include isoforms")
    ap.add_argument("--max_records", type=int, default=0, help="Stop early after N records (0=all)")
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--min_len", type=int, default=60)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--family", default="alpha beta hydrolase", help='Used in prompt')
    ap.add_argument("--motif", default="GXSXG", help='Used in prompt (regex string OK)')
    ap.add_argument("--secreted", type=int, default=0, help="Append 'secreted' to prompt")
    ap.add_argument("--approx_cluster", type=int, default=1, help="1=use approximate clustering for family-aware split")
    ap.add_argument("--write_topk", type=int, default=0, help="1=compute retrieval_topk.jsonl (slow on very large sets)")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.out_root)
    raw = root / "raw"; processed = root / "processed"; interim = root / "interim"
    for p in (raw, processed, interim): p.mkdir(parents=True, exist_ok=True)

    fasta_path = raw / "uniprot.fasta"
    if args.in_fasta:
        fasta_path = Path(args.in_fasta)
        if not fasta_path.exists():
            print(f"[ERR] --in_fasta '{fasta_path}' not found", file=sys.stderr); sys.exit(1)
        print(f"[info] using local FASTA: {fasta_path}")
    else:
        print(f"[info] fetching UniProt: query='{args.query}', include_isoform={bool(args.include_isoform)}")
        s = session()
        cursor=None; seen=0
        with fasta_path.open("wb") as out:
            while True:
                accs, nxt = search_accessions(s, args.query, size=args.page_size, include_isoform=bool(args.include_isoform), cursor=cursor)
                if not accs: break
                # stream in smaller chunks to avoid huge OR queries
                for i in range(0, len(accs), args.batch_size):
                    chunk = accs[i:i+args.batch_size]
                    blob = stream_fasta_for_accs(s, chunk, bool(args.include_isoform))
                    out.write(blob)
                    # estimate seqs by counting '>'
                    nseq = blob.count(b"\n>") + (1 if blob.startswith(b">") else 0)
                    seen += nseq
                    print(f"[page] wrote ~{nseq} seqs (total ~{seen})")
                    if args.max_records and seen >= args.max_records:
                        print(f"[info] reached max_records={args.max_records}; stopping fetch")
                        break
                if args.max_records and seen >= args.max_records: break
                if not nxt: break
                cursor=nxt
        print(f"[done] FASTA: {fasta_path}")

    # Parse & filter
    print("[info] parsing & filtering FASTA...")
    records=[]
    for name, seq in fasta_iter(fasta_path):
        if not filter_seq(seq, args.min_len, args.max_len): continue
        lo, hi = length_window(len(seq))
        prompt = make_prompt(args.family, args.motif, lo, hi, bool(args.secreted))
        acc = name.split("|")[-1] if "|" in name else name
        records.append({"accession": acc, "prompt": prompt, "sequence": seq, "len_bucket": bucket_len(len(seq))})
    if not records:
        print("[ERR] no sequences passed filters", file=sys.stderr); sys.exit(2)
    print(f"[info] kept {len(records)} sequences")

    # Splits
    print("[info] making splits.tsv ... (approx_cluster=%d)" % args.approx_cluster)
    if args.approx_cluster:
        mapping = stratified_cluster_split(records)
    else:
        mapping = stratified_split(records)
    with (processed/"splits.tsv").open("w") as f:
        f.write("accession\tsplit\n")
        for r in records:
            f.write(f"{r['accession']}\t{mapping[r['accession']]}\n")
    print(f"[done] {processed/'splits.tsv'}")

    # prompt_pairs.jsonl
    with (processed/"prompt_pairs.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({"accession": r["accession"], "prompt": r["prompt"], "sequence": r["sequence"]})+"\n")
    print(f"[done] {processed/'prompt_pairs.jsonl'}")

    # interim/filtered.jsonl
    with (interim/"filtered.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({"accession": r["accession"], "sequence": r["sequence"]})+"\n")
    print(f"[done] {interim/'filtered.jsonl'}")

    # retrieval db
    retr_dir = Path("retrieval"); retr_dir.mkdir(parents=True, exist_ok=True)
    with (retr_dir/"db.fasta").open("w") as f:
        for r in records:
            f.write(f">{r['accession']}\n")
            s=r["sequence"]
            for i in range(0, len(s), 60): f.write(s[i:i+60]+"\n")
    print(f"[done] {retr_dir/'db.fasta'}")

    # refs: split FASTAs
    paths={ "train": processed/"refs_train.fasta", "val": processed/"refs_val.fasta", "test": processed/"refs_test.fasta" }
    tmp = { "train": [], "val": [], "test": [] }
    for r in records:
        tmp[ mapping[r['accession']] ].append(r)
    for sp, path in paths.items():
        with path.open("w") as f:
            for r in tmp[sp]:
                f.write(f">{r['accession']}\n")
                s=r["sequence"]
                for i in range(0, len(s), 60): f.write(s[i:i+60]+"\n")
        print(f"[done] {path}")

    # Optional: retrieval_topk.jsonl
    if args.write_topk:
        print("[info] computing retrieval_topk.jsonl (this can be slow for large N) ...")
        # For memory safety, cap to 200k; otherwise advise using FAISS offline.
        if len(records) > 200000:
            print("[warn] too many sequences for brute-force top-k; skipping")
        else:
            acc2neighbors = topk_neighbors(records, topk=args.topk, dim=2048)
            with (processed/"retrieval_topk.jsonl").open("w") as f:
                for r in records:
                    f.write(json.dumps({"accession": r["accession"], "exemplars": acc2neighbors.get(r["accession"], [])})+"\n")
            print(f"[done] {processed/'retrieval_topk.jsonl'}")

    # Summary
    summary = {
        "n_sequences": len(records),
        "splits": {sp: len(tmp[sp]) for sp in ("train","val","test")},
        "paths": {
            "prompt_pairs": str(processed/"prompt_pairs.jsonl"),
            "splits": str(processed/"splits.tsv"),
            "interim_filtered": str(interim/"filtered.jsonl"),
            "retrieval_db": str(retr_dir/"db.fasta"),
            "refs_train": str(paths["train"]),
            "refs_val": str(paths["val"]),
            "refs_test": str(paths["test"])
        }
    }
    with (processed/"build_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
'''
out_path = "/mnt/data/build_dataset.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(script)

out_path
