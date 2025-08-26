#!/usr/bin/env python3
import json, argparse, math, os
import matplotlib.pyplot as plt

def load_ledger(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]

def copy_rate_vs_conservation(entries, cons_by_entry, out='copy_vs_conservation.png'):
    xs=[]; ys=[]
    for i,e in enumerate(entries):
        prov = e.get('provenance', [])
        cons = cons_by_entry.get(i)
        if cons is None or not prov: 
            continue
        # align by min length
        L = min(len(cons), len(prov))
        for t in range(L):
            xs.append(cons[t])
            ys.append(prov[t].get('gate', 0.0))
    if xs:
        plt.figure(); plt.scatter(xs, ys, s=4)
        plt.xlabel('Conservation (PSSM max prob)'); plt.ylabel('Gate/copy prob')
        plt.title('Copy-rate vs conservation'); plt.savefig(out, dpi=160); plt.close()

def gate_hist(entries, out='gate_hist.png'):
    vals=[]
    for e in entries:
        vals += [p.get('gate',0.0) for p in e.get('provenance', [])]
    if vals:
        plt.figure(); plt.hist(vals, bins=30)
        plt.xlabel('Gate'); plt.ylabel('Count'); plt.title('Gate histogram')
        plt.savefig(out, dpi=160); plt.close()

def duration_vs_realized(entries, out='duration_vs_realized.png'):
    xs=[]; ys=[]
    for e in entries:
        plan_hi = (e.get('controller') or {}).get('length_hi')
        if plan_hi is None: continue
        xs.append(plan_hi); ys.append(len(e.get('sequence','')))
    if xs:
        plt.figure(); plt.scatter(xs, ys); plt.xlabel('Planned upper length'); plt.ylabel('Realized length')
        plt.title('Duration (upper) vs realized'); plt.savefig(out, dpi=160); plt.close()

def tier_occupancy(entries, out='tier_occupancy.png'):
    tiers = {'normal':0,'stretched':0,'sparse':0}
    for e in entries:
        t = (e.get('controller') or {}).get('z_tier','normal')
        tiers[t] = tiers.get(t,0)+1
    plt.figure(); plt.bar(list(tiers.keys()), list(tiers.values()))
    plt.title('z-tier occupancy'); plt.savefig(out, dpi=160); plt.close()

def shock_stats(entries, out='shock_stats.png'):
    vals=[(e.get('metrics') or {}).get('shocks',0) for e in entries]
    plt.figure(); plt.hist(vals, bins=20)
    plt.xlabel('Shocks per design'); plt.ylabel('Count'); plt.title('Shock statistics')
    plt.savefig(out, dpi=160); plt.close()

def compute_conservation_for_entries(entries, exemplar_fasta=None):
    # Returns dict: entry_idx -> [conservation per token] (0..1). If no exemplar_fasta, returns {}
    if exemplar_fasta is None or not os.path.exists(exemplar_fasta):
        return {}
    # Load exemplars
    def load_fasta(path):
        name, buf = None, []
        recs = []
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                if line.startswith('>'):
                    if name is not None:
                        recs.append((name, "".join(buf)))
                    name = line[1:]; buf=[]
                else:
                    buf.append(line)
        if name is not None:
            recs.append((name, "".join(buf)))
        return recs
    recs = load_fasta(exemplar_fasta)
    if not recs:
        return {}
    # Build PSSM and compute mapping
    from retrieval.profile_hmm import build_pssm, viterbi_profile
    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"; idx={a:i for i,a in enumerate(ALPHABET)}
    pssm, consensus, conservation = build_pssm([s for _,s in recs])
    cons_by_entry = {}
    if pssm.shape[0] == 0:
        return cons_by_entry
    for i,e in enumerate(entries):
        seq = e.get('sequence',"")
        if not seq: 
            continue
        mapping, score = viterbi_profile(pssm, seq)
        cons_seq = []
        for pos, col in enumerate(mapping):
            if col < 0:
                cons_seq.append(0.0)
            else:
                # conservation as max prob at that column (0..1 via softmax inverse)
                import numpy as np
                probs = np.exp(pssm[col] - pssm[col].max())
                probs = probs / probs.sum()
                cons_seq.append(float(probs.max()))
        cons_by_entry[i] = cons_seq
    return cons_by_entry

def motif_window_gate_hist(entries, motif_regex='G.[ST][AGST]G', window=9, out='motif_window_gate_hist.png'):
    import re
    vals=[]
    for e in entries:
        seq = e.get('sequence','')
        if not seq:  # Skip entries without sequence
            continue
        prov = e.get('provenance', [])
        for m in re.finditer(motif_regex, seq):
            s = max(0, m.start()-window//2)
            t = min(len(seq), m.start()+5+window//2)
            for i in range(s, t):
                if i < len(prov):
                    vals.append(prov[i].get('gate',0.0))
    if vals:
        import matplotlib.pyplot as plt
        plt.figure(); plt.hist(vals, bins=30)
        plt.xlabel('Gate in motif windows'); plt.ylabel('Count'); plt.title('Motif-window gate histogram')
        plt.savefig(out, dpi=160); plt.close()

def tier_occupancy_over_time(entries, out='tier_occupancy_over_time.png'):
    # Count designs per tier per run order (proxy for time)
    tiers = {'normal':[], 'stretched':[], 'sparse':[]}
    counts={'normal':0,'stretched':0,'sparse':0}
    for e in entries:
        t = (e.get('controller') or {}).get('z_tier','normal')
        counts[t]+=1
        for k in tiers.keys():
            tiers[k].append(counts[k])
    import matplotlib.pyplot as plt
    plt.figure()
    for k,v in tiers.items():
        plt.plot(range(len(v)), v, label=k)
    plt.legend(); plt.xlabel('Design index'); plt.ylabel('Cumulative count'); plt.title('Tier occupancy over time')
    plt.savefig(out, dpi=160); plt.close()

def per_segment_duration_error(entries, out='per_segment_duration_error.png'):
    xs=[]; ys=[]
    for e in entries:
        seg = e.get('segments') or {}
        planned = seg.get('planned') or []
        realized = seg.get('realized') or []
        L = min(len(planned), len(realized))
        for i in range(L):
            xs.append(i); ys.append(realized[i] - planned[i])
    if ys:
        import matplotlib.pyplot as plt
        plt.figure(); plt.scatter(xs, ys, s=6)
        plt.xlabel('Segment index'); plt.ylabel('Realized - Planned (aa)')
        plt.title('Per-segment duration error'); plt.savefig(out, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', default='data_pipeline/data/safety_ledger_generated.jsonl')
    ap.add_argument('--exemplar_fasta', default=None, help='FASTA to compute PSSM for conservation (optional)')
    a = ap.parse_args()
    ents = load_ledger(a.ledger)
    if not ents:
        print('No entries'); return
    cons = compute_conservation_for_entries(ents, a.exemplar_fasta)
    copy_rate_vs_conservation(ents, cons)
    gate_hist(ents)
    duration_vs_realized(ents)
    tier_occupancy(ents)
    shock_stats(ents)
    motif_window_gate_hist(ents)
    tier_occupancy_over_time(ents)
    per_segment_duration_error(ents)
    print('Wrote diagnostics PNGs.')

if __name__ == '__main__':
    main()
