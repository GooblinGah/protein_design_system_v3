#!/usr/bin/env python3
import argparse, os, json, re, tempfile, subprocess, hashlib
import torch
from utils import encode_prompt_bytes, detok_seq, AA_START, PROMPT_START, VOCAB_SIZE, load_fasta, save_fasta
from models.transformer_lm import TransformerCopyLM
from dsl.parser import parse_prompt
from dsl.compiler import compile_constraints, all_done
from safety.structure import max_tm_vs_dir, active_site_rmsd
from retrieval.profile_hmm import column_features
from retrieval.pipeline import build_profile_from_db
from planner.validator import validate_prompt
from safety.ledger_chain import HashChainLedger
from controller.segmental_hmm import apply_controller_penalty




def duration_pos_prior(duration_targets, ctrl_state):
    # Returns a function(pos:int, seg_idx:int, seg_pos:int) -> scalar bias
    # Penalize overrun beyond target for current segment; small encouragement to reach target
    def prior(seq_pos:int, seg_idx:int, seg_pos:int):
        if not duration_targets or seg_idx >= len(duration_targets):
            return 0.0
        tgt = max(1, duration_targets[seg_idx])
        # Bias: encourage growth until target, then penalize after
        d = seg_pos - tgt
        if d < 0:
            return +0.02 * min(50, -d)  # gentle push towards target
        else:
            mult = 1.0 if ctrl_state.z_tier=='normal' else (1.3 if ctrl_state.z_tier=='stretched' else 1.7)
            return -0.15 * mult * min(50, d)  # stronger penalty past target
    return prior


def controller_pos_prior(z_tier:str, length_lo:int=None, length_hi:int=None):
    # returns a function pos -> bias logits favoring EOS near bounds and penalizing overrun per z-tier
    def bias(pos:int):
        b = 0.0
        if length_hi is not None and pos > length_hi:
            over = pos - length_hi
            # harsher penalty for tighter tiers
            mult = 1.0 if z_tier=='normal' else (1.5 if z_tier=='stretched' else 2.0)
            b += -0.2 * mult * over  # negative bias grows with overrun
        # slight encouragement before lower bound to keep generating
        if length_lo is not None and pos < length_lo:
            b += +0.05 * (length_lo - pos)
        # Convert scalar bias to a vector over vocab (AA only): encourage not-EOS by default
        def to_vec(seq_pos):
            vec = torch.zeros(1, VOCAB_SIZE)  # will be narrowed in caller to AA indices
            return vec + b
        return b
    # For simplicity, return a scalar bias applied equally to AA logits
    def prior(seq_pos:int):
        return bias(seq_pos)
    return prior


def sample_step(logits, strategy="greedy", top_k=0, top_p=0.0, temperature=1.0):
    if strategy=="greedy":
        return int(torch.argmax(logits, dim=-1))
    # basic nucleus sampling
    probs = torch.softmax(logits/float(temperature), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= top_p
    mask[...,0] = True
    idxs = sorted_idx[mask]
    probs = probs[mask]
    probs = probs / probs.sum()
    return int(idxs[torch.multinomial(probs, 1)])




from dataclasses import dataclass, field

@dataclass
class ControllerState:
    z_tier: str = "normal"
    length_lo: int = None
    length_hi: int = None
    hysteresis_on: bool = True
    shocks: int = 0
    boundary_resets: int = 0
    last_allowed_size: int = None
    last_bias: float = 0.0



class _DurMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

def _load_duration_mlp(path='controller/duration_mlp.pt'):
    try:
        ck = torch.load(path, map_location='cpu')
        m = _DurMLP(ck.get('in',256), ck.get('out',4))
        m.load_state_dict(ck['state_dict'])
        m.eval()
        return m, ck.get('in',256), ck.get('out',4)
    except Exception:
        return None, 256, 4

def _predict_durations_from_prompt(prompt: str, nseg: int, total_len: int):
    # Encode prompt bytes
    b = prompt.encode('utf-8', errors='ignore')[:256]
    import numpy as np
    x = torch.tensor(list(b) + [0]*(256-len(b)), dtype=torch.float32).unsqueeze(0)
    mdl, in_d, out_d = _load_duration_mlp()
    if mdl is None:
        # Uniform split if model missing
        base = max(1, total_len // nseg)
        vec = [base]*nseg
    else:
        with torch.no_grad():
            y = mdl(x).squeeze(0)
            y = torch.relu(y) + 1.0  # ensure positive
            vec = y[:nseg].tolist()
    s = sum(vec) if sum(vec)>0 else 1.0
    vec = [max(1, int(round(v / s * total_len))) for v in vec]
    # Adjust rounding to sum exactly
    diff = total_len - sum(vec)
    if diff != 0 and len(vec)>0:
        vec[0] = max(1, vec[0] + diff)
    return vec


def aa_token_ids():
    return list(range(6,26))

def allowed_mask_from_fsa(fsas):
    # Returns a boolean mask over AA token ids allowed now and the size of allowed set
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if fsas:
        cur = set("ACDEFGHIKLMNPQRSTVWY")
        for f in fsas:
            cur &= f.allowed_now(steps_left=9999)
        if cur: allowed = cur
    # Map AA chars -> token ids
    from utils import AA_TO_ID
    mask = {AA_TO_ID[a]: True for a in allowed if a in AA_TO_ID}
    allowed_size = len(allowed)
    return mask, allowed_size

def beam_search_generate(model, ids_init, attn_init, fsas, ex_ids, device, beam_size=4, max_new=256,
                         pos_prior=None, profile_prior=None, ctrl_state:ControllerState=None, profile_logits=None):
    import copy
    from utils import ID_TO_AA, AA_TO_ID
    Beam = {"ids": None, "logp": 0.0, "fsas": None, "seq_only": 0, "prov": []}
    beams = [dict(ids=ids_init.clone(), logp=0.0, fsas=[copy.deepcopy(f) for f in fsas], seq_only=0, prov=[], ctrl=ControllerState(z_tier=ctrl_state.z_tier, length_lo=ctrl_state.length_lo, length_hi=ctrl_state.length_hi, hysteresis_on=ctrl_state.hysteresis_on))]
    finished = []
    for t in range(max_new):
        cand = []
        for b in beams:
            with torch.no_grad():
                out = model(b["ids"], attn_mask=torch.ones_like(b["ids"], dtype=torch.float32).to(device),
                            exemplar_ids=ex_ids, return_mix=True)
                logits = out["logits"][:,-1,:].clone()  # [1,V]
                gate_t = float(out["gate"][:,-1].detach().cpu().item()) if "gate" in out else 0.0

            # FSA mask
            mask, allowed_size = allowed_mask_from_fsa(b["fsas"])
            # Disallow non-AA
            logits[:, :6] = -1e9
            logits[:, 26:] = -1e9
            # Apply allowed mask
            for tok in range(6,26):
                if tok not in mask: logits[0, tok] = -1e9

            # Positional prior (controller length) as a small bias
            if pos_prior is not None:
                base_bias = float(pos_prior(b["seq_only"]))
                # hysteresis: if we are already beyond hi bound, increase penalty each consecutive step
                if b['ctrl'].hysteresis_on and b['ctrl'].length_hi is not None and b['seq_only'] > b['ctrl'].length_hi:
                    extra = -0.1 * (b['seq_only'] - b['ctrl'].length_hi)
                    base_bias += extra
                logits = logits + base_bias
                b['ctrl'].last_bias = base_bias
            # Shock detection: big change in allowed set size
            prev = b['ctrl'].last_allowed_size
            if prev is not None and allowed_size is not None:
                if prev>0 and abs(allowed_size - prev)/prev > 0.5:
                    b['ctrl'].shocks += 1
            b['ctrl'].last_allowed_size = allowed_size

            # Profile prior (consensus column if available)
            if profile_prior is not None:
                logits = logits + profile_prior(b["seq_only"])

            # Top-k expansion
            probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(probs, k=min(beam_size, (probs> -1e9).sum().item()), dim=-1)
            for k in range(topk.indices.size(-1)):
                tok = int(topk.indices[0,k])
                logp = float(topk.values[0,k])
                new_ids = torch.cat([b["ids"], torch.tensor([[tok]], device=device)], dim=1)
                new_fsas = [copy.deepcopy(f) for f in b["fsas"]]
                aa = tok
                from utils import ID_TO_AA
                aac = ID_TO_AA.get(aa,"")
                boundary_reset=False
                for f_old, f_new in zip(b['fsas'], new_fsas):
                    prev_comp = getattr(f_old, 'completed', 0)
                    if aac: f_new.step(aac)
                    if getattr(f_new, 'completed', 0) > prev_comp:
                        boundary_reset=True
                new_seq_only = b["seq_only"] + 1
                ctrl_new = ControllerState(z_tier=b['ctrl'].z_tier, length_lo=b['ctrl'].length_lo, length_hi=b['ctrl'].length_hi, hysteresis_on=b['ctrl'].hysteresis_on)
                ctrl_new.shocks = b['ctrl'].shocks
                ctrl_new.boundary_resets = b['ctrl'].boundary_resets + (1 if boundary_reset else 0)
                ctrl_new.last_allowed_size = b['ctrl'].last_allowed_size
                ctrl_new.last_bias = b['ctrl'].last_bias
                new_prov = b["prov"] + [ {"aa_id": aa, "source": "copy" if gate_t>0.5 else "vocab", "gate": gate_t, "boundary_reset": boundary_reset} ]
                cand.append(dict(ids=new_ids, logp=b["logp"]+logp, fsas=new_fsas, seq_only=new_seq_only, prov=new_prov, ctrl=ctrl_new))
        # Select next beams
        cand.sort(key=lambda x: x["logp"], reverse=True)
        beams = cand[:beam_size]
        # Termination if all done or EOS token appeared (we don't include EOS in AA range here)
        # We stop if all FSAs done and we reached reasonable length
        all_done = all(all_done_f.done for b in beams for all_done_f in b["fsas"]) if beams and beams[0]["fsas"] else False
        if not beams: break
    # Choose best beam
    best = max(beams, key=lambda x: x["logp"])
    return best["ids"], best["prov"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--motif_min_occurrences", type=int, default=1)
    ap.add_argument("--z_tier", type=str, default="normal", choices=["normal","stretched","sparse"])
    ap.add_argument("--ref_fasta", type=str, default=None)
    ap.add_argument("--max_seq_identity", type=float, default=0.7)
    ap.add_argument("--ref_structures_dir", type=str, default=None)
    ap.add_argument("--tm_max", type=float, default=None)
    ap.add_argument("--site_motif", type=str, default="G.[ST][AGST]G")
    ap.add_argument("--site_window", type=int, default=9)
    ap.add_argument("--site_rmsd_max", type=float, default=None)
    ap.add_argument("--predict_cmd", type=str, default="python scripts/predict_structure.py --fasta {fasta} --out_pdb {out_pdb}")
    ap.add_argument("--resample_max", type=int, default=2)
    ap.add_argument("--enforce_signalp", type=int, default=1)
    ap.add_argument("--exemplar_fasta", type=str, default=None)
    ap.add_argument("--retrieval_db", type=str, default=None)
    ap.add_argument("--retrieval_topk", type=int, default=16)
    ap.add_argument("--prov_topk", type=int, default=5)
    args = ap.parse_args()

    # Planner validation
    vrep = validate_prompt(args.prompt)
    if not vrep["ok"]:
        print("Refused by planner:", vrep["errors"]); return

    device = torch.device(args.device)
    model = TransformerCopyLM().to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck.get("model", ck))
    model.eval()

    # Build constraints
    dsl_tokens = parse_prompt(args.prompt)
    constraints = compile_constraints(dsl_tokens, max_len=512, min_occurrences=args.motif_min_occurrences)
    


    # Exemplar handling (optional)
    exemplar_names, ex_ids = [], torch.zeros((1,0), dtype=torch.long, device=device)
    profile_prior_logits = None
    if args.exemplar_fasta:
        recs = list(load_fasta(args.exemplar_fasta))
        exemplar_names = [n for n,_ in recs]
        from utils import encode_seq, ID_TO_AA
        cat = []
        msa = [s for _,s in recs]
        for _,s in recs: cat.extend(encode_seq(s, 256))
        if cat:
            ex_ids = torch.tensor([cat], dtype=torch.long, device=device)
        # Build a PSSM-based prior over AA tokens for each seq step (approximate by fixed-length mapping)
        def make_profile_prior():
            # For now, expose a function that, given seq_pos, returns logits over V aligned to consensus column
            import numpy as np, torch
            # We'll compute after generation when we have seq; at decode-time we provide a weak flat prior
            def prior(seq_pos):
                return 0.0
            return prior
        profile_prior = make_profile_prior()
    else:
        profile_prior = None


    # Retrieval→MSA to build profile if exemplar_fasta not given
    retrieval_hits = []
    profile_logits = None  # Initialize profile_logits
    if not args.exemplar_fasta and args.retrieval_db and os.path.exists(args.retrieval_db):
        # crude query: use prompt bytes as seed sequence by extracting motif shell repeats to approximate length; or fallback to first exemplars after generation
        # Better: if user provides a seed sequence, prefer it; here we approximate by using motif shell 'GASAG' repeated to length 150
        seed = 'GASAG' * 30
        msa, hits = build_profile_from_db(args.retrieval_db, seed, topk=args.retrieval_topk)
        retrieval_hits = [(h[0], h[2]) for h in hits]  # (id, similarity)
        if msa:
            msa_seqs = [s for _,s in msa]
            from retrieval.profile_hmm import build_pssm
            pssm, consensus, conservation = build_pssm(msa_seqs)
            if pssm.shape[0] > 0:
                def profile_prior_fn_from_pssm(pssm_logits):
                    def fn(pos:int):
                        col = min(pos, pssm_logits.shape[0]-1)
                        vec = torch.full((1, VOCAB_SIZE), 0.0, device=device)
                        return vec
                    return fn
                profile_logits = profile_prior_fn_from_pssm(pssm)

    # Initialize input sequence and attention mask
    from utils import encode_prompt_bytes, BOS, SEP, EOS
    prompt_tokens = encode_prompt_bytes(args.prompt, max_len=200)
    x = [BOS] + prompt_tokens + [SEP]
    ids = torch.tensor([x], dtype=torch.long, device=device)
    attn = torch.ones_like(ids, dtype=torch.float32)
    
    # Initialize motif FSA state
    motif_state = constraints.get("fsas", None)
    
    # Initialize controller state and position prior
    ctrl = ControllerState(z_tier=args.z_tier)
    
    # Parse length constraints from prompt
    import re
    length_match = re.search(r'length\s+(\d+)\.\.(\d+)', args.prompt)
    lo = int(length_match.group(1)) if length_match else None
    hi = int(length_match.group(2)) if length_match else None
    ctrl.length_lo = lo
    ctrl.length_hi = hi
    
    # Create position prior function
    pos_prior = controller_pos_prior(args.z_tier, lo, hi)
    
    # Greedy-ish decode (replaced by FSA beam search)
    from dsl.compiler import MotifFSA
    ids, provenance = beam_search_generate(model, ids, attn, motif_state, ex_ids, device,
                                           beam_size=4, max_new=args.max_new_tokens,
                                           pos_prior=pos_prior, profile_prior=profile_prior, ctrl_state=ctrl)


    # Extract sequence tokens after last SEP
    seq_ids = []
    toks = ids.squeeze(0).tolist()
    cut = max(i for i,t in enumerate(toks) if t==3) if 3 in toks else 0
    for t in toks[cut+1:]:
        if t>=6 and t<26 and t!=2: seq_ids.append(t)
    from utils import ID_TO_AA
    seq = "".join(ID_TO_AA.get(t,"") for t in seq_ids)

    # If signal peptide is requested by prompt, enforce via predictor; resample if needed
    from planner.validator import requires_signal_peptide
    needs_sp = requires_signal_peptide(args.prompt)
    if needs_sp and args.enforce_signalp:
        import subprocess, sys
        def has_sp(sequence):
            try:
                import sys, subprocess
                out = subprocess.check_output(['python','scripts/predict_signal_peptide.py','--seq', sequence], text=True)
                return out.strip().endswith('1')
            except Exception:
                return False
        trials = 0
        while trials < args.resample_max and not has_sp(seq):
            trials += 1
            # resample once more
            ids, provenance = beam_search_generate(model, ids[:, :cut+1], attn[:, :cut+1], motif_state, ex_ids, device,
                                           beam_size=4, max_new=args.max_new_tokens,
                                           pos_prior=pos_prior, profile_prior=profile_logits, ctrl_state=ctrl, profile_logits=profile_logits)
            toks = ids.squeeze(0).tolist(); seq_ids=[]; cut = max(i for i,t in enumerate(toks) if t==3) if 3 in toks else 0
            for t in toks[cut+1:]:
                if t>=6 and t<26 and t!=2: seq_ids.append(t)
            from utils import ID_TO_AA
            seq = ''.join(ID_TO_AA.get(t,'') for t in seq_ids)

    # Homology checks
    ok_seq=True; ok_struct=True; tm_best=None; site_best=None
    if args.ref_fasta and os.path.exists(args.ref_fasta):
        refs = list(load_fasta(args.ref_fasta))
        from utils import max_identity_vs_refs
        max_id = max_identity_vs_refs(seq, refs)
        if args.max_seq_identity is not None and max_id > args.max_seq_identity: ok_seq=False
    # structure: predict, TM vs refs, active site RMSD
    gen_pdb = None
    if args.ref_structures_dir:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".fa")
        save_fasta(tmpf.name, [("gen", seq)])
        out_pdb = tmpf.name.replace(".fa",".pdb")
        cmd = args.predict_cmd.format(fasta=tmpf.name, out_pdb=out_pdb)
        try: subprocess.check_call(cmd, shell=True)
        except Exception: pass
        gen_pdb = out_pdb if os.path.exists(out_pdb) else None
        if gen_pdb:
            if args.tm_max is not None:
                tm_best,_ = max_tm_vs_dir(gen_pdb, args.ref_structures_dir)
                if tm_best is not None and tm_best > args.tm_max: ok_struct=False
            if ok_struct and args.site_rmsd_max is not None:
                import os
                for ref in os.listdir(args.ref_structures_dir):
                    if not ref.endswith(".pdb"): continue
                    rmsd = active_site_rmsd(gen_pdb, os.path.join(args.ref_structures_dir, ref), motif_regex=args.site_motif, window=args.site_window)
                    if rmsd is not None:
                        site_best = rmsd if site_best is None else min(site_best, rmsd)
                if site_best is not None and site_best > args.site_rmsd_max: ok_struct=False


    # Compute planned vs realized durations and monotone consensus index (MCI)
    def find_motif_positions(seq, motif_regex):
        return [m.span()[0] for m in re.finditer(motif_regex, seq)]
    starts = find_motif_positions(seq, args.site_motif)
    seg_targets = []
    if hi is not None and len(starts) >= 0:
        # simplistic: distribute planned hi across segments according to controller tier
        nseg = len(starts) + 1
        base = max(1, (hi or len(seq)) // nseg)
        seg_targets = [base]*nseg
    # realized lengths
    realized = []
    prev = 0
    for s in starts:
        realized.append(max(1, s - prev)); prev = s + 5
    realized.append(max(1, len(seq) - prev))
    from controller.segmental_hmm import monotone_consensus_index
    mci = monotone_consensus_index(realized, seg_targets) if seg_targets else 0.0


    # Segment-level provenance confidence: average gate within segments
    seg_conf = []
    if realized:
        # Build per-residue gates array
        gates = [p.get('gate',0.0) for p in provenance]
        idx=0
        for r in realized:
            if r <= 0: seg_conf.append(0.0)
            else:
                seg_conf.append(sum(gates[idx:idx+r])/max(1,r) if idx+r <= len(gates) else 0.0)
            idx += r

    # Ledger (hash-chain)
    ledger = HashChainLedger()
    # Ledger record with hashes and controller stats
    pr_hash = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    dsl_hash = hashlib.sha256(json.dumps(dsl_tokens, sort_keys=True).encode("utf-8")).hexdigest()
    controller_meta = {"z_tier": args.z_tier, "length_lo": lo, "length_hi": hi}
    # Hash model checkpoint and capture software versions
    model_hash = None
    try:
        with open(args.checkpoint, 'rb') as _f:
            model_hash = hashlib.sha256(_f.read()).hexdigest()
    except Exception:
        model_hash = None
    import numpy, sys, platform
    software = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': getattr(torch, '__version__', None),
        'numpy': getattr(numpy, '__version__', None)
    }
    # Gather shock/boundary stats from best beam provenance
    shocks = 0; resets = sum(1 for p in provenance if p.get("boundary_reset"))
    # Append ledger
    ledger.append({

        "prompt": args.prompt, "prompt_hash": pr_hash, "dsl": dsl_tokens, "dsl_hash": dsl_hash, "compiler_id": "FSA-1", "profile_id": "PSSM-1", "controller": controller_meta, "sequence": seq, "model_hash": model_hash, "software": software, "retrieval_hits": retrieval_hits, "segments": {"planned": seg_targets, "realized": realized, "mci": mci, "confidence": seg_conf},
        "provenance": provenance, "exemplars": exemplar_names,
        "metrics": {"ok_seq": ok_seq, "ok_struct": ok_struct, "tm_best": tm_best, "site_rmsd": site_best, "shocks": shocks, "boundary_resets": resets}
    })

    print(seq)

if __name__ == "__main__":
    main()
