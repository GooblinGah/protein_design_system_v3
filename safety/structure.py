
# safety/structure.py
import os, subprocess, glob
def which(cmd):
    from shutil import which as _w; return _w(cmd)

def _run_tmalign(gen_pdb: str, ref_pdb: str):
    if which("TMalign") is None: return None
    try:
        out = subprocess.check_output(["TMalign", gen_pdb, ref_pdb], stderr=subprocess.STDOUT, text=True)
        for line in out.splitlines():
            if "TM-score=" in line:
                try: return float(line.split("TM-score=")[1].split()[0])
                except Exception: continue
    except Exception:
        return None
    return None

def max_tm_vs_dir(gen_pdb: str, ref_dir: str):
    if not os.path.isdir(ref_dir): return None, None
    best_tm, best_ref = None, None
    pdbs = glob.glob(os.path.join(ref_dir, "*.pdb")) + glob.glob(os.path.join(ref_dir, "*.cif"))
    if not pdbs: return None, None
    for ref in pdbs:
        tm = _run_tmalign(gen_pdb, ref)
        if tm is None: continue
        if best_tm is None or tm > best_tm:
            best_tm, best_ref = tm, ref
    return best_tm, best_ref

AA3_TO_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def parse_pdb_ca(pdb_path):
    seq = []; coords=[]; last=None
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'): continue
                name = line[12:16].strip()
                if name != 'CA': continue
                resn = line[17:20].strip()
                chain = line[21].strip()
                resi = line[22:26].strip()
                key = (chain, resi)
                if key == last: continue
                last = key
                aa = AA3_TO_1.get(resn, 'X')
                if aa == 'X': continue
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                seq.append(aa); coords.append((x,y,z))
    except Exception:
        return "", []
    import numpy as np
    return "".join(seq), np.array(coords, dtype=float)

def find_motif_positions(seq, motif_regex):
    import re
    return [m.span()[0] for m in re.finditer(motif_regex, seq)]

def kabsch(P, Q):
    import numpy as np
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1; R = Vt.T @ U.T
    P_aligned = Pc @ R
    rmsd = np.sqrt(np.mean(((P_aligned - Qc)**2).sum(axis=1)))
    return rmsd

def active_site_rmsd(gen_pdb, ref_pdb, motif_regex='G.[ST][AGST]G', window=9):
    gseq, gcoords = parse_pdb_ca(gen_pdb); rseq, rcoords = parse_pdb_ca(ref_pdb)
    if not gseq or not rseq or len(gcoords)==0 or len(rcoords)==0: return None
    gstarts = find_motif_positions(gseq, motif_regex); rstarts = find_motif_positions(rseq, motif_regex)
    if not gstarts or not rstarts: return None
    half = window//2; best=None
    for gs in gstarts:
        gL = slice(max(0, gs-half), min(len(gcoords), gs+5+half)); G = gcoords[gL]
        for rs in rstarts:
            rL = slice(max(0, rs-half), min(len(rcoords), rs+5+half)); R = rcoords[rL]
            if len(G) != len(R) or len(G) < 5: continue
            rmsd = kabsch(G, R)
            if best is None or rmsd < best: best = rmsd
    return best
