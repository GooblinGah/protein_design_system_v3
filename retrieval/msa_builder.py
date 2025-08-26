# retrieval/msa_builder.py
import os, subprocess, tempfile

def write_fasta(path, recs):
    with open(path,'w') as f:
        for n,s in recs: f.write(f'>{n}\n{s}\n')

def run_muscle(in_fa, out_fa):
    cmd = os.environ.get('MUSCLE_CMD', 'muscle -align {in_fa} -output {out_fa} -quiet')
    try:
        subprocess.check_call(cmd.format(in_fa=in_fa, out_fa=out_fa), shell=True)
        return True
    except Exception:
        return False

def simple_stack(seqs):
    # naive column stack by padding to max length
    L = max(len(s) for _,s in seqs) if seqs else 0
    return [(n, s.ljust(L,'-')) for n,s in seqs]

def build_msa(top_hits):
    # Try MUSCLE, fallback to simple pad
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fa') as fin,          tempfile.NamedTemporaryFile(delete=False, suffix='.fa') as fout:
        write_fasta(fin.name, [(n,s) for n,s,_ in top_hits])
        ok = run_muscle(fin.name, fout.name)
        if ok:
            # return records from aligned fasta
            recs=[]; name=None; seq=[]
            for line in open(fout.name):
                line=line.strip()
                if not line: continue
                if line.startswith('>'):
                    if name is not None: recs.append((name,"".join(seq))); seq=[]
                    name=line[1:]
                else:
                    seq.append(line)
            if name is not None: recs.append((name,"".join(seq)))
            return recs
    return simple_stack([(n,s) for n,s,_ in top_hits])
