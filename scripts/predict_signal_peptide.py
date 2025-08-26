#!/usr/bin/env python3
import argparse, re, sys, os, subprocess

def heuristic_signal_peptide(seq):
    # Simplified Sec signal peptide heuristic: n-region (1-5 + charge), h-region (7-15 hydrophobics), c-region (AXA motif near 20-35)
    n = seq[:5]
    h = seq[5:25]
    c = seq[20:40] if len(seq)>=40 else seq[max(0,len(seq)-20):]
    charge = sum(1 for x in n if x in 'KRH')
    hydros = set('AILMFWV')
    hlen = 0; best=0
    for ch in h:
        if ch in hydros: hlen += 1; best = max(best, hlen)
        else: hlen = 0
    axa = bool(re.search(r'.[ASTVILMFYW]A', c))  # loose A-X-A
    return (charge>=1 and best>=7 and axa)

def run_signalp(seq):
    # If external SignalP is installed, try to call it (placeholder, not guaranteed)
    if os.environ.get('SIGNALP_CMD'):
        try:
            out = subprocess.check_output(os.environ['SIGNALP_CMD'], shell=True, text=True)
            return 'SP=' in out or 'signal peptide' in out.lower()
        except Exception:
            return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    a = ap.parse_args()
    ext = run_signalp(a.seq)
    if ext is True:
        print('1'); sys.exit(0)
    if ext is False:
        print('0'); sys.exit(1)
    print('1' if heuristic_signal_peptide(a.seq) else '0')

if __name__ == '__main__':
    main()
