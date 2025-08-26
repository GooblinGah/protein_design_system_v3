
# safety/liability.py
import re
def has_homopolymer(seq, k=6):
    return any(ch*k in seq for ch in "ACDEFGHIKLMNPQRSTVWY")
def hydrophobic_run(seq, k=7):
    H = set("AILMFWV")
    cnt=0
    for c in seq:
        cnt = cnt+1 if c in H else 0
        if cnt>=k: return True
    return False
def low_complexity(seq, k=12, distinct=3):
    for i in range(0, max(0,len(seq)-k+1)):
        if len(set(seq[i:i+k]))<=distinct: return True
    return False
