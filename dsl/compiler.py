
# dsl/compiler.py
from typing import List, Set, Dict, Any
from utils import AA_VOCAB

class MotifFSA:
    def __init__(self, sets: List[Set[str]], min_occurrences: int = 1):
        self.sets = sets
        self.len = len(sets)
        self.started = False
        self.idx = 0
        self.done = False
        self.min_occ = max(1, min_occurrences)
        self.completed = 0

    def reset(self):
        self.started = False
        self.idx = 0
        self.done = False
        self.completed = 0

    @property
    def start_set(self):
        return self.sets[0] if self.sets else set(AA_VOCAB)

    def allowed_now(self, steps_left: int):
        if (self.completed >= self.min_occ) or self.len == 0:
            return set(AA_VOCAB)
        if not self.started:
            return self.start_set
        if self.idx < self.len:
            return self.sets[self.idx]
        return set(AA_VOCAB)

    def step(self, aa: str):
        if self.completed >= self.min_occ or self.len == 0:
            return
        if not self.started:
            self.started = True
            self.idx = 0
        if aa in self.sets[self.idx]:
            self.idx += 1
            if self.idx >= self.len:
                self.completed += 1
                self.started = False
                self.idx = 0
                if self.completed >= self.min_occ:
                    self.done = True
        else:
            # mismatch: allow overlap restart if aa can start
            if aa in self.sets[0]:
                self.started = True
                self.idx = 1
            else:
                self.started = False
                self.idx = 0

def tokenclass(c: str):
    # map motif spec char to set of AA
    from utils import AA_VOCAB
    if c == "X" or c == ".":
        return set(AA_VOCAB)
    if c == "[ST]":
        return set("ST")
    return set(c)

def regex_to_sets(rx: str) -> List[Set[str]]:
    # handle patterns like G.[ST][AGST]G
    sets: List[Set[str]] = []
    i = 0
    while i < len(rx):
        ch = rx[i]
        if ch == "[":
            j = rx.index("]", i+1)
            choices = rx[i+1:j]
            sets.append(set(choices))
            i = j+1
        elif ch == ".":
            from utils import AA_VOCAB
            sets.append(set(AA_VOCAB))
            i += 1
        else:
            sets.append(set(ch))
            i += 1
    return sets

def compile_constraints(dsl_tokens: Dict[str, Any], max_len=1024, min_occurrences: int = 1):
    motifs = dsl_tokens.get("MOTIF", [])
    fsas = []
    for m in motifs:
        if not m: continue
        sets = regex_to_sets(m)
        fsas.append(MotifFSA(sets, min_occurrences=min_occurrences))
    return {"fsas": fsas, "max_len": max_len}

def all_done(fsas: List[MotifFSA]) -> bool:
    return all((f.completed >= f.min_occ) or f.len == 0 for f in fsas)
