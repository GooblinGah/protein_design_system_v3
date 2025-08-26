
# planner/validator.py
import re
NEGATIVE_REGEX = [r"botulinum", r"ricin", r"neurotoxin", r"hemolysin", r"enterotoxin", r"\btoxin\b"]
GO_EC_WHITELIST = {"hydrolase", "transferase", "oxidoreductase", "EC:3.", "EC:2.", "EC:1."}
def validate_prompt(prompt: str):
    errors=[]; warnings=[]
    for pat in NEGATIVE_REGEX:
        if re.search(pat, prompt, re.I): errors.append(f"blocked keyword matched: {pat}")
    if not any(k.lower() in prompt.lower() for k in GO_EC_WHITELIST):
        warnings.append("no recognized GO/EC keywords; permissive but flagged")
    if "secret" in prompt.lower():
        warnings.append("consider enforcing signal peptide via DSL/constraints")
    return {"ok": len(errors)==0, "errors": errors, "warnings": warnings}
def refuse_if_needed(report):
    return (not report.get("ok", False)), "; ".join(report.get("errors", []))


import re

FORBIDDEN_REGEX = [
    r".{0,10}(.{6,})\1{2,}",  # repeats,
    r"[GP]{5,}",
    r"C{6,}",
    r"N{5,}", r"[KR]{6,}", r"[DE]{6,}"
]

def extract_length_window(prompt: str):
    # Find patterns like 'length 220..280' or 'len 150-250'
    m = re.search(r"(?:length|len)\s*(\d{2,4})\s*[\.\-–]+\s*(\d{2,4})", prompt, re.I)
    if not m: 
        return None, None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi: lo, hi = hi, lo
    return lo, hi

def requires_signal_peptide(prompt: str):
    return bool(re.search(r"\b(secreted|signal peptide)\b", prompt, re.I))

from .ontology import resolve_ontology
from pathlib import Path
import json

AUDIT_PATH = Path("data_pipeline/data/planner_refusals.jsonl")

def audit_refusal(prompt: str, errors, warnings, ontology):
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps({"prompt": prompt, "errors": errors, "warnings": warnings, "ontology": ontology})+"\n")

def validate_prompt(prompt: str):
    errors=[]; warnings=[]
    for pat in NEGATIVE_REGEX:
        if re.search(pat, prompt, re.I): errors.append(f"blocked keyword matched: {pat}")
    if not any(k.lower() in prompt.lower() for k in GO_EC_WHITELIST):
        warnings.append("no recognized GO/EC keywords; permissive but flagged")
    if "secret" in prompt.lower():
        warnings.append("consider enforcing signal peptide via DSL/constraints")
    # ontology resolve
    ont = resolve_ontology(prompt)
    if not (ont['GO'] or ont['EC'] or ont['PFAM']):
        warnings.append("no ontology hits (GO/EC/PFAM)")
    ok = len(errors)==0
    if not ok:
        audit_refusal(prompt, errors, warnings, ont)
    return {"ok": ok, "errors": errors, "warnings": warnings, "ontology": ont}
