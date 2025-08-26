
# dsl/parser.py
import re
from typing import Dict, Any

def parse_prompt(prompt: str) -> Dict[str, Any]:
    # Extract motif from prompt, handling various formats
    # Look for "motif GXSXG" or similar patterns
    motif_patterns = [
        r"motif\s+(G\.\[?S[^\]]*\]?XG|G.?S.?G|G\.[ST]\[[A-Z]+\]G|GXSXG)",
        r"G\.\[?S[^\]]*\]?XG|G.?S.?G|G\.[ST]\[[A-Z]+\]G|GXSXG"
    ]
    
    motif = None
    for pattern in motif_patterns:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            motif = m.group(1) if m.groups() else m.group(0)
            break
    
    # If no motif found, use default
    if not motif:
        motif = "G.[ST][AGST]G"
    
    return {"MOTIF": [motif]}
