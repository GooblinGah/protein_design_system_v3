# planner/ontology.py
import re

# Extremely lightweight keyword/regex resolvers for GO/EC/Pfam
GO_HINTS = {
    'hydrolase': r'GO:0016787',
    'transferase': r'GO:0016740',
    'oxidoreductase': r'GO:0016491',
}
EC_HINTS = [
    r'EC:1\.[0-9]+\.[0-9]+\.[0-9]+',
    r'EC:2\.[0-9]+\.[0-9]+\.[0-9]+',
    r'EC:3\.[0-9]+\.[0-9]+\.[0-9]+'
]
PFAM_HINTS = {
    'alpha beta hydrolase': 'PF12697|PF00561',
    'serine protease': 'PF00089|PF13365',
    'TIM-barrel': 'PF00198|PF14670'
}

def resolve_ontology(prompt: str):
    hits = {'GO': [], 'EC': [], 'PFAM': []}
    low = prompt.lower()
    for k,v in GO_HINTS.items():
        if k in low: hits['GO'].append(v)
    for rx in EC_HINTS:
        for m in re.findall(rx, prompt):
            hits['EC'].append(m)
    for k,v in PFAM_HINTS.items():
        if k in low: hits['PFAM'].append(v)
    return hits


# Expanded hints
GO_HINTS.update({
    'kinase': 'GO:0016301',
    'phosphatase': 'GO:0016791',
    'glycosidase': 'GO:0016798',
    'lipase': 'GO:0016298',
    'protease': 'GO:0008233',
    'secreted': 'GO:0005576',  # extracellular region
    'membrane': 'GO:0016020',
})
PFAM_HINTS.update({
    'kinase': 'PF00069|PF07714',
    'lipase': 'PF00151|PF01764',
    'serpin': 'PF00079',
    'beta-lactamase': 'PF00144',
    'alpha/beta hydrolase': 'PF12697|PF00561',
    'thioredoxin': 'PF00085',
})
# Keep EC regex scan but add common text shortcuts -> canonical examples
EC_SHORTCUTS = {
    'beta-lactamase': 'EC:3.5.2.6',
    'alkaline phosphatase': 'EC:3.1.3.1',
    'glucose oxidase': 'EC:1.1.3.4',
    'lactate dehydrogenase': 'EC:1.1.1.27'
}

def resolve_ontology(prompt: str):
    hits = {'GO': [], 'EC': [], 'PFAM': []}
    low = prompt.lower()
    for k,v in GO_HINTS.items():
        if k in low: hits['GO'].append(v)
    for rx in EC_HINTS:
        for m in re.findall(rx, prompt):
            hits['EC'].append(m)
    for k,v in EC_SHORTCUTS.items():
        if k in low: hits['EC'].append(v)
    for k,v in PFAM_HINTS.items():
        if k in low: hits['PFAM'].append(v)
    # Deduplicate
    for k in hits:
        dedup = []
        for x in hits[k]:
            if x not in dedup: dedup.append(x)
        hits[k] = dedup
    return hits
