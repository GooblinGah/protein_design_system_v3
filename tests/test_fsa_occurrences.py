
from dsl.compiler import compile_constraints, MotifFSA, all_done, regex_to_sets
from dsl.parser import parse_prompt

def test_fsa_min_occurrences():
    dsl = {"MOTIF": ["G.[ST][AGST]G"]}
    cons = compile_constraints(dsl, min_occurrences=2)
    fsa = cons["fsas"][0]
    
    for _ in range(2):
        for a in list("GASAG"): fsa.step(a)
    assert fsa.completed >= 2
    assert all_done(cons["fsas"])

def test_fsa_single_occurrence():
    dsl = {"MOTIF": ["GXSXG"]}
    cons = compile_constraints(dsl, min_occurrences=1)
    fsa = cons["fsas"][0]
    
    for a in list("GXSXG"): fsa.step(a)
    assert fsa.completed >= 1
    assert all_done(cons["fsas"])

def test_fsa_no_motifs():
    dsl = {"MOTIF": []}
    cons = compile_constraints(dsl, min_occurrences=1)
    assert len(cons["fsas"]) == 0
    assert all_done(cons["fsas"])

def test_fsa_reset():
    dsl = {"MOTIF": ["G.[ST][AGST]G"]}
    cons = compile_constraints(dsl, min_occurrences=1)
    fsa = cons["fsas"][0]
    
    for a in list("GASAG"): fsa.step(a)
    assert fsa.completed >= 1
    
    fsa.reset()
    assert fsa.completed == 0
    assert not fsa.done
    assert not fsa.started

def test_regex_to_sets():
    sets = regex_to_sets("G.[ST][AGST]G")
    assert len(sets) == 5
    assert sets[0] == {'G'}
    assert sets[1] == {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    assert sets[2] == {'S', 'T'}
    assert sets[3] == {'A', 'G', 'S', 'T'}
    assert sets[4] == {'G'}

def test_parse_prompt():
    prompt1 = "alpha beta hydrolase with motif GXSXG, length 260..320, secreted"
    result1 = parse_prompt(prompt1)
    assert "MOTIF" in result1
    assert len(result1["MOTIF"]) > 0
    
    prompt2 = "Design a protein with G.[ST][AGST]G pattern"
    result2 = parse_prompt(prompt2)
    assert "MOTIF" in result2
    assert len(result2["MOTIF"]) > 0
    
    prompt3 = "Just a regular protein description"
    result3 = parse_prompt(prompt3)
    assert "MOTIF" in result3
    assert len(result3["MOTIF"]) > 0

def test_fsa_allowed_now():
    dsl = {"MOTIF": ["G.[ST][AGST]G"]}
    cons = compile_constraints(dsl, min_occurrences=1)
    fsa = cons["fsas"][0]
    
    allowed = fsa.allowed_now(steps_left=9999)
    assert 'G' in allowed
    
    fsa.step('G')
    allowed = fsa.allowed_now(steps_left=9999)
    assert len(allowed) > 0

def test_fsa_overlap_restart():
    dsl = {"MOTIF": ["G.[ST][AGST]G"]}
    cons = compile_constraints(dsl, min_occurrences=1)
    fsa = cons["fsas"][0]
    
    fsa.step('G')
    fsa.step('X')
    assert not fsa.started
    
    fsa.step('G')
    assert fsa.started
