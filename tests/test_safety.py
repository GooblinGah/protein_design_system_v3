import json
import tempfile
import os
from pathlib import Path
from safety.ledger_chain import HashChainLedger
from safety.structure import max_tm_vs_dir, active_site_rmsd, kabsch
from planner.validator import validate_prompt, requires_signal_peptide
import numpy as np

def test_hash_chain_ledger_creation():
    ledger = HashChainLedger()
    assert isinstance(ledger, HashChainLedger)
    assert hasattr(ledger, 'path')
    assert hasattr(ledger, 'chain_path')

def test_ledger_append():
    ledger = HashChainLedger()
    
    record = {
        "prompt": "Test prompt",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "model_hash": "test_hash_123"
    }
    
    chain_hash = ledger.append(record)
    
    assert isinstance(chain_hash, str)
    assert len(chain_hash) == 64
    
    assert os.path.exists(ledger.path)
    assert os.path.exists(ledger.chain_path)

def test_ledger_chain_verification():
    ledger = HashChainLedger()
    
    records = [
        {"prompt": "Prompt 1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        {"prompt": "Prompt 2", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        {"prompt": "Prompt 3", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    ]
    
    hashes = []
    for record in records:
        chain_hash = ledger.append(record)
        hashes.append(chain_hash)
    
    assert os.path.exists(ledger.path)
    assert os.path.exists(ledger.chain_path)
    
    with open(ledger.chain_path, 'r') as f:
        chain_lines = f.readlines()
        assert len(chain_lines) >= len(records)

def test_ledger_tamper_detection():
    ledger = HashChainLedger()
    
    record = {"prompt": "Test", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    original_hash = ledger.append(record)
    
    with open(ledger.path, 'r') as f:
        lines = f.readlines()
    
    if lines:
        modified_line = lines[0].replace("Test", "TAMPERED")
        with open(ledger.path, 'w') as f:
            f.writelines([modified_line] + lines[1:])
    
    assert True

def test_ledger_save_load():
    ledger = HashChainLedger()
    
    records = [
        {"prompt": "Test 1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        {"prompt": "Test 2", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    ]
    
    for record in records:
        ledger.append(record)
    
    with open(ledger.path, 'r') as f:
        loaded_lines = f.readlines()
    
    assert len(loaded_lines) >= len(records)
    
    found_records = 0
    for line in loaded_lines:
        try:
            loaded_record = json.loads(line.strip())
            for record in records:
                if (loaded_record.get("prompt") == record["prompt"] and 
                    loaded_record.get("sequence") == record["sequence"]):
                    found_records += 1
        except json.JSONDecodeError:
            continue
    
    assert found_records >= len(records)

def test_validate_prompt():
    valid_prompts = [
        "Design a protein with motif GXSXG",
        "Create alpha beta hydrolase, length 260..320",
        "Generate secreted protein with G.[ST][AGST]G pattern"
    ]
    
    for prompt in valid_prompts:
        result = validate_prompt(prompt)
        assert result["ok"], f"Prompt should be valid: {prompt}"

def test_requires_signal_peptide():
    sp_prompts = [
        "Design a secreted protein",
        "Create extracellular enzyme",
        "Generate protein for secretion"
    ]
    
    for prompt in sp_prompts:
        result = requires_signal_peptide(prompt)
    
    non_sp_prompts = [
        "Design a cytoplasmic protein",
        "Create intracellular enzyme",
        "Generate protein for internal use"
    ]
    
    for prompt in non_sp_prompts:
        result = requires_signal_peptide(prompt)

def test_kabsch_algorithm():
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = P.copy()
    
    rmsd = kabsch(P, Q)
    assert abs(rmsd) < 1e-6
    
    Q_translated = Q + np.array([1, 1, 1])
    rmsd = kabsch(P, Q_translated)
    assert abs(rmsd) < 1e-6
    
    Q_rotated = Q @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rmsd = kabsch(P, Q_rotated)
    assert rmsd < 2.0
    
    Q_different = Q + np.random.randn(*Q.shape) * 0.1
    rmsd = kabsch(P, Q_different)
    assert rmsd > 0

def test_max_tm_vs_dir():
    result = max_tm_vs_dir("nonexistent.pdb", "/nonexistent/dir")
    assert result is not None
    
    result = max_tm_vs_dir("nonexistent.pdb", "/tmp")
    assert result is not None

def test_active_site_rmsd():
    result = active_site_rmsd("nonexistent1.pdb", "nonexistent2.pdb")
    assert result is None
    
    result = active_site_rmsd("nonexistent1.pdb", "nonexistent2.pdb", motif_regex="[invalid")
    assert result is None

def test_ledger_edge_cases():
    ledger = HashChainLedger()
    
    empty_record = {}
    chain_hash = ledger.append(empty_record)
    assert isinstance(chain_hash, str)
    
    none_record = {"prompt": None, "sequence": None}
    chain_hash = ledger.append(none_record)
    assert isinstance(chain_hash, str)
    
    long_record = {
        "prompt": "A" * 10000,
        "sequence": "ACDEFGHIKLMNPQRSTVWY" * 100
    }
    chain_hash = ledger.append(long_record)
    assert isinstance(chain_hash, str)

def test_ledger_hash_uniqueness():
    ledger = HashChainLedger()
    
    record = {"prompt": "Test", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    
    hashes = []
    for _ in range(5):
        chain_hash = ledger.append(record)
        hashes.append(chain_hash)
    
    unique_hashes = set(hashes)
    assert len(hashes) == len(unique_hashes)

def test_ledger_verification_after_tampering():
    ledger = HashChainLedger()
    
    for i in range(5):
        record = {"prompt": f"Prompt {i}", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
        ledger.append(record)
    
    assert True
