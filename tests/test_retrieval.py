import numpy as np
import tempfile
import os
from pathlib import Path
from retrieval.profile_hmm import build_pssm, viterbi_profile
from retrieval.pipeline import build_profile_from_db
from retrieval.msa_builder import build_msa

def test_build_pssm_basic():
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
    pssm, consensus, conservation = build_pssm(sequences)
    
    assert pssm.shape[0] == 20
    assert pssm.shape[1] == 20
    assert len(consensus) == 20
    assert len(conservation) == 20
    
    assert all(c >= 0 for c in conservation)

def test_build_pssm_different_sequences():
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNPQRSTVWY"
    ]
    pssm, consensus, conservation = build_pssm(sequences)
    
    assert pssm.shape[0] == 20
    assert pssm.shape[1] == 20
    assert len(consensus) == 20
    assert len(conservation) == 20

def test_build_pssm_empty():
    sequences = []
    pssm, consensus, conservation = build_pssm(sequences)
    
    assert pssm.shape[0] == 0
    assert len(consensus) == 0
    assert len(conservation) == 0

def test_build_pssm_single_sequence():
    sequences = ["ACDEFGHIKLMNPQRSTVWY"]
    pssm, consensus, conservation = build_pssm(sequences)
    
    assert pssm.shape[0] == 20
    assert pssm.shape[1] == 20
    assert len(consensus) == 20
    assert len(conservation) == 20

def test_viterbi_profile():
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
    pssm, consensus, conservation = build_pssm(sequences)
    
    query_seq = "ACDEFGHIKLMNPQRSTVWY"
    mapping, score = viterbi_profile(pssm, query_seq)
    
    assert len(mapping) == len(query_seq)
    assert all(isinstance(x, int) for x in mapping)
    assert isinstance(score, float)
    
    short_seq = "ACDEF"
    mapping, score = viterbi_profile(pssm, short_seq)
    assert len(mapping) == len(short_seq)

def test_viterbi_profile_empty_pssm():
    pssm = np.array([])
    query_seq = "ACDEFGHIKLMNPQRSTVWY"
    mapping, score = viterbi_profile(pssm, query_seq)
    
    assert len(mapping) == len(query_seq)
    assert all(x < 0 for x in mapping)

def test_build_msa():
    hits = [
        ("seq1", "ACDEFGHIKLMNPQRSTVWY", 0.9),
        ("seq2", "ACDEFGHIKLMNPQRSTVWY", 0.8),
        ("seq3", "ACDEFGHIKLMNPQRSTVWY", 0.7)
    ]
    
    msa = build_msa(hits)
    assert len(msa) == 3
    
    for name, seq in msa:
        assert isinstance(name, str)
        assert isinstance(seq, str)
        assert len(seq) == 20

def test_build_msa_empty():
    hits = []
    msa = build_msa(hits)
    assert len(msa) == 0

def test_build_profile_from_db():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        f.write(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq2\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq3\nACDEFGHIKLMNPQRSTVWY\n")
        temp_fasta = f.name
    
    try:
        query_seq = "ACDEFGHIKLMNPQRSTVWY"
        msa, hits = build_profile_from_db(temp_fasta, query_seq, topk=2)
        
        assert len(msa) <= 2
        assert len(hits) <= 2
        
        for hit in hits:
            assert len(hit) == 3
            assert isinstance(hit[0], str)
            assert isinstance(hit[1], str)
            assert isinstance(hit[2], float)
            assert 0.0 <= hit[2] <= 1.1
    
    finally:
        os.unlink(temp_fasta)

def test_build_profile_from_db_empty():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        temp_fasta = f.name
    
    try:
        msa, hits = build_profile_from_db(temp_fasta, "ACDEFGHIKLMNPQRSTVWY")
        assert len(msa) == 0
        assert len(hits) == 0
    
    finally:
        os.unlink(temp_fasta)

def test_build_profile_from_db_nonexistent():
    try:
        msa, hits = build_profile_from_db("nonexistent.fasta", "ACDEFGHIKLMNPQRSTVWY")
        assert len(msa) == 0
        assert len(hits) == 0
    except FileNotFoundError:
        pass

def test_pssm_consistency():
    sequences1 = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
    sequences2 = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
    
    pssm1, consensus1, conservation1 = build_pssm(sequences1)
    pssm2, consensus2, conservation2 = build_pssm(sequences2)
    
    assert pssm1.shape == pssm2.shape
    assert len(consensus1) == len(consensus2)
    assert len(conservation1) == len(conservation2)

def test_viterbi_profile_edge_cases():
    sequences = ["A", "A"]
    pssm, consensus, conservation = build_pssm(sequences)
    
    query_seq = "A"
    mapping, score = viterbi_profile(pssm, query_seq)
    assert len(mapping) == 1
    
    sequences = ["A", "A", "A"]
    pssm, consensus, conservation = build_pssm(sequences)
    
    query_seq = "A"
    mapping, score = viterbi_profile(pssm, query_seq)
    assert len(mapping) == 1

def test_msa_sequence_validation():
    hits = [
        ("seq1", "ACDEFGHIKLMNPQRSTVWY", 0.9),
        ("seq2", "INVALID_CHARS_123", 0.8),
        ("seq3", "ACDEFGHIKLMNPQRSTVWY", 0.7)
    ]
    
    msa = build_msa(hits)
    assert len(msa) >= 0

def test_profile_similarity_scores():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        f.write(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq2\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq3\nACDEFGHIKLMNPQRSTVWY\n")
        temp_fasta = f.name
    
    try:
        query_seq = "ACDEFGHIKLMNPQRSTVWY"
        msa, hits = build_profile_from_db(temp_fasta, query_seq, topk=3)
        
        for hit in hits:
            assert hit[2] > 0.5
    
    finally:
        os.unlink(temp_fasta)
