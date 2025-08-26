import torch
import numpy as np
from utils import (
    encode_prompt_bytes, encode_seq, detok_seq,
    load_fasta, save_fasta, max_identity_vs_refs,
    BOS, EOS, SEP, EXB, EXE, PAD, AA_START, PROMPT_START,
    VOCAB_SIZE, ID_TO_AA, AA_TO_ID
)
import tempfile
import os

def test_special_tokens():
    assert BOS == 1
    assert EOS == 2
    assert SEP == 3
    assert EXB == 4
    assert EXE == 5
    assert PAD == 0
    
    assert AA_START == 6
    assert PROMPT_START == 26
    assert VOCAB_SIZE > PROMPT_START

def test_encode_prompt_bytes():
    prompt = "Design a protein with motif GXSXG"
    encoded = encode_prompt_bytes(prompt, max_len=100)
    
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert all(isinstance(x, int) for x in encoded)
    assert all(x >= PROMPT_START for x in encoded)
    
    long_prompt = "A" * 1000
    encoded_long = encode_prompt_bytes(long_prompt, max_len=100)
    assert len(encoded_long) == 100
    
    encoded_empty = encode_prompt_bytes("", max_len=100)
    assert len(encoded_empty) == 0
    
    special_prompt = "Design protein with G.[ST][AGST]G pattern"
    encoded_special = encode_prompt_bytes(special_prompt, max_len=100)
    assert len(encoded_special) > 0

def test_encode_seq():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encoded = encode_seq(seq, max_len=100)
    
    assert isinstance(encoded, list)
    assert len(encoded) == len(seq)
    assert all(isinstance(x, int) for x in encoded)
    assert all(AA_START <= x < PROMPT_START for x in encoded)
    
    long_seq = "A" * 1000
    encoded_long = encode_seq(long_seq, max_len=100)
    assert len(encoded_long) == 100
    
    encoded_empty = encode_seq("", max_len=100)
    assert len(encoded_empty) == 0
    
    invalid_seq = "ACDEFGHIKLMNPQRSTVWY123"
    encoded_invalid = encode_seq(invalid_seq, max_len=100)
    assert len(encoded_invalid) > 0

def test_detok_seq():
    seq_ids = [AA_START, AA_START + 1, AA_START + 2]
    decoded = detok_seq(seq_ids)
    
    assert isinstance(decoded, str)
    assert decoded == "ACD"
    
    seq_with_special = [BOS, AA_START, AA_START + 1, SEP, AA_START + 2, EOS]
    decoded_special = detok_seq(seq_with_special)
    assert "ACD" in decoded_special
    
    decoded_empty = detok_seq([])
    assert decoded_empty == ""

def test_fasta_io():
    test_records = [
        ("seq1", "ACDEFGHIKLMNPQRSTVWY"),
        ("seq2", "ACDEFGHIKLMNPQRSTVWY"),
        ("seq3", "ACDEFGHIKLMNPQRSTVWY")
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        temp_fasta = f.name
    
    try:
        save_fasta(temp_fasta, test_records)
        
        assert os.path.exists(temp_fasta)
        
        loaded_records = list(load_fasta(temp_fasta))
        
        assert len(loaded_records) == len(test_records)
        for i, (name, seq) in enumerate(loaded_records):
            assert name == test_records[i][0]
            assert seq == test_records[i][1]
    
    finally:
        if os.path.exists(temp_fasta):
            os.unlink(temp_fasta)

def test_fasta_edge_cases():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        temp_fasta = f.name
    
    try:
        empty_records = list(load_fasta(temp_fasta))
        assert len(empty_records) == 0
        
        with open(temp_fasta, 'w') as f:
            f.write(">seq1\n>seq2\n")
        
        header_only_records = list(load_fasta(temp_fasta))
        assert len(header_only_records) == 2
        assert all(seq == "" for _, seq in header_only_records)
        
        with open(temp_fasta, 'w') as f:
            f.write(">seq1\n\n>seq2\n\n")
        
        empty_seq_records = list(load_fasta(temp_fasta))
        assert len(empty_seq_records) == 2
        assert all(seq == "" for _, seq in empty_seq_records)
    
    finally:
        if os.path.exists(temp_fasta):
            os.unlink(temp_fasta)

def test_max_identity_vs_refs():
    query = "ACDEFGHIKLMNPQRSTVWY"
    refs = [
        ("ref1", "ACDEFGHIKLMNPQRSTVWY"),
        ("ref2", "ACDEFGHIKLMNPQRSTVWY")
    ]
    
    max_id = max_identity_vs_refs(query, refs)
    assert abs(max_id - 1.0) < 1e-6
    
    refs_different = [
        ("ref1", "ACDEFGHIKLMNPQRSTVWY"),
        ("ref2", "ACDEFGHIKLMNPQRSTVWX")
    ]
    
    max_id_different = max_identity_vs_refs(query, refs_different)
    assert 0.9 < max_id_different <= 1.0
    
    max_id_empty = max_identity_vs_refs(query, [])
    assert max_id_empty == 0.0

def test_aa_mapping():
    assert AA_TO_ID['A'] == AA_START
    assert AA_TO_ID['C'] == AA_START + 1
    assert AA_TO_ID['Y'] == AA_START + 19
    
    assert ID_TO_AA[AA_START] == 'A'
    assert ID_TO_AA[AA_START + 1] == 'C'
    assert ID_TO_AA[AA_START + 19] == 'Y'
    
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_id = AA_TO_ID[aa]
        aa_back = ID_TO_AA[aa_id]
        assert aa == aa_back

def test_encoding_edge_cases():
    long_seq = "A" * 10000
    encoded_long = encode_seq(long_seq, max_len=1000)
    assert len(encoded_long) == 1000
    
    mixed_case = "AcDeFgHiKlMnPqRsTvWy"
    encoded_mixed = encode_seq(mixed_case, max_len=100)
    assert len(encoded_mixed) <= len(mixed_case)
    
    mixed_chars = "ACDEFGHIKLMNPQRSTVWY123!@#"
    encoded_mixed_chars = encode_seq(mixed_chars, max_len=100)
    assert len(encoded_mixed_chars) <= len(mixed_chars)

def test_prompt_encoding_edge_cases():
    unicode_prompt = "Design protein with motif GXSXG"
    encoded_unicode = encode_prompt_bytes(unicode_prompt, max_len=100)
    assert len(encoded_unicode) > 0
    
    long_prompt = "Design a protein with motif GXSXG " * 1000
    encoded_long = encode_prompt_bytes(long_prompt, max_len=500)
    assert len(encoded_long) == 500
    
    assert len(encode_prompt_bytes("", max_len=100)) == 0
    assert len(encode_prompt_bytes("   ", max_len=100)) == 3

def test_fasta_io_edge_cases():
    long_seq = "A" * 10000
    test_records = [("long_seq", long_seq)]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        temp_fasta = f.name
    
    try:
        save_fasta(temp_fasta, test_records)
        loaded_records = list(load_fasta(temp_fasta))
        
        assert len(loaded_records) == 1
        assert loaded_records[0][0] == "long_seq"
        assert loaded_records[0][1] == long_seq
    
    finally:
        if os.path.exists(temp_fasta):
            os.unlink(temp_fasta)
    
    special_records = [
        ("seq_with_spaces", "ACDEFGHIKLMNPQRSTVWY"),
        ("seq_with_underscores", "ACDEFGHIKLMNPQRSTVWY"),
        ("seq-with-dashes", "ACDEFGHIKLMNPQRSTVWY")
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        temp_fasta = f.name
    
    try:
        save_fasta(temp_fasta, special_records)
        loaded_special = list(load_fasta(temp_fasta))
        
        assert len(loaded_special) == len(special_records)
        for i, (name, seq) in enumerate(loaded_special):
            assert name == special_records[i][0]
            assert seq == special_records[i][1]
    
    finally:
        if os.path.exists(temp_fasta):
            os.unlink(temp_fasta)
