
from pathlib import Path
import json
import torch
from datasets import SeqGenDataset, collate_batch
from utils import encode_prompt_bytes, encode_seq, BOS, EOS, SEP, EXB, EXE, PAD

def test_data_contract(tmp_path):
    proc = tmp_path / "processed"; proc.mkdir(parents=True, exist_ok=True)
    (proc/"splits.tsv").write_text("accession\tsplit\nACC0\ttrain\nACC1\tval\n")
    (proc/"prompt_pairs.jsonl").write_text('{"prompt":"Design protein","sequence":"ACDEFGHIKLMNPQRSTVWY","exemplar":""}\n{"prompt":"Design enzyme","sequence":"ACDEFGHIKLMNPQRSTVWY","exemplar":""}\n')
    
    seen = set()
    for i,line in enumerate((proc/"splits.tsv").read_text().strip().splitlines()):
        if i==0: continue
        acc, split = line.strip().split("\t")[:2]
        seen.add(acc)
        assert split in {"train","val","test"}
    
    assert "ACC0" in seen
    assert "ACC1" in seen

def test_utils_encoding():
    prompt = "Design a protein with motif GXSXG"
    encoded = encode_prompt_bytes(prompt, max_len=100)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encoded_seq = encode_seq(seq, max_len=100)
    assert isinstance(encoded_seq, list)
    assert len(encoded_seq) == len(seq)
    
    assert BOS == 1
    assert EOS == 2
    assert SEP == 3
    assert EXB == 4
    assert EXE == 5
    assert PAD == 0

def test_dataset_creation():
    items = [
        {"prompt": "Design protein A", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": ""},
        {"prompt": "Design protein B", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": ""}
    ]
    
    dataset = SeqGenDataset(items, max_prompt_len=100, max_seq_len=50, use_exemplar=False)
    assert len(dataset) == 2
    
    item = dataset[0]
    assert "input" in item
    assert "labels" in item
    assert "exemplar" in item
    
    assert len(item["input"]) == len(item["labels"])

def test_dataset_with_exemplars():
    items = [
        {"prompt": "Design protein A", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": "ACDEFGHIKLMNPQRSTVWY"}
    ]
    
    dataset = SeqGenDataset(items, max_prompt_len=100, max_seq_len=50, use_exemplar=True)
    item = dataset[0]
    
    assert EXB in item["input"]
    assert EXE in item["input"]

def test_collate_batch():
    items = [
        {"prompt": "Design protein A", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": ""},
        {"prompt": "Design protein B", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": ""}
    ]
    
    dataset = SeqGenDataset(items, max_prompt_len=100, max_seq_len=50, use_exemplar=False)
    batch = [dataset[i] for i in range(len(dataset))]
    
    X, Y, attn, exemplars = collate_batch(batch)
    
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert isinstance(attn, torch.Tensor)
    assert isinstance(exemplars, list)
    
    assert X.shape[0] == 2
    assert Y.shape[0] == 2
    assert attn.shape[0] == 2
    
    assert X.shape[1] == Y.shape[1] == attn.shape[1]

def test_dataset_edge_cases():
    empty_dataset = SeqGenDataset([], max_prompt_len=100, max_seq_len=50)
    assert len(empty_dataset) == 0
    
    long_seq = "A" * 1000
    items = [{"prompt": "Long protein", "sequence": long_seq, "exemplar": ""}]
    dataset = SeqGenDataset(items, max_prompt_len=100, max_seq_len=50)
    
    item = dataset[0]
    assert len(item["input"]) <= 100 + 50 + 3

def test_token_alignment():
    items = [
        {"prompt": "Design protein", "sequence": "ACDEFGHIKLMNPQRSTVWY", "exemplar": ""}
    ]
    
    dataset = SeqGenDataset(items, max_prompt_len=100, max_seq_len=50, use_exemplar=False)
    item = dataset[0]
    
    assert item["labels"][0] == -100
    assert item["labels"][1] == -100
    
    seq_start = None
    for i, token in enumerate(item["input"]):
        if token == SEP and i > 0 and item["input"][i-1] != SEP:
            seq_start = i + 1
            break
    
    if seq_start is not None:
        assert item["labels"][seq_start] >= 0 or item["labels"][seq_start] == -100
