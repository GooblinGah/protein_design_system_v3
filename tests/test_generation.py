import torch
import tempfile
import os
from pathlib import Path
from generate import (
    allowed_mask_from_fsa, beam_search_generate, 
    controller_pos_prior, duration_pos_prior, sample_step
)
from dsl.compiler import MotifFSA, compile_constraints
from dsl.parser import parse_prompt
from models.transformer_lm import TransformerCopyLM
from utils import VOCAB_SIZE, AA_START, PROMPT_START

class MockModel:
    def __init__(self):
        self.device = 'cpu'
    
    def __call__(self, x, attn_mask=None, exemplar_ids=None, exemplar_mask=None, return_mix=True, gate_bias=0.0, gate_features=None):
        batch_size, seq_len = x.shape
        vocab_size = VOCAB_SIZE
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        gate = torch.rand(batch_size, seq_len)
        
        if return_mix:
            return {
                "logits": logits,
                "gate": gate,
                "vocab_logits": logits,
                "copy_logits": logits
            }
        else:
            return logits

class MockControllerState:
    def __init__(self, z_tier="normal", length_lo=None, length_hi=None):
        self.z_tier = z_tier
        self.length_lo = length_lo
        self.length_hi = length_hi
        self.hysteresis_on = False

def test_allowed_mask_from_fsa():
    mask, size = allowed_mask_from_fsa([])
    assert isinstance(mask, dict)
    assert size > 0
    
    fsa = MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)
    mask, size = allowed_mask_from_fsa([fsa])
    
    assert isinstance(mask, dict)
    assert size > 0
    
    fsa2 = MotifFSA([{'A'}, {'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'G'}], min_occurrences=1)
    mask, size = allowed_mask_from_fsa([fsa, fsa2])
    
    assert isinstance(mask, dict)
    assert size > 0

def test_controller_pos_prior():
    prior = controller_pos_prior("normal", length_lo=100, length_hi=200)
    assert callable(prior)
    
    prior = controller_pos_prior("stretched", length_lo=100, length_hi=200)
    assert callable(prior)
    
    prior = controller_pos_prior("sparse", length_lo=100, length_hi=200)
    assert callable(prior)
    
    prior = controller_pos_prior("normal")
    assert callable(prior)

def test_duration_pos_prior():
    duration_targets = [50, 100, 150]
    ctrl_state = MockControllerState(z_tier='normal')
    
    prior = duration_pos_prior(duration_targets, ctrl_state)
    assert callable(prior)
    
    result = prior(25, 0, 25)
    assert isinstance(result, float)
    
    result = prior(75, 0, 75)
    assert result < 0

def test_sample_step():
    logits = torch.randn(1, 10)
    token = sample_step(logits, strategy="greedy")
    assert isinstance(token, int)
    assert 0 <= token < logits.shape[-1]
    
    token = sample_step(logits, strategy="greedy", top_p=0.9)
    assert isinstance(token, int)
    assert 0 <= token < logits.shape[-1]

def test_beam_search_generate_basic():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas, 
        ex_ids=None, device=device, beam_size=2, max_new=5,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)
    assert ids.shape[0] == 1
    assert ids.shape[1] > ids_init.shape[1]

def test_beam_search_generate_with_exemplars():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    ex_ids = torch.randint(AA_START, PROMPT_START, (1, 10))
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=ex_ids, device=device, beam_size=2, max_new=5,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)

def test_beam_search_generate_with_position_prior():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    pos_prior = lambda pos: 0.1 if pos < 10 else -0.1
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=2, max_new=5,
        pos_prior=pos_prior, ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)

def test_fsa_integration():
    model = MockModel()
    device = 'cpu'
    
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=2, max_new=10,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)

def test_generation_edge_cases():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    fsas = []
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=2, max_new=5,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=1, max_new=5,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)

def test_provenance_tracking():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=2, max_new=5,
        ctrl_state=ctrl_state
    )
    
    for entry in provenance:
        assert "aa_id" in entry
        assert "source" in entry
        assert "gate" in entry
        assert "boundary_reset" in entry
        
        assert isinstance(entry["aa_id"], int)
        assert entry["source"] in ["copy", "vocab"]
        assert isinstance(entry["gate"], float)
        assert isinstance(entry["boundary_reset"], bool)

def test_beam_search_termination():
    model = MockModel()
    device = 'cpu'
    
    ids_init = torch.tensor([[1, 2, 3, 4, 5]])
    attn_init = torch.ones_like(ids_init, dtype=torch.float32)
    fsas = [MotifFSA([{'G'}, {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}, {'S', 'T'}, {'A', 'G', 'S', 'T'}, {'G'}], min_occurrences=1)]
    
    ctrl_state = MockControllerState(z_tier="normal")
    
    ids, provenance = beam_search_generate(
        model, ids_init, attn_init, fsas,
        ex_ids=None, device=device, beam_size=2, max_new=1,
        ctrl_state=ctrl_state
    )
    
    assert isinstance(ids, torch.Tensor)
    assert isinstance(provenance, list)
    assert ids.shape[1] <= ids_init.shape[1] + 1
