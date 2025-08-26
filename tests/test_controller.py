import torch
import torch.nn as nn
from controller.segmental_hmm import (
    DurationMLP, z_adjust, boundary_anneal, shock_detector,
    monotone_consensus_index, segments_from_motifs, build_plan
)
from controller.train_duration import SmallMLP, encode_prompt_bytes
import numpy as np

def test_duration_mlp_creation():
    mlp = DurationMLP(d_model=384, n_segments=3)
    assert isinstance(mlp, DurationMLP)
    assert mlp.fc[0].in_features == 384
    assert mlp.fc[-1].out_features == 3

def test_duration_mlp_forward():
    mlp = DurationMLP(d_model=384, n_segments=3)
    batch_size = 2
    x = torch.randn(batch_size, 384)
    
    output = mlp(x)
    assert output.shape == (batch_size, 3)
    assert torch.all(output > 0)

def test_z_adjust():
    lengths = torch.tensor([100, 200, 300])
    
    normal = z_adjust(lengths, "normal")
    assert torch.allclose(normal, lengths)
    
    stretched = z_adjust(lengths, "stretched")
    assert torch.allclose(stretched, lengths * 1.3)
    
    sparse = z_adjust(lengths, "sparse")
    assert torch.allclose(sparse, lengths * 1.7)
    
    unknown = z_adjust(lengths, "unknown")
    assert torch.allclose(unknown, lengths)

def test_boundary_anneal():
    assert boundary_anneal(10, 20) == 0.0
    assert boundary_anneal(20, 20) == 0.0
    assert boundary_anneal(25, 20) == 0.25
    assert boundary_anneal(40, 20) == 1.0
    assert boundary_anneal(30, 20, width=10) == 1.0

def test_shock_detector():
    assert not shock_detector(100, 110, thresh=0.5)
    assert shock_detector(100, 150, thresh=0.49)
    assert shock_detector(100, 50, thresh=0.49)
    assert not shock_detector(100, 150, thresh=0.6)
    assert not shock_detector(0, 100, thresh=0.5)
    assert not shock_detector(-10, 100, thresh=0.5)

def test_monotone_consensus_index():
    actual = [100, 200, 300]
    planned = [100, 200, 300]
    mci = monotone_consensus_index(actual, planned)
    assert abs(mci - 1.0) < 1e-6
    
    actual = [80, 180, 320]
    planned = [100, 200, 300]
    mci = monotone_consensus_index(actual, planned)
    assert 0.0 < mci < 1.0
    
    assert monotone_consensus_index([], []) == 0.0
    assert monotone_consensus_index([100], []) == 0.0
    assert monotone_consensus_index([], [100]) == 0.0

def test_segments_from_motifs():
    assert segments_from_motifs(100, 0) == 1
    assert segments_from_motifs(100, 1) == 2
    assert segments_from_motifs(100, 5) == 6
    assert segments_from_motifs(0, 0) == 1
    assert segments_from_motifs(100, 100) == 101

def test_build_plan():
    prompt_vec = torch.randn(256)
    plan = build_plan(prompt_vec, d_model=256, n_segments=3, z_tier="normal")
    
    assert isinstance(plan.targets, list)
    assert len(plan.targets) == 3
    assert plan.z_tier == "normal"
    assert all(t > 0 for t in plan.targets)
    
    plan_stretched = build_plan(prompt_vec, d_model=256, n_segments=3, z_tier="stretched")
    assert plan_stretched.z_tier == "stretched"

def test_small_mlp_creation():
    mlp = SmallMLP(in_dim=256, out_dim=3)
    assert isinstance(mlp, SmallMLP)
    assert mlp.net[0].in_features == 256
    assert mlp.net[-1].out_features == 3

def test_small_mlp_forward():
    mlp = SmallMLP(in_dim=256, out_dim=3)
    batch_size = 2
    x = torch.randn(batch_size, 256)
    
    output = mlp(x)
    assert output.shape == (batch_size, 3)

def test_encode_prompt_bytes():
    prompt = "Design a protein with motif GXSXG"
    encoded = encode_prompt_bytes(prompt)
    
    assert hasattr(encoded, 'shape')
    assert len(encoded.shape) == 1
    assert all(isinstance(x, (int, np.integer)) for x in encoded)
    assert all(0 <= x <= 255 for x in encoded)
    
    long_prompt = "A" * 1000
    encoded_long = encode_prompt_bytes(long_prompt, max_len=100)
    assert len(encoded_long) == 100
    
    encoded_empty = encode_prompt_bytes("")
    assert len(encoded_empty) == 0

def test_controller_integration():
    mlp = DurationMLP(d_model=384, n_segments=3)
    x = torch.randn(1, 384)
    
    raw = mlp(x)
    
    normal = z_adjust(raw, "normal")
    stretched = z_adjust(raw, "stretched")
    sparse = z_adjust(raw, "sparse")
    
    assert torch.allclose(normal, raw)
    assert torch.allclose(stretched, raw * 1.3)
    assert torch.allclose(sparse, raw * 1.7)
    
    target = float(stretched[0, 0])
    assert boundary_anneal(target, target) == 0.0
    assert boundary_anneal(target + 10, target) > 0.0

def test_edge_cases():
    lengths = torch.tensor([1, 2, 3])
    adjusted = z_adjust(lengths, "stretched")
    assert torch.all(adjusted > 0)
    
    lengths = torch.tensor([1000, 2000, 3000])
    adjusted = z_adjust(lengths, "sparse")
    assert torch.all(adjusted > 0)
    
    assert not shock_detector(100, 101, thresh=0.5)
    assert shock_detector(100, 101, thresh=0.009)
    
    assert boundary_anneal(25, 20, width=0) == 1.0
