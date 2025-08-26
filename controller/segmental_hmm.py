
# controller/segmental_hmm.py
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class SegmentPlan:
    targets: List[int]
    z_tier: str

class DurationMLP(nn.Module):
    def __init__(self, d_model=384, n_segments=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, n_segments),
        )
    def forward(self, x):
        return F.softplus(self.fc(x)) + 1.0

def z_adjust(lengths: torch.Tensor, tier: str):
    if tier == "normal": return lengths
    if tier == "stretched": return lengths * 1.3
    if tier == "sparse": return lengths * 1.7
    return lengths

def boundary_anneal(step_in_seg: int, target_len: int, width: int = 20):
    over = step_in_seg - target_len
    if over <= 0: return 0.0
    return min(1.0, over/max(1,width))

def shock_detector(prev_target: int, new_target: int, thresh: float = 0.5):
    if prev_target <= 0: return False
    return abs(new_target - prev_target) / prev_target > thresh

def monotone_consensus_index(actual: List[int], planned: List[int]):
    if not planned or not actual: return 0.0
    s=0.0
    for a,p in zip(actual, planned):
        s += min(a,p)/max(a,p) if max(a,p)>0 else 0.0
    return s/len(planned)

def segments_from_motifs(seq_len_est: int, n_motifs: int):
    return max(1, n_motifs + 1)

def build_plan(prompt_vec: torch.Tensor, d_model: int, n_segments: int, z_tier: str) -> SegmentPlan:
    mlp = DurationMLP(d_model=d_model, n_segments=n_segments)
    with torch.no_grad():
        raw = mlp(prompt_vec.view(1, -1))
        adj = z_adjust(raw, z_tier)
        lens = adj.round().clamp(1, 1024).int().tolist()[0]
    return SegmentPlan(targets=lens[:n_segments], z_tier=z_tier)

def apply_controller_penalty(step_idx: int, seg_idx: int, seg_step: int, seg_target: int):
    ann = boundary_anneal(seg_step, seg_target, width=20)
    return -5.0 * ann
