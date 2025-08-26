
# trainers/monitors.py
import torch

class CopyMonitor:
    def __init__(self, target_copy_rate=0.25, lr=0.2):
        self.target = target_copy_rate
        self.lr = lr
        self.state = 0.0
    def update(self, gate_probs, conserved_mask):
        if gate_probs is None or conserved_mask is None or gate_probs.numel()==0:
            return 0.0
        m = conserved_mask.float()
        denom = m.sum().clamp_min(1.0)
        rate = (gate_probs * m).sum() / denom
        self.state = 0.9*self.state + 0.1*float(rate.item())
        err = self.state - self.target
        return -self.lr * err

def binary_entropy(p, eps=1e-6):
    p = torch.clamp(p, eps, 1-eps)
    return -(p*torch.log2(p) + (1-p)*torch.log2(1-p))

class GateEntropyMonitor:
    def __init__(self, target_entropy_bits=0.5, lr=0.1):
        self.target = target_entropy_bits
        self.lr = lr
        self.state = 0.0
    def update(self, gate_probs, mask):
        if gate_probs is None or mask is None or gate_probs.numel()==0:
            return 0.0
        ent = binary_entropy(gate_probs)
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        mean_ent = (ent * m).sum() / denom
        self.state = 0.9*self.state + 0.1*float(mean_ent.item())
        err = self.state - self.target
        return +self.lr * err
