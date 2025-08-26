
# models/transformer_lm.py
import torch, torch.nn as nn
from utils import VOCAB_SIZE, AA_START, PROMPT_START

class TransformerCopyLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=384, n_layers=6, n_heads=6, d_ff=1536, pdrop=0.1, max_len=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(pdrop)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=pdrop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.gate_head = nn.Linear(d_model, 1)
        self.gate_feat = nn.Linear(2, 1, bias=False)  # [cons_prob, match_flag] -> logit bias
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, attn_mask=None, exemplar_ids=None, exemplar_mask=None, return_mix=True, gate_bias: float = 0.0, gate_features=None):
        B,T = x.size()
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.drop(self.tok(x) + self.pos(pos))
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        padmask = (attn_mask == 0) if attn_mask is not None else None
        h = self.encoder(h, mask=causal, src_key_padding_mask=padmask)

        vocab_logits = self.lm_head(h)  # [B,T,V]

        # simple exemplar-copy distribution from histogram
        copy_logits = torch.full_like(vocab_logits, -1e9)
        if exemplar_ids is not None and exemplar_ids.numel() > 0:
            V = vocab_logits.size(-1)
            hist = torch.zeros(B, V, device=device)
            for b in range(B):
                ex = exemplar_ids[b] if exemplar_ids.dim()==2 else exemplar_ids[b,0]
                for t in ex.tolist():
                    if AA_START <= t < PROMPT_START:
                        hist[b, t] += 1.0
            hist = torch.where(hist>0, hist, torch.ones_like(hist))
            hist = hist / hist.sum(dim=-1, keepdim=True)
            copy_logits = torch.log(hist + 1e-9).unsqueeze(1).expand_as(vocab_logits)

        gate_logits = self.gate_head(h).squeeze(-1) + float(gate_bias)
        if gate_features is not None:
            # gate_features: [B,T,2]
            gf = self.gate_feat(gate_features).squeeze(-1)
            gate_logits = gate_logits + gf
        gate = torch.sigmoid(gate_logits)

        if not return_mix:
            return vocab_logits

        gate_v = gate.unsqueeze(-1)
        mix = torch.logaddexp(
            torch.log(torch.clamp(1-gate_v, 1e-6, 1.0)) + vocab_logits,
            torch.log(torch.clamp(gate_v, 1e-6, 1.0)) + copy_logits
        )
        return {"logits": mix, "gate": gate, "vocab_logits": vocab_logits, "copy_logits": copy_logits}
