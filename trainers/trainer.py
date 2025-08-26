
# trainers/trainer.py
import os, math, importlib
import torch, torch.nn as nn
from trainers.monitors import CopyMonitor, GateEntropyMonitor

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, save_path,
                 amp=False, gate_reg_weight=0.0, curriculum=None, grad_accum=1,
                 identity_tau=0.7, identity_weight=0.0, identity_warmup_epochs=0,
                 use_wandb=False, wandb=None):
        self.model = model
        self.optim = optimizer
        self.sched = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_path = save_path
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.curriculum = curriculum or {"phase_epochs":[math.inf], "gate_reg":[0.0]}
        self.grad_accum = max(1, grad_accum)
        self.identity_tau = identity_tau
        self.identity_weight = identity_weight
        self.identity_warmup_epochs = identity_warmup_epochs
        self.copy_monitor = CopyMonitor(target_copy_rate=0.25, lr=0.2)
        self.ent_monitor = GateEntropyMonitor(target_entropy_bits=0.5, lr=0.1)
        self.gate_scale = 1.0
        self.gate_bias = 0.0
        self.use_wandb = use_wandb
        self.wandb = wandb

    def _curriculum_gate_weight(self, epoch):
        w = self.curriculum.get("gate_reg",[0.0])
        ph = self.curriculum.get("phase_epochs",[math.inf])
        s = 0
        for ep,wei in zip(ph,w):
            s += ep
            if epoch <= s: return wei
        return w[-1]

    def _expected_identity(self, logits, y, ex_ids, ex_mask):
        B,T,V = logits.shape
        if ex_ids is None or ex_ids.numel()==0: return logits.new_zeros(())
        L = min(T, ex_ids.size(1))
        probs = torch.softmax(logits[:, :L, :], dim=-1)
        gather = torch.gather(probs, dim=-1, index=ex_ids[:, :L].unsqueeze(-1)).squeeze(-1)
        m = ex_mask[:, :L] if ex_mask is not None else torch.ones_like(gather)
        num = (gather * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0)
        return (num/den).mean()

    def _compute_loss(self, out, y, epoch=1, ex_ids=None, ex_mask=None, conserved_mask=None):
        logits = out["logits"]
        gate = out.get("gate")
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        # gate regularizer (encourage lower gate on conserved positions)
        gre = 0.0
        if gate is not None and conserved_mask is not None:
            m = conserved_mask.float()
            denom = m.sum().clamp_min(1.0)
            gre = (gate * m).sum() / denom

        # dynamic scaling
        bias_delta = 0.0; scale_delta = 0.0
        if gate is not None and conserved_mask is not None:
            bias_delta = float(self.copy_monitor.update(gate.detach(), conserved_mask.detach()))
            scale_delta = float(self.ent_monitor.update(gate.detach(), conserved_mask.detach()))
        self.gate_bias = 5.0 * bias_delta
        self.gate_scale = max(0.5, min(2.0, 1.0 + scale_delta))

        reg_w = self._curriculum_gate_weight(epoch) * self.gate_scale

        id_pen = 0.0
        if self.identity_weight > 0 and epoch > self.identity_warmup_epochs:
            exid = self._expected_identity(logits, y, ex_ids, ex_mask)
            id_pen = torch.relu(exid - self.identity_tau)

        total = ce + reg_w * (gre if isinstance(gre, torch.Tensor) else 0.0) + self.identity_weight * (id_pen if isinstance(id_pen, torch.Tensor) else 0.0)
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({"train/reg_weight_eff": float(reg_w),
                            "train/gate_bias": float(self.gate_bias),
                            "train/copy_rate_ema": self.copy_monitor.state,
                            "train/gate_entropy_ema": self.ent_monitor.state}, commit=False)
        return total

    def fit(self, train_loader, val_loader, epochs=5):
        best_val = float("inf")
        for epoch in range(1, epochs+1):
            self.model.train()
            total, n = 0.0, 0
            for i,(xb, yb, attn, exemplars) in enumerate(train_loader, start=1):
                xb, yb, attn = xb.to(self.device), yb.to(self.device), attn.to(self.device)
                # Build exemplar tensors
                ex_ids = None; ex_mask = None
                if any(bool(e) for e in exemplars):
                    from utils import encode_seq
                    ex_list = [encode_seq(e, 256) for e in exemplars]
                    maxl = max((len(x) for x in ex_list), default=0)
                    ex_ids = torch.full((xb.size(0), maxl), 0, dtype=torch.long, device=self.device)
                    ex_mask = torch.zeros_like(ex_ids, dtype=torch.float32)
                    for b, arr in enumerate(ex_list):
                        for j,t in enumerate(arr):
                            ex_ids[b,j] = t; ex_mask[b,j] = 1.0

                with torch.cuda.amp.autocast(enabled=self.amp):
                    # Build gate features from exemplars using profile PSSM (cons_prob, match_flag)
                    gate_features = None
                    if ex_ids is not None and ex_ids.numel()>0:
                        from retrieval.profile_hmm import build_pssm
                        from utils import ID_TO_AA
                        # Decode yb to AA chars (approximate by ID_TO_AA; -100 ignored)
                        B,T = xb.size()  # Use input sequence length, not target length
                        gfeat = torch.zeros(B,T,2, device=self.device)
                        for b in range(B):
                            ex_seq = ''.join(ID_TO_AA.get(int(t), '') for t in ex_ids[b].tolist() if t>=6 and t<26)
                            msa = [ex_seq] if len(ex_seq)>0 else []
                            import numpy as np
                            pssm, consensus, consv = build_pssm(msa)
                            for j in range(T):
                                col = min(j, pssm.shape[0]-1) if pssm.shape[0]>0 else -1
                                if col>=0:
                                    gfeat[b,j,0] = float(np.exp(pssm[col].max()))  # cons prob
                                    # For gate features, we need to check if the input token at position j matches consensus
                                    # Since we're building features for the input sequence, we should use xb instead of yb
                                    aa_id = int(xb[b,j].item())
                                    if aa_id>=6 and aa_id<26:
                                        aa = ID_TO_AA.get(aa_id, '')
                                        gfeat[b,j,1] = 1.0 if (len(consensus)>col and aa == consensus[col]) else 0.0
                        gate_features = gfeat
                    out = self.model(xb, attn_mask=attn, exemplar_ids=ex_ids, exemplar_mask=ex_mask, return_mix=True, gate_bias=self.gate_bias, gate_features=gate_features)
                    # proxy conserved mask: last 60% tokens (sequence region)
                    start = int(xb.size(1)*0.4)
                    cons_mask = torch.zeros_like(xb, dtype=torch.float32); cons_mask[:, start:] = 1.0
                    loss = self._compute_loss(out, yb, epoch, ex_ids, ex_mask, cons_mask)

                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
                if self.sched is not None:
                    try: self.sched.step()
                    except Exception: pass
                total += float(loss.item()); n += 1

            train_loss = total/max(1,n)

            # val
            self.model.eval()
            vtot=0.0; vn=0
            with torch.no_grad():
                for xb, yb, attn, exemplars in val_loader:
                    xb, yb, attn = xb.to(self.device), yb.to(self.device), attn.to(self.device)
                    out = self.model(xb, attn_mask=attn, return_mix=True, gate_bias=0.0)
                    loss = self.loss_fn(out["logits"].view(-1, out["logits"].size(-1)), yb.view(-1))
                    vtot += float(loss.item()); vn += 1
            val_loss = vtot/max(1,vn)

            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save({"model": self.model.state_dict()}, self.save_path)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} gate_scale={self.gate_scale:.3f} gate_bias={self.gate_bias:.3f}")
            if self.use_wandb and self.wandb is not None:
                self.wandb.log({"train/loss": train_loss, "val/loss": val_loss,
                                "gate/scale": self.gate_scale, "gate/bias": self.gate_bias}, step=epoch)
