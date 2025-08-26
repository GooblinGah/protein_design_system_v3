import torch
import torch.nn as nn
from models.transformer_lm import TransformerCopyLM
from trainers.trainer import Trainer
from utils import VOCAB_SIZE, AA_START, PROMPT_START

def test_transformer_model_creation():
    model = TransformerCopyLM()
    assert isinstance(model, TransformerCopyLM)
    assert model.tok.embedding_dim == 384
    assert model.lm_head.out_features == VOCAB_SIZE

def test_transformer_forward_pass():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    output = model(x, return_mix=False)
    assert output.shape == (batch_size, seq_len, VOCAB_SIZE)
    
    attn_mask = torch.ones(batch_size, seq_len)
    output = model(x, attn_mask=attn_mask, return_mix=False)
    assert output.shape == (batch_size, seq_len, VOCAB_SIZE)

def test_transformer_with_exemplars():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    ex_ids = torch.randint(AA_START, PROMPT_START, (batch_size, 5))
    ex_mask = torch.ones(batch_size, 5)
    
    output = model(x, exemplar_ids=ex_ids, exemplar_mask=ex_mask, return_mix=True)
    assert "logits" in output
    assert "gate" in output
    assert "vocab_logits" in output
    assert "copy_logits" in output
    
    assert output["logits"].shape == (batch_size, seq_len, VOCAB_SIZE)
    assert output["gate"].shape == (batch_size, seq_len)

def test_transformer_gate_features():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    gate_features = torch.randn(batch_size, seq_len, 2)
    output = model(x, gate_features=gate_features, return_mix=True)
    
    assert "gate" in output
    assert output["gate"].shape == (batch_size, seq_len)

def test_transformer_copy_logits():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    ex_ids = torch.randint(AA_START, PROMPT_START, (batch_size, 5))
    ex_mask = torch.ones(batch_size, 5)
    
    output = model(x, exemplar_ids=ex_ids, exemplar_mask=ex_mask, return_mix=True)
    
    assert not torch.allclose(output["copy_logits"], output["vocab_logits"])
    assert torch.isfinite(output["copy_logits"]).all()

def test_trainer_creation():
    model = TransformerCopyLM()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        save_path="test_checkpoint.pt",
        amp=False,
        use_wandb=False
    )
    assert isinstance(trainer, Trainer)
    assert trainer.model == model

def test_trainer_loss_computation():
    model = TransformerCopyLM()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        save_path="test_checkpoint.pt",
        amp=False,
        use_wandb=False
    )
    
    batch_size, seq_len = 2, 10
    vocab_size = VOCAB_SIZE
    
    mock_output = {
        "logits": torch.randn(batch_size, seq_len, vocab_size, requires_grad=True),
        "gate": torch.rand(batch_size, seq_len)
    }
    
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    ex_ids = torch.randint(AA_START, PROMPT_START, (batch_size, 5))
    ex_mask = torch.ones(batch_size, 5)
    cons_mask = torch.ones(batch_size, seq_len)
    
    loss = trainer._compute_loss(mock_output, y, epoch=1, ex_ids=ex_ids, ex_mask=ex_mask, conserved_mask=cons_mask)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)

def test_trainer_gate_monitoring():
    model = TransformerCopyLM()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        save_path="test_checkpoint.pt",
        amp=False,
        use_wandb=False
    )
    
    gates = torch.rand(10, 20)
    conserved_mask = torch.ones(10, 20)
    
    copy_rate = trainer.copy_monitor.update(gates, conserved_mask)
    entropy = trainer.ent_monitor.update(gates, conserved_mask)
    
    assert isinstance(copy_rate, float)
    assert isinstance(entropy, float)
    assert 0.0 <= copy_rate <= 1.0
    assert torch.isfinite(torch.tensor(entropy))

def test_model_device_handling():
    model = TransformerCopyLM()
    
    model_cpu = model.to('cpu')
    assert next(model_cpu.parameters()).device.type == 'cpu'
    
    for param in model_cpu.parameters():
        assert torch.isfinite(param).all()

def test_transformer_attention_mask():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    full_mask = torch.ones(batch_size, seq_len)
    output_full = model(x, attn_mask=full_mask, return_mix=False)
    
    partial_mask = torch.ones(batch_size, seq_len)
    partial_mask[:, 5:] = 0
    output_partial = model(x, attn_mask=partial_mask, return_mix=False)
    
    assert not torch.allclose(output_full, output_partial)

def test_transformer_position_embeddings():
    model = TransformerCopyLM()
    batch_size, seq_len = 2, 10
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    output1 = model(x, return_mix=False)
    
    x_shifted = torch.roll(x, shifts=1, dims=1)
    output2 = model(x_shifted, return_mix=False)
    
    assert not torch.allclose(output1, output2)
