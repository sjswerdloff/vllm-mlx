#!/usr/bin/env python3
"""Verify gradient checkpointing actually works (class patch, not instance patch)."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.set_cache_limit(4 * 1024**3)

from train_qwopus_merger import (
    load_model_and_processor,
    freeze_all_except_merger,
    enable_llm_gradient_checkpointing,
    LLaVAPretrainDataset,
    DATASET_PATH,
    MODEL_PATH,
    compute_loss,
)

print("Loading model...")
model, processor, tokenizer = load_model_and_processor(MODEL_PATH)
freeze_all_except_merger(model)
model.train()
enable_llm_gradient_checkpointing(model)

optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

print("Loading dataset...")
ds = LLaVAPretrainDataset(DATASET_PATH, processor, tokenizer, max_tokens=64)
ds.shuffle()
data_iter = iter(ds)

print("\nWith CLASS-level checkpointing + cache limit:")
print("Step | Peak MB | Active MB | Cache MB | s/step | Loss")
print("-" * 65)

import time
for step in range(1, 11):
    t0 = time.time()
    batch = next(data_iter)
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())
    elapsed = time.time() - t0

    peak = mx.get_peak_memory() / 1e6
    active = mx.get_active_memory() / 1e6
    cache = mx.get_cache_memory() / 1e6
    print(f"  {step:2d}  | {peak:8.0f} | {active:8.0f} | {cache:7.0f} | {elapsed:5.1f}  | {loss.item():.4f}")

print("\nExpected: ~6s/step (checkpointed), active+cache stable, no growth to 50GB+")
