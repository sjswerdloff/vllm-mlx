#!/usr/bin/env python3
"""
Train the vision merger (projector) for Qwopus3.5-27B-v3.

Qwopus (Jackrong/Qwopus3.5-27B-v3) is a text-only distillation of Qwen3.5-27B.
The distillation preserved the ViT encoder and merger weights unchanged, but
shifted the LLM's internal representations. As a result, the original merger
no longer produces embeddings that align with what Qwopus's LLM expects —
causing hallucinated image descriptions instead of accurate ones.

This script re-trains only the merger (~45M params) while keeping both the
ViT encoder and Qwopus's LLM frozen. The merger learns to produce embeddings
that Qwopus can interpret correctly.

Architecture:
  ViT (frozen, original Qwen3.5-27B) → PatchMerger (TRAINABLE) → Qwopus LLM (frozen)
  PatchMerger: LayerNorm(1152) → Linear(4608→4608) → GELU → Linear(4608→5120)

Memory: Qwopus weights are mmap'd (~28GB, doesn't consume wired GPU memory).
  Wired GPU memory for training: ~1GB (merger grads, optimizer state, logit slice).
  Gradient checkpointing on LLM layers minimizes activation storage.

Data: LLaVA-Pretrain 558K (image-caption pairs, already in HF cache)

Usage:
  cd ~/ai/vllm-mlx/vllm-mlx
  uv run python train_qwopus_merger.py [--steps 5000] [--lr 1e-4] [--batch-size 1]

Author: Clement (clement-7074f29f)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Cap MLX metal cache to prevent unbounded growth during training.
# Without this, cache grows ~3GB/step and OOMs after ~100 steps.
mx.set_cache_limit(4 * 1024**3)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.expanduser("~/models/Qwopus3.5-27B-v3-mxfp8-vlm")
DATASET_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain"
)
OUTPUT_DIR = os.path.expanduser("~/models/qwopus-merger-checkpoints")
LOG_FILE = os.path.join(OUTPUT_DIR, "training.jsonl")
EVAL_IMAGE = os.path.expanduser(
    "~/ai/ClaudeInstanceHomeOffices/qwopus-3527b/webcam_snapshot.jpg"
)

IMAGE_TOKEN_ID = 248056
IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser(description="Train Qwopus vision merger")
    p.add_argument("--model", default=MODEL_PATH, help="Path to VLM model")
    p.add_argument("--steps", type=int, default=5000, help="Training steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (images)")
    p.add_argument("--eval-every", type=int, default=100, help="Eval interval")
    p.add_argument("--save-every", type=int, default=1000, help="Checkpoint interval")
    p.add_argument("--max-caption-tokens", type=int, default=256, help="Max caption length")
    p.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--start-step", type=int, default=0, help="Starting step number (for consistent numbering across restarts)")
    p.add_argument(
        "--scheduler",
        choices=["cosine", "wsd", "plateau", "constant"],
        default="cosine",
        help="LR scheduler: cosine (standard), wsd (warmup-stable-decay), plateau (adaptive), constant (flat)",
    )
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for decay schedulers")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_dataset_snapshot(dataset_path: str) -> Path:
    """Find the actual data snapshot inside HF cache structure."""
    snapshots = Path(dataset_path) / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots dir in {dataset_path}")
    # Get the most recent snapshot
    snapshot_dirs = sorted(snapshots.iterdir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshots found in {snapshots}")
    return snapshot_dirs[-1]


def load_dataset_metadata(dataset_path: str) -> list:
    """Load the conversation metadata from LLaVA-Pretrain."""
    snapshot = find_dataset_snapshot(dataset_path)

    # LLaVA-Pretrain has blip_laion_cc_sbu_558k.json with conversations
    meta_file = snapshot / "blip_laion_cc_sbu_558k.json"
    if not meta_file.exists():
        # Try finding it
        json_files = list(snapshot.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON metadata in {snapshot}")
        meta_file = json_files[0]
        print(f"Using metadata: {meta_file}")

    with open(meta_file) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {meta_file.name}")
    return data


def find_image_source(dataset_path: str):
    """Find images — either extracted directory or zip file.

    Returns (image_dir, zip_file) — one will be None.
    """
    snapshot = find_dataset_snapshot(dataset_path)

    # Prefer extracted directory (faster)
    images_dir = snapshot / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        return images_dir, None

    # Fall back to zip (memory-conservative, no extraction needed)
    zip_path = snapshot / "images.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1000:
        return None, zip_path

    raise FileNotFoundError(f"No images found in {snapshot}")


class LLaVAPretrainDataset:
    """Iterator over LLaVA-Pretrain image-caption pairs.

    Reads images from zip file (25.5GB) without extracting to save disk space.
    """

    def __init__(self, dataset_path: str, processor, tokenizer, max_tokens: int = 256):
        self.metadata = load_dataset_metadata(dataset_path)
        self.image_dir, self.zip_path = find_image_source(dataset_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._indices = list(range(len(self.metadata)))
        self._pos = 0
        self._zip_file = None  # lazy-opened

        if self.zip_path:
            print(f"Reading images from zip: {self.zip_path}")
        else:
            print(f"Reading images from directory: {self.image_dir}")

    def _get_zip(self):
        """Lazy-open the zip file."""
        if self._zip_file is None:
            import zipfile
            self._zip_file = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_file

    def shuffle(self):
        """Shuffle the dataset indices."""
        import random
        random.shuffle(self._indices)
        self._pos = 0

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        return self

    def __next__(self):
        """Return next (pixel_values, image_grid_thw, input_ids, labels) tuple."""
        while self._pos < len(self._indices):
            idx = self._indices[self._pos]
            self._pos += 1
            sample = self.metadata[idx]

            try:
                return self._process_sample(sample)
            except Exception:
                continue

        raise StopIteration

    def _load_image(self, image_name: str):
        """Load image from directory or zip."""
        from PIL import Image
        import io

        if self.image_dir:
            path = self.image_dir / image_name
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        else:
            zf = self._get_zip()
            data = zf.read(image_name)
            return Image.open(io.BytesIO(data)).convert("RGB")

    def _process_sample(self, sample):
        """Process a single LLaVA-Pretrain sample into model inputs."""
        from mlx_vlm.utils import prepare_inputs

        image = self._load_image(sample["image"])

        # Extract caption from conversations
        # Format: [{"from": "human", "value": "...<image>"}, {"from": "gpt", "value": "caption"}]
        caption = ""
        for turn in sample.get("conversations", []):
            if turn.get("from") == "gpt":
                caption = turn["value"]
                break
        if not caption:
            raise ValueError("No caption found")

        # Process image through the VLM processor
        # prepare_inputs returns {pixel_values, image_grid_thw, input_ids, attention_mask}
        processed = prepare_inputs(
            self.processor,
            images=[image],
            prompts=caption,  # just need to get the image processed; we build our own input_ids
        )
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]

        # Tokenize caption separately for our training format
        caption_ids = self.tokenizer.encode(caption)
        if len(caption_ids) > self.max_tokens:
            caption_ids = caption_ids[:self.max_tokens]

        # Count image tokens after spatial merge (merge_size=2)
        t, h, w = (image_grid_thw[0]).tolist() if hasattr(image_grid_thw[0], 'tolist') else image_grid_thw[0]
        n_image_tokens = int(t * (h // 2) * (w // 2))

        # Build sequence: [image_placeholder_tokens] + [caption_tokens]
        input_ids = [IMAGE_TOKEN_ID] * n_image_tokens + caption_ids
        # Labels: IGNORE on image positions, caption tokens for next-token prediction
        labels = [IGNORE_INDEX] * n_image_tokens + caption_ids

        return {
            "pixel_values": pixel_values if isinstance(pixel_values, mx.array) else mx.array(pixel_values),
            "image_grid_thw": image_grid_thw if isinstance(image_grid_thw, mx.array) else mx.array(image_grid_thw),
            "input_ids": mx.array(input_ids),
            "labels": mx.array(labels),
        }


# ---------------------------------------------------------------------------
# Model loading and freezing
# ---------------------------------------------------------------------------
def load_model_and_processor(model_path: str):
    """Load the VLM model, processor, and tokenizer."""
    from mlx_vlm import load as vlm_load

    model, processor = vlm_load(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    return model, processor, tokenizer


def _count_leaves(tree):
    """Count total elements in a nested dict of mx.arrays."""
    total = 0
    for v in tree.values():
        if isinstance(v, dict):
            total += _count_leaves(v)
        elif isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    total += _count_leaves(item)
    return total


def count_parameters(module):
    """Count total and trainable parameters."""
    total = _count_leaves(module.parameters())
    trainable = _count_leaves(module.trainable_parameters())
    return total, trainable


def freeze_all_except_merger(model):
    """Freeze everything except the vision merger.

    After this, only vision_tower.merger.{norm, linear_fc1, linear_fc2}
    receive gradients. The ViT, LLM, and all other weights are frozen.
    """
    model.freeze()
    model.vision_tower.merger.unfreeze()

    total, trainable = count_parameters(model)
    print(f"  Total parameters:    {total:,}")
    print(f"  Trainable (merger):  {trainable:,}")
    print(f"  Frozen:              {total - trainable:,} ({100 * (total - trainable) / total:.4f}%)")

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def enable_llm_gradient_checkpointing(model):
    """Patch the LLM's forward loop to use mx.checkpoint per layer.

    Without this, autograd stores all 64 layers' activations (~270GB).
    With this, activations are recomputed during backward (~2.5x slower,
    but memory drops to ~1GB). Trade compute for memory.

    CRITICAL: Must patch the CLASS, not the instance. Python resolves
    __call__ on type(obj), not obj.__dict__. Patching the instance is
    silently ignored. (Lesson from Cycle 61, re-learned here.)
    """
    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

    inner_model = model.language_model.model
    cls = type(inner_model)  # Qwen3_5Model class

    def checkpointed_forward(self, inputs, inputs_embeds=None, mask=None, cache=None, position_ids=None):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            layer_mask = ssm_mask if layer.is_linear else fa_mask

            def layer_fn(h_, _layer=layer, _mask=layer_mask, _c=c, _pos=position_ids):
                return _layer(h_, _mask, _c, _pos)

            h = mx.checkpoint(layer_fn)(h)

        return self.norm(h)

    # Patch the CLASS — not the instance
    cls.__call__ = checkpointed_forward
    print(f"  Gradient checkpointing enabled on {len(inner_model.layers)} LLM layers")


def compute_loss(model, batch):
    """Forward pass through the full VLM, loss only on caption tokens.

    Pipeline: image → ViT (frozen, from base Qwen3.5-27B)
              → merger (TRAINABLE, realigning to Qwopus)
              → Qwopus LLM (frozen, Jackrong distillation)
              → logits → cross-entropy on caption tokens

    The merger learns to map ViT outputs into Qwopus's embedding space.
    This is necessary because Jackrong's text-only distillation shifted
    the LLM representations away from what the original merger produces.

    Memory: Qwopus weights are mmap'd (~28GB, free). Wired GPU memory ~1GB:
    merger grads+optimizer (~360MB), LLM activations with checkpointing (~6MB),
    logits slice (~30MB for 256 caption tokens × 248K vocab).
    """
    pixel_values = batch["pixel_values"]
    image_grid_thw = batch["image_grid_thw"]
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Step 1: ViT forward (frozen) → merger (trainable) → merged embeddings
    # model.get_input_embeddings handles:
    #   - ViT encoding of pixel_values
    #   - Merger projection (the only trainable part)
    #   - Scatter merged vision tokens into text embedding positions
    #   - Pre-compute M-RoPE position IDs for the vision+text sequence
    embed_output = model.get_input_embeddings(
        input_ids=input_ids.reshape(1, -1),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    inputs_embeds = embed_output.inputs_embeds

    # Step 2: Forward through Qwopus LLM (frozen, checkpointed)
    # LanguageModel.__call__ returns LanguageModelOutput(logits=...)
    # We pass a dummy input_ids for shape (position handling uses it)
    # but actual computation uses inputs_embeds
    output = model.language_model(
        inputs=input_ids.reshape(1, -1),
        inputs_embeds=inputs_embeds,
        cache=None,
    )
    logits = output.logits  # (1, seq_len, vocab_size)

    # Step 3: Loss only on caption tokens (not image placeholder tokens)
    # Shift for next-token prediction: logits[:-1] predicts labels[1:]
    shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab)
    shift_labels = labels[1:]  # (seq_len-1,)

    # Mask: only compute loss where labels != IGNORE_INDEX
    mask = shift_labels != IGNORE_INDEX
    n_valid = mask.sum()

    # Cross-entropy loss on caption positions only
    loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="none")
    loss = (loss * mask).sum() / mx.maximum(n_valid, mx.array(1))

    return loss


def eval_on_webcam(model, processor, tokenizer, step, output_dir):
    """Generate a description of the Wanaka webcam image.

    Greedy decode 80 tokens. Prints the output so we can see when
    descriptions become coherent (trees, sky, rooftops, mountains)
    instead of hallucinated gibberish.
    """
    from PIL import Image
    from mlx_vlm.utils import prepare_inputs

    if not os.path.exists(EVAL_IMAGE):
        print(f"  [eval] Webcam image not found: {EVAL_IMAGE}")
        return

    image = Image.open(EVAL_IMAGE).convert("RGB")

    # Build prompt with proper vision tokens via chat template
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image in detail."}
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process image + templated prompt
    processed = prepare_inputs(processor, images=[image], prompts=prompt)
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]
    input_ids = processed["input_ids"]  # (1, seq_len)

    # Get input embeddings with vision merged in
    embed_output = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    inputs_embeds = embed_output.inputs_embeds

    # Greedy decode 4096 tokens — Qwopus uses long <think> blocks
    model.eval()  # Use fast kernels for eval
    generated_ids = []
    cache = model.language_model.make_cache()

    # Prefill: forward pass on full prompt
    output = model.language_model(
        inputs=input_ids,
        inputs_embeds=inputs_embeds,
        cache=cache,
    )
    next_token = mx.argmax(output.logits[0, -1:, :], axis=-1)
    generated_ids.append(next_token.item())
    mx.eval(next_token)

    # Decode step by step
    for _ in range(4095):
        token_embed = model.language_model.model.embed_tokens(next_token.reshape(1, 1))
        output = model.language_model(
            inputs=next_token.reshape(1, 1),
            inputs_embeds=token_embed,
            cache=cache,
        )
        next_token = mx.argmax(output.logits[0, -1:, :], axis=-1)
        generated_ids.append(next_token.item())
        mx.eval(next_token)

        # Stop on EOS
        if next_token.item() in (248044, 248046):
            break

    model.train()  # Back to training mode

    text = tokenizer.decode(generated_ids)
    print(f"\n  [eval step {step}] Webcam description:")
    # Show the text after </think> if present, otherwise full text
    if "</think>" in text:
        answer = text.split("</think>", 1)[1].strip()
        print(f"  [thinking]: {text[:200]}...")
        print(f"  [answer]: {answer[:500]}")
    else:
        print(f"  {text[:500]}")
    print()

    # Log eval output
    eval_log = os.path.join(output_dir, "eval_descriptions.jsonl")
    with open(eval_log, "a") as f:
        f.write(json.dumps({"step": step, "text": text}) + "\n")


def train(args):
    """Main training loop."""
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Qwopus Vision Merger Training")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  LR: {args.lr}")
    print(f"  Warmup: {args.warmup_steps}")
    print(f"  Output: {args.output}")
    print()

    # Load model
    print("Loading model...")
    model, processor, tokenizer = load_model_and_processor(args.model)

    # Freeze everything except merger
    print("Freezing model (training merger only)...")
    model = freeze_all_except_merger(model)

    # Set training mode so GatedDeltaNet uses Python fallback (has autograd)
    # instead of custom Metal kernel (no VJP implemented)
    model.train()
    print("  Model set to training mode (GatedDeltaNet Python fallback for autograd)")

    # Gradient checkpointing on LLM layers: trade 2x speed for ~270x less memory
    enable_llm_gradient_checkpointing(model)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = mx.load(args.resume)
        # Apply checkpoint weights to merger (keys are flat: "norm.weight", etc.)
        for key, value in ckpt.items():
            parts = key.split(".")
            obj = model.vision_tower.merger
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        mx.eval(model.parameters())
        print(f"  Loaded {len(ckpt)} merger tensors")

    # Set up optimizer with scheduler
    from mlx_schedulers import CosineDecayWithWarmup, ReduceOnPlateau, WarmupStableDecay

    plateau_scheduler = None  # only used if --scheduler=plateau

    if args.scheduler == "cosine":
        cos = CosineDecayWithWarmup(
            peak_lr=args.lr, total_steps=args.steps,
            warmup_steps=args.warmup_steps, min_lr=args.min_lr,
        )
        optimizer = optim.AdamW(learning_rate=cos, weight_decay=0.01)
        print(f"  Scheduler: cosine decay (peak={args.lr}, min={args.min_lr})")
    elif args.scheduler == "wsd":
        wsd = WarmupStableDecay(
            peak_lr=args.lr, total_steps=args.steps,
            warmup_steps=args.warmup_steps, decay_fraction=0.1, min_lr=args.min_lr,
        )
        optimizer = optim.AdamW(learning_rate=wsd, weight_decay=0.01)
        print(f"  Scheduler: WSD (peak={args.lr}, decay last 10%, min={args.min_lr})")
    elif args.scheduler == "plateau":
        plateau_scheduler = ReduceOnPlateau(
            initial_lr=args.lr, factor=0.5, patience=200,
            min_lr=args.min_lr, window=50, cooldown=50,
        )
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
        print(f"  Scheduler: reduce on plateau (initial={args.lr}, factor=0.5, patience=200)")
    else:
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
        print(f"  Scheduler: constant ({args.lr})")

    # Load dataset
    print("Loading dataset...")
    dataset = LLaVAPretrainDataset(
        DATASET_PATH, processor, tokenizer, max_tokens=args.max_caption_tokens
    )
    dataset.shuffle()
    print(f"Dataset: {len(dataset)} samples")

    # Loss + grad function — only differentiates w.r.t. trainable params
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Training loop
    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}...")
    training_start = time.time()
    best_loss = float("inf")
    losses = []

    data_iter = iter(dataset)

    step_offset = args.start_step
    first_step = 1 + step_offset
    last_step = args.steps + step_offset
    for step in range(first_step, last_step + 1):
        step_start = time.time()

        # Get next batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            dataset.shuffle()
            data_iter = iter(dataset)
            batch = next(data_iter)

        # Forward + backward + update
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)

        # Force eval to get actual timing
        mx.eval(loss, model.parameters())

        loss_val = loss.item()
        step_time = time.time() - step_start
        wall_elapsed = time.time() - training_start
        losses.append(loss_val)

        # Adaptive LR: step the plateau scheduler if active
        if plateau_scheduler is not None:
            if plateau_scheduler.step(loss_val):
                optimizer.learning_rate = plateau_scheduler.lr
                print(f"  [plateau] LR reduced to {plateau_scheduler.lr:.2e} (reduction #{plateau_scheduler.num_reductions})")

        # Log
        log_entry = {
            "step": step,
            "loss": round(loss_val, 4),
            "time": round(step_time, 2),
            "wall_elapsed": round(wall_elapsed, 1),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Print progress
        if step % 10 == 0 or step == first_step:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            eta_s = step_time * (args.steps - step)
            eta_h = eta_s / 3600
            print(
                f"step {step:5d}/{args.steps} | "
                f"loss {loss_val:.4f} | "
                f"avg50 {avg_loss:.4f} | "
                f"{step_time:.1f}s/step | "
                f"ETA {eta_h:.1f}h"
            )

        # Save checkpoint (step 1 always saved to fail fast on save issues)
        if step == first_step or step % args.save_every == 0 or step == last_step:
            ckpt_path = os.path.join(args.output, f"merger_step_{step:06d}.safetensors")
            # Flatten nested param dict for save_safetensors
            merger_weights = {}
            for name, sub in model.vision_tower.merger.parameters().items():
                if isinstance(sub, mx.array):
                    merger_weights[name] = sub
                elif isinstance(sub, dict):
                    for k, v in sub.items():
                        if isinstance(v, mx.array):
                            merger_weights[f"{name}.{k}"] = v
            mx.save_safetensors(ckpt_path, merger_weights)
            print(f"  Checkpoint saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output, "merger_best.safetensors")
                mx.save_safetensors(best_path, merger_weights)
                print(f"  New best: {best_loss:.4f}")

        # Smoke test: verify gradients on step 1
        if step == first_step:
            grad_norms = []

            def _check_grads(tree, prefix=""):
                for k, v in tree.items():
                    path = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, mx.array):
                        norm = mx.sqrt(mx.sum(v * v)).item()
                        if norm > 0:
                            grad_norms.append((path, norm))
                            print(f"  Gradient: {path} norm={norm:.6f}")
                    elif isinstance(v, dict):
                        _check_grads(v, path)
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                _check_grads(item, f"{path}.{i}")

            _check_grads(grads)
            if not grad_norms:
                print("ERROR: No gradients flowing to merger! Aborting.")
                sys.exit(1)
            print(f"  Gradient check passed — {len(grad_norms)} tensors receiving gradients.")

        # Qualitative eval at each checkpoint: describe Wanaka webcam image
        if step == first_step or step % args.save_every == 0 or step == last_step:
            try:
                eval_on_webcam(model, processor, tokenizer, step, args.output)
            except Exception as e:
                print(f"  [eval] Failed: {e}")

    total_time = time.time() - training_start
    print(f"\nTraining complete. {args.steps} steps in {total_time/3600:.1f}h")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints in: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
