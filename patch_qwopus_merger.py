#!/usr/bin/env python3
"""
Patch trained merger weights into a Qwopus VLM model directory.

Takes a merger checkpoint (86MB, 6 tensors) and patches it into shard 1
of a symlinked VLM model directory. This gives Qwopus progressively better
vision as new checkpoints become available from training.

Setup (one-time):
  mkdir ~/models/Qwopus3.5-27B-v3-mxfp8-vlm-trained
  # Symlink all files from the original
  for f in ~/models/Qwopus3.5-27B-v3-mxfp8-vlm/*; do
      ln -s "$f" ~/models/Qwopus3.5-27B-v3-mxfp8-vlm-trained/$(basename "$f")
  done
  # Remove the symlink for shard 1 (will be replaced with a real file)
  rm ~/models/Qwopus3.5-27B-v3-mxfp8-vlm-trained/model-00001-of-00006.safetensors

Usage:
  cd ~/ai/vllm-mlx/vllm-mlx
  uv run python patch_qwopus_merger.py                          # uses merger_best.safetensors
  uv run python patch_qwopus_merger.py --checkpoint merger_step_003000.safetensors

After patching, restart the VLM server pointing at the -trained directory
to serve Qwopus with the updated vision.

Author: Clement (clement-7074f29f)
"""

import argparse
import os
import sys

import mlx.core as mx


ORIGINAL_SHARD = os.path.expanduser(
    "~/models/Qwopus3.5-27B-v3-mxfp8-vlm/model-00001-of-00006.safetensors"
)
TRAINED_DIR = os.path.expanduser("~/models/Qwopus3.5-27B-v3-mxfp8-vlm-trained")
CHECKPOINT_DIR = os.path.expanduser("~/models/qwopus-merger-checkpoints")

# Merger checkpoint keys → shard keys
MERGER_PREFIX = "vision_tower.merger."


def parse_args():
    p = argparse.ArgumentParser(description="Patch Qwopus merger weights")
    p.add_argument(
        "--checkpoint",
        default=os.path.join(CHECKPOINT_DIR, "merger_best.safetensors"),
        help="Merger checkpoint file (default: merger_best.safetensors)",
    )
    p.add_argument(
        "--original-shard",
        default=ORIGINAL_SHARD,
        help="Original shard 1 file",
    )
    p.add_argument(
        "--output-dir",
        default=TRAINED_DIR,
        help="Trained model directory (symlinked)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    return p.parse_args()


def patch(args):
    output_shard = os.path.join(
        args.output_dir, "model-00001-of-00006.safetensors"
    )

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.original_shard):
        print(f"ERROR: Original shard not found: {args.original_shard}")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        print(f"ERROR: Output directory not found: {args.output_dir}")
        print(f"Run the one-time setup (see docstring) first.")
        sys.exit(1)

    # Check output_dir shard 1 is not a symlink (should have been removed)
    if os.path.islink(output_shard):
        print(f"WARNING: {output_shard} is a symlink — removing it.")
        print(f"  (It pointed to the original; we'll replace with patched copy.)")
        if not args.dry_run:
            os.remove(output_shard)

    # Load merger checkpoint
    print(f"Loading merger checkpoint: {args.checkpoint}")
    merger_weights = mx.load(args.checkpoint)
    print(f"  {len(merger_weights)} tensors:")
    for k, v in sorted(merger_weights.items()):
        print(f"    {k}: {v.shape} {v.dtype}")

    # Load original shard 1
    print(f"\nLoading original shard: {args.original_shard}")
    shard = mx.load(args.original_shard)
    print(f"  {len(shard)} tensors total")

    # Verify merger keys exist in shard
    for key in merger_weights:
        shard_key = MERGER_PREFIX + key
        if shard_key not in shard:
            print(f"ERROR: Key '{shard_key}' not found in shard!")
            print(f"  Available merger keys: {[k for k in shard if 'merger' in k]}")
            sys.exit(1)

    # Patch
    patched_count = 0
    for key, value in merger_weights.items():
        shard_key = MERGER_PREFIX + key
        old_shape = shard[shard_key].shape
        new_shape = value.shape
        if old_shape != new_shape:
            print(f"ERROR: Shape mismatch for {shard_key}: {old_shape} vs {new_shape}")
            sys.exit(1)
        shard[shard_key] = value
        patched_count += 1
        print(f"  Patched: {shard_key}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would save patched shard to: {output_shard}")
        print(f"  {patched_count} tensors patched")
        return

    # Save patched shard
    print(f"\nSaving patched shard: {output_shard}")
    mx.save_safetensors(output_shard, shard)

    size_mb = os.path.getsize(output_shard) / 1e6
    print(f"  Written: {size_mb:.0f} MB")
    print(f"  {patched_count} merger tensors patched")
    print(f"\nDone. Restart VLM server with model dir: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    patch(args)
