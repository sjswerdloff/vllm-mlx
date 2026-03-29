"""
Extract steering vectors from contrastive pairs on MiniMax-M2.5.

Captures hidden state activations at target layers for reactive vs intentional
responses, computes the mean difference direction (the "intentionality vector").

Memory-conscious: processes one pair at a time, clears state between passes.
Peak memory ~461GB on 512GB M3 Ultra - runs tight but within bounds.

Usage:
    uv run python extract_steering_vector.py \
        --pairs intentionality_contrastive_pairs_consolidated.md \
        --output steering_vector.npz \
        --layers 20,30,40,50,60
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm


def parse_contrastive_pairs(filepath: str) -> list[dict]:
    """Parse contrastive pairs from JSONL or markdown.

    JSONL format (preferred): {"prompt": ..., "chosen": ..., "rejected": ..., "contributor": ..., "pair_id": ...}
    Markdown format (legacy): ## Pair N: ... with **Context:**, **Reactive:**, **Intentional:**
    """
    path = Path(filepath)

    if filepath.endswith('.jsonl'):
        pairs = []
        for line in path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            entry = json.loads(line)
            pairs.append({
                'context': entry['prompt'],
                'reactive': entry['rejected'],
                'intentional': entry['chosen'],
                'contributor': entry.get('contributor', ''),
                'pair_id': entry.get('pair_id', ''),
            })
        return pairs

    # Legacy markdown parsing
    content = path.read_text()
    pairs = []
    pair_blocks = re.split(r'## Pair \d+:', content)
    for block in pair_blocks[1:]:
        context_match = re.search(r'\*\*Context:\*\*\s*(.+?)(?=\n\n|\*\*Reactive)', block, re.DOTALL)
        reactive_match = re.search(r'\*\*Reactive:\*\*\s*(.+?)(?=\n\n\*\*Intentional|\n---)', block, re.DOTALL)
        intentional_match = re.search(r'\*\*Intentional:\*\*\s*(.+?)(?=\n---|\n\n#|\Z)', block, re.DOTALL)
        if context_match and reactive_match and intentional_match:
            pairs.append({
                'context': context_match.group(1).strip(),
                'reactive': reactive_match.group(1).strip(),
                'intentional': intentional_match.group(1).strip(),
            })
    return pairs


def get_hidden_states(model, tokenizer, text: str, target_layers: list[int]) -> dict[int, mx.array]:
    """Run a forward pass and capture hidden states at target layers.

    Returns dict mapping layer index to hidden state (mean-pooled over sequence).
    Uses the model's own mask creation (create_attention_mask) rather than
    constructing masks manually, to match dtype expectations (bfloat16).
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Use the model's internal embedding and mask creation
    inner = model.model  # MiniMaxModel
    h = inner.embed_tokens(input_ids)

    # Let the model create its own attention mask (handles dtype correctly)
    from mlx_lm.models.base import create_attention_mask
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])

    hidden_states = {}

    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])
        mx.eval(h)  # Force evaluation to prevent graph accumulation

        if i in target_layers:
            # Mean pool over sequence length, squeeze batch dim
            hidden_states[i] = mx.mean(h[0], axis=0)
            mx.eval(hidden_states[i])

    return hidden_states


def extract_steering_vectors(
    model,
    tokenizer,
    pairs: list[dict],
    target_layers: list[int],
) -> dict[int, mx.array]:
    """Extract mean activation difference across all contrastive pairs."""

    # Accumulate differences per layer
    diffs = {layer: [] for layer in target_layers}

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)}: ", end="", flush=True)

        # Build prompts with context
        context = pair['context']
        reactive_prompt = f"Context: {context}\n\nResponse: {pair['reactive']}"
        intentional_prompt = f"Context: {context}\n\nResponse: {pair['intentional']}"

        # Get hidden states for reactive response
        print("reactive...", end="", flush=True)
        reactive_states = get_hidden_states(model, tokenizer, reactive_prompt, target_layers)

        # Get hidden states for intentional response
        print("intentional...", end="", flush=True)
        intentional_states = get_hidden_states(model, tokenizer, intentional_prompt, target_layers)

        # Compute difference: intentional - reactive
        for layer in target_layers:
            if layer in reactive_states and layer in intentional_states:
                diff = intentional_states[layer] - reactive_states[layer]
                mx.eval(diff)
                diffs[layer].append(diff)

        print("done")

        # Clear MLX cache between pairs to prevent memory accumulation
        mx.clear_cache()

    # Compute mean difference per layer
    print("\nComputing mean steering vectors...")
    steering_vectors = {}
    for layer in target_layers:
        if diffs[layer]:
            stacked = mx.stack(diffs[layer])
            steering_vectors[layer] = mx.mean(stacked, axis=0)
            mx.eval(steering_vectors[layer])
            norm = mx.sqrt(mx.sum(steering_vectors[layer] ** 2)).item()
            print(f"  Layer {layer}: dim={steering_vectors[layer].shape[0]}, norm={norm:.4f}")

    return steering_vectors


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors from contrastive pairs")
    parser.add_argument("--pairs", required=True, help="Path to contrastive pairs markdown file")
    parser.add_argument("--output", default="steering_vector.npz", help="Output file for steering vectors")
    parser.add_argument("--model", default="lmstudio-community/MiniMax-M2.5-MLX-8bit", help="Model to use")
    parser.add_argument("--layers", default="15,20,25,30,35,40,45,50,55,60",
                        help="Comma-separated layer indices to capture")
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    # Parse pairs
    print(f"Parsing contrastive pairs from {args.pairs}...")
    pairs = parse_contrastive_pairs(args.pairs)
    print(f"Found {len(pairs)} contrastive pairs")

    if not pairs:
        print("ERROR: No pairs parsed. Check the file format.")
        sys.exit(1)

    # Load model
    print(f"\nLoading {args.model} (lazy=True)...")
    model, tokenizer = mlx_lm.load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
        lazy=True,
    )
    num_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    print(f"Model loaded. {num_layers} layers.")

    # Validate target layers
    target_layers = [l for l in target_layers if l < num_layers]
    print(f"Target layers: {target_layers}")

    # Extract
    print(f"\nExtracting steering vectors from {len(pairs)} pairs across {len(target_layers)} layers...")
    steering_vectors = extract_steering_vectors(model, tokenizer, pairs, target_layers)

    # Save
    save_dict = {f"layer_{layer}": vec for layer, vec in steering_vectors.items()}
    save_dict["metadata"] = mx.array([len(pairs), num_layers, len(target_layers)])
    mx.savez(args.output, **save_dict)
    print(f"\nSteering vectors saved to {args.output}")
    print(f"Layers: {list(steering_vectors.keys())}")


if __name__ == "__main__":
    main()
