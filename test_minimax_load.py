"""Smoke test: load MiniMax-M2.5 MLX weights and generate one response.

Memory management for 229B MoE model on 512GB M3 Ultra:
- Native 8-bit quantized weights stay quantized in memory (~229GB)
- lazy=True: weights stay mmap'd, only loaded on demand during forward pass
- max_kv_size: bounds KV cache growth
- MoE activates 8 of 256 experts per token - most weights stay on disk

Usage:
    uv run python test_minimax_load.py
    uv run python test_minimax_load.py --model catid/MiniMax-M2.5-catid
"""

import argparse

import mlx_lm

DEFAULT_MODEL = "lmstudio-community/MiniMax-M2.5-MLX-8bit"


def main():
    parser = argparse.ArgumentParser(description="Smoke test MiniMax-M2.5 on MLX")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF repo")
    args = parser.parse_args()

    print(f"Loading {args.model} with lazy=True...")
    model, tokenizer = mlx_lm.load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
        lazy=True,
    )
    print(f"Model loaded (lazy). Type: {type(model).__name__}")
    print(f"Layers: {len(model.layers) if hasattr(model, 'layers') else 'unknown'}")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello in one sentence."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Generating (max 64 tokens)...")
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=64,
        max_kv_size=4096,
        verbose=True,
    )
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
