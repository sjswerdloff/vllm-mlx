"""
Test that the Llama EAGLE3 patch doesn't change the model's own output.

The patch captures hidden states at intermediate layers for the EAGLE3
draft head. If this capture changes the model's logits, the verify pass
will produce different results than what the EAGLE3 head was trained
against, causing 0% acceptance.

This test patches the model via class swap (matching the real
inject_eagle3_llama) but WITHOUT loading EAGLE3 head weights,
isolating the question: does the patched forward pass produce
the same logits as the original?

IMPORTANT: Python resolves __call__ on the TYPE, not the instance.
Setting __call__ via object.__setattr__ does NOT affect model(tokens).
Must use class swap (model.__class__ = NewClass) like the real patch.

Usage:
    # Requires Llama 70B — run from vllm-mlx-test directory
    uv run python tests/test_eagle3_patch_parity.py

    # Or with pytest:
    uv run python -m pytest tests/test_eagle3_patch_parity.py -v -s
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "mlx-community/Llama-3.3-70B-Instruct-8bit"


def _apply_capture_patch(model, aux_layer_ids):
    """Apply ONLY the hidden-state capture patch via class swap.

    Replicates the forward-pass changes from inject_eagle3_llama without
    requiring the EAGLE3 head weights. Uses the same class-swap mechanism
    as the real patch — NOT object.__setattr__ on __call__, because
    Python resolves dunder methods on the type, not the instance.
    """
    inner = model.model
    args = model.args
    lm_head = getattr(model, "lm_head", None)

    # Store refs via object.__setattr__ to bypass nn.Module
    object.__setattr__(model, "_test_inner", inner)
    object.__setattr__(model, "_test_args", args)
    object.__setattr__(model, "_test_lm_head", lm_head)
    object.__setattr__(model, "_test_aux_layer_ids", aux_layer_ids)
    object.__setattr__(model, "_test_capture_set", set(aux_layer_ids))
    object.__setattr__(model, "_test_captured", None)

    original_class = model.__class__

    class _TestCaptureModel(original_class):
        """Model with hidden state capture — no EAGLE3 head."""

        def __call__(self, inputs=None, cache=None, input_embeddings=None, **kwargs):
            inner = self._test_inner

            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = inner.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(inner.layers)

            from mlx_lm.models.llama import create_attention_mask
            fa_mask = create_attention_mask(h, cache[inner.fa_idx])
            swa_mask = None
            if inner.swa_idx is not None:
                swa_mask = create_attention_mask(
                    h, cache[inner.swa_idx], window_size=inner.sliding_window
                )

            captured = {}
            for i, (layer, c) in enumerate(zip(inner.layers, cache)):
                mask = swa_mask if layer.use_sliding else fa_mask
                h = layer(h, mask, cache=c)
                if i in self._test_capture_set:
                    captured[i] = h

            normed = inner.norm(h)

            if self._test_args.tie_word_embeddings:
                out = inner.embed_tokens.as_linear(normed)
            else:
                out = self._test_lm_head(normed)

            self._test_captured = [captured[lid] for lid in self._test_aux_layer_ids]
            return out

    model.__class__ = _TestCaptureModel


def test_patch_does_not_change_logits():
    """Patched and unpatched Llama must produce identical logits."""
    print(f"\nLoading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    num_layers = len(model.model.layers)
    aux_layer_ids = [2, num_layers // 2, num_layers - 3]

    prompt = "The quick brown fox jumps over the lazy dog"
    tokens = mx.array([tokenizer.encode(prompt)])
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens.shape}, Layers: {num_layers}, Capture: {aux_layer_ids}")

    # Cache model reference — same for both runs
    cache_model = model.model

    # --- Unpatched forward ---
    print("\nRunning unpatched forward...")
    cache_unpatched = make_prompt_cache(cache_model)
    logits_unpatched = model(tokens, cache=cache_unpatched)
    if hasattr(logits_unpatched, "logits"):
        logits_unpatched = logits_unpatched.logits
    mx.eval(logits_unpatched)

    argmax_unpatched = mx.argmax(logits_unpatched[0, -1, :]).item()
    print(f"  Logits shape: {logits_unpatched.shape}")
    print(f"  Last token argmax: {argmax_unpatched} ({tokenizer.decode([argmax_unpatched])})")
    print(f"  Logits [0,-1,:5]: {logits_unpatched[0, -1, :5]}")

    # --- Apply capture patch via class swap ---
    print("\nApplying capture patch (class swap)...")
    _apply_capture_patch(model, aux_layer_ids)
    print(f"  Model class: {model.__class__.__name__}")

    # --- Patched forward (fresh cache) ---
    print("Running patched forward...")
    cache_patched = make_prompt_cache(cache_model)
    logits_patched = model(tokens, cache=cache_patched)
    if hasattr(logits_patched, "logits"):
        logits_patched = logits_patched.logits
    mx.eval(logits_patched)

    argmax_patched = mx.argmax(logits_patched[0, -1, :]).item()
    print(f"  Logits shape: {logits_patched.shape}")
    print(f"  Last token argmax: {argmax_patched} ({tokenizer.decode([argmax_patched])})")
    print(f"  Logits [0,-1,:5]: {logits_patched[0, -1, :5]}")

    # --- Also verify captured hidden states exist ---
    captured = model._test_captured
    if captured is not None:
        print(f"\n  Captured {len(captured)} hidden states:")
        for i, (lid, h) in enumerate(zip(aux_layer_ids, captured)):
            print(f"    Layer {lid}: shape={h.shape}, mean={mx.mean(h).item():.4f}")
    else:
        print("\n  WARNING: No hidden states captured — patch may not have run")

    # --- Compare ---
    max_diff = mx.max(mx.abs(logits_unpatched - logits_patched)).item()
    mean_diff = mx.mean(mx.abs(logits_unpatched - logits_patched)).item()

    print(f"\n{'='*50}")
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Argmax match: {argmax_unpatched == argmax_patched}")

    top5_un = sorted(mx.argpartition(-logits_unpatched[0, -1, :], kth=5)[:5].tolist())
    top5_pa = sorted(mx.argpartition(-logits_patched[0, -1, :], kth=5)[:5].tolist())
    print(f"Top-5 match: {top5_un == top5_pa}")
    if top5_un != top5_pa:
        print(f"  Unpatched: {top5_un}")
        print(f"  Patched:   {top5_pa}")

    if max_diff > 1e-4:
        print(f"\nFAIL: Patch changes model logits (max_diff={max_diff:.2e})")
        print("This explains 0% EAGLE3 acceptance.")
    else:
        print(f"\nPASS: Patch does not change model logits")
        print("Bug is elsewhere — check hidden state values,")
        print("EAGLE3 head input construction, or cache handling.")

    assert max_diff < 1e-4, f"Patch changes logits: max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_patch_does_not_change_logits()
