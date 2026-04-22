"""
Diagnose why EAGLE3 produces identical draft tokens within a round.

Tests:
1. KV cache growth: does the EAGLE3 cache accumulate across draft steps?
2. RoPE position advancement: are positions incrementing between steps?
3. Input diversity: does each draft step receive a different token embedding?
4. Output diversity: are logits actually identical or just argmax-identical?
5. Hidden state dominance: how much do hidden states vs token embedding contribute?

Usage:
    uv run python tests/test_eagle3_draft_diversity.py
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


MODEL_PATH = "mlx-community/Llama-3.3-70B-Instruct-8bit"
EAGLE3_HEAD_PATH = "/Users/stuartswerdloff/.cache/huggingface/hub/models--RedHatAI--Llama-3.3-70B-Instruct-speculator.eagle3/snapshots/42864f78d503693ae5fd317419f099ab3b7c13e4"


def run_diagnostics():
    from vllm_mlx.patches.llama_eagle3 import inject_eagle3_llama
    from vllm_mlx.spec_decode import prefill_model

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    inject_eagle3_llama(model, EAGLE3_HEAD_PATH)

    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    tokens = tokenizer.encode(prompt)
    tokens_arr = mx.array([tokens])

    cache_model = getattr(model, "_eagle3_inner", model.model)
    cache = make_prompt_cache(cache_model)
    sampler = make_sampler(temp=0.0)

    # Prefill
    target_logits = prefill_model(model, tokens_arr, cache, 2048)
    mx.eval(target_logits)

    # Advance with first token
    first_logprobs = target_logits - mx.logsumexp(target_logits, keepdims=True)
    first_token = sampler(first_logprobs[None, :]).item()
    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")

    first_out = model(mx.array([[first_token]]), cache=cache, return_hidden=True)
    if isinstance(first_out, tuple):
        lm_out, _ = first_out
        target_logits = lm_out.logits if hasattr(lm_out, "logits") else lm_out
    else:
        target_logits = first_out.logits if hasattr(first_out, "logits") else first_out
    target_logits = target_logits.reshape(-1)
    mx.eval(target_logits)

    # === Test 1: KV cache growth ===
    print("\n=== Test 1: EAGLE3 KV Cache Growth ===")
    eagle3_cache = None
    token_id = first_token
    n_drafts = 5
    cache_sizes = []
    draft_tokens = []
    draft_logits_list = []

    for step in range(n_drafts):
        logits, eagle3_cache = model.eagle3_forward(
            mx.array([[token_id]]), eagle3_cache=eagle3_cache
        )
        mx.eval(logits)
        if eagle3_cache is not None:
            if isinstance(eagle3_cache, tuple):
                k_cache = eagle3_cache[0]
                cache_sizes.append(k_cache.shape)
            else:
                cache_sizes.append(str(type(eagle3_cache)))
        else:
            cache_sizes.append(None)

        logits_1d = logits[0, -1, :]
        draft_logits_list.append(logits_1d)
        token_id = mx.argmax(logits_1d).item()
        draft_tokens.append(token_id)
        print(f"  Step {step}: token={token_id} ('{tokenizer.decode([token_id])}'), cache={cache_sizes[-1]}")

    # Check if cache is actually growing
    if all(s is None for s in cache_sizes):
        print("  FAIL: EAGLE3 cache is always None — no KV caching between draft steps")
    elif len(set(str(s) for s in cache_sizes)) == 1:
        print("  FAIL: EAGLE3 cache size not growing — stuck at same size every step")
    else:
        print("  PASS: EAGLE3 cache is growing between steps")

    # === Test 2: Draft token diversity ===
    print("\n=== Test 2: Draft Token Diversity ===")
    unique_tokens = set(draft_tokens)
    print(f"  Draft tokens: {draft_tokens}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in draft_tokens]}")
    print(f"  Unique: {len(unique_tokens)}/{n_drafts}")
    if len(unique_tokens) == 1:
        print("  FAIL: All draft tokens identical — head is not differentiating steps")
    else:
        print("  PASS: Draft tokens vary between steps")

    # === Test 3: Logit diversity ===
    print("\n=== Test 3: Logit Diversity (are outputs numerically identical?) ===")
    for i in range(1, len(draft_logits_list)):
        diff = mx.max(mx.abs(draft_logits_list[i] - draft_logits_list[0])).item()
        top5_i = sorted(mx.argpartition(-draft_logits_list[i], kth=5)[:5].tolist())
        print(f"  Step 0 vs {i}: max_diff={diff:.4e}, top5={top5_i}")
    if all(mx.max(mx.abs(draft_logits_list[i] - draft_logits_list[0])).item() < 1e-6
           for i in range(1, len(draft_logits_list))):
        print("  FAIL: Logits are numerically identical across all steps")
        print("  → EAGLE3 head is producing the exact same output regardless of input/cache")
    else:
        print("  PASS: Logits differ between steps")

    # === Test 4: Hidden state contribution ===
    print("\n=== Test 4: Hidden State vs Token Embedding Contribution ===")
    aux_states = model._eagle3_aux_hidden_states
    head = model.eagle3
    # Get the fc layer output for the hidden states
    concat_hidden = mx.concatenate([h[:, -1:, :] for h in aux_states], axis=-1)
    fused = head.fc(concat_hidden)
    mx.eval(fused)
    # Get token embedding
    emb = head.embed_tokens(mx.array([[first_token]]))
    mx.eval(emb)
    fused_norm = mx.sqrt(mx.sum(fused * fused)).item()
    emb_norm = mx.sqrt(mx.sum(emb * emb)).item()
    print(f"  Fused hidden norm: {fused_norm:.2f}")
    print(f"  Token embedding norm: {emb_norm:.2f}")
    print(f"  Ratio (fused/embed): {fused_norm/emb_norm:.1f}x")
    if fused_norm > 100 * emb_norm:
        print("  WARNING: Hidden states dominate by 100x+ — token embedding is noise")
    elif fused_norm > 10 * emb_norm:
        print("  WARNING: Hidden states dominate by 10x+ — may overwhelm token signal")
    else:
        print("  OK: Reasonable balance between hidden states and embeddings")


if __name__ == "__main__":
    run_diagnostics()
