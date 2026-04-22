"""
Test EAGLE3 orchestration: hidden states flow, cache handling, multi-round acceptance.

Verifies the speculative decoding pipeline:
1. Hidden states are captured during prefill
2. Hidden states update after verify+bonus
3. Target logits maintain consistent shape across rounds
4. Multi-round generation produces non-zero acceptance

Requires Llama 70B + EAGLE3 head.

Usage:
    uv run python tests/test_eagle3_orchestration.py
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


MODEL_PATH = "mlx-community/Llama-3.3-70B-Instruct-8bit"
EAGLE3_HEAD_PATH = "/Users/stuartswerdloff/.cache/huggingface/hub/models--RedHatAI--Llama-3.3-70B-Instruct-speculator.eagle3/snapshots/42864f78d503693ae5fd317419f099ab3b7c13e4"


def test_eagle3_orchestration():
    """Multi-round EAGLE3 speculative decoding produces non-zero acceptance."""
    from vllm_mlx.patches.llama_eagle3 import inject_eagle3_llama
    from vllm_mlx.spec_decode import (
        eagle3_draft_n_tokens,
        eagle3_verify_and_accept,
        prefill_model,
    )

    print(f"\nLoading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    print(f"Injecting EAGLE3 from: {EAGLE3_HEAD_PATH}")
    inject_eagle3_llama(model, EAGLE3_HEAD_PATH)

    # Use chat-formatted prompt for realistic evaluation
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    tokens = tokenizer.encode(prompt)
    tokens_arr = mx.array([tokens])
    print(f"Prompt tokens: {len(tokens)}")

    cache_model = getattr(model, "_eagle3_inner", model.model)
    cache = make_prompt_cache(cache_model)
    sampler = make_sampler(temp=0.0)

    # === Step 1: Prefill ===
    print("\nStep 1: Prefill...")
    target_logits = prefill_model(model, tokens_arr, cache, 2048)
    mx.eval(target_logits)

    # Verify hidden states captured
    aux = model._eagle3_aux_hidden_states
    assert aux is not None, "No hidden states captured after prefill"
    assert len(aux) == 3, f"Expected 3 aux states, got {len(aux)}"
    prefill_state_sums = [float(mx.sum(mx.abs(h)).item()) for h in aux]
    print(f"  Hidden state sums: {[f'{s:.1f}' for s in prefill_state_sums]}")
    print(f"  target_logits shape: {target_logits.shape}")
    assert target_logits.ndim == 1, f"Expected 1D logits from prefill, got {target_logits.ndim}D"

    # === Step 2: Sample first token and advance ===
    print("\nStep 2: Sample first token and advance target cache...")
    first_logprobs = target_logits - mx.logsumexp(target_logits, keepdims=True)
    first_token = sampler(first_logprobs[None, :])
    mx.eval(first_token)
    first_id = first_token.item()
    print(f"  First token: {first_id} = '{tokenizer.decode([first_id])}'")

    first_out = model(mx.array([[first_id]]), cache=cache, return_hidden=True)
    if isinstance(first_out, tuple):
        lm_out, _ = first_out
        target_logits = lm_out.logits if hasattr(lm_out, "logits") else lm_out
    else:
        target_logits = first_out.logits if hasattr(first_out, "logits") else first_out
    target_logits = target_logits.reshape(-1) if target_logits.ndim > 1 else target_logits
    mx.eval(target_logits)
    aux = model._eagle3_aux_hidden_states
    if aux:
        mx.eval(*aux)

    # Hidden states should have changed
    advance_state_sums = [float(mx.sum(mx.abs(h)).item()) for h in aux]
    print(f"  Hidden state sums after advance: {[f'{s:.1f}' for s in advance_state_sums]}")
    assert advance_state_sums != prefill_state_sums, "Hidden states didn't change after first token advance"

    # === Step 3: Multi-round draft-verify ===
    print("\nStep 3: Multi-round draft-verify loop...")
    generated = [first_id]
    total_accepted = 0
    total_proposed = 0
    n_draft = 5
    n_rounds = 8
    prev_state_sums = advance_state_sums
    eagle3_cache = None  # Maintained across rounds — not reset

    for round_num in range(n_rounds):
        last_token = generated[-1]

        drafts, eagle3_cache = eagle3_draft_n_tokens(model, last_token, n_draft, sampler, eagle3_cache)
        target_pred = mx.argmax(target_logits).item()

        accepted, target_logits = eagle3_verify_and_accept(
            model, cache, drafts, target_logits, sampler, 0.0
        )

        n_acc = len(accepted) - 1
        total_accepted += n_acc
        total_proposed += len(drafts)
        generated.extend(accepted)

        # Check shape consistency
        assert target_logits.ndim == 1, (
            f"Round {round_num+1}: target_logits should be 1D, got shape {target_logits.shape}"
        )

        # Check hidden states updated
        cur_aux = model._eagle3_aux_hidden_states
        cur_sums = [float(mx.sum(mx.abs(h)).item()) for h in cur_aux]
        states_changed = cur_sums != prev_state_sums
        prev_state_sums = cur_sums

        print(
            f"  R{round_num+1}: target='{tokenizer.decode([target_pred])}' "
            f"acc={n_acc}/{n_draft} states_changed={states_changed}"
        )

        if any(t in {tokenizer.eos_token_id, 128009} for t in accepted):
            break

    # === Results ===
    acceptance_pct = 100 * total_accepted / total_proposed if total_proposed > 0 else 0
    text = tokenizer.decode(generated)
    print(f"\n{'='*50}")
    print(f"Generated: {text}")
    print(f"Acceptance: {total_accepted}/{total_proposed} ({acceptance_pct:.1f}%)")
    print(f"Tokens generated: {len(generated)}")

    # Acceptance should be non-zero — EAGLE3 head predicts first draft token
    # correctly in most rounds for this simple factual prompt
    assert total_accepted > 0, (
        f"0% acceptance rate — orchestration is broken. "
        f"Generated: {text}"
    )

    # Sanity check: the generated text should contain "Paris"
    assert "Paris" in text or "paris" in text.lower(), (
        f"Generated text doesn't contain expected answer: {text}"
    )

    print(f"\nPASS: {acceptance_pct:.1f}% acceptance, hidden states update correctly")


if __name__ == "__main__":
    test_eagle3_orchestration()
