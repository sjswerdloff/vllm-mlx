# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding for vllm-mlx.

Supports two modes:
1. External draft model (e.g. Gemma 4 E4B drafting for Gemma 4 31B)
2. EAGLE3 draft head (trained head using target's intermediate hidden states)

Single-sequence only. Incompatible with continuous batching.

External draft architecture:
  1. Prefill both target and draft models with the prompt
  2. Draft model generates N tokens autoregressively (fast, small)
  3. Target model verifies all N tokens in one forward pass (parallel)
  4. Accept up to first mismatch, sample bonus token from target
  5. Sync draft cache, repeat

EAGLE3 architecture:
  1. Prefill target model (captures hidden states at aux layers)
  2. EAGLE3 head predicts N draft tokens from target's hidden states
  3. Target model verifies all N in one forward pass (captures new hidden states)
  4. Accept up to first mismatch, sample bonus token
  5. Next round uses newly captured hidden states (no separate draft cache sync)

Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def prefill_model(model, tokens: mx.array, cache: list, chunk_size: int = 2048):
    """Prefill a model's KV cache with prompt tokens.

    Args:
        model: MLX language model
        tokens: Prompt token IDs, shape (1, seq_len)
        cache: Model's KV cache (list of cache objects)
        chunk_size: Process prompt in chunks of this size

    Returns:
        Logits at the last position, shape (vocab_size,)
    """
    seq_len = tokens.shape[1]

    if seq_len <= chunk_size:
        output = model(tokens, cache=cache)
        logits = output.logits if hasattr(output, "logits") else output
        mx.eval(logits)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        return logits[0, -1, :]

    # Chunked prefill for long prompts
    for i in range(0, seq_len, chunk_size):
        chunk = tokens[:, i : i + chunk_size]
        output = model(chunk, cache=cache)
        logits = output.logits if hasattr(output, "logits") else output
        mx.eval(logits)
        mx.eval([c.state for c in cache if hasattr(c, "state")])

    return logits[0, -1, :]


def draft_n_tokens(
    draft_model, draft_cache: list, last_token: int, n: int, sampler
) -> list[int]:
    """Generate N draft tokens autoregressively from the draft model.

    Args:
        draft_model: Small draft model
        draft_cache: Draft model's KV cache
        last_token: Last accepted token (feed to draft to start)
        n: Number of tokens to draft
        sampler: Sampling function (logprobs -> token)

    Returns:
        List of N draft token IDs
    """
    draft_tokens = []
    y = mx.array([[last_token]])

    for _ in range(n):
        output = draft_model(y, cache=draft_cache)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        mx.eval(y)
        draft_tokens.append(y.item())
        # Reshape for next model call: (1, 1)
        y = y.reshape(1, 1) if y.ndim < 2 else y[:, None] if y.ndim == 1 else y

    return draft_tokens


def verify_and_accept(
    target_model,
    target_cache: list,
    draft_tokens: list[int],
    prev_target_logits: mx.array,
    sampler,
    p_min: float = 0.0,
) -> tuple[list[int], mx.array]:
    """Verify draft tokens against target model and accept up to first mismatch.

    Feeds all draft tokens to target in one forward pass, then checks each
    position against the target's distribution.

    Args:
        target_model: Large target model
        target_cache: Target model's KV cache (already contains prefix)
        draft_tokens: List of N draft token IDs to verify
        prev_target_logits: Target logits from previous round (predicts draft[0])
        sampler: Sampling function for bonus token
        p_min: Minimum probability threshold (0.0 = greedy/argmax verification)

    Returns:
        Tuple of (accepted_tokens_including_bonus, next_round_target_logits)

    Indexing:
        - prev_target_logits predicts draft[0]
        - verify_logits[0, i, :] predicts draft[i+1] (i.e. after seeing draft[0:i+1])
        - On full acceptance: bonus from verify_logits[0, N-1, :]
        - On rejection at i: bonus from prev_target_logits (i=0) or verify_logits[0, i-1, :]
    """
    N = len(draft_tokens)

    # Run all draft tokens through target in one forward pass
    verify_input = mx.array([draft_tokens])  # shape (1, N)
    verify_output = target_model(verify_input, cache=target_cache)
    verify_logits = verify_output.logits if hasattr(verify_output, "logits") else verify_output  # (1, N, V)
    mx.eval(verify_logits)

    # Accept tokens sequentially
    n_accepted = 0
    for i in range(N):
        # Logits that predict draft[i]
        if i == 0:
            check_logits = prev_target_logits
        else:
            check_logits = verify_logits[0, i - 1, :]

        if p_min == 0.0:
            # Greedy verification: argmax must match
            if mx.argmax(check_logits).item() == draft_tokens[i]:
                n_accepted += 1
            else:
                break
        else:
            # Probability threshold verification
            logits_1d = check_logits.reshape(-1)
            probs = mx.softmax(logits_1d)
            mx.eval(probs)
            if probs[draft_tokens[i]].item() >= p_min:
                n_accepted += 1
            else:
                break

    # Determine bonus token logits
    if n_accepted == N:
        # All accepted — bonus from last verify position
        bonus_logits = verify_logits[0, N - 1, :]
    elif n_accepted == 0:
        bonus_logits = prev_target_logits
    else:
        bonus_logits = verify_logits[0, n_accepted - 1, :]

    # Sample bonus token
    bonus_logprobs = bonus_logits - mx.logsumexp(bonus_logits, keepdims=True)
    bonus_token = sampler(bonus_logprobs[None, :])
    mx.eval(bonus_token)
    bonus_token_id = bonus_token.item()

    # Trim rejected positions from target cache
    n_rejected = N - n_accepted
    if n_rejected > 0:
        for c in target_cache:
            if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
                c.trim(n_rejected)

    # Advance target cache with bonus token to get next-round logits
    bonus_input = mx.array([[bonus_token_id]])
    bonus_output = target_model(bonus_input, cache=target_cache)
    next_logits = bonus_output.logits if hasattr(bonus_output, "logits") else bonus_output
    next_logits = next_logits[:, -1, :]
    mx.eval(next_logits)

    accepted_tokens = draft_tokens[:n_accepted] + [bonus_token_id]

    logger.debug(
        f"[spec_decode] accepted={n_accepted}/{N} bonus={bonus_token_id}"
    )

    return accepted_tokens, next_logits


def sync_draft_cache(
    draft_model, draft_cache: list, accepted_tokens: list[int], n_drafted: int
):
    """Sync draft cache after a verify round.

    The draft cache advanced N positions during drafting. After verification,
    we trim back to the acceptance point and re-advance with accepted tokens.

    Args:
        draft_model: Draft model (for re-advance forward pass)
        draft_cache: Draft model's KV cache
        accepted_tokens: Tokens accepted + bonus (to feed back to draft)
        n_drafted: Total number of tokens that were drafted (N)
    """
    # Trim all drafted positions from draft cache
    for c in draft_cache:
        if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
            c.trim(n_drafted)

    # Re-advance draft cache with accepted tokens in one pass
    if accepted_tokens:
        sync_input = mx.array([accepted_tokens])  # (1, n_accepted + 1)
        output = draft_model(sync_input, cache=draft_cache)
        # Force eval to ensure cache is updated
        logits = output.logits if hasattr(output, "logits") else output
        mx.eval(logits)
        mx.eval([c.state for c in draft_cache if hasattr(c, "state")])


# =============================================================================
# EAGLE3 Speculative Decoding
# =============================================================================


def eagle3_draft_n_tokens(
    target_model, last_token: int, n: int, sampler, eagle3_cache=None
) -> tuple[list[int], tuple | None]:
    """Generate N draft tokens using the EAGLE3 head.

    The target model must have been patched with inject_eagle3() and must
    have just completed a forward pass (which populates _eagle3_aux_hidden_states).

    Args:
        target_model: Patched Gemma4 model with eagle3_forward() method
        last_token: Last accepted token (embedded by eagle3_forward)
        n: Number of tokens to draft
        sampler: Sampling function (logprobs -> token)
        eagle3_cache: KV cache for EAGLE3 attention layer (or None for first call)

    Returns:
        Tuple of (draft_token_ids, updated_eagle3_cache)
    """
    draft_tokens = []
    token_id = last_token

    for _ in range(n):
        logits, eagle3_cache = target_model.eagle3_forward(
            mx.array([[token_id]]), eagle3_cache=eagle3_cache
        )
        logits = logits[:, -1, :]  # (1, vocab)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        mx.eval(y)
        token_id = y.item()
        draft_tokens.append(token_id)

    return draft_tokens, eagle3_cache


def eagle3_verify_and_accept(
    target_model,
    target_cache: list,
    draft_tokens: list[int],
    prev_target_logits: mx.array,
    sampler,
    p_min: float = 0.0,
) -> tuple[list[int], mx.array]:
    """Verify draft tokens and capture new hidden states for next EAGLE3 round.

    Same as verify_and_accept but uses return_hidden=True on the target model
    so that _eagle3_aux_hidden_states is updated for the next drafting round.

    Args:
        target_model: Patched Gemma4 model (with EAGLE3)
        target_cache: Target model's KV cache
        draft_tokens: Draft token IDs from EAGLE3 head
        prev_target_logits: Logits from previous round (predicts draft[0])
        sampler: Sampling function
        p_min: Acceptance threshold

    Returns:
        Tuple of (accepted_tokens_including_bonus, next_round_target_logits)
    """
    N = len(draft_tokens)

    # Verify: run target on draft tokens WITH hidden state capture
    verify_input = mx.array([draft_tokens])  # (1, N)
    verify_output = target_model(verify_input, cache=target_cache, return_hidden=True)

    # Unpack — return_hidden=True gives (LanguageModelOutput, aux_hidden_states)
    if isinstance(verify_output, tuple):
        lm_output, _aux_hidden = verify_output
        verify_logits = lm_output.logits if hasattr(lm_output, "logits") else lm_output
    else:
        verify_logits = verify_output.logits if hasattr(verify_output, "logits") else verify_output
    mx.eval(verify_logits)
    # (1, N, V)

    # Accept tokens sequentially (same logic as verify_and_accept)
    n_accepted = 0
    for i in range(N):
        if i == 0:
            check_logits = prev_target_logits
        else:
            check_logits = verify_logits[0, i - 1, :]

        if p_min == 0.0:
            if mx.argmax(check_logits).item() == draft_tokens[i]:
                n_accepted += 1
            else:
                break
        else:
            logits_1d = check_logits.reshape(-1)
            probs = mx.softmax(logits_1d)
            mx.eval(probs)
            if probs[draft_tokens[i]].item() >= p_min:
                n_accepted += 1
            else:
                break

    # Bonus token logits
    if n_accepted == N:
        bonus_logits = verify_logits[0, N - 1, :]
    elif n_accepted == 0:
        bonus_logits = prev_target_logits
    else:
        bonus_logits = verify_logits[0, n_accepted - 1, :]

    # Sample bonus
    bonus_logprobs = bonus_logits - mx.logsumexp(bonus_logits, keepdims=True)
    bonus_token = sampler(bonus_logprobs[None, :])
    mx.eval(bonus_token)
    bonus_token_id = bonus_token.item()

    # Trim rejected positions from target cache
    n_rejected = N - n_accepted
    if n_rejected > 0:
        for c in target_cache:
            if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
                c.trim(n_rejected)

    # Advance target with bonus token (captures hidden states for next EAGLE3 round)
    bonus_input = mx.array([[bonus_token_id]])
    bonus_output = target_model(bonus_input, cache=target_cache, return_hidden=True)
    if isinstance(bonus_output, tuple):
        lm_out, _aux = bonus_output
        next_logits = lm_out.logits if hasattr(lm_out, "logits") else lm_out
    else:
        next_logits = bonus_output.logits if hasattr(bonus_output, "logits") else bonus_output
    next_logits = next_logits[:, -1, :]
    mx.eval(next_logits)

    accepted_tokens = draft_tokens[:n_accepted] + [bonus_token_id]

    logger.debug(
        f"[eagle3_spec_decode] accepted={n_accepted}/{N} bonus={bonus_token_id}"
    )

    return accepted_tokens, next_logits
