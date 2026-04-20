# SPDX-License-Identifier: Apache-2.0
"""
External draft model speculative decoding for vllm-mlx.

Implements the draft-verify-accept loop using a separate small model
as the draft (e.g. Gemma 4 E4B drafting for Gemma 4 31B).

Single-sequence only. Incompatible with continuous batching.

Architecture:
  1. Prefill both target and draft models with the prompt
  2. Draft model generates N tokens autoregressively (fast, small)
  3. Target model verifies all N tokens in one forward pass (parallel)
  4. Accept up to first mismatch, sample bonus token from target
  5. Sync draft cache, repeat

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
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        return logits[0, -1, :]

    # Chunked prefill for long prompts
    for i in range(0, seq_len, chunk_size):
        chunk = tokens[:, i : i + chunk_size]
        logits = model(chunk, cache=cache)
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
        logits = draft_model(y, cache=draft_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        mx.eval(y)
        draft_tokens.append(y.item())

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
    verify_logits = target_model(verify_input, cache=target_cache)  # (1, N, V)
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
            probs = mx.softmax(check_logits)
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
    next_logits = target_model(bonus_input, cache=target_cache)
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
        draft_model(sync_input, cache=draft_cache)
        mx.eval([c.state for c in draft_cache if hasattr(c, "state")])
