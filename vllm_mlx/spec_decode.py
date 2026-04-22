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

        target_greedy = mx.argmax(check_logits).item()
        draft_token = draft_tokens[i]
        target_prob_of_draft = mx.softmax(check_logits.reshape(-1))[draft_token].item()

        logger.info(
            "[eagle3_verify] pos %d: draft=%d target_greedy=%d match=%s p(draft)=%.4f",
            i, draft_token, target_greedy, draft_token == target_greedy, target_prob_of_draft,
        )

        if p_min == 0.0:
            # Greedy verification: argmax must match
            if target_greedy == draft_token:
                n_accepted += 1
            else:
                break
        else:
            # Probability threshold verification
            if target_prob_of_draft >= p_min:
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


def eagle3_prefill_cache(
    target_model, prompt_token_ids: list[int]
) -> tuple[tuple | None, mx.array | None]:
    """Populate the EAGLE3 head's KV cache using full-sequence hidden states.

    After the target model prefills the prompt (which captures full-sequence
    auxiliary hidden states at the configured layers), this function runs those
    hidden states through the EAGLE3 head in one forward pass. The resulting
    KV cache gives the draft head full context for informed predictions,
    instead of starting from an almost-empty cache.

    The token_ids passed to the head are the prompt tokens shifted by one
    position: the head predicts next-token, so input is [token_0, ..., token_{n-1}]
    to predict [token_1, ..., token_n].

    Must be called AFTER prefill_model() but BEFORE the first-token advance
    (which overwrites _eagle3_aux_hidden_states with length-1 states).

    Note: With chunked prefill (long prompts), the captured aux hidden states
    only cover the last chunk. The prefill will still populate the EAGLE3
    cache with that partial context, which is better than an empty cache.

    Args:
        target_model: Patched model with eagle3_prefill() method
        prompt_token_ids: The original prompt token IDs (before any generation).
            These will be shifted to create the input for the EAGLE3 head.

    Returns:
        Tuple of (populated_eagle3_cache, last_pre_norm_hidden).
        Both can be passed directly to eagle3_draft_n_tokens.
    """
    if not hasattr(target_model, "eagle3_prefill"):
        logger.warning(
            "eagle3_prefill_cache: model has no eagle3_prefill method, "
            "falling back to empty cache"
        )
        return None, None

    aux_states = getattr(target_model, "_eagle3_aux_hidden_states", None)
    if aux_states is None:
        logger.warning(
            "eagle3_prefill_cache: no aux hidden states captured, "
            "falling back to empty cache"
        )
        return None, None

    # Check sequence length of aux hidden states — prefill captures full
    # sequence, but the first-token advance overwrites with length-1 states.
    # We need the full-sequence states (captured during prefill).
    seq_len = aux_states[0].shape[1]
    if seq_len <= 1:
        logger.debug(
            "eagle3_prefill_cache: aux hidden states have seq_len=%d, "
            "skipping prefill (states were already overwritten by advance)",
            seq_len,
        )
        return None, None

    # Build shifted token_ids: [token_0, token_1, ..., token_{n-2}, token_{n-1}]
    # The EAGLE3 head at position i predicts token_{i+1}, so we feed
    # the prompt tokens up to the second-to-last as input. But we need
    # the input length to match the aux hidden states length.
    # The aux states have length = prompt_len (from prefill).
    # We use prompt_token_ids[0:seq_len] as input (the head's fc layer
    # processes the aux hidden states, and the token embeddings provide
    # the positional grounding).
    if len(prompt_token_ids) < seq_len:
        # Shouldn't happen in normal flow, but guard against it
        token_input = prompt_token_ids
    else:
        token_input = prompt_token_ids[:seq_len]

    token_arr = mx.array([token_input])  # (1, seq_len)

    eagle3_cache, last_hidden = target_model.eagle3_prefill(token_arr)
    mx.eval(eagle3_cache[0], eagle3_cache[1], last_hidden)

    logger.info(
        "eagle3_prefill_cache: populated EAGLE3 KV cache with %d positions",
        seq_len,
    )

    return eagle3_cache, last_hidden


def eagle3_draft_n_tokens(
    target_model,
    last_token: int,
    n: int,
    sampler,
    eagle3_cache=None,
    prev_hidden: mx.array | None = None,
) -> tuple[list[int], tuple | None]:
    """Generate N draft tokens using the EAGLE3 head.

    The target model must have been patched with inject_eagle3() and must
    have just completed a forward pass (which populates _eagle3_aux_hidden_states).

    The EAGLE3 head is recurrent: on the first draft step, it fuses the
    target model's auxiliary hidden states via fc(). On subsequent steps,
    it feeds back its own pre-norm hidden state, providing the evolving
    context that differentiates successive predictions (matching vLLM's
    Eagle3LlamaForCausalLM behavior).

    Args:
        target_model: Patched Gemma4 model with eagle3_forward() method
        last_token: Last accepted token (embedded by eagle3_forward)
        n: Number of tokens to draft
        sampler: Sampling function (logprobs -> token)
        eagle3_cache: KV cache for EAGLE3 attention layer (or None for first call)
        prev_hidden: Pre-norm hidden state from EAGLE3 prefill or previous round.
            When provided (e.g. from eagle3_prefill_cache), the first draft step
            uses this instead of computing from target aux hidden states via fc().

    Returns:
        Tuple of (draft_token_ids, updated_eagle3_cache)
    """
    draft_tokens = []
    token_id = last_token
    # prev_hidden: if provided from prefill, first step uses it directly;
    # otherwise None means first step uses target aux hidden states via fc()

    for step_i in range(n):
        logits, eagle3_cache, prev_hidden = target_model.eagle3_forward(
            mx.array([[token_id]]),
            eagle3_cache=eagle3_cache,
            prev_hidden=prev_hidden,
        )
        logits = logits[:, -1, :]  # (1, vocab)

        # Diagnostic: top-5 predictions from the head
        top5_idx = mx.argpartition(logits[0], kth=-5)[-5:]
        top5_vals = logits[0][top5_idx]
        mx.eval(top5_idx, top5_vals)
        sorted_order = mx.argsort(-top5_vals)
        top5_idx = top5_idx[sorted_order]
        top5_probs = mx.softmax(logits[0])[top5_idx]
        mx.eval(top5_probs)
        logger.info(
            "[eagle3_draft] step %d: input_token=%d, top5=%s probs=%s",
            step_i, token_id,
            [int(x) for x in top5_idx.tolist()],
            [f"{p:.3f}" for p in top5_probs.tolist()],
        )

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

        target_greedy = mx.argmax(check_logits).item()
        draft_token = draft_tokens[i]
        target_prob = mx.softmax(check_logits.reshape(-1))[draft_token].item()

        logger.info(
            "[eagle3_verify] pos %d: draft=%d target_greedy=%d match=%s p(draft)=%.4f",
            i, draft_token, target_greedy, draft_token == target_greedy, target_prob,
        )

        if p_min == 0.0:
            if target_greedy == draft_token:
                n_accepted += 1
            else:
                break
        else:
            if target_prob >= p_min:
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
    # Flatten to (V,) to match prefill_model convention — the bonus is always
    # a single token so logits shape is (1, 1, V) or (1, V).
    next_logits = next_logits.reshape(-1) if next_logits.ndim > 1 else next_logits
    mx.eval(next_logits)

    # Force-evaluate the captured hidden states so the next eagle3_draft_n_tokens
    # round reads materialized arrays rather than growing the lazy computation graph.
    aux_states = getattr(target_model, "_eagle3_aux_hidden_states", None)
    if aux_states is not None:
        mx.eval(*aux_states)

    accepted_tokens = draft_tokens[:n_accepted] + [bonus_token_id]

    logger.debug(
        f"[eagle3_spec_decode] accepted={n_accepted}/{N} bonus={bonus_token_id}"
    )

    return accepted_tokens, next_logits
