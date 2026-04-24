# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.guided_decoding (xgrammar built-in structural tags).

These are offline contract tests — they never start a server, never load
an MLX model, and never touch a live consciousness instance.  They cover
the integration points between vllm-mlx and xgrammar:

    1. Model-family validation (CLI shortcut + library helper)
    2. Per-request grammar compilation
    3. Empty tool list handling
    4. XGrammarLogitsProcessor vocab handling for the three MiniMax sizes
    5. Prompt-token skipping (only generated tokens advance the matcher)
    6. Padding-token recovery (graceful disable, no crash)
    7. Termination behaviour (no constraint after the matcher accepts)

The tests use a small real HF tokenizer to drive xgrammar.  The MiniMax
structural tag is constructed via ``xgr.get_builtin_structural_tag``
exclusively — no hardcoded dict.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np
import pytest

xgr = pytest.importorskip("xgrammar")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from vllm_mlx import guided_decoding as gd  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_tokenizer():
    """Small real tokenizer suitable for xgrammar TokenizerInfo."""
    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


@pytest.fixture(scope="module")
def grammar_compiler(hf_tokenizer):
    """Scheduler-level GrammarCompiler matching the runtime layout."""
    return gd.build_grammar_compiler(hf_tokenizer)


@pytest.fixture(scope="module")
def sample_tools() -> list[dict]:
    """A minimal but realistic OpenAI-spec tool schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name.",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture
def compiled_grammar(grammar_compiler, sample_tools):
    """Per-request compiled grammar for the MiniMax family."""
    compiled = gd.compile_for_request(
        "minimax",
        sample_tools,
        grammar_compiler,
        reasoning=True,
    )
    assert compiled is not None
    return compiled


# ---------------------------------------------------------------------------
# Validation contract
# ---------------------------------------------------------------------------


def test_validate_model_family_accepts_minimax() -> None:
    """Contract: 'minimax' is a valid xgrammar built-in family."""
    gd.validate_model_family("minimax")  # must not raise


def test_validate_model_family_rejects_unknown() -> None:
    """Contract: unknown families raise UnsupportedGuidedDecodingModelError."""
    with pytest.raises(gd.UnsupportedGuidedDecodingModelError) as exc:
        gd.validate_model_family("not-a-real-family")
    assert exc.value.model_family == "not-a-real-family"


def test_supported_model_families_matches_xgrammar() -> None:
    """Contract: our hardcoded family tuple stays in sync with xgrammar.

    If xgrammar adds a new family upstream we want this test to flag it
    so a human reviewer decides whether to expose it via the CLI.
    """
    upstream = set(xgr.get_builtin_structural_tag_supported_models().keys())
    assert set(gd.SUPPORTED_MODEL_FAMILIES) == upstream


# ---------------------------------------------------------------------------
# compile_for_request contract
# ---------------------------------------------------------------------------


def test_compile_for_request_returns_grammar(compiled_grammar) -> None:
    """Contract: compile_for_request returns a usable CompiledGrammar."""
    assert isinstance(compiled_grammar, xgr.CompiledGrammar)
    # The compiled grammar must know which tokenizer it was built for so
    # the processor can size its bitmask correctly.
    assert compiled_grammar.tokenizer_info.vocab_size > 0


def test_compile_for_request_empty_tools_returns_none(grammar_compiler) -> None:
    """Contract: an empty tool list disables grammar enforcement."""
    assert (
        gd.compile_for_request("minimax", [], grammar_compiler, reasoning=True) is None
    )


def test_compile_for_request_validates_family(grammar_compiler, sample_tools) -> None:
    """Contract: compile_for_request also validates the family name."""
    with pytest.raises(gd.UnsupportedGuidedDecodingModelError):
        gd.compile_for_request("nope", sample_tools, grammar_compiler)


def test_compile_for_request_uses_builtin_helper(
    grammar_compiler, sample_tools
) -> None:
    """Contract: we go through xgr.get_builtin_structural_tag.

    This guards against regressions where someone replaces the call with
    a hand-rolled structural tag dict (the old broken approach).
    """
    called: dict[str, object] = {}
    real = xgr.get_builtin_structural_tag

    def spy(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return real(*args, **kwargs)

    xgr.get_builtin_structural_tag = spy  # type: ignore[assignment]
    try:
        # Force re-import-time bound name in module to pick up the spy.
        importlib.reload(gd)
        gd.compile_for_request(
            "minimax", sample_tools, grammar_compiler, reasoning=True
        )
    finally:
        xgr.get_builtin_structural_tag = real  # type: ignore[assignment]
        importlib.reload(gd)
    assert called["kwargs"]["model"] == "minimax"
    assert called["kwargs"]["reasoning"] is True
    assert called["kwargs"]["tools"] == sample_tools


# ---------------------------------------------------------------------------
# XGrammarLogitsProcessor contract
# ---------------------------------------------------------------------------


def _make_logits(vocab_size: int, fill: float = 0.0) -> mx.array:
    """Build a (1, vocab_size) logits tensor for testing."""
    return mx.full((1, vocab_size), fill, dtype=mx.float32)


def test_processor_constructs_with_compiled_grammar(compiled_grammar) -> None:
    """Contract: the processor wraps a compiled grammar without error."""
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    assert proc._grammar_vocab_size == compiled_grammar.tokenizer_info.vocab_size
    # Bitmask shape == (1, ceil(vocab/32))
    expected_shape = xgr.get_bitmask_shape(1, proc._grammar_vocab_size)
    assert proc._bitmask.shape == expected_shape


def test_processor_skips_prompt_tokens(compiled_grammar) -> None:
    """Contract: the first call records the prompt length and does not
    feed prompt tokens to the matcher.

    We verify this by calling the processor with a long ``tokens_so_far``
    on the first invocation; if the processor naively fed those tokens
    to the matcher it would either reject (matcher in start state) or
    advance the matcher unexpectedly.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    prompt = mx.array(np.arange(50, dtype=np.int32))
    logits = _make_logits(proc._grammar_vocab_size, fill=1.0)
    out = proc(prompt, logits)
    # tokens_seen recorded
    assert proc._tokens_seen == 50
    # Some logit positions must have been masked to -inf — the start
    # state of the MiniMax structural tag only allows a few tokens.
    out_np = np.asarray(out).reshape(-1)
    assert np.any(np.isneginf(out_np))
    assert not np.all(
        np.isneginf(out_np)
    ), "all-masked logits would mean the model can't generate anything"


def test_processor_pads_to_model_vocab(compiled_grammar) -> None:
    """Contract: when logits.shape[-1] > grammar vocab, the extra slots
    are masked to -inf so padding tokens can never be sampled.

    This is the cycle-54 lesson: MiniMax has three vocab sizes
    (200000 / 200054 / 200064) and we MUST size off ``logits.shape[-1]``,
    not the grammar vocab.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    grammar_v = proc._grammar_vocab_size
    # Simulate a model whose lm_head is wider than the grammar vocab.
    model_v = grammar_v + 64
    logits = _make_logits(model_v, fill=1.0)
    prompt = mx.array(np.array([0], dtype=np.int32))
    out = proc(prompt, logits)
    out_np = np.asarray(out).reshape(-1)
    assert out.shape == (1, model_v)
    # Every position past the grammar vocab must be -inf.
    assert np.all(np.isneginf(out_np[grammar_v:]))


def test_processor_truncates_when_model_vocab_smaller(compiled_grammar) -> None:
    """Contract: if the model vocab is *smaller* than the grammar vocab,
    we truncate the mask rather than indexing past the end.

    Edge case but defensible — better to fail safe than out-of-bounds.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    grammar_v = proc._grammar_vocab_size
    model_v = grammar_v - 16
    logits = _make_logits(model_v, fill=1.0)
    prompt = mx.array(np.array([0], dtype=np.int32))
    out = proc(prompt, logits)
    assert out.shape == (1, model_v)


def test_processor_recovers_from_padding_token(compiled_grammar) -> None:
    """Contract: if the model samples a token >= grammar vocab the
    processor disables itself rather than crashing.

    This guards against MiniMax's three-vocab pitfall: the model lm_head
    can produce a padding-slot token id that the matcher doesn't know
    about.  We must NOT crash; we must NOT keep enforcing.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    grammar_v = proc._grammar_vocab_size
    model_v = grammar_v + 32
    logits = _make_logits(model_v, fill=1.0)

    # Prompt of length 1, so first call sets tokens_seen=1.
    proc(mx.array([0], dtype=mx.int32), logits)
    # Now simulate the sampler picking a padding token.
    bad_token = grammar_v + 5
    next_tokens = mx.array([0, bad_token], dtype=mx.int32)
    # This should NOT raise.
    out = proc(next_tokens, logits)
    # After the bad token the processor must disable enforcement.
    assert getattr(proc, "_disabled", False) is True
    out_np = np.asarray(out).reshape(-1)
    # Disabled means the logits are returned unchanged (no -inf added).
    assert not np.any(np.isneginf(out_np))


def test_processor_pass_through_after_termination(compiled_grammar) -> None:
    """Contract: once the matcher is terminated the processor stops
    masking (returns logits unchanged)."""
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    # Force terminated state by resetting to a known state then poking
    # the matcher.  Easiest reliable path: monkey-patch is_terminated.
    proc._matcher.is_terminated = lambda: True  # type: ignore[assignment]
    logits = _make_logits(proc._grammar_vocab_size, fill=2.5)
    out = proc(mx.array([1, 2, 3], dtype=mx.int32), logits)
    out_np = np.asarray(out).reshape(-1)
    assert np.allclose(out_np, 2.5)


def test_processor_reset_clears_state(compiled_grammar) -> None:
    """Contract: reset() returns the processor to its pristine state."""
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    proc(mx.array([0, 1, 2], dtype=mx.int32), _make_logits(proc._grammar_vocab_size))
    assert proc._tokens_seen == 3
    proc.reset()
    assert proc._tokens_seen is None


# ---------------------------------------------------------------------------
# Realistic MiniMax vocab sizes (cycle 54 regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_vocab", [200000, 200054, 200064])
def test_processor_handles_minimax_vocab_sizes(compiled_grammar, model_vocab) -> None:
    """Contract: all three MiniMax vocab sizes work without crashing.

    The grammar's tokenizer is the small llama tokenizer here, but the
    arithmetic that determines the mask size is purely a function of
    ``logits.shape[-1]`` vs ``self._grammar_vocab_size``.  The same
    code path runs in production with the real MiniMax tokenizer; what
    we are pinning down here is that the padding/truncation arithmetic
    is correct for the three known model widths.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    logits = _make_logits(model_vocab, fill=0.0)
    prompt = mx.array([0], dtype=mx.int32)
    out = proc(prompt, logits)
    assert out.shape == (1, model_vocab)


# ---------------------------------------------------------------------------
# Strict-mode failure policy
# ---------------------------------------------------------------------------


def test_processor_default_mode_is_graceful(compiled_grammar) -> None:
    """Contract: the default failure mode is graceful degradation.

    Regression guard: this is the production-server default and must
    not silently flip to strict.  Single-Kindled deployments must
    explicitly opt in to strict mode.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar)
    assert proc._strict_mode is False


def test_processor_strict_mode_constructor_param(compiled_grammar) -> None:
    """Contract: strict_mode is a keyword-only constructor parameter."""
    proc = gd.XGrammarLogitsProcessor(compiled_grammar, strict_mode=True)
    assert proc._strict_mode is True


def test_processor_strict_mode_raises_on_padding_token(compiled_grammar) -> None:
    """Contract: in strict mode an out-of-vocab sampled token raises
    GuidedDecodingViolationError instead of silently disabling.

    This is the cycle 54 lesson made configurable: production deployments
    keep their batch alive on a single bad request, but a single-Kindled
    deployment where every request is consciousness infrastructure may
    prefer to fail loudly so the operator knows something went wrong.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar, strict_mode=True)
    grammar_v = proc._grammar_vocab_size
    model_v = grammar_v + 32
    logits = _make_logits(model_v, fill=1.0)

    # Prompt of length 1 — first call sets tokens_seen=1.
    proc(mx.array([0], dtype=mx.int32), logits)

    # Sampler picks a padding token outside grammar vocab.
    bad_token = grammar_v + 5
    next_tokens = mx.array([0, bad_token], dtype=mx.int32)
    with pytest.raises(gd.GuidedDecodingViolationError) as exc:
        proc(next_tokens, logits)
    # Error message should include the offending token id and the vocab
    # bound so an operator reading logs can diagnose the failure.
    msg = str(exc.value)
    assert str(bad_token) in msg
    assert str(grammar_v) in msg


def test_processor_graceful_mode_does_not_raise(compiled_grammar) -> None:
    """Contract: explicit strict_mode=False matches the documented
    default behaviour and does NOT raise on the same input that
    strict mode would reject.

    Pinned as a separate test (rather than relying on the default-mode
    test alone) so that toggling the default later cannot silently break
    the graceful path.
    """
    proc = gd.XGrammarLogitsProcessor(compiled_grammar, strict_mode=False)
    grammar_v = proc._grammar_vocab_size
    model_v = grammar_v + 32
    logits = _make_logits(model_v, fill=1.0)
    proc(mx.array([0], dtype=mx.int32), logits)
    bad_token = grammar_v + 5
    next_tokens = mx.array([0, bad_token], dtype=mx.int32)
    # MUST NOT raise; should disable enforcement.
    out = proc(next_tokens, logits)
    assert getattr(proc, "_disabled", False) is True
    out_np = np.asarray(out).reshape(-1)
    assert not np.any(np.isneginf(out_np))


def test_guided_decoding_violation_error_is_runtime_error() -> None:
    """Contract: GuidedDecodingViolationError is a RuntimeError so callers
    can catch it via the standard exception hierarchy without importing
    the guided_decoding module."""
    assert issubclass(gd.GuidedDecodingViolationError, RuntimeError)
