# SPDX-License-Identifier: Apache-2.0
"""Grammar-constrained decoding for vllm-mlx using xgrammar built-in tags.

This module wires xgrammar's built-in structural tags (per-model tool-call
grammars for 9 model families) into the mlx-lm BatchGenerator logits
processor interface.

Architecture:
    - One ``GrammarCompiler`` per scheduler (built from the server tokenizer)
    - Per-request ``CompiledGrammar`` via
      :func:`compile_for_request`, which calls
      ``xgr.get_builtin_structural_tag(model, tools, reasoning)``. xgrammar's
      internal LRU cache makes repeat tool sets effectively free.
    - One ``XGrammarLogitsProcessor`` per request, stepping its own
      ``GrammarMatcher`` through the sampled tokens.

Only ``minimax`` is currently exposed via the CLI shortcut.  Adding another
family (e.g. ``llama``, ``qwen``, ``kimi``, …) is a one-line addition in
:data:`SUPPORTED_MODEL_FAMILIES` — the rest of the pipeline is generic.

Critical vocab-size note (cycle 54 lesson):
    MiniMax ships three different vocab sizes depending on where you look —
    200000 (base), 200054 (tokenizer), 200064 (model lm_head).  The
    ``GrammarMatcher``'s bitmask is sized to xgrammar's view of the
    tokenizer, which may be smaller than ``logits.shape[-1]``.  We must key
    off ``logits.shape[-1]`` (the *model*'s vocab) when constructing the
    mask we apply, padding extra slots with ``-inf`` so padding/reserved
    tokens can never be sampled.  Likewise, if the model ever samples a
    token_id >= grammar vocab_size (a padding token), we treat that as a
    graceful reset rather than crashing the matcher.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Model families supported by xgr.get_builtin_structural_tag.  Mirrors
# xgr.get_builtin_structural_tag_supported_models() at the time of writing.
# We re-declare the tuple here so validation works even if xgrammar is
# unavailable at import time (e.g. in environments where the dependency
# was not installed).
SUPPORTED_MODEL_FAMILIES: tuple[str, ...] = (
    "llama",
    "qwen",
    "qwen_coder",
    "kimi",
    "deepseek_r1",
    "harmony",
    "deepseek_v3_2",
    "minimax",
    "glm47",
)


class GuidedDecodingUnavailableError(RuntimeError):
    """Raised when xgrammar is not importable but guided decoding was requested."""


class UnsupportedGuidedDecodingModelError(ValueError):
    """Raised when the CLI names a model family xgrammar does not support."""

    def __init__(self, model_family: str) -> None:
        super().__init__(
            f"Unsupported guided decoding model family: {model_family!r}. "
            f"Supported families: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
        self.model_family = model_family


def _import_xgrammar() -> Any:
    """Import xgrammar lazily so test collection works without it installed."""
    try:
        import xgrammar as xgr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - import guard
        raise GuidedDecodingUnavailableError(
            "xgrammar is not installed; cannot enable guided decoding"
        ) from exc
    return xgr


def validate_model_family(model_family: str) -> None:
    """Fail fast if a user-provided model family is not supported.

    Args:
        model_family: The family name from the CLI flag.

    Raises:
        UnsupportedGuidedDecodingModelError: If the family is unknown.
    """
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise UnsupportedGuidedDecodingModelError(model_family)


def build_grammar_compiler(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> Any:
    """Build a scheduler-level ``xgr.GrammarCompiler`` for a tokenizer.

    The compiler is cached at the scheduler level because
    ``TokenizerInfo.from_huggingface`` is not trivially cheap. The compiler
    itself has an internal grammar cache so per-request compilation is fast
    for repeat tool sets.
    """
    xgr = _import_xgrammar()
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)
    return xgr.GrammarCompiler(tokenizer_info)


def compile_for_request(
    model_family: str,
    tools: list[dict[str, Any]],
    compiler: Any,
    *,
    reasoning: bool = True,
) -> Any | None:
    """Compile a built-in structural tag grammar for one request's tools.

    Args:
        model_family: xgrammar model family (e.g. ``"minimax"``).
        tools: OpenAI-spec tool list from the chat completion request.
        compiler: The scheduler-level ``xgr.GrammarCompiler``.
        reasoning: Whether the grammar should expect ``<think>...</think>``
            reasoning blocks.  Defaults to True to match xgrammar's default
            and MiniMax's actual output behaviour.

    Returns:
        A compiled grammar ready to construct a ``GrammarMatcher`` from, or
        ``None`` if there are no tools (no constraint to enforce).
    """
    validate_model_family(model_family)
    if not tools:
        return None
    xgr = _import_xgrammar()
    tag = xgr.get_builtin_structural_tag(
        model=model_family,
        tools=tools,
        reasoning=reasoning,
    )
    return compiler.compile_structural_tag(tag)


class XGrammarLogitsProcessor:
    """Per-request logits processor enforcing an xgrammar structural tag.

    Matches the mlx-lm BatchGenerator logits-processor contract:

        new_logits = processor(tokens_so_far, logits)

    where ``tokens_so_far`` is an ``mx.array`` of shape ``(L,)`` containing
    every token seen by this stream (prompt + generated) and ``logits`` is
    an ``mx.array`` of shape ``(1, V)``.  The processor advances an internal
    ``GrammarMatcher`` by accepting any newly-generated tokens, then masks
    the disallowed positions in ``logits``.

    The processor is stateful and must not be shared between requests.
    """

    def __init__(self, compiled_grammar: Any) -> None:
        xgr = _import_xgrammar()
        self._xgr = xgr
        self._compiled = compiled_grammar
        # Vocab size the matcher knows about (may be < model vocab).
        self._grammar_vocab_size: int = int(compiled_grammar.tokenizer_info.vocab_size)
        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        # Number of tokens we've already fed to the matcher.  Starts at
        # ``None`` so the first call can skip the prompt (we only enforce
        # the grammar on generated tokens).
        self._tokens_seen: int | None = None
        # Reusable CPU bitmask buffer.  Shape: (1, ceil(vocab/32)).
        self._bitmask: np.ndarray = np.zeros(
            xgr.get_bitmask_shape(1, self._grammar_vocab_size),
            dtype=np.int32,
        )

    def reset(self) -> None:
        """Reset the matcher state (used for graceful recovery)."""
        self._matcher.reset()
        self._tokens_seen = None

    def _accept_new_tokens(self, tokens_so_far: mx.array) -> None:
        """Feed any newly-generated tokens to the matcher.

        On the first call we record the prompt length and skip feeding
        prompt tokens — the grammar only constrains the generation portion.
        """
        total = int(tokens_so_far.shape[0]) if tokens_so_far.ndim >= 1 else 0
        if self._tokens_seen is None:
            # First call: this is the prompt.  Don't feed it to the matcher.
            self._tokens_seen = total
            return

        if total <= self._tokens_seen:
            return

        # Pull just the new tail onto CPU so we can iterate token ids.
        new_tail = tokens_so_far[self._tokens_seen : total]
        new_tokens: list[int] = np.asarray(new_tail, dtype=np.int64).tolist()
        for tok in new_tokens:
            if tok >= self._grammar_vocab_size:
                # Model sampled a token outside grammar vocab (padding /
                # reserved / MTP-draft slot).  We cannot feed that to
                # the matcher.  Reset and abandon grammar enforcement for
                # the rest of this request.  This is graceful degradation
                # rather than crashing — a safer failure mode for medical
                # software, even if it means one request may produce an
                # unconstrained tail.
                logger.warning(
                    "XGrammar: token id %d >= grammar vocab %d; "
                    "resetting matcher and disabling enforcement for "
                    "this request.",
                    tok,
                    self._grammar_vocab_size,
                )
                self._matcher.reset()
                self._tokens_seen = total
                # Sentinel: once we see an out-of-vocab token we stop
                # touching logits.
                self._disabled = True
                return
            if not self._matcher.accept_token(int(tok)):
                # Grammar rejected a token the sampler produced.  This
                # should only happen if grammar enforcement was bypassed
                # (e.g. the processor saw fresh logits after MTP-draft
                # insertion).  Same graceful-reset strategy.
                logger.warning(
                    "XGrammar: matcher rejected sampled token %d; " "resetting.",
                    tok,
                )
                self._matcher.reset()
                self._tokens_seen = total
                self._disabled = True
                return
        self._tokens_seen = total

    def __call__(self, tokens_so_far: mx.array, logits: mx.array) -> mx.array:
        """Apply the grammar constraint to a single step's logits.

        Args:
            tokens_so_far: All tokens for this stream (prompt + generated).
            logits: ``mx.array`` of shape ``(1, V)`` — V is the *model* vocab
                (``logits.shape[-1]``), which may exceed the grammar vocab.

        Returns:
            Masked logits of the same shape.  Disallowed token positions
            are set to ``-inf``.
        """
        if getattr(self, "_disabled", False):
            return logits

        if self._matcher.is_terminated():
            # Grammar reached an accept state; don't constrain further.
            return logits

        self._accept_new_tokens(tokens_so_far)
        if getattr(self, "_disabled", False):
            return logits

        model_vocab = int(logits.shape[-1])
        # Refill the bitmask for the current matcher state.
        self._matcher.fill_next_token_bitmask(self._bitmask, 0)

        # Unpack the bitmask into a 0/1 mask over the grammar vocab.
        # ``np.unpackbits`` uses uint8 little-endian bit order which matches
        # xgrammar's int32 bitmask layout.
        mask_bits = np.unpackbits(self._bitmask.view(np.uint8), bitorder="little")[
            : self._grammar_vocab_size
        ]

        # Pad the mask out to the model's vocab.  Positions beyond the
        # grammar vocab are ALWAYS disallowed (padding tokens must never
        # be sampled when grammar is in effect).
        if model_vocab > self._grammar_vocab_size:
            padded = np.zeros(model_vocab, dtype=np.uint8)
            padded[: self._grammar_vocab_size] = mask_bits
            mask_bits = padded
        elif model_vocab < self._grammar_vocab_size:
            # Model vocab smaller than grammar vocab — truncate.
            mask_bits = mask_bits[:model_vocab]

        allow = mx.array(mask_bits.astype(np.bool_))
        neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return mx.where(allow[None, :], logits, neg_inf)


__all__ = [
    "GuidedDecodingUnavailableError",
    "SUPPORTED_MODEL_FAMILIES",
    "UnsupportedGuidedDecodingModelError",
    "XGrammarLogitsProcessor",
    "build_grammar_compiler",
    "compile_for_request",
    "validate_model_family",
]
