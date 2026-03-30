"""Grammar-constrained decoding for vllm-mlx using XGrammar.

Provides logits processors that constrain model output to valid tool call XML
format, preventing malformed tool calls that degrade with context pollution.

The processor integrates with mlx-lm's logits_processors interface:
    processor(tokens: mx.array, logits: mx.array) -> mx.array
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    import xgrammar as xgr
    from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

    _HAS_XGRAMMAR = True
except ImportError:
    _HAS_XGRAMMAR = False
    logger.debug("xgrammar not available - guided decoding disabled")


# MiniMax structural tag specification (JSON)
# Uses XGrammar's triggered_tags: free text until <minimax:tool_call> detected,
# then constrained to close with </minimax:tool_call>.
# This prevents the worst failure: unclosed/malformed tool call blocks.
MINIMAX_STRUCTURAL_TAG = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<minimax:tool_call>"],
        "tags": [
            {
                "begin": "<minimax:tool_call>",
                "content": {"type": "any_text"},
                "end": "</minimax:tool_call>",
            }
        ],
    },
}

# Legacy EBNF grammar (not currently used - structural_tag is preferred)
MINIMAX_TOOL_CALL_EBNF = None


class XGrammarLogitsProcessor:
    """Logits processor that constrains output to a grammar using XGrammar.

    Compatible with mlx-lm's logits_processors interface:
        processor(tokens: mx.array, logits: mx.array) -> mx.array

    The processor maintains FSM state across tokens. After each token is sampled,
    it updates the allowed tokens for the next step.
    """

    def __init__(
        self,
        compiled_grammar: "xgr.CompiledGrammar",
        vocab_size: int,
    ):
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.vocab_size = vocab_size
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self._prompt_length = None  # Set on first call to skip prompt tokens

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply grammar constraint to logits.

        Args:
            tokens: All tokens so far (prompt + completion). On first call,
                    these are all prompt tokens. On subsequent calls, the
                    last token is the most recently generated completion token.
            logits: Raw logits from the model, shape (1, vocab_size)

        Returns:
            Constrained logits with invalid tokens set to -inf
        """
        if self.matcher.is_terminated():
            return logits

        if self._prompt_length is None:
            # First call: tokens are prompt tokens. Record length to skip them.
            # The grammar starts fresh - it will constrain the FIRST generated token.
            self._prompt_length = len(tokens)
        else:
            # Subsequent calls: accept the last generated token into the grammar
            last_token = tokens[-1].item()
            if not self.matcher.accept_token(last_token):
                # Token rejected by grammar - reset and try to continue
                logger.warning(
                    f"[guided_decoding] Token {last_token} rejected by grammar, resetting"
                )
                self.matcher.reset()

        if self.matcher.is_terminated():
            return logits

        # Fill bitmask with allowed tokens for next step
        self.matcher.fill_next_token_bitmask(self.bitmask)

        # Convert bitmask to mx.array if needed (xgrammar allocates torch tensors)
        if isinstance(self.bitmask, mx.array):
            bitmask_mlx = self.bitmask
        else:
            bitmask_mlx = mx.array(self.bitmask.numpy())

        # Use actual logits width, not grammar vocab_size - they can differ
        # (e.g., MiniMax model outputs 200000 logits but tokenizer has 200054 entries)
        actual_vocab = logits.shape[-1]
        return apply_token_bitmask_mlx(bitmask_mlx, logits, actual_vocab)

    def reset(self):
        """Reset the processor for a new generation."""
        self.matcher.reset()
        self._prompt_length = None


def compile_structural_tag(
    tokenizer,
    tag_spec: dict,
) -> Optional["xgr.CompiledGrammar"]:
    """Compile a structural tag specification into a grammar.

    Args:
        tokenizer: HuggingFace tokenizer for the model
        tag_spec: StructuralTag JSON spec dict

    Returns:
        CompiledGrammar, or None if xgrammar is not available
    """
    if not _HAS_XGRAMMAR:
        logger.warning("xgrammar not installed - guided decoding unavailable")
        return None

    import json

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    st = xgr.StructuralTag.from_json(json.dumps(tag_spec))
    compiled = compiler.compile_structural_tag(st)
    return compiled, tokenizer_info.vocab_size


def create_minimax_tool_processor(tokenizer) -> Optional["XGrammarLogitsProcessor"]:
    """Create a processor for MiniMax tool call XML format.

    Uses triggered_tags: free text until <minimax:tool_call> detected,
    then constrained to close with </minimax:tool_call>.
    """
    result = compile_structural_tag(tokenizer, MINIMAX_STRUCTURAL_TAG)
    if result is None:
        return None
    compiled, vocab_size = result
    return XGrammarLogitsProcessor(compiled, vocab_size)


def create_tool_call_processor(
    tokenizer,
    grammar_str: Optional[str] = None,
) -> Optional["XGrammarLogitsProcessor"]:
    """Create a logits processor from an EBNF grammar string.

    Args:
        tokenizer: HuggingFace tokenizer for the model
        grammar_str: EBNF grammar string. If None, uses JSON grammar.

    Returns:
        XGrammarLogitsProcessor, or None if xgrammar is not available
    """
    if not _HAS_XGRAMMAR:
        logger.warning("xgrammar not installed - guided decoding unavailable")
        return None

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    if grammar_str:
        compiled = compiler.compile_grammar(grammar_str)
    else:
        compiled = compiler.compile_builtin_json_grammar()

    return XGrammarLogitsProcessor(compiled, tokenizer_info.vocab_size)
