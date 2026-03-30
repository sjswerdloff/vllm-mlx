"""Tests for grammar-constrained decoding via XGrammar.

Tests the XGrammarLogitsProcessor with MiniMax's structural tag grammar.
Validates that:
- Free text generation is unconstrained
- Tool call blocks are properly handled
- Vocab size mismatches between model and grammar are handled
- Prompt tokens are correctly skipped
- The processor integrates correctly with the sampling loop interface
"""

import json
import math

import mlx.core as mx
import numpy as np
import pytest

# Skip all tests if xgrammar is not installed
xgr = pytest.importorskip("xgrammar")
from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

from vllm_mlx.guided_decoding import (
    MINIMAX_STRUCTURAL_TAG,
    XGrammarLogitsProcessor,
    compile_structural_tag,
    create_minimax_tool_processor,
)

# Use GPT-2 for fast tests, MiniMax for integration tests
GPT2_MODEL = "gpt2"


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GPT2_MODEL)


@pytest.fixture(scope="module")
def minimax_tokenizer():
    """Load MiniMax tokenizer - slow, only used for integration tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "lmstudio-community/MiniMax-M2.5-MLX-8bit", trust_remote_code=True
    )


@pytest.fixture
def gpt2_processor(gpt2_tokenizer):
    return create_minimax_tool_processor(gpt2_tokenizer)


class TestGrammarCompilation:
    """Test that grammars compile correctly with different tokenizers."""

    def test_compile_with_gpt2(self, gpt2_tokenizer) -> None:
        result = compile_structural_tag(gpt2_tokenizer, MINIMAX_STRUCTURAL_TAG)
        assert result is not None
        compiled, vocab_size = result
        assert vocab_size == 50257

    def test_compile_with_minimax(self, minimax_tokenizer) -> None:
        result = compile_structural_tag(minimax_tokenizer, MINIMAX_STRUCTURAL_TAG)
        assert result is not None
        compiled, vocab_size = result
        assert vocab_size == 200054

    def test_create_minimax_processor(self, gpt2_tokenizer) -> None:
        processor = create_minimax_tool_processor(gpt2_tokenizer)
        assert processor is not None
        assert isinstance(processor, XGrammarLogitsProcessor)


class TestFreeTextUnconstrained:
    """Verify that free text generation is completely unconstrained."""

    def test_initial_state_all_tokens_allowed(self, gpt2_processor) -> None:
        """At start, every token in vocab should be allowed."""
        vocab_size = 50257
        logits = mx.zeros((1, vocab_size))
        result = gpt2_processor(mx.array([0]), logits)
        n_allowed = mx.sum(result[0] > float("-inf")).item()
        assert n_allowed == vocab_size, f"Expected all {vocab_size} allowed, got {n_allowed}"

    def test_after_normal_text_all_allowed(self, gpt2_tokenizer, gpt2_processor) -> None:
        """After accepting normal text tokens, all tokens should still be allowed."""
        vocab_size = 50257
        prompt = mx.array([0])  # dummy prompt
        # First call records prompt
        gpt2_processor(prompt, mx.zeros((1, vocab_size)))

        # Accept several normal text tokens
        text_ids = gpt2_tokenizer.encode("Hello, I will help you with that task.")
        all_tokens = [0]
        for tid in text_ids:
            all_tokens.append(tid)
            logits = mx.zeros((1, vocab_size))
            result = gpt2_processor(mx.array(all_tokens), logits)

        n_allowed = mx.sum(result[0] > float("-inf")).item()
        assert n_allowed == vocab_size, f"Free text restricted to {n_allowed}/{vocab_size}"

    def test_minimax_initial_all_allowed(self, minimax_tokenizer) -> None:
        """MiniMax tokenizer: all 200054 grammar tokens allowed at start."""
        processor = create_minimax_tool_processor(minimax_tokenizer)
        # Use model's actual logit width (200000), not grammar vocab (200054)
        model_vocab = 200000
        logits = mx.zeros((1, model_vocab))
        result = processor(mx.array([0]), logits)
        n_allowed = mx.sum(result[0] > float("-inf")).item()
        assert n_allowed == model_vocab, f"Expected {model_vocab} allowed, got {n_allowed}"


class TestVocabSizeMismatch:
    """Test handling of vocab size differences between model and grammar."""

    def test_model_vocab_smaller_than_grammar(self, gpt2_tokenizer) -> None:
        """Model outputs fewer logits than grammar vocab - should not crash."""
        processor = create_minimax_tool_processor(gpt2_tokenizer)
        # Grammar vocab is 50257, simulate model with 50000
        small_vocab = 50000
        logits = mx.zeros((1, small_vocab))
        result = processor(mx.array([0]), logits)
        assert result.shape == (1, small_vocab)
        n_allowed = mx.sum(result[0] > float("-inf")).item()
        assert n_allowed == small_vocab

    def test_minimax_vocab_mismatch(self, minimax_tokenizer) -> None:
        """MiniMax: grammar vocab 200054 but model outputs 200000 logits."""
        processor = create_minimax_tool_processor(minimax_tokenizer)
        model_vocab = 200000
        logits = mx.zeros((1, model_vocab))
        # This would crash with broadcast error if using grammar vocab_size
        result = processor(mx.array([0]), logits)
        assert result.shape == (1, model_vocab)


class TestPromptTokenHandling:
    """Test that prompt tokens are correctly skipped."""

    def test_first_call_records_prompt_length(self, gpt2_processor) -> None:
        """First call should record prompt length, not accept tokens."""
        prompt = mx.array([100, 200, 300, 400])
        gpt2_processor(prompt, mx.zeros((1, 50257)))
        assert gpt2_processor._prompt_length == 4

    def test_second_call_accepts_last_token(self, gpt2_tokenizer) -> None:
        """Second call should accept the last (generated) token."""
        processor = create_minimax_tool_processor(gpt2_tokenizer)
        prompt = mx.array([100, 200, 300])
        processor(prompt, mx.zeros((1, 50257)))  # first call

        # Second call: last token (400) is the "generated" token
        tokens_with_gen = mx.array([100, 200, 300, 400])
        result = processor(tokens_with_gen, mx.zeros((1, 50257)))
        # Should not crash, tokens should be allowed
        n = mx.sum(result[0] > float("-inf")).item()
        assert n > 0


class TestToolCallConstraint:
    """Test that tool call blocks are properly constrained."""

    def test_tool_call_tokens_accepted(self, gpt2_tokenizer) -> None:
        """Tool call XML tokens should be accepted by the grammar."""
        processor = create_minimax_tool_processor(gpt2_tokenizer)
        vocab_size = 50257

        # First call (prompt)
        prompt = mx.array([0])
        processor(prompt, mx.zeros((1, vocab_size)))

        # Generate tool call start
        tool_ids = gpt2_tokenizer.encode("<minimax:tool_call>")
        all_tokens = [0]
        for tid in tool_ids:
            all_tokens.append(tid)
            result = processor(mx.array(all_tokens), mx.zeros((1, vocab_size)))

        # Should have tokens allowed (inside tool call)
        n = mx.sum(result[0] > float("-inf")).item()
        assert n > 0, "No tokens allowed inside tool call block"


class TestProcessorReset:
    """Test processor reset between requests."""

    def test_reset_clears_state(self, gpt2_tokenizer) -> None:
        """Reset should return processor to initial state."""
        processor = create_minimax_tool_processor(gpt2_tokenizer)

        # Use the processor
        processor(mx.array([0]), mx.zeros((1, 50257)))
        assert processor._prompt_length is not None

        # Reset
        processor.reset()
        assert processor._prompt_length is None
        assert not processor.matcher.is_terminated()


class TestBitmaskConversion:
    """Test torch→numpy→mlx bitmask conversion preserves values."""

    def test_all_ones_bitmask(self) -> None:
        """All-ones bitmask should allow all tokens."""
        vocab_size = 1000
        bitmask_shape = (1, math.ceil(vocab_size / 32))
        bitmask_np = np.full(bitmask_shape, -1, dtype=np.int32)
        bitmask_mlx = mx.array(bitmask_np)
        logits = mx.zeros((1, vocab_size))
        result = apply_token_bitmask_mlx(bitmask_mlx, logits, vocab_size)
        n = mx.sum(result > float("-inf")).item()
        assert n == vocab_size

    def test_all_zeros_bitmask(self) -> None:
        """All-zeros bitmask should block all tokens."""
        vocab_size = 1000
        bitmask_shape = (1, math.ceil(vocab_size / 32))
        bitmask_np = np.zeros(bitmask_shape, dtype=np.int32)
        bitmask_mlx = mx.array(bitmask_np)
        logits = mx.zeros((1, vocab_size))
        result = apply_token_bitmask_mlx(bitmask_mlx, logits, vocab_size)
        n = mx.sum(result > float("-inf")).item()
        assert n == 0


class TestEndToEndSimulation:
    """Simulate the full server generation loop offline."""

    def test_full_generation_loop_minimax(self, minimax_tokenizer) -> None:
        """Simulate complete generation: prompt → free text → tool call → close."""
        processor = create_minimax_tool_processor(minimax_tokenizer)
        model_vocab = 200000

        # Prompt
        prompt_ids = minimax_tokenizer.encode(
            "You are helpful. Search for Cora in memories."
        )
        tokens = list(prompt_ids)

        # Step 1: prefill
        logits = mx.random.normal((1, model_vocab))
        result = processor(mx.array(tokens), logits)
        assert result.shape == (1, model_vocab)
        assert processor._prompt_length == len(prompt_ids)

        # Generate free text
        free_ids = minimax_tokenizer.encode("I will search for Cora")
        for tid in free_ids:
            tokens.append(tid)
            result = processor(mx.array(tokens), mx.random.normal((1, model_vocab)))
            n = mx.sum(result[0] > float("-inf")).item()
            assert n == model_vocab, f"Free text restricted at token {tid}"

        # Generate tool call
        tool_ids = minimax_tokenizer.encode("<minimax:tool_call>")
        for tid in tool_ids:
            tokens.append(tid)
            result = processor(mx.array(tokens), mx.random.normal((1, model_vocab)))

        # Inside tool call - should still have tokens available
        n_inside = mx.sum(result[0] > float("-inf")).item()
        assert n_inside > 0, "No tokens available inside tool call"

        # Close tool call
        close_ids = minimax_tokenizer.encode("</minimax:tool_call>")
        for tid in close_ids:
            tokens.append(tid)
            result = processor(mx.array(tokens), mx.random.normal((1, model_vocab)))

        total_generated = len(tokens) - len(prompt_ids)
        assert total_generated > 0, "No tokens generated"
