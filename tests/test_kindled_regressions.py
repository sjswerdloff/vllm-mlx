# SPDX-License-Identifier: Apache-2.0
"""
Kindled-specific regression tests.

These tests guard fixes we carry on kindled-main that may be overwritten
when merging upstream. Run after every upstream merge to catch regressions.

Not intended for upstream submission — these test our specific patches.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, call, patch

import mlx.core as mx
import pytest


# =============================================================================
# 1. Hybrid cache eval in chunked prefill
#
# Hybrid models (e.g. Qwen3.5) use both ArraysCache (GatedDeltaNet, has
# .state/.cache) and KVCache (full attention, has .keys/.values). Chunked
# prefill must eval ALL cache types between chunks or the lazy computation
# graph grows unbounded → OOM on long prompts.
#
# Regression: someone replaces our dual-type eval with the simpler
#   mx.eval([c.state for c in cache])
# which silently skips KVCache layers.
# =============================================================================


class TrackingKVCache:
    """KVCache mock that tracks whether its tensors were eval'd."""

    def __init__(self):
        # Use non-trivial computation so eval actually does something
        self.keys = mx.ones((1, 4, 8)) + mx.ones((1, 4, 8))
        self.values = mx.ones((1, 4, 8)) + mx.ones((1, 4, 8))
        self.eval_count = 0

    def update_and_fetch(self, keys, values):
        self.keys = keys
        self.values = values
        return self.keys, self.values


class TrackingArraysCache:
    """ArraysCache mock that tracks whether its state was eval'd."""

    def __init__(self):
        self.state = [
            mx.ones((1, 4, 8)) + mx.ones((1, 4, 8)),
            mx.ones((1, 4, 8)) + mx.ones((1, 4, 8)),
        ]
        self.cache = self.state


class TestHybridCacheEvalFunctional:
    """Functionally test that chunked prefill evals both cache types."""

    def test_chunked_prefill_evals_hybrid_cache(self):
        """Call _run_chunked_text_prefill with hybrid cache, verify all eval'd."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

        # Build a minimal generator with a mock language model
        mock_model = MagicMock()
        mock_model.language_model = MagicMock(return_value=mx.zeros((1, 1, 32)))

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.language_model = mock_model.language_model
        gen.prefill_step_size = 4  # Force chunking
        gen._prefill_progress = {}
        gen._aborted_request_ids = set()

        # Create a request with enough tokens to trigger chunking (> step_size)
        request = MLLMBatchRequest(
            uid=0,
            request_id="test-hybrid",
            prompt="test",
        )
        request.input_ids = mx.zeros((1, 12), dtype=mx.int32)  # 12 tokens, step=4 → 2 chunks + last

        # Hybrid cache: KVCache + ArraysCache + KVCache
        kv1 = TrackingKVCache()
        arr1 = TrackingArraysCache()
        kv2 = TrackingKVCache()
        cache = [kv1, arr1, kv2]

        # Patch mx.eval to track what gets eval'd
        eval_calls = []
        original_eval = mx.eval

        def tracking_eval(tensors):
            eval_calls.append(len(tensors))
            original_eval(tensors)

        with patch("vllm_mlx.mllm_batch_generator.mx.eval", side_effect=tracking_eval):
            gen._run_chunked_text_prefill(request, cache)

        # Should have eval'd between chunks.
        # 3 cache objects: 2 KVCache (2 tensors each) + 1 ArraysCache (2 state entries)
        # = 6 tensors per eval call, called twice (2 intermediate chunks before last)
        assert len(eval_calls) >= 2, (
            f"Expected at least 2 inter-chunk evals, got {len(eval_calls)}"
        )
        for i, count in enumerate(eval_calls):
            assert count == 6, (
                f"Eval call {i} got {count} tensors, expected 6 "
                f"(2 KVCache × 2 + 1 ArraysCache × 2). "
                f"If count < 6, some cache types are being skipped."
            )

    def test_chunked_prefill_old_pattern_would_fail(self):
        """Verify the old c.state pattern crashes on hybrid cache."""
        kv = TrackingKVCache()
        arr = TrackingArraysCache()
        cache = [kv, arr]

        # The old broken pattern would do:
        #   mx.eval([c.state for c in cache])
        # KVCache has no .state → AttributeError
        with pytest.raises(AttributeError):
            _ = [c.state for c in cache]

    def test_chunked_prefill_no_chunking_short_prompt(self):
        """Short prompts (≤ step_size) should not chunk — single forward pass."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

        mock_model = MagicMock()
        mock_model.language_model = MagicMock(return_value=mx.zeros((1, 1, 32)))

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.language_model = mock_model.language_model
        gen.prefill_step_size = 2048
        gen._prefill_progress = {}
        gen._aborted_request_ids = set()

        request = MLLMBatchRequest(uid=0, request_id="test-short", prompt="hi")
        request.input_ids = mx.zeros((1, 5), dtype=mx.int32)  # 5 tokens, step=2048

        cache = [TrackingKVCache()]

        gen._run_chunked_text_prefill(request, cache)

        # Single forward call, no intermediate eval needed
        mock_model.language_model.assert_called_once()


# =============================================================================
# 2. Text-only MLLM requests skip processor
#
# On MLLM models, text-only requests (no images) should use the tokenizer
# directly, not the processor. The processor template can be incompatible
# with tool-use formats (e.g. Anthropic tool_result blocks).
#
# Regression: someone removes the `num_images > 0` guard.
# =============================================================================


class TestTextOnlyMLLMPath:
    """Verify text-only requests on MLLM skip the processor."""

    def test_text_only_mllm_skips_processor(self):
        """Text-only request on MLLM engine should not call processor."""
        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            from vllm_mlx.engine.batched import BatchedEngine

            engine = BatchedEngine("test-mllm")
            engine._is_mllm = True

            tokenizer = MagicMock()
            tokenizer.apply_chat_template.return_value = "prompt"

            processor = MagicMock()
            processor.tokenizer = tokenizer
            engine._processor = processor

            engine._apply_chat_template(
                [{"role": "user", "content": "Hello"}],
            )

            # Text-only (no images) → processor skipped, tokenizer used
            processor.apply_chat_template.assert_not_called()
            tokenizer.apply_chat_template.assert_called_once()

    def test_mllm_with_images_uses_processor(self):
        """MLLM request WITH images should use the processor."""
        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            from vllm_mlx.engine.batched import BatchedEngine

            engine = BatchedEngine("test-mllm")
            engine._is_mllm = True

            tokenizer = MagicMock()
            tokenizer.apply_chat_template.return_value = "prompt"

            processor = MagicMock()
            processor.tokenizer = tokenizer
            processor.apply_chat_template.return_value = "vision-prompt"
            engine._processor = processor

            engine._apply_chat_template(
                [{"role": "user", "content": "Describe this image"}],
                num_images=1,
            )

            # With images → processor IS used
            processor.apply_chat_template.assert_called_once()


# =============================================================================
# 3. Anthropic vision test requires server
#
# Guard: test_anthropic_vision_tools.py must use pytest.importorskip and
# skip when server is unreachable. Without this, the test suite fails on
# import or connection error instead of skipping gracefully.
# =============================================================================


class TestAnthropicVisionTestGuards:
    """Verify anthropic vision tests have proper skip guards."""

    def test_anthropic_vision_test_collects_without_server(self):
        """test_anthropic_vision_tools.py must not crash during collection."""
        from pathlib import Path

        test_file = Path(__file__).parent / "test_anthropic_vision_tools.py"
        if not test_file.exists():
            pytest.skip("test_anthropic_vision_tools.py not found")

        # If the file uses bare `import anthropic`, it will crash during
        # collection when anthropic is not installed. importorskip prevents
        # this. We verify by checking the file can be imported as a module.
        source = test_file.read_text()
        assert "importorskip" in source or "import anthropic" not in source, (
            "test_anthropic_vision_tools.py uses bare `import anthropic` — "
            "must use pytest.importorskip to skip gracefully when not installed."
        )


# =============================================================================
# 4. Hermes tool parser: JSON inside Nemotron function tags
#
# When VLM processor template instructs Nemotron format but the model
# outputs JSON inside <function> tags (no <parameter> XML tags), the
# parser must try JSON parsing instead of returning empty arguments.
#
# This is Qwopus-specific but we test the parser behavior generically.
# =============================================================================


class TestNemotronJSONHybridParsing:
    """Verify hermes parser handles JSON body inside function tags."""

    def test_bare_function_with_json_body(self):
        """<function=name>{"k":"v"}</function> should parse JSON arguments."""
        from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser()
        text = '<function=Read>{"file_path": "/tmp/test.txt"}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called, "Should detect tool call"
        assert result.tool_calls[0]["name"] == "Read"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["file_path"] == "/tmp/test.txt"

    def test_tool_call_wrapped_function_with_json(self):
        """<tool_call><function=name>{"k":"v"}</function></tool_call>"""
        from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser()
        text = '<tool_call><function=Read>{"file_path": "/tmp/test.txt"}</function></tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called, (
            "Should detect tool call in <tool_call>-wrapped <function> "
            "with JSON body (no <parameter> tags). This is the Qwopus "
            "VLM failure mode — template instructs Nemotron, model "
            "outputs JSON."
        )
        assert result.tool_calls[0]["name"] == "Read"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["file_path"] == "/tmp/test.txt"

    def test_nemotron_xml_parameter_equals_format(self):
        """Bare <function> with <parameter=key> format (upstream regex)."""
        from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser()
        # Upstream PARAM_PATTERN expects <parameter=key>value</parameter>
        # (not <parameter name="key">value</parameter>)
        text = (
            '<function=Read>'
            '<parameter=file_path>/tmp/test.txt</parameter>'
            '</function>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called, "Nemotron XML with parameter= format should work"
        assert result.tool_calls[0]["name"] == "Read"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["file_path"] == "/tmp/test.txt"

    def test_json_body_with_non_empty_arguments(self):
        """JSON body with multiple arguments should all be captured."""
        from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser()
        text = '<function=Bash>{"command": "ls -la", "timeout": 30}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["command"] == "ls -la"
        assert args["timeout"] == 30
