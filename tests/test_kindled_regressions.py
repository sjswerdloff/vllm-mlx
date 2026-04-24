# SPDX-License-Identifier: Apache-2.0
"""
Kindled-specific regression tests.

These tests guard fixes we carry on kindled-main that may be overwritten
when merging upstream. Run after every upstream merge to catch regressions.

Not intended for upstream submission — these test our specific patches.
"""

import json
from unittest.mock import MagicMock, patch

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


class FakeKVCache:
    """Simulates KVCache (full attention) — has .keys/.values, no .state."""

    def __init__(self):
        self.keys = mx.zeros((1, 4, 8))
        self.values = mx.zeros((1, 4, 8))


class FakeArraysCache:
    """Simulates ArraysCache (GatedDeltaNet) — has .state/.cache, no .keys."""

    def __init__(self):
        self.state = [mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8))]
        self.cache = self.state  # ArraysCache exposes both


class TestHybridCacheEval:
    """Verify that chunked prefill eval handles both cache types."""

    def test_eval_pattern_handles_kv_cache(self):
        """KVCache layers must be eval'd via .keys/.values."""
        cache = [FakeKVCache(), FakeKVCache()]

        # This is the pattern that MUST be in mllm_batch_generator.py
        cache_tensors = []
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                cache_tensors.extend([c.keys, c.values])
            elif hasattr(c, "state"):
                cache_tensors.extend([s for s in c.state if s is not None])

        assert len(cache_tensors) == 4  # 2 layers × (keys + values)
        mx.eval(cache_tensors)  # Should not raise

    def test_eval_pattern_handles_arrays_cache(self):
        """ArraysCache layers must be eval'd via .state."""
        cache = [FakeArraysCache(), FakeArraysCache()]

        cache_tensors = []
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                cache_tensors.extend([c.keys, c.values])
            elif hasattr(c, "state"):
                cache_tensors.extend([s for s in c.state if s is not None])

        assert len(cache_tensors) == 4  # 2 layers × 2 state arrays
        mx.eval(cache_tensors)

    def test_eval_pattern_handles_hybrid(self):
        """Mixed KVCache + ArraysCache (the real scenario) must eval all."""
        cache = [FakeKVCache(), FakeArraysCache(), FakeKVCache(), FakeArraysCache()]

        cache_tensors = []
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                cache_tensors.extend([c.keys, c.values])
            elif hasattr(c, "state"):
                cache_tensors.extend([s for s in c.state if s is not None])

        assert len(cache_tensors) == 8  # 2 KV × 2 + 2 Arrays × 2
        mx.eval(cache_tensors)

    def test_old_pattern_misses_kv_cache(self):
        """The OLD pattern silently skips KVCache — this is the bug we guard."""
        cache = [FakeKVCache(), FakeArraysCache()]

        # Old broken pattern: mx.eval([c.state for c in cache])
        # FakeKVCache has no .state → AttributeError
        with pytest.raises(AttributeError):
            _ = [c.state for c in cache]

    def test_source_uses_correct_pattern(self):
        """Verify mllm_batch_generator.py uses the hybrid eval pattern."""
        import inspect
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._run_chunked_text_prefill)

        # Must NOT contain the old broken pattern
        assert "mx.eval([c.state for c in cache])" not in source, (
            "_run_chunked_text_prefill uses old c.state pattern — "
            "this will OOM on hybrid models (Qwen3.5). "
            "Must eval both .keys/.values (KVCache) and .state (ArraysCache)."
        )

        # Must contain the hybrid eval
        assert "hasattr(c, \"keys\")" in source, (
            "_run_chunked_text_prefill missing KVCache detection"
        )


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

    def test_source_has_num_images_guard(self):
        """_apply_chat_template must check num_images > 0 for MLLM."""
        import inspect
        from vllm_mlx.engine.batched import BatchedEngine

        source = inspect.getsource(BatchedEngine._apply_chat_template)

        assert "num_images > 0" in source, (
            "_apply_chat_template missing num_images > 0 guard — "
            "text-only MLLM requests will hit processor template which "
            "breaks Anthropic tool_result blocks."
        )

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


# =============================================================================
# 3. Anthropic vision test requires server
#
# Guard: test_anthropic_vision_tools.py must use pytest.importorskip and
# skip when server is unreachable. Without this, the test suite fails on
# import or connection error instead of skipping gracefully.
# =============================================================================


class TestAnthropicVisionTestGuards:
    """Verify anthropic vision tests have proper skip guards."""

    def test_importorskip_in_test_file(self):
        """test_anthropic_vision_tools.py must use importorskip for anthropic."""
        from pathlib import Path

        test_file = Path(__file__).parent / "test_anthropic_vision_tools.py"
        if not test_file.exists():
            pytest.skip("test_anthropic_vision_tools.py not found")

        source = test_file.read_text()
        assert "importorskip" in source, (
            "test_anthropic_vision_tools.py must use pytest.importorskip "
            "for anthropic — bare import causes collection failure when "
            "anthropic is not installed."
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
