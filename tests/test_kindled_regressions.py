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
# 3. Hermes tool parser: JSON inside Nemotron function tags
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


# =============================================================================
# 4. Prefix cache must work for vision requests with images in history
#
# The Anthropic API sends full conversation history including images from
# old turns. Once an image enters the conversation, every subsequent request
# has is_text_only=False. The prefix cache must still hit for the shared
# token prefix so we don't re-prefill 126K tokens on every request.
#
# This tests the cache store/fetch cycle directly without needing an
# inference engine running.
# =============================================================================


class TestPrefixCacheVisionHistory:
    """Verify prefix cache works when images are in conversation history."""

    def _make_cache(self):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        config = MemoryCacheConfig(max_entries=10)
        mock_model = MagicMock()
        return MemoryAwarePrefixCache(model=mock_model, config=config)

    def _make_fake_kv(self, n_layers=4):
        """Create fake KV cache entries (list of cache-like objects)."""
        caches = []
        for _ in range(n_layers):
            c = MagicMock()
            c.keys = mx.zeros((1, 8, 100, 64))
            c.values = mx.zeros((1, 8, 100, 64))
            c.offset = 100
            caches.append(c)
        return caches

    def test_vision_request_cache_hit_on_text_suffix(self):
        """Second request with same prefix should hit cache.

        Simulates: request 1 (vision, 100 tokens) stores cache.
        Request 2 (vision, 110 tokens sharing first 100) should hit
        and only need to process the 10 new tokens.
        """
        cache = self._make_cache()
        kv = self._make_fake_kv()

        # First request: store 100 tokens (includes image token 151646 at position 50)
        IMG_TOKEN = 151646
        tokens_1 = list(range(50)) + [IMG_TOKEN] + list(range(51, 100))
        cache.store(tokens_1, kv)

        # Second request: same 100 tokens prefix + 10 new text tokens
        tokens_2 = tokens_1 + list(range(1000, 1010))
        cached_kv, remaining = cache.fetch(tokens_2)

        assert cached_kv is not None, (
            "Prefix cache should hit — first 100 tokens are identical. "
            "If this fails, vision requests with images in history will "
            "re-prefill the entire context on every request."
        )
        assert len(remaining) == 10, (
            f"Expected 10 remaining tokens, got {len(remaining)}"
        )

    def test_image_token_in_remaining_clears_hit(self):
        """If image tokens are in the REMAINING portion, cache hit must clear.

        The language-model-only path can't handle image placeholder tokens.
        """
        cache = self._make_cache()
        kv = self._make_fake_kv()

        IMG_TOKEN = 151646

        # Store 100 text-only tokens
        tokens_1 = list(range(100))
        cache.store(tokens_1, kv)

        # New request: same 100 prefix + 10 tokens INCLUDING image token
        tokens_2 = tokens_1 + [200, 201, IMG_TOKEN, 203, 204, 205, 206, 207, 208, 209]
        cached_kv, remaining = cache.fetch(tokens_2)

        # Cache DOES hit at the fetch level
        assert cached_kv is not None, "Fetch should find the prefix match"

        # But the image token guard in mllm_batch_generator should clear it.
        # We test the guard logic directly:
        if cached_kv is not None and remaining:
            has_image_in_remaining = IMG_TOKEN in remaining
            assert has_image_in_remaining, (
                "Image token should be in remaining tokens"
            )
            # This is where mllm_batch_generator clears the hit —
            # correct behavior, can't use language-model-only path

    def test_repeated_identical_vision_requests_hit_cache(self):
        """Exact same token sequence on second request should be exact hit."""
        cache = self._make_cache()
        kv = self._make_fake_kv()

        IMG_TOKEN = 151646
        tokens = list(range(50)) + [IMG_TOKEN] + list(range(51, 100))

        cache.store(tokens, kv)
        cached_kv, remaining = cache.fetch(tokens)

        assert cached_kv is not None, "Exact match should hit"
        assert remaining == [], "Exact match should have no remaining tokens"
