# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for speculative decoding core logic.

Tests verify_and_accept, draft_n_tokens, and cache sync without
requiring actual models — uses mock objects.

Author: Clement (clement-7074f29f)
"""

import mlx.core as mx
import pytest

from vllm_mlx.spec_decode import (
    draft_n_tokens,
    sync_draft_cache,
    verify_and_accept,
)


class MockCache:
    """Mock KV cache that tracks trim calls."""

    def __init__(self):
        self.trim_calls = []
        self.state = [None]

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trim_calls.append(n)


class MockModel:
    """Mock model that returns predictable logits."""

    def __init__(self, vocab_size=100, predict_token=5):
        self.vocab_size = vocab_size
        self.predict_token = predict_token
        self.call_count = 0
        self.last_input = None

    def __call__(self, input_ids, cache=None):
        self.call_count += 1
        self.last_input = input_ids
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Return logits that strongly predict self.predict_token
        logits = mx.full((batch_size, seq_len, self.vocab_size), -10.0)
        # Set high logit for predicted token at every position
        logits[:, :, self.predict_token] = 10.0
        return logits


class MockDraftModel:
    """Mock draft model that generates a fixed sequence of tokens."""

    def __init__(self, vocab_size=100, token_sequence=None):
        self.vocab_size = vocab_size
        self.token_sequence = token_sequence or [5, 5, 5, 5, 5]
        self._idx = 0
        self.call_count = 0
        self.last_input = None

    def __call__(self, input_ids, cache=None):
        self.call_count += 1
        self.last_input = input_ids

        # Ensure 2D
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # For sync calls (multi-token input), just return dummy logits
        if seq_len > 1:
            return mx.zeros((batch_size, seq_len, self.vocab_size))

        # Single token: return logits predicting next token in sequence
        tok = self.token_sequence[self._idx % len(self.token_sequence)]
        self._idx += 1
        logits = mx.full((batch_size, 1, self.vocab_size), -10.0)
        logits[:, :, tok] = 10.0
        return logits


def greedy_sampler(logprobs):
    """Simple greedy sampler for testing."""
    return mx.argmax(logprobs, axis=-1)


class TestVerifyAndAccept:
    """Tests for verify_and_accept logic."""

    def test_all_accepted(self):
        """All draft tokens match target argmax → all accepted + bonus."""
        target = MockModel(predict_token=5)
        cache = [MockCache()]

        # prev_target_logits predicts token 5 (matching draft[0])
        prev_logits = mx.full((100,), -10.0)
        prev_logits[5] = 10.0

        # All drafts are token 5 — target also predicts 5 everywhere
        draft_tokens = [5, 5, 5, 5, 5]

        accepted, next_logits = verify_and_accept(
            target, cache, draft_tokens, prev_logits, greedy_sampler, p_min=0.0
        )

        # All 5 drafts accepted + 1 bonus = 6 tokens
        assert len(accepted) == 6
        assert accepted[:5] == [5, 5, 5, 5, 5]
        # Bonus is also 5 (target always predicts 5)
        assert accepted[5] == 5
        # No trims (all accepted)
        assert cache[0].trim_calls == []

    def test_first_rejected(self):
        """Draft[0] doesn't match target → 0 accepted, bonus only."""
        target = MockModel(predict_token=5)
        cache = [MockCache()]

        # prev_target_logits predicts token 5, but draft[0] is 7
        prev_logits = mx.full((100,), -10.0)
        prev_logits[5] = 10.0

        draft_tokens = [7, 5, 5, 5, 5]

        accepted, next_logits = verify_and_accept(
            target, cache, draft_tokens, prev_logits, greedy_sampler, p_min=0.0
        )

        # 0 drafts accepted + 1 bonus = 1 token
        assert len(accepted) == 1
        # Bonus sampled from prev_logits (predicts 5)
        assert accepted[0] == 5
        # 5 positions trimmed (all rejected)
        assert cache[0].trim_calls == [5]

    def test_partial_accept(self):
        """Accept 3 of 5 drafts."""
        # Target predicts 5 for positions 0,1,2 then something different
        target = MockModel(predict_token=5)
        cache = [MockCache()]

        prev_logits = mx.full((100,), -10.0)
        prev_logits[5] = 10.0

        # First 3 match (5), then mismatch (9)
        draft_tokens = [5, 5, 5, 9, 9]

        accepted, next_logits = verify_and_accept(
            target, cache, draft_tokens, prev_logits, greedy_sampler, p_min=0.0
        )

        # 3 accepted + 1 bonus = 4 tokens
        assert len(accepted) == 4
        assert accepted[:3] == [5, 5, 5]
        # Bonus from verify_logits[0, 2, :] which predicts 5
        assert accepted[3] == 5
        # 2 positions trimmed (drafts 3,4 rejected)
        assert cache[0].trim_calls == [2]

    def test_probability_threshold(self):
        """p_min > 0 uses probability threshold instead of argmax."""
        target = MockModel(predict_token=5)
        cache = [MockCache()]

        # prev_logits: token 5 has high probability
        prev_logits = mx.full((100,), -10.0)
        prev_logits[5] = 10.0

        # Draft token 5 has very high target probability (~1.0)
        draft_tokens = [5, 5, 5]

        accepted, _ = verify_and_accept(
            target, cache, draft_tokens, prev_logits, greedy_sampler, p_min=0.5
        )

        # All should be accepted (target gives ~1.0 prob to token 5)
        assert len(accepted) == 4  # 3 + bonus


class TestDraftNTokens:
    """Tests for draft_n_tokens."""

    def test_generates_n_tokens(self):
        """Generates exactly N tokens."""
        draft = MockDraftModel(token_sequence=[3, 7, 2, 9, 1])
        cache = [MockCache()]

        tokens = draft_n_tokens(draft, cache, last_token=0, n=5, sampler=greedy_sampler)

        assert len(tokens) == 5
        assert tokens == [3, 7, 2, 9, 1]

    def test_generates_fewer_than_sequence(self):
        """Can generate fewer tokens than the model's sequence length."""
        draft = MockDraftModel(token_sequence=[3, 7, 2, 9, 1])
        cache = [MockCache()]

        tokens = draft_n_tokens(draft, cache, last_token=0, n=3, sampler=greedy_sampler)

        assert len(tokens) == 3
        assert tokens == [3, 7, 2]


class TestSyncDraftCache:
    """Tests for sync_draft_cache."""

    def test_trims_all_drafted_positions(self):
        """Trims exactly n_drafted positions from cache."""
        draft = MockDraftModel()
        cache = [MockCache()]

        sync_draft_cache(draft, cache, accepted_tokens=[5, 5, 5], n_drafted=5)

        assert cache[0].trim_calls == [5]

    def test_readvances_with_accepted(self):
        """Feeds accepted tokens back to draft model."""
        draft = MockDraftModel()
        cache = [MockCache()]

        sync_draft_cache(draft, cache, accepted_tokens=[3, 7, 2], n_drafted=5)

        # Draft model should have been called with the accepted tokens
        assert draft.call_count == 1
        # Input should be [3, 7, 2]
        assert draft.last_input.tolist() == [[3, 7, 2]]


class TestStartupValidation:
    """Tests for CLI argument validation."""

    def test_rejects_continuous_batching(self):
        """--speculative-draft-model + --continuous-batching should error."""
        import subprocess

        result = subprocess.run(
            [
                "uv", "run", "python3", "-m", "vllm_mlx.cli", "serve", "dummy-model",
                "--speculative-draft-model", "/tmp/dummy",
                "--continuous-batching",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/stuartswerdloff/ai/vllm-mlx/vllm-mlx-test",
            timeout=10,
        )
        combined = result.stdout.lower() + result.stderr.lower()
        assert result.returncode != 0
        assert "incompatible" in combined, f"Expected 'incompatible' in output: {combined[:500]}"

    def test_rejects_mtp_conflict(self):
        """--speculative-draft-model + --enable-mtp should error."""
        import subprocess

        result = subprocess.run(
            [
                "uv", "run", "python3", "-m", "vllm_mlx.cli", "serve", "dummy-model",
                "--speculative-draft-model", "/tmp/dummy",
                "--enable-mtp",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/stuartswerdloff/ai/vllm-mlx/vllm-mlx-test",
            timeout=10,
        )
        combined = result.stdout.lower() + result.stderr.lower()
        assert result.returncode != 0
        assert "incompatible" in combined, f"Expected 'incompatible' in output: {combined[:500]}"
