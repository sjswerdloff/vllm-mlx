# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI-compatible logprobs support."""


from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogprobs,
    Message,
    TokenLogprob,
    TopLogprob,
    Usage,
)
from vllm_mlx.request import RequestOutput, SamplingParams

# =============================================================================
# Pydantic model tests
# =============================================================================


class TestLogprobsModels:
    """Test logprobs Pydantic models serialize correctly."""

    def test_top_logprob_serialization(self):
        tlp = TopLogprob(token="Hello", logprob=-0.317, bytes=[72, 101, 108, 108, 111])
        d = tlp.model_dump()
        assert d["token"] == "Hello"
        assert d["logprob"] == -0.317
        assert d["bytes"] == [72, 101, 108, 108, 111]

    def test_top_logprob_no_bytes(self):
        tlp = TopLogprob(token="Hi", logprob=-1.2)
        d = tlp.model_dump()
        assert d["bytes"] is None

    def test_token_logprob_with_alternatives(self):
        tl = TokenLogprob(
            token="Hello",
            logprob=-0.317,
            bytes=[72, 101, 108, 108, 111],
            top_logprobs=[
                TopLogprob(
                    token="Hello", logprob=-0.317, bytes=[72, 101, 108, 108, 111]
                ),
                TopLogprob(token="Hi", logprob=-1.319, bytes=[72, 105]),
            ],
        )
        d = tl.model_dump()
        assert len(d["top_logprobs"]) == 2
        assert d["top_logprobs"][0]["token"] == "Hello"
        assert d["top_logprobs"][1]["token"] == "Hi"

    def test_token_logprob_empty_alternatives(self):
        tl = TokenLogprob(token="the", logprob=-0.5)
        d = tl.model_dump()
        assert d["top_logprobs"] == []

    def test_choice_logprobs_structure(self):
        cl = ChoiceLogprobs(
            content=[
                TokenLogprob(token="Hello", logprob=-0.317),
                TokenLogprob(token=" world", logprob=-0.5),
            ]
        )
        d = cl.model_dump()
        assert len(d["content"]) == 2
        assert d["content"][0]["token"] == "Hello"
        assert d["content"][1]["token"] == " world"

    def test_choice_logprobs_null_content(self):
        cl = ChoiceLogprobs(content=None)
        d = cl.model_dump()
        assert d["content"] is None

    def test_chat_completion_choice_with_logprobs(self):
        """Non-streaming choice has logprobs at choice level, not in message."""
        choice = ChatCompletionChoice(
            message=AssistantMessage(content="Hello"),
            logprobs=ChoiceLogprobs(
                content=[TokenLogprob(token="Hello", logprob=-0.317)]
            ),
            finish_reason="stop",
        )
        d = choice.model_dump()
        assert d["logprobs"]["content"][0]["token"] == "Hello"
        assert "logprobs" not in d["message"]

    def test_chat_completion_choice_no_logprobs(self):
        """Without logprobs request, logprobs is None."""
        choice = ChatCompletionChoice(
            message=AssistantMessage(content="Hello"),
            finish_reason="stop",
        )
        d = choice.model_dump()
        assert d["logprobs"] is None

    def test_streaming_chunk_choice_with_logprobs(self):
        """Streaming chunk has logprobs at choice level."""
        chunk_choice = ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(content="Hello"),
            logprobs=ChoiceLogprobs(
                content=[TokenLogprob(token="Hello", logprob=-0.317)]
            ),
        )
        d = chunk_choice.model_dump()
        assert d["logprobs"]["content"][0]["token"] == "Hello"

    def test_streaming_chunk_choice_no_logprobs(self):
        chunk_choice = ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(content="Hello"),
        )
        d = chunk_choice.model_dump()
        assert d["logprobs"] is None


# =============================================================================
# Request model tests
# =============================================================================


class TestLogprobsRequest:
    """Test logprobs fields in request models."""

    def test_chat_request_logprobs_fields(self):
        req = ChatCompletionRequest(
            model="phi-4",
            messages=[Message(role="user", content="Hello")],
            logprobs=True,
            top_logprobs=5,
        )
        assert req.logprobs is True
        assert req.top_logprobs == 5

    def test_chat_request_no_logprobs(self):
        req = ChatCompletionRequest(
            model="phi-4",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.logprobs is None
        assert req.top_logprobs is None

    def test_sampling_params_logprobs(self):
        sp = SamplingParams(logprobs=True, top_logprobs=3)
        assert sp.logprobs is True
        assert sp.top_logprobs == 3

    def test_sampling_params_default_no_logprobs(self):
        sp = SamplingParams()
        assert sp.logprobs is False
        assert sp.top_logprobs == 0


# =============================================================================
# RequestOutput tests
# =============================================================================


class TestRequestOutputLogprobs:
    """Test logprobs in RequestOutput."""

    def test_request_output_with_logprobs(self):
        output = RequestOutput(
            request_id="test-123",
            new_token_ids=[42],
            new_text="Hello",
            token_logprobs=[
                {
                    "token": "Hello",
                    "logprob": -0.317,
                    "bytes": [72, 101, 108, 108, 111],
                    "top_logprobs": [
                        {
                            "token": "Hello",
                            "logprob": -0.317,
                            "bytes": [72, 101, 108, 108, 111],
                        },
                        {"token": "Hi", "logprob": -1.319, "bytes": [72, 105]},
                    ],
                }
            ],
        )
        assert output.token_logprobs is not None
        assert len(output.token_logprobs) == 1
        assert output.token_logprobs[0]["token"] == "Hello"

    def test_request_output_no_logprobs(self):
        output = RequestOutput(request_id="test-123")
        assert output.token_logprobs is None


# =============================================================================
# Output collector merge tests
# =============================================================================


class TestOutputCollectorLogprobsMerge:
    """Test that logprobs merge correctly when outputs are aggregated."""

    def test_merge_both_have_logprobs(self):
        from vllm_mlx.output_collector import RequestOutputCollector

        collector = RequestOutputCollector(aggregate=True)

        out1 = RequestOutput(
            request_id="r1",
            new_token_ids=[1],
            new_text="A",
            token_logprobs=[{"token": "A", "logprob": -0.5, "top_logprobs": []}],
        )
        out2 = RequestOutput(
            request_id="r1",
            new_token_ids=[2],
            new_text="B",
            token_logprobs=[{"token": "B", "logprob": -0.3, "top_logprobs": []}],
        )

        collector.put(out1)
        collector.put(out2)

        merged = collector.get_nowait()
        assert merged is not None
        assert merged.token_logprobs is not None
        assert len(merged.token_logprobs) == 2
        assert merged.token_logprobs[0]["token"] == "A"
        assert merged.token_logprobs[1]["token"] == "B"

    def test_merge_no_logprobs(self):
        from vllm_mlx.output_collector import RequestOutputCollector

        collector = RequestOutputCollector(aggregate=True)

        out1 = RequestOutput(request_id="r1", new_token_ids=[1], new_text="A")
        out2 = RequestOutput(request_id="r1", new_token_ids=[2], new_text="B")

        collector.put(out1)
        collector.put(out2)

        merged = collector.get_nowait()
        assert merged is not None
        assert merged.token_logprobs is None

    def test_merge_mixed_logprobs(self):
        from vllm_mlx.output_collector import RequestOutputCollector

        collector = RequestOutputCollector(aggregate=True)

        out1 = RequestOutput(request_id="r1", new_token_ids=[1], new_text="A")
        out2 = RequestOutput(
            request_id="r1",
            new_token_ids=[2],
            new_text="B",
            token_logprobs=[{"token": "B", "logprob": -0.3, "top_logprobs": []}],
        )

        collector.put(out1)
        collector.put(out2)

        merged = collector.get_nowait()
        assert merged is not None
        assert merged.token_logprobs is not None
        assert len(merged.token_logprobs) == 1


# =============================================================================
# Full response round-trip test
# =============================================================================


class TestLogprobsResponseRoundTrip:
    """Test that a complete response with logprobs serializes correctly."""

    def test_full_response_json_structure(self):
        """Verify the JSON matches OpenAI spec structure."""
        response = ChatCompletionResponse(
            model="phi-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hello world"),
                    logprobs=ChoiceLogprobs(
                        content=[
                            TokenLogprob(
                                token="Hello",
                                logprob=-0.317,
                                bytes=[72, 101, 108, 108, 111],
                                top_logprobs=[
                                    TopLogprob(
                                        token="Hello",
                                        logprob=-0.317,
                                        bytes=[72, 101, 108, 108, 111],
                                    ),
                                    TopLogprob(
                                        token="Hi",
                                        logprob=-1.319,
                                        bytes=[72, 105],
                                    ),
                                ],
                            ),
                            TokenLogprob(
                                token=" world",
                                logprob=-0.5,
                                bytes=[32, 119, 111, 114, 108, 100],
                                top_logprobs=[],
                            ),
                        ]
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

        d = response.model_dump()

        # Verify structure matches OpenAI spec
        choice = d["choices"][0]

        # logprobs is at choice level
        assert "logprobs" in choice
        assert choice["logprobs"] is not None
        assert "content" in choice["logprobs"]
        assert len(choice["logprobs"]["content"]) == 2

        # First token
        t0 = choice["logprobs"]["content"][0]
        assert t0["token"] == "Hello"
        assert t0["logprob"] == -0.317
        assert t0["bytes"] == [72, 101, 108, 108, 111]
        assert len(t0["top_logprobs"]) == 2

        # logprobs NOT in message
        assert "logprobs" not in choice["message"]
