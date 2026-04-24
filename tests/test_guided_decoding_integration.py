# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for xgrammar guided decoding.

These tests require a running vllm-mlx server with guided decoding enabled.
They are NOT run by default — use the integration marker:

    # Start the server first (Stuart runs this):
    bash /Users/stuartswerdloff/models/start_minimax25_xgrammar_strict_vllm_mlx.sh

    # Then run integration tests against it:
    pytest tests/test_guided_decoding_integration.py \
        --server-url http://localhost:8899/v1 \
        -m integration

Tests cover the four critical paths:
    1. Normal chat WITHOUT tools — grammar enforcement must not break it
    2. Tool call generation — model produces valid, parseable tool calls
    3. Reasoning + tools — <think> blocks work alongside tool calls
    4. No-tool request — tools defined but model chooses not to call them
"""

from __future__ import annotations

import json

import pytest
import requests

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_url(request) -> str:
    """Server URL from --server-url CLI option."""
    return request.config.getoption("--server-url")


@pytest.fixture(scope="module")
def model_name(server_url) -> str:
    """Discover the model name from the running server."""
    resp = requests.get(f"{server_url}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    assert len(models) > 0, "No models available on server"
    return models[0]["id"]


@pytest.fixture(scope="module")
def weather_tools() -> list[dict]:
    """A realistic tool schema for testing."""
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
                            "description": "City name, e.g. 'London'.",
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


@pytest.fixture(scope="module")
def multi_tools() -> list[dict]:
    """Multiple tools to test tool selection."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _chat_completion(
    server_url: str,
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> dict | list[dict]:
    """Send a chat completion request and return the response."""
    payload: dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools

    resp = requests.post(
        f"{server_url}/chat/completions",
        json=payload,
        timeout=120,
        stream=stream,
    )
    resp.raise_for_status()

    if not stream:
        return resp.json()

    # Collect streaming chunks
    chunks = []
    for line in resp.iter_lines():
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunks.append(json.loads(data))
    return chunks


# ---------------------------------------------------------------------------
# Test 1: Normal chat without tools — grammar must not interfere
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNormalChatUnaffected:
    """Contract: guided decoding does not break normal conversation."""

    def test_simple_chat_produces_content(self, server_url, model_name):
        """A simple chat request with no tools should work normally."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            max_tokens=64,
        )
        choice = result["choices"][0]
        assert choice["message"]["content"] is not None
        content = choice["message"]["content"].strip()
        assert len(content) > 0, "Empty response from normal chat"
        assert "4" in content, f"Expected '4' in response, got: {content}"

    def test_simple_chat_no_tool_calls(self, server_url, model_name):
        """Normal chat should not produce tool_calls."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=64,
        )
        choice = result["choices"][0]
        tool_calls = choice["message"].get("tool_calls")
        assert tool_calls is None or len(tool_calls) == 0, (
            f"Normal chat produced unexpected tool_calls: {tool_calls}"
        )

    def test_streaming_chat_produces_content(self, server_url, model_name):
        """Streaming chat without tools should produce text chunks."""
        chunks = _chat_completion(
            server_url,
            model_name,
            messages=[{"role": "user", "content": "What color is the sky?"}],
            stream=True,
            max_tokens=64,
        )
        assert len(chunks) > 0, "No streaming chunks received"
        # Reconstruct full text from deltas
        full_text = ""
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                full_text += delta["content"]
        assert len(full_text.strip()) > 0, "Streaming produced empty content"


# ---------------------------------------------------------------------------
# Test 2: Tool call generation — valid, parseable tool calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestToolCallGeneration:
    """Contract: when tools are provided and the prompt elicits a tool call,
    the model produces a valid, parseable tool call."""

    def test_tool_call_structure(self, server_url, model_name, weather_tools):
        """Tool call response has correct OpenAI structure."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in London?",
                }
            ],
            tools=weather_tools,
        )
        choice = result["choices"][0]
        # Model should either produce a tool call or text content
        tool_calls = choice["message"].get("tool_calls")
        if tool_calls is None or len(tool_calls) == 0:
            # Model chose not to call the tool — acceptable but note it
            pytest.skip("Model chose text response over tool call")

        tc = tool_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        # Arguments should be valid JSON
        args = json.loads(tc["function"]["arguments"])
        assert "location" in args, f"Missing 'location' in args: {args}"

    def test_tool_call_arguments_valid_json(
        self, server_url, model_name, weather_tools
    ):
        """Tool call arguments must be valid JSON, not malformed."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Check the weather in Tokyo please.",
                }
            ],
            tools=weather_tools,
        )
        choice = result["choices"][0]
        tool_calls = choice["message"].get("tool_calls")
        if not tool_calls:
            pytest.skip("Model chose text response over tool call")

        for tc in tool_calls:
            # This must not raise json.JSONDecodeError
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict), (
                f"Arguments should be a dict, got {type(args)}"
            )

    def test_tool_selection_from_multiple(
        self, server_url, model_name, multi_tools
    ):
        """Model selects the correct tool from multiple options."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Search the web for 'New Zealand weather forecast'.",
                }
            ],
            tools=multi_tools,
        )
        choice = result["choices"][0]
        tool_calls = choice["message"].get("tool_calls")
        if not tool_calls:
            pytest.skip("Model chose text response over tool call")

        tc = tool_calls[0]
        assert tc["function"]["name"] == "search_web", (
            f"Expected search_web, got {tc['function']['name']}"
        )

    def test_streaming_tool_call(self, server_url, model_name, weather_tools):
        """Streaming mode produces parseable tool calls."""
        chunks = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris?",
                }
            ],
            tools=weather_tools,
            stream=True,
        )
        assert len(chunks) > 0, "No streaming chunks received"

        # Reconstruct tool call from streaming deltas
        tool_call_parts: dict[int, dict] = {}
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            tc_deltas = delta.get("tool_calls", [])
            for tc_delta in tc_deltas:
                idx = tc_delta.get("index", 0)
                if idx not in tool_call_parts:
                    tool_call_parts[idx] = {
                        "name": "",
                        "arguments": "",
                    }
                fn = tc_delta.get("function", {})
                if "name" in fn:
                    tool_call_parts[idx]["name"] += fn["name"]
                if "arguments" in fn:
                    tool_call_parts[idx]["arguments"] += fn["arguments"]

        if not tool_call_parts:
            pytest.skip("Model chose text response over tool call in streaming")

        tc = tool_call_parts[0]
        assert tc["name"] == "get_weather"
        args = json.loads(tc["arguments"])
        assert "location" in args


# ---------------------------------------------------------------------------
# Test 3: Reasoning + tools — <think> blocks alongside tool calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReasoningWithTools:
    """Contract: reasoning (<think> blocks) works alongside tool calls."""

    def test_reasoning_does_not_corrupt_tool_call(
        self, server_url, model_name, weather_tools
    ):
        """If the model reasons before calling a tool, both must be valid."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Think step by step about what information you need, "
                        "then check the weather in Berlin."
                    ),
                }
            ],
            tools=weather_tools,
            max_tokens=1024,
        )
        choice = result["choices"][0]
        content = choice["message"].get("content", "") or ""
        tool_calls = choice["message"].get("tool_calls")

        # Model may reason in content and then call the tool,
        # or just call the tool directly. Both are acceptable.
        if tool_calls:
            tc = tool_calls[0]
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args
        else:
            # If no tool call, content should at least mention Berlin
            assert len(content.strip()) > 0, "Neither content nor tool call produced"


# ---------------------------------------------------------------------------
# Test 4: No-tool-call response — tools defined but not needed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNoToolCallWhenUnnecessary:
    """Contract: when tools are defined but the prompt doesn't need them,
    the model responds with text content, not a forced tool call."""

    def test_greeting_with_tools_defined(
        self, server_url, model_name, weather_tools
    ):
        """A greeting should get a text response even with tools available."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! How are you today?",
                }
            ],
            tools=weather_tools,
            max_tokens=128,
        )
        choice = result["choices"][0]
        content = choice["message"].get("content", "") or ""
        tool_calls = choice["message"].get("tool_calls")

        # Should have text content, not a tool call
        assert len(content.strip()) > 0, (
            "Greeting produced empty content — grammar may be forcing tool calls"
        )
        if tool_calls and len(tool_calls) > 0:
            pytest.fail(
                "Grammar enforcement forced a tool call on a greeting. "
                f"Tool calls: {tool_calls}"
            )

    def test_factual_question_with_tools_defined(
        self, server_url, model_name, weather_tools
    ):
        """A factual question unrelated to tools should get a text response."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                }
            ],
            tools=weather_tools,
            max_tokens=64,
        )
        choice = result["choices"][0]
        content = choice["message"].get("content", "") or ""
        assert "Paris" in content or "paris" in content.lower(), (
            f"Expected 'Paris' in response, got: {content}"
        )


# ---------------------------------------------------------------------------
# Test 5: Empty tool list — grammar should be disabled
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmptyToolList:
    """Contract: an empty tool list should behave identically to no tools."""

    def test_empty_tools_produces_normal_response(
        self, server_url, model_name
    ):
        """Empty tools=[] should not activate grammar enforcement."""
        result = _chat_completion(
            server_url,
            model_name,
            messages=[{"role": "user", "content": "What is 3 * 7?"}],
            tools=[],
            max_tokens=64,
        )
        choice = result["choices"][0]
        content = choice["message"].get("content", "") or ""
        assert len(content.strip()) > 0, "Empty tool list broke generation"
        assert "21" in content, f"Expected '21' in response, got: {content}"
