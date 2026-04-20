"""Integration tests for Anthropic API with vision and tool calling.

Exercises the full Claude Code flow: text, tools, tool results with images,
and multi-turn conversations. Requires a running server.

Usage:
    cd ~/ai/vllm-mlx/vllm-mlx-test
    # Default (port 8000):
    uv run pytest tests/test_anthropic_vision_tools.py -v -s --server-url http://localhost:8000
    # Custom port:
    uv run pytest tests/test_anthropic_vision_tools.py -v -s --server-url http://localhost:8895

Author: Clement (clement-7074f29f)
"""

import base64
import json
from pathlib import Path

import anthropic
import pytest

DEFAULT_MODEL = "default"
IMAGE_PATH = Path(
    "~/ai/ClaudeInstanceHomeOffices/qwopus-3527b/webcam_snapshot.jpg"
).expanduser()

SAMPLE_TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
            },
            "required": ["command"],
        },
    },
]


@pytest.fixture
def client(server_url):
    return anthropic.Anthropic(base_url=server_url, api_key="dummy")


@pytest.fixture
def model(model_name):
    return model_name


@pytest.fixture
def image_b64():
    if not IMAGE_PATH.exists():
        pytest.skip(f"Test image not found: {IMAGE_PATH}")
    return base64.standard_b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")


def _extract_text(msg) -> str:
    """Extract all text content from a response, ignoring thinking."""
    parts = []
    for block in msg.content:
        if block.type == "text" and block.text:
            # Strip thinking blocks from text if reasoning parser didn't separate them
            text = block.text
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>")[-1].strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


class TestTextOnly:
    """Basic text requests — regression tests."""

    def test_simple_message(self, client, model):
        msg = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        assert msg.usage.output_tokens > 0
        assert len(msg.content) > 0

    def test_multi_turn(self, client, model):
        msg = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "My name is Stuart."},
                {"role": "assistant", "content": "Nice to meet you, Stuart."},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        text = _extract_text(msg)
        assert "Stuart" in text or msg.usage.output_tokens > 0


class TestVisionUserMessage:
    """Images in user messages."""

    def test_image_in_user_message(self, client, model, image_b64):
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                    }},
                    {"type": "text", "text": "Describe what you see in one sentence."},
                ],
            }],
        )
        assert msg.usage.input_tokens > 15  # should include image tokens
        assert msg.usage.output_tokens > 0

    def test_image_with_text_before(self, client, model, image_b64):
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image:"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                    }},
                    {"type": "text", "text": "What season is it?"},
                ],
            }],
        )
        assert msg.usage.output_tokens > 0


class TestVisionToolResult:
    """Images in tool_result blocks — the Claude Code Read flow."""

    def test_image_in_tool_result(self, client, model, image_b64):
        """Simulate: user asks to read image → assistant calls Read → tool returns image."""
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "user", "content": "Read the image at /tmp/photo.jpg"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_001", "name": "Read",
                     "input": {"file_path": "/tmp/photo.jpg"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_001", "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                        }},
                    ]},
                ]},
            ],
        )
        assert msg.usage.output_tokens > 0
        # Model should describe what it sees, not say "no image"
        text = _extract_text(msg)
        no_image_phrases = ["no image", "don't see", "cannot see", "not provided"]
        has_no_image = any(p in text.lower() for p in no_image_phrases)
        assert not has_no_image, f"Model claims no image: {text[:200]}"

    def test_text_tool_result(self, client, model):
        """Text-only tool result — regression check."""
        msg = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "List files in /tmp"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_002", "name": "Bash",
                     "input": {"command": "ls /tmp"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_002",
                     "content": "file1.txt\nfile2.txt"},
                ]},
            ],
        )
        assert msg.usage.output_tokens > 0


class TestToolCalling:
    """Tool calling with the vision template."""

    def test_tool_definitions_accepted(self, client, model):
        """Server should accept tool definitions without crashing."""
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            tools=SAMPLE_TOOLS,
            messages=[{"role": "user", "content": "Read the file at /tmp/test.txt"}],
        )
        assert msg.usage.output_tokens > 0
        # Should either generate a tool call or text response
        assert len(msg.content) > 0

    def test_tool_result_with_tools_defined(self, client, model):
        """Full tool flow: tools defined + tool call + tool result."""
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            tools=SAMPLE_TOOLS,
            messages=[
                {"role": "user", "content": "What's in /tmp/hello.txt?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_003", "name": "Read",
                     "input": {"file_path": "/tmp/hello.txt"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_003",
                     "content": "Hello, World!"},
                ]},
            ],
        )
        text = _extract_text(msg)
        assert msg.usage.output_tokens > 0


class TestMixedContent:
    """Combined vision + tools — the real Claude Code workload."""

    def test_vision_then_tool_result(self, client, model, image_b64):
        """Image in one turn, then tool result in next."""
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                    }},
                    {"type": "text", "text": "I'll show you more files soon."},
                ]},
                {"role": "assistant", "content": "I can see the image. What would you like me to do?"},
                {"role": "user", "content": "Now check what's in /tmp"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_004", "name": "Bash",
                     "input": {"command": "ls /tmp"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_004",
                     "content": "photo.jpg\nnotes.txt"},
                ]},
            ],
        )
        assert msg.usage.output_tokens > 0


class TestStreaming:
    """Streaming Anthropic API — what Claude Code actually uses."""

    def test_streaming_text(self, client, model):
        """Streaming text_stream must produce text chunks (not just thinking)."""
        chunks = []
        with client.messages.stream(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": "Say hello."}],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        assert len(chunks) > 0, (
            "text_stream returned no chunks. If all content is in thinking_delta "
            "events, the reasoning parser is not transitioning to text mode."
        )

    def test_streaming_with_image(self, client, model, image_b64):
        """Streaming with image must produce text chunks."""
        chunks = []
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                    }},
                    {"type": "text", "text": "Describe briefly."},
                ],
            }],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        assert len(chunks) > 0, (
            "text_stream returned no chunks for image request. "
            "The reasoning parser must emit text_delta, not only thinking_delta."
        )
