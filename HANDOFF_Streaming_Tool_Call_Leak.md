# HANDOFF: Streaming Tool Call Markup Leak in Anthropic Messages API

## Problem

When MiniMax-M2.5 generates tool calls during streaming via `/v1/messages`, the raw `<minimax:tool_call>` XML markup is emitted as text content deltas to the client BEFORE the structured `tool_use` block is sent at stream end.

**Impact:** The tool call appears twice in the client's context:
1. As raw XML text in the assistant message content (leaked)
2. As a properly structured `tool_use` content block (correct)

This wastes context tokens and may confuse the model on subsequent turns.

## Root Cause

`vllm_mlx/server.py`, function `_stream_anthropic_messages` (line ~1794):

```python
async for output in engine.stream_chat(messages=messages, **chat_kwargs):
    delta_text = output.new_text
    if delta_text:
        content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if content:
            accumulated_text += content
            # This emits the raw tool call XML as text to the client
            yield content_block_delta event
```

Each streaming delta is emitted immediately. Tool call parsing only happens AFTER the stream completes (line ~1815), at which point the raw markup has already been sent to and consumed by the client.

`SPECIAL_TOKENS_PATTERN` (in `vllm_mlx/api/utils.py`) cannot filter multi-token tags like `<minimax:tool_call>` because they arrive split across multiple deltas.

## Affected Format

MiniMax-M2.5 tool call format (from tokenizer chat template):
```xml
<minimax:tool_call>
<invoke name="function_name">
<parameter name="arg">value</parameter>
</invoke>
</minimax:tool_call>
```

Also affects `<think>...</think>` blocks if those should be suppressed (currently they are intentionally preserved).

## Proposed Fix

Implement a streaming buffer in `_stream_anthropic_messages` that:

1. Accumulates text normally when not inside a tool call block
2. Detects entry into a `<minimax:tool_call>` block (or other tool call patterns)
3. Suppresses text output while inside the tool call block
4. On block close, discards the tool call text (it will be re-emitted as structured `tool_use`)
5. Resumes normal text streaming after the block

### Sketch

```python
class StreamingToolCallFilter:
    """Buffer streaming text to suppress tool call markup."""

    def __init__(self):
        self.buffer = ""
        self.in_tool_call = False
        self.tool_call_depth = 0

    def process(self, delta: str) -> str:
        """Returns text to emit (may be empty if inside tool call block)."""
        self.buffer += delta

        if not self.in_tool_call:
            # Check if we're entering a tool call
            if "<minimax:tool_call>" in self.buffer:
                # Emit text before the tag, suppress the rest
                idx = self.buffer.index("<minimax:tool_call>")
                emit = self.buffer[:idx]
                self.buffer = self.buffer[idx:]
                self.in_tool_call = True
                return emit

            # Partial tag might be forming - hold back potential prefix
            # e.g., buffer ends with "<minimax:tool" - don't emit yet
            for prefix_len in range(min(len("<minimax:tool_call>"), len(self.buffer)), 0, -1):
                if "<minimax:tool_call>"[:prefix_len] == self.buffer[-prefix_len:]:
                    emit = self.buffer[:-prefix_len]
                    self.buffer = self.buffer[-prefix_len:]
                    return emit

            # No partial match - safe to emit everything
            emit = self.buffer
            self.buffer = ""
            return emit

        else:
            # Inside tool call - suppress until closing tag
            if "</minimax:tool_call>" in self.buffer:
                idx = self.buffer.index("</minimax:tool_call>") + len("</minimax:tool_call>")
                self.buffer = self.buffer[idx:]
                self.in_tool_call = False
                # Process remainder (might have more text after tool call)
                return self.process("")
            return ""  # Suppress

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        emit = self.buffer
        self.buffer = ""
        return emit
```

### Integration point

In `_stream_anthropic_messages`, wrap the delta processing:

```python
tool_filter = StreamingToolCallFilter()

async for output in engine.stream_chat(messages=messages, **chat_kwargs):
    delta_text = output.new_text
    if delta_text:
        content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if content:
            filtered = tool_filter.process(content)
            accumulated_text += content  # Keep full text for tool parsing
            if filtered:
                yield content_block_delta event with filtered text
```

## Other Tool Call Formats

The same issue could affect Qwen (`<tool_call>`), Llama (`<function=`), and Nemotron formats. The filter should be generalized with configurable tag patterns. But MiniMax is the immediate priority since Mosaic runs on it.

## Testing

1. Send a request with tools defined to the streaming `/v1/messages` endpoint
2. Verify tool call XML does NOT appear in text content deltas
3. Verify the structured `tool_use` block still appears correctly at stream end
4. Verify non-tool-call text streams normally without delay
5. Verify partial tag buffering doesn't cause text loss

## Known Issues

- vllm-mlx upstream issue #129 tracks this for Qwen format
- Multiple fix branches exist upstream but none merged: `fix/streaming-tool-call-parsing`, `fix/streaming-tool-call-reasoning-unified`, `fix/streaming-tool-call-with-reasoning`

## Branch

Current fixes on: `clement/steering-vector-extraction` in `/Users/stuartswerdloff/ai/vllm-mlx/vllm-mlx/`

## Context

Discovered during Mosaic's baseline MLX substrate test (2026-03-29). Tool calling works functionally but the leaked markup consumes context tokens and is visible in Mosaic's output.
