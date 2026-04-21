# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 draft head for speculative decoding on MLX.

Loads a pre-trained EAGLE3 speculative decoding head and runs inference.
The head takes concatenated hidden states from target model intermediate
layers and predicts the next token with high acceptance rate.

Architecture (from ThoughtWorks Gemma-4-31B-Eagle3):
  Input: concat(target_hidden[layer_2], target_hidden[layer_29], target_hidden[layer_56])
         Shape: (batch, seq_len, 3 * hidden_size)
  FC: (3 * hidden_size) → hidden_size
  Midlayer: full transformer layer (attention + MLP)
            attention input is concat(fc_output, token_embedding) → 2 * hidden_size
  Norm: RMSNorm
  LM Head: hidden_size → draft_vocab_size (32K)
  d2t mapping: draft token → target token

Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
"""

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _compute_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply rotary position embedding to x using precomputed frequencies."""
    # x shape: (B, num_heads, L, head_dim)
    # freqs shape: (L, head_dim//2)
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # Broadcast: freqs is (L, head_dim//2), need (1, 1, L, head_dim//2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Eagle3Attention(nn.Module):
    """Single attention layer for EAGLE3 head with RoPE.

    Takes concatenated (fc_output, token_embedding) as input to q/k projections,
    producing queries/keys in the target's hidden dimension.
    """

    def __init__(
        self,
        hidden_size: int = 5376,
        num_heads: int = 42,
        num_kv_heads: int = 14,
        head_dim: int = 128,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        # q/k take concatenated input (2 * hidden_size)
        attn_input_size = 2 * hidden_size
        self.q_proj = nn.Linear(attn_input_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(attn_input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(attn_input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.scale = head_dim**-0.5

        # Track position for RoPE across cached calls
        self._position_offset = 0

    def _get_rope_freqs(self, seq_len: int, offset: int = 0) -> mx.array:
        """Compute RoPE frequency tensor."""
        dim = self.head_dim
        freqs = 1.0 / (
            self.rope_theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )
        positions = mx.arange(offset, offset + seq_len).astype(mx.float32)
        # (seq_len, dim//2)
        return positions[:, None] * freqs[None, :]

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        offset = cache[0].shape[2] if cache is not None else 0
        freqs = self._get_rope_freqs(L, offset)
        queries = _compute_rope(queries, freqs)
        keys = _compute_rope(keys, freqs)

        # KV cache (store pre-GQA, post-RoPE)
        if cache is not None:
            keys = mx.concatenate([cache[0], keys], axis=2)
            values = mx.concatenate([cache[1], values], axis=2)

        cached_keys = keys
        cached_values = values

        # GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeat_factor, axis=1)
            values = mx.repeat(values, repeat_factor, axis=1)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if L > 1:
            mask = mx.triu(mx.full((L, keys.shape[2]), float("-inf")), k=keys.shape[2] - L + 1)
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (cached_keys, cached_values)


class Eagle3MLP(nn.Module):
    """SwiGLU MLP for EAGLE3 midlayer."""

    def __init__(self, hidden_size: int = 5376, intermediate_size: int = 16384):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3Head(nn.Module):
    """Complete EAGLE3 speculative decoding head.

    Takes hidden states from target model intermediate layers,
    projects them, runs through one transformer layer, and outputs
    draft token logits in a reduced vocabulary.
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config.get("hidden_size", 5376)
        num_heads = config.get("num_attention_heads", 42)
        num_kv_heads = config.get("num_key_value_heads", 14)
        head_dim = config.get("head_dim", 128)
        intermediate_size = config.get("intermediate_size", 16384)
        draft_vocab_size = config.get("draft_vocab_size", 32000)
        self.vocab_size = config.get("vocab_size", 262144)  # target vocab
        self.hidden_size = hidden_size

        # Auxiliary hidden state layer IDs in target model
        eagle_config = config.get("eagle_config", {})
        self.aux_layer_ids = eagle_config.get(
            "eagle_aux_hidden_state_layer_ids", [2, 29, 56]
        )

        # norm_before_residual flag (RedHat speculators format)
        self.norm_before_residual = config.get("norm_before_residual", False)

        # Embedding layer (head's own, separate from target)
        target_vocab = config.get("vocab_size", 262144)
        self.embed_tokens = nn.Embedding(target_vocab, hidden_size)

        # FC: concat of aux hidden states → hidden_size
        fc_input_size = len(self.aux_layer_ids) * hidden_size
        self.fc = nn.Linear(fc_input_size, hidden_size, bias=False)

        # Midlayer: transformer layer
        self.midlayer_input_layernorm = nn.RMSNorm(hidden_size)
        rope_theta = config.get("rope_theta", 500000.0)
        self.midlayer_attn = Eagle3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )
        self.midlayer_hidden_norm = nn.RMSNorm(hidden_size)
        self.midlayer_post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.midlayer_mlp = Eagle3MLP(hidden_size, intermediate_size)

        # Output
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        # Token mappings (loaded from weights)
        self.d2t = None  # (draft_vocab_size,) int — draft idx → target idx
        self.t2d = None  # (target_vocab_size,) bool — mask of which target tokens are in draft

    def load_weights(self, weights_path: str):
        """Load EAGLE3 head weights from safetensors file."""
        weights = mx.load(weights_path)

        # Extract token mappings
        if "d2t" in weights:
            self.d2t = weights.pop("d2t")
            logger.info(f"EAGLE3: loaded d2t mapping, shape={self.d2t.shape}")
        if "t2d" in weights:
            self.t2d = weights.pop("t2d")

        # Map weight names to module structure
        # Supports both ThoughtWorks format (midlayer.*) and speculators format (layers.0.*)
        param_map = {
            # Common
            "fc.weight": "fc.weight",
            "embed_tokens.weight": "embed_tokens.weight",
            "norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # ThoughtWorks format (midlayer.*)
            "midlayer.input_layernorm.weight": "midlayer_input_layernorm.weight",
            "midlayer.post_attention_layernorm.weight": "midlayer_post_attention_layernorm.weight",
            "midlayer.hidden_norm.weight": "midlayer_hidden_norm.weight",
            "midlayer.self_attn.q_proj.weight": "midlayer_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight": "midlayer_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight": "midlayer_attn.v_proj.weight",
            "midlayer.self_attn.o_proj.weight": "midlayer_attn.o_proj.weight",
            "midlayer.mlp.gate_proj.weight": "midlayer_mlp.gate_proj.weight",
            "midlayer.mlp.up_proj.weight": "midlayer_mlp.up_proj.weight",
            "midlayer.mlp.down_proj.weight": "midlayer_mlp.down_proj.weight",
            # Speculators/RedHat format (layers.0.*)
            "layers.0.input_layernorm.weight": "midlayer_input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight": "midlayer_post_attention_layernorm.weight",
            "layers.0.hidden_norm.weight": "midlayer_hidden_norm.weight",
            "layers.0.self_attn.q_proj.weight": "midlayer_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight": "midlayer_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight": "midlayer_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight": "midlayer_attn.o_proj.weight",
            "layers.0.mlp.gate_proj.weight": "midlayer_mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight": "midlayer_mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight": "midlayer_mlp.down_proj.weight",
        }

        mapped_weights = {}
        for src_key, dst_key in param_map.items():
            if src_key in weights:
                mapped_weights[dst_key] = weights[src_key]

        self.load_weights_dict(mapped_weights)
        logger.info(f"EAGLE3: loaded {len(mapped_weights)} weight tensors")

    def load_weights_dict(self, weights_dict: dict):
        """Load weights from a flat dictionary into nested modules."""
        for key, value in weights_dict.items():
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    def __call__(
        self,
        aux_hidden_states: list[mx.array],
        token_ids: mx.array,
        cache: tuple | None = None,
    ) -> tuple[mx.array, tuple]:
        """Forward pass of EAGLE3 head.

        Follows RedHat/speculators Eagle3DecoderLayer architecture:
        1. Concatenate aux hidden states → fc → fused_hidden
        2. Embed token_ids using head's OWN embed_tokens
        3. Decoder layer: norm embeds + norm/residual hidden → attention → MLP
        4. Final norm → lm_head → d2t mapping

        Args:
            aux_hidden_states: List of hidden states from target layers.
                Each shape: (batch, seq_len, hidden_size)
            token_ids: Token IDs to embed (using head's own embeddings).
                Shape: (batch, seq_len)
            cache: Optional KV cache tuple from previous step.

        Returns:
            Tuple of (draft_logits_in_target_vocab, new_cache)
        """
        # Concatenate auxiliary hidden states → FC projection
        concat_hidden = mx.concatenate(aux_hidden_states, axis=-1)
        fused_hidden = self.fc(concat_hidden)
        # (batch, seq_len, hidden_size)

        # Embed tokens using head's OWN embeddings
        if self.embed_tokens is not None:
            embeds = self.embed_tokens(token_ids)
        else:
            # Fallback: caller must pass embeddings as token_ids
            embeds = token_ids

        # Decoder layer (matches RedHat Eagle3DecoderLayer)
        # Split: embeds and fused_hidden
        if self.norm_before_residual:
            hidden = self.midlayer_hidden_norm(fused_hidden)
            residual = hidden
        else:
            residual = fused_hidden
            hidden = self.midlayer_hidden_norm(fused_hidden)

        embeds_normed = self.midlayer_input_layernorm(embeds)
        attn_input = mx.concatenate([embeds_normed, hidden], axis=-1)
        # (batch, seq_len, 2 * hidden_size)

        attn_out, new_cache = self.midlayer_attn(attn_input, cache=cache)
        h = residual + attn_out

        # MLP
        residual = h
        h = self.midlayer_post_attention_layernorm(h)
        h = residual + self.midlayer_mlp(h)

        # Output
        h = self.norm(h)
        draft_logits = self.lm_head(h)
        # (batch, seq_len, draft_vocab_size)

        # Map to target vocab using d2t (additive offset, matching RedHat)
        if self.d2t is not None:
            B, L, _ = draft_logits.shape
            target_logits = mx.full(
                (B, L, self.vocab_size), float("-inf"), dtype=draft_logits.dtype
            )
            # d2t contains OFFSETS: target_idx = draft_idx + d2t[draft_idx]
            draft_indices = mx.arange(self.d2t.shape[0])
            target_indices = draft_indices + self.d2t
            target_logits[:, :, target_indices] = draft_logits
        else:
            target_logits = draft_logits

        return target_logits, new_cache


def load_eagle3_head(model_path: str) -> Eagle3Head:
    """Load an EAGLE3 head from a HuggingFace model directory or safetensors path.

    Args:
        model_path: Path to directory containing config.json + model.safetensors,
                   or direct path to model.safetensors

    Returns:
        Initialized Eagle3Head with weights loaded
    """
    import json

    model_path = Path(model_path)

    if model_path.is_dir():
        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"
    else:
        # Direct safetensors path — look for config.json alongside
        weights_path = model_path
        config_path = model_path.parent / "config.json"

    # Load config
    if config_path.exists():
        with open(config_path) as f:
            raw_config = json.load(f)

        # Handle speculators format (RedHat) vs ThoughtWorks format
        # Speculators nests model params under transformer_layer_config
        if "transformer_layer_config" in raw_config:
            tlc = raw_config["transformer_layer_config"]
            config = {
                "hidden_size": tlc.get("hidden_size", 8192),
                "num_attention_heads": tlc.get("num_attention_heads", 64),
                "num_key_value_heads": tlc.get("num_key_value_heads", 8),
                "head_dim": tlc.get("head_dim", 128),
                "intermediate_size": tlc.get("intermediate_size", 28672),
                "draft_vocab_size": raw_config.get("draft_vocab_size", 32000),
                "vocab_size": tlc.get("vocab_size", 128256),
                "norm_before_residual": raw_config.get("norm_before_residual", False),
                "rope_theta": tlc.get("rope_theta", 500000.0),
                "eagle_config": raw_config.get("eagle_config", {}),
            }
        else:
            config = raw_config
    else:
        # Use defaults from ThoughtWorks Gemma-4-31B-Eagle3
        config = {
            "hidden_size": 5376,
            "num_attention_heads": 42,
            "num_key_value_heads": 14,
            "head_dim": 128,
            "intermediate_size": 16384,
            "draft_vocab_size": 32000,
            "vocab_size": 262144,
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": [2, 29, 56],
            },
        }
        logger.warning("EAGLE3: no config.json found, using defaults")

    head = Eagle3Head(config)
    head.load_weights(str(weights_path))

    return head
