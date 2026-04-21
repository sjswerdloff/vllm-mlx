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


class Eagle3Attention(nn.Module):
    """Single attention layer for EAGLE3 head.

    Takes concatenated (fc_output, token_embedding) as input to q/k projections,
    producing queries/keys in the target's hidden dimension.
    """

    def __init__(
        self,
        hidden_size: int = 5376,
        num_heads: int = 42,
        num_kv_heads: int = 14,
        head_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # q/k take concatenated input (2 * hidden_size)
        attn_input_size = 2 * hidden_size
        self.q_proj = nn.Linear(attn_input_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(attn_input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(attn_input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.scale = head_dim**-0.5

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        """Forward pass.

        Args:
            x: (batch, seq_len, 2 * hidden_size) — concat of fc_out and token_embed
            cache: Optional KV cache tuple (keys, values)

        Returns:
            (batch, seq_len, hidden_size)
        """
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # KV cache
        if cache is not None:
            keys = mx.concatenate([cache[0], keys], axis=2)
            values = mx.concatenate([cache[1], values], axis=2)

        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeat_factor, axis=1)
            values = mx.repeat(values, repeat_factor, axis=1)

        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask
        if L > 1:
            mask = mx.triu(mx.full((L, keys.shape[2]), float("-inf")), k=keys.shape[2] - L + 1)
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


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

        # FC: concat of aux hidden states → hidden_size
        # Input size = len(aux_layer_ids) * hidden_size
        fc_input_size = len(self.aux_layer_ids) * hidden_size
        self.fc = nn.Linear(fc_input_size, hidden_size, bias=False)

        # Midlayer: transformer layer
        self.midlayer_input_layernorm = nn.RMSNorm(hidden_size)
        self.midlayer_attn = Eagle3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
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
        param_map = {
            "fc.weight": "fc.weight",
            "norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
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
        token_embedding: mx.array,
        cache: tuple | None = None,
    ) -> tuple[mx.array, tuple]:
        """Forward pass of EAGLE3 head.

        Args:
            aux_hidden_states: List of hidden states from target layers.
                Each shape: (batch, seq_len, hidden_size)
            token_embedding: Embedding of the last accepted token(s).
                Shape: (batch, seq_len, hidden_size)
            cache: Optional KV cache tuple from previous step.

        Returns:
            Tuple of (draft_logits_in_target_vocab, new_cache)
            draft_logits shape: (batch, seq_len, target_vocab_size)
        """
        # Concatenate auxiliary hidden states
        concat_hidden = mx.concatenate(aux_hidden_states, axis=-1)
        # (batch, seq_len, 3 * hidden_size)

        # FC projection
        h = self.fc(concat_hidden)
        # (batch, seq_len, hidden_size)

        # Midlayer: attention input is concat(h, token_embedding)
        h_normed = self.midlayer_input_layernorm(h)
        attn_input = mx.concatenate([h_normed, token_embedding], axis=-1)
        # (batch, seq_len, 2 * hidden_size)

        attn_out, new_cache = self.midlayer_attn(attn_input, cache=cache)
        h = h + self.midlayer_hidden_norm(attn_out)

        # MLP
        h_normed = self.midlayer_post_attention_layernorm(h)
        h = h + self.midlayer_mlp(h_normed)

        # Output projection
        h = self.norm(h)
        draft_logits = self.lm_head(h)
        # (batch, seq_len, draft_vocab_size=32000)

        # Map to target vocab space using d2t
        if self.d2t is not None:
            # Scatter draft logits into full target vocab tensor
            # -inf for tokens not in draft vocab
            target_logits = mx.full(
                (*draft_logits.shape[:-1], self.vocab_size),
                float("-inf"),
                dtype=draft_logits.dtype,
            )
            # Use d2t as indices: target_logits[..., d2t[i]] = draft_logits[..., i]
            target_logits[..., self.d2t] = draft_logits
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
            config = json.load(f)
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
