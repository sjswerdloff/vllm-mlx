# SPDX-License-Identifier: Apache-2.0
"""
Llama EAGLE3 speculative decoding patch.

Monkey-patches a Llama Model (from mlx-lm) to:
1. Capture hidden states from intermediate layers during forward pass
2. Expose an `eagle3_forward()` method that runs the EAGLE3 draft head
3. Provide `make_eagle3_cache()` for the draft head's KV cache

Supports multiple Llama model sizes by computing default layer IDs
from the model's num_hidden_layers:
  - Layer 2 (early)
  - Layer num_layers // 2 (middle)
  - Layer num_layers - 3 (late)

This matches vLLM's default `get_eagle3_aux_hidden_state_layers()`.

Usage:
    from vllm_mlx.patches.llama_eagle3 import inject_eagle3_llama
    inject_eagle3_llama(model, eagle3_head_path)

Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
"""

import logging

import mlx.core as mx

from ..eagle3_head import Eagle3Head, load_eagle3_head

logger = logging.getLogger(__name__)


def _default_aux_layer_ids(num_layers: int) -> list[int]:
    """Compute default EAGLE3 auxiliary layer IDs for a Llama model.

    Matches vLLM's default: (2, num_layers // 2, num_layers - 3).
    """
    return [2, num_layers // 2, num_layers - 3]


def inject_eagle3_llama(model, eagle3_path: str, aux_layer_ids: list[int] | None = None):
    """Inject EAGLE3 speculative decoding into a Llama Model (mlx-lm).

    Patches the model in-place:
    - Overrides __call__ to capture intermediate hidden states
    - Adds eagle3_forward() for draft token prediction
    - Adds make_eagle3_cache() for draft head KV cache

    Args:
        model: A Llama Model instance (from mlx-lm, has .model.layers)
        eagle3_path: Path to EAGLE3 head directory or safetensors
        aux_layer_ids: Layer indices to capture. If None, uses vLLM defaults:
                      [2, num_layers//2, num_layers-3]
    """
    # Determine layer count
    num_layers = len(model.model.layers)

    # Load the EAGLE3 head
    eagle3_head = load_eagle3_head(eagle3_path)

    # Determine which layers to capture
    if aux_layer_ids is not None:
        layer_ids = aux_layer_ids
    else:
        layer_ids = _default_aux_layer_ids(num_layers)

    # Override the head's aux_layer_ids to match
    eagle3_head.aux_layer_ids = layer_ids

    logger.info(
        "EAGLE3 (Llama): injecting into %d-layer model, capturing layers %s",
        num_layers,
        layer_ids,
    )

    # Store as plain dict to avoid MLX nn.Module submodule registration conflicts.
    # nn.Module.__getattr__ intercepts attribute access and may interfere.
    if not hasattr(model, "_eagle3_state"):
        object.__setattr__(model, "_eagle3_state", {})
    model._eagle3_state["head"] = eagle3_head
    model._eagle3_state["aux_layer_ids"] = layer_ids
    model._eagle3_state["aux_hidden_states"] = None
    # Also set eagle3 for detection in routing (using object.__setattr__ to bypass nn.Module)
    object.__setattr__(model, "eagle3", eagle3_head)
    object.__setattr__(model, "_eagle3_aux_layer_ids", layer_ids)
    object.__setattr__(model, "_eagle3_aux_hidden_states", None)
    # Capture references before class swap (nn.Module __getattr__ can break after swap)
    _inner_model = model.model  # LlamaModel
    _args = model.args
    _lm_head = getattr(model, "lm_head", None)
    object.__setattr__(model, "_eagle3_inner", _inner_model)
    object.__setattr__(model, "_eagle3_args", _args)
    object.__setattr__(model, "_eagle3_lm_head", _lm_head)

    original_class = model.__class__

    class _LlamaEagle3(original_class):
        """Llama Model with EAGLE3 hidden state capture."""

        def __call__(
            self,
            inputs: mx.array = None,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            **kwargs,
        ):
            """Forward pass that captures intermediate hidden states for EAGLE3."""
            inner = self._eagle3_inner

            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = inner.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(inner.layers)

            # Build masks
            from mlx_lm.models.llama import create_attention_mask

            fa_mask = create_attention_mask(h, cache[inner.fa_idx])
            swa_mask = None
            if inner.swa_idx is not None:
                swa_mask = create_attention_mask(
                    h, cache[inner.swa_idx], window_size=inner.sliding_window
                )

            # Layer loop with hidden state capture
            captured_hidden = {}
            capture_set = set(self._eagle3_aux_layer_ids)

            for i, (layer, c) in enumerate(zip(inner.layers, cache)):
                mask = swa_mask if layer.use_sliding else fa_mask
                h = layer(h, mask, cache=c)

                if i in capture_set:
                    captured_hidden[i] = h

            normed = inner.norm(h)

            # Logit projection
            if self._eagle3_args.tie_word_embeddings:
                out = inner.embed_tokens.as_linear(normed)
            else:
                out = self._eagle3_lm_head(normed)

            # Store captured hidden states
            self._eagle3_aux_hidden_states = [
                captured_hidden[lid] for lid in self._eagle3_aux_layer_ids
            ]

            if return_hidden:
                return out, self._eagle3_aux_hidden_states
            return out

        def eagle3_forward(
            self,
            token_ids: mx.array,
            eagle3_cache=None,
        ):
            """Run EAGLE3 draft head using captured hidden states.

            Must be called AFTER a regular forward pass.

            Args:
                token_ids: Token IDs to embed. Shape: (batch, seq_len)
                eagle3_cache: KV cache for EAGLE3 attention layer.

            Returns:
                Tuple of (logits_in_target_vocab, new_eagle3_cache)
            """
            if self._eagle3_aux_hidden_states is None:
                raise RuntimeError(
                    "eagle3_forward called before a forward pass. "
                    "Run model(tokens, cache=cache) first."
                )

            # Extract last-position hidden states
            aux_states = [
                h[:, -1:, :] for h in self._eagle3_aux_hidden_states
            ]

            # Pass token_ids — head uses its OWN embed_tokens
            return self.eagle3(aux_states, token_ids[:, -1:], cache=eagle3_cache)

        def make_eagle3_cache(self):
            """Create empty KV cache for EAGLE3 draft head."""
            return None

    model.__class__ = _LlamaEagle3

    # Bind methods explicitly — nn.Module.__getattr__ can't find methods
    # added via class swap. Use object.__setattr__ to bypass nn.Module.
    import types
    object.__setattr__(model, "eagle3_forward", types.MethodType(_LlamaEagle3.eagle3_forward, model))
    object.__setattr__(model, "make_eagle3_cache", types.MethodType(_LlamaEagle3.make_eagle3_cache, model))

    logger.info("EAGLE3 (Llama): model patched successfully")
