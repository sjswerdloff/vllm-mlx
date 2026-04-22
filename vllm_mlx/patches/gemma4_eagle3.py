# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 EAGLE3 speculative decoding patch.

Monkey-patches the Gemma4 LanguageModel to:
1. Capture hidden states from intermediate layers during forward pass
2. Expose an `eagle3_forward()` method that runs the EAGLE3 draft head
3. Provide `make_eagle3_cache()` for the draft head's KV cache

The patch wraps the model's `__call__` to intercept the layer loop and
capture hidden states at the EAGLE3 auxiliary layer indices (default: [2, 29, 56]).

Usage:
    from vllm_mlx.patches.gemma4_eagle3 import inject_eagle3
    inject_eagle3(model, eagle3_head_path)

Co-Authored-By: clement-7074f29f <clement-7074f29f@sjstargetedsolutions.co.nz>
"""

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..eagle3_head import Eagle3Head, load_eagle3_head

logger = logging.getLogger(__name__)


def inject_eagle3(model, eagle3_path: str):
    """Inject EAGLE3 speculative decoding into a Gemma4 LanguageModel.

    Patches the model in-place:
    - Overrides __call__ to capture intermediate hidden states
    - Adds eagle3_forward() for draft token prediction
    - Adds make_eagle3_cache() for draft head KV cache

    Args:
        model: A Gemma4 LanguageModel instance (from mlx-vlm)
        eagle3_path: Path to EAGLE3 head (directory with config.json + model.safetensors,
                    or HuggingFace model ID)
    """
    # Load the EAGLE3 head
    eagle3_head = load_eagle3_head(eagle3_path)
    aux_layer_ids = eagle3_head.aux_layer_ids

    logger.info(
        "EAGLE3: injecting into Gemma4, capturing layers %s", aux_layer_ids
    )

    # Store on the model
    model.eagle3 = eagle3_head
    model._eagle3_aux_layer_ids = aux_layer_ids
    model._eagle3_aux_hidden_states = None  # Populated during forward

    # Save original class and create patched version
    original_class = model.__class__

    class _Gemma4Eagle3(original_class):
        """Gemma4 LanguageModel with EAGLE3 hidden state capture."""

        def __call__(
            self,
            inputs=None,
            inputs_embeds=None,
            mask=None,
            cache=None,
            per_layer_inputs=None,
            return_hidden: bool = False,
            **kwargs,
        ):
            """Forward pass that captures intermediate hidden states for EAGLE3.

            When return_hidden=True, returns (LanguageModelOutput, aux_hidden_states)
            where aux_hidden_states is a list of tensors from the configured layers.
            """
            text_model = self.model

            # Compute embeddings
            if inputs_embeds is None:
                h = text_model.embed_tokens(inputs)
                h = h * text_model.embed_scale
            else:
                h = inputs_embeds

            # Handle per-layer inputs (PLE)
            if text_model.hidden_size_per_layer_input:
                if inputs is not None and per_layer_inputs is None:
                    per_layer_inputs = text_model.get_per_layer_inputs(inputs)
                elif per_layer_inputs is not None:
                    target_len = h.shape[1]
                    if per_layer_inputs.shape[1] != target_len:
                        cache_offset = next(
                            (
                                int(c.offset)
                                for c in (cache or [])
                                if c is not None and hasattr(c, "offset")
                            ),
                            0,
                        )
                        max_start = max(
                            per_layer_inputs.shape[1] - target_len, 0
                        )
                        start = min(cache_offset, max_start)
                        per_layer_inputs = per_layer_inputs[
                            :, start : start + target_len
                        ]
                if per_layer_inputs is not None or inputs is not None:
                    per_layer_inputs = text_model.project_per_layer_inputs(
                        h, per_layer_inputs
                    )

            if cache is None:
                cache = [None] * text_model.first_kv_shared_layer_idx

            # Build masks
            from mlx_vlm.models.base import create_attention_mask

            if mask is None:
                first_full = text_model.first_full_cache_idx
                first_sliding = text_model.first_sliding_cache_idx
                global_mask = create_attention_mask(
                    h,
                    cache[first_full] if first_full < len(cache) else None,
                )
                sliding_window_mask = create_attention_mask(
                    h,
                    (
                        cache[first_sliding]
                        if first_sliding < len(cache)
                        else None
                    ),
                    window_size=text_model.window_size,
                )

            # Layer loop with hidden state capture
            captured_hidden = {}
            capture_set = set(self._eagle3_aux_layer_ids)

            for i, layer in enumerate(text_model.layers):
                c = cache[text_model.layer_idx_to_cache_idx[i]]
                is_global = layer.layer_type == "full_attention"

                local_mask = mask
                if mask is None and is_global:
                    local_mask = global_mask
                elif mask is None:
                    local_mask = sliding_window_mask

                per_layer_input = None
                if per_layer_inputs is not None:
                    per_layer_input = per_layer_inputs[:, :, i, :]

                h = layer(h, local_mask, c, per_layer_input=per_layer_input)

                # Capture hidden state at EAGLE3 auxiliary layers
                if i in capture_set:
                    captured_hidden[i] = h

            # Final norm + logit projection
            normed = text_model.norm(h)
            out = text_model.embed_tokens.as_linear(normed)
            if self.final_logit_softcapping is not None:
                from mlx_vlm.models.gemma4.language import logit_softcap
                out = logit_softcap(self.final_logit_softcapping, out)

            from mlx_vlm.models.base import LanguageModelOutput
            result = LanguageModelOutput(logits=out)

            # Store captured hidden states for eagle3_forward
            self._eagle3_aux_hidden_states = [
                captured_hidden[lid] for lid in self._eagle3_aux_layer_ids
            ]

            if return_hidden:
                return result, self._eagle3_aux_hidden_states
            return result

        def eagle3_forward(
            self,
            token_ids: mx.array,
            eagle3_cache: tuple | None = None,
            prev_hidden: mx.array | None = None,
        ) -> tuple[mx.array, tuple, mx.array]:
            """Run EAGLE3 draft head using captured hidden states.

            Must be called AFTER a regular forward pass (which populates
            _eagle3_aux_hidden_states).

            The EAGLE3 head is recurrent: on the first draft step, it uses
            the target model's captured auxiliary hidden states. On subsequent
            steps, it uses its own pre-norm output (prev_hidden) as the
            hidden state input.

            Args:
                token_ids: Token IDs to embed for the draft head input.
                    Shape: (batch, seq_len)
                eagle3_cache: KV cache for the EAGLE3 attention layer.
                prev_hidden: Pre-norm hidden state from previous draft step.
                    None on first step (uses target aux hidden states).

            Returns:
                Tuple of (logits_in_target_vocab, new_eagle3_cache, pre_norm_hidden)
            """
            if self._eagle3_aux_hidden_states is None:
                raise RuntimeError(
                    "eagle3_forward called before a forward pass. "
                    "Run model(tokens, cache=cache) first to capture hidden states."
                )

            # Get token embeddings from target model's embedding layer
            token_embeds = self.model.embed_tokens(token_ids)
            token_embeds = token_embeds * self.model.embed_scale

            # Extract last-position hidden states for EAGLE3
            aux_states = [
                h[:, -1:, :] for h in self._eagle3_aux_hidden_states
            ]

            return self.eagle3(
                aux_states,
                token_embeds[:, -1:, :],
                cache=eagle3_cache,
                prev_hidden=prev_hidden,
            )

        def make_eagle3_cache(self):
            """Create an empty KV cache for the EAGLE3 draft head.

            Returns None (EAGLE3 head manages cache internally via tuple).
            """
            return None

    # Patch the model's class
    model.__class__ = _Gemma4Eagle3
    logger.info("EAGLE3: Gemma4 model patched successfully")
