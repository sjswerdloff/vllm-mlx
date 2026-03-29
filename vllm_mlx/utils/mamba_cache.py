# SPDX-License-Identifier: Apache-2.0
"""
BatchMambaCache implementation for continuous batching with Mamba models.

mlx-lm's BatchGenerator requires cache objects to have an `extract` method,
but MambaCache (which extends ArraysCache) doesn't have one. This module
provides a BatchMambaCache wrapper that adds batching support.

Note: MambaCache was removed in mlx-lm 0.31.x. This module gracefully
degrades when MambaCache is unavailable (non-Mamba models are unaffected).
"""

import logging
from typing import List, Optional

import mlx.core as mx

try:
    from mlx_lm.models.cache import MambaCache

    _MAMBA_AVAILABLE = True
except ImportError:
    MambaCache = None
    _MAMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


if _MAMBA_AVAILABLE:

    class BatchMambaCache(MambaCache):
        """
        Batch-aware MambaCache for continuous batching.

        This extends MambaCache to support batch operations required by
        mlx-lm's BatchGenerator, specifically the `extract` method.
        """

        def __init__(self, left_padding: Optional[List[int]] = None):
            super().__init__(left_padding=left_padding)
            self._batch_size = len(left_padding) if left_padding else 0

        def extract(self, idx: int) -> MambaCache:
            cache = MambaCache()
            cache.cache = [
                mx.contiguous(c[idx : idx + 1]) if c is not None else None
                for c in self.cache
            ]
            cache.left_padding = None
            return cache

        @classmethod
        def merge(cls, caches: List[MambaCache]) -> "BatchMambaCache":
            if not caches:
                return cls([])

            batch_size = len(caches)
            merged_cache = cls([0] * batch_size)

            num_arrays = len(caches[0].cache)
            merged_cache.cache = []

            for i in range(num_arrays):
                arrays = [c.cache[i] for c in caches if c.cache[i] is not None]
                if arrays:
                    merged_cache.cache.append(mx.concatenate(arrays, axis=0))
                else:
                    merged_cache.cache.append(None)

            return merged_cache

else:
    BatchMambaCache = None


def patch_mlx_lm_for_mamba():
    """
    Patch mlx-lm to support MambaCache in BatchGenerator.

    This modifies the _make_cache function to handle MambaCache by
    converting it to BatchMambaCache.
    """
    if not _MAMBA_AVAILABLE:
        logger.debug("MambaCache not available in mlx-lm, skipping Mamba patch")
        return

    import importlib

    gen_module = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import (
        KVCache,
        ArraysCache,
        RotatingKVCache,
        CacheList,
        MambaCache as OrigMambaCache,
    )
    from mlx_lm.generate import BatchKVCache, BatchRotatingKVCache

    _original_make_cache = gen_module._make_cache

    def _patched_make_cache(model, left_padding):
        def to_batch_cache(c):
            if isinstance(c, KVCache):
                return BatchKVCache(left_padding)
            elif isinstance(c, OrigMambaCache):
                return BatchMambaCache(left_padding)
            elif isinstance(c, ArraysCache):
                c.left_padding = mx.array(left_padding)
                return c
            elif isinstance(c, RotatingKVCache):
                if c.keep > 0:
                    raise ValueError(
                        "RotatingKVCache with keep tokens is not supported."
                    )
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        else:
            return [BatchKVCache(left_padding) for _ in model.layers]

    gen_module._make_cache = _patched_make_cache

    _original_merge_caches = gen_module._merge_caches

    def _patched_merge_caches(caches):
        batch_cache = []
        for i in range(len(caches[0])):
            cache = None
            if isinstance(caches[0][i], KVCache):
                cache = BatchKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], RotatingKVCache):
                cache = BatchRotatingKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], (OrigMambaCache, BatchMambaCache)):
                cache = BatchMambaCache.merge([c[i] for c in caches])
            else:
                raise ValueError(
                    f"{type(caches[0][i])} does not yet support batching with history"
                )
            batch_cache.append(cache)
        return batch_cache

    gen_module._merge_caches = _patched_merge_caches

    logger.info("Patched mlx-lm for MambaCache batching support")


_patched = False


def ensure_mamba_support():
    """Ensure MambaCache batching support is enabled."""
    global _patched
    if not _patched:
        patch_mlx_lm_for_mamba()
        _patched = True
