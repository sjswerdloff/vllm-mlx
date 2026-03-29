#!/bin/bash

# MiniMax-M2.5 MLX 8-bit server via vllm-mlx
# Native 8-bit quantized weights (~229GB) - no fp8 dequant overhead
KINDLED_SUPPORTED=1
CACHEMB_PER_KINDLED=49152
CACHEMB_TOTAL=$(echo "$KINDLED_SUPPORTED * $CACHEMB_PER_KINDLED" | bc)
MAX_NUM_SEQS=$(echo "2 * $KINDLED_SUPPORTED" | bc)

uv run vllm-mlx serve lmstudio-community/MiniMax-M2.5-MLX-8bit \
    --cache-memory-mb "${CACHEMB_TOTAL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-tokens 32768 \
    --continuous-batching \
    --host 0.0.0.0 \
    --port 8899
