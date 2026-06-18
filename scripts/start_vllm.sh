#!/usr/bin/env bash
# Start vLLM for Qwen2.5-VL-7B-AWQ (Linux + RTX 5080, vLLM 0.23).
# Usage: ./scripts/start_vllm.sh [model_dir]
set -euo pipefail

MODEL_DIR="${1:-$HOME/models/Qwen2.5-VL-7B-Instruct-AWQ}"
ENV_BIN="${CONDA_PREFIX:-$HOME/miniforge3/envs/qwenvl}/bin"
CU13="$ENV_BIN/../lib/python3.11/site-packages/nvidia/cu13"

export CUDA_HOME="$CU13"
export PATH="$ENV_BIN:$CUDA_HOME/bin:$PATH"

# One-time symlinks (FlashInfer linker expects lib64/ and libcudart.so)
mkdir -p "$CU13/lib"
ln -sf libcudart.so.13 "$CU13/lib/libcudart.so" 2>/dev/null || true
ln -sfn lib "$CU13/lib64" 2>/dev/null || true

exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen \
  --quantization awq_marlin \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":40}' \
  --gpu-memory-utilization 0.92
