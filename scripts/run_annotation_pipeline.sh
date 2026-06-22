#!/usr/bin/env bash
# End-to-end annotation pipeline for one black_smash dataset:
# state-only -> Qwen-only -> fused -> multi-track visualization.
#
# Example:
#   DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
#   PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
#   bash scripts/run_annotation_pipeline.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT, e.g. /home/hillbot/black_smash_07}"
DATASET_ID="${DATASET_ID:-$(basename "${DATASET_ROOT}" | sed -E 's/.*_([0-9]+)$/\1/')}"
DATA_CHUNK="${DATA_CHUNK:-${DATASET_ROOT}/data/chunk-000}"
META_PATH="${META_PATH:-${DATASET_ROOT}/meta/tasks.jsonl}"
META_ROOT="${META_ROOT:-${DATASET_ROOT}/meta}"
OUT_ROOT="${OUT_ROOT:-.}"
FPS="${FPS:-30}"
EPS="${EPS:-}"

RUN_STATE="${RUN_STATE:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_FUSED="${RUN_FUSED:-1}"
RUN_VIZ="${RUN_VIZ:-1}"
RUN_GEMINI="${RUN_GEMINI:-0}"

STATE_OUT="${STATE_OUT:-${OUT_ROOT}/annotations_state_${DATASET_ID}}"
QWEN_OUT="${QWEN_OUT:-${OUT_ROOT}/annotations_qwen_${DATASET_ID}}"
FUSED_OUT="${FUSED_OUT:-${OUT_ROOT}/annotations_fused_${DATASET_ID}}"
VIZ_OUT="${VIZ_OUT:-${OUT_ROOT}/compare_tracks_${DATASET_ID}}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-${OUT_ROOT}/annotations_gemini_stage_${DATASET_ID}}"
GEMINI_JSONL="${GEMINI_JSONL:-}"

QWEN_BACKEND="${QWEN_BACKEND:-openai}"
QWEN_MODEL="${QWEN_MODEL:-qwen}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://localhost:8000/v1}"
QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
QWEN_CAM="${QWEN_CAM:-observation.images.camera1}"
QWEN_N_FRAMES="${QWEN_N_FRAMES:-32}"
QWEN_SIZE="${QWEN_SIZE:-256}"
QWEN_CROP="${QWEN_CROP:-0.6}"
QWEN_FINE="${QWEN_FINE:-1}"
QWEN_P2_HISTORY="${QWEN_P2_HISTORY:-1}"
QWEN_P2_HISTORY_WINDOW_S="${QWEN_P2_HISTORY_WINDOW_S:-3.0}"
QWEN_P2_HISTORY_FRAMES="${QWEN_P2_HISTORY_FRAMES:-15}"

echo "dataset_id=${DATASET_ID}"
echo "data_chunk=${DATA_CHUNK}"
echo "state_out=${STATE_OUT}"
echo "qwen_out=${QWEN_OUT}"
echo "fused_out=${FUSED_OUT}"
echo "viz_out=${VIZ_OUT}"

eps_args=()
if [ -n "${EPS}" ]; then
  eps_args=(--eps "${EPS}")
fi

if [ "${RUN_STATE}" = "1" ]; then
  "${PYTHON_BIN}" batch_annotate.py \
    --data "${DATA_CHUNK}" \
    --meta "${META_PATH}" \
    --out "${STATE_OUT}" \
    --fps "${FPS}" \
    "${eps_args[@]}"
fi

if [ "${RUN_QWEN}" = "1" ]; then
  qwen_fine_args=(--fine)
  if [ "${QWEN_FINE}" = "0" ]; then
    qwen_fine_args=(--no-fine)
  fi
  qwen_p2_args=()
  if [ "${QWEN_P2_HISTORY}" = "1" ]; then
    qwen_p2_args=(
      --state-ref "${STATE_OUT}"
      --p2-history
      --p2-history-window-s "${QWEN_P2_HISTORY_WINDOW_S}"
      --p2-history-frames "${QWEN_P2_HISTORY_FRAMES}"
    )
  fi
  "${PYTHON_BIN}" vlm_annotate.py \
    --backend "${QWEN_BACKEND}" \
    --model "${QWEN_MODEL}" \
    --base-url "${QWEN_BASE_URL}" \
    --api-key "${QWEN_API_KEY}" \
    --data "${DATA_CHUNK}" \
    --meta "${META_PATH}" \
    --out "${QWEN_OUT}" \
    --fps "${FPS}" \
    --cam "${QWEN_CAM}" \
    --n-frames "${QWEN_N_FRAMES}" \
    --size "${QWEN_SIZE}" \
    --crop "${QWEN_CROP}" \
    "${qwen_fine_args[@]}" \
    "${qwen_p2_args[@]}" \
    "${eps_args[@]}"
fi

if [ "${RUN_GEMINI}" = "1" ]; then
  DATASET_ROOT="${DATASET_ROOT}" \
  META_ROOT="${META_ROOT}" \
  OUT_ROOT="${GEMINI_OUT_ROOT}" \
  EPISODES="${EPS}" \
  PROVIDER="${GEMINI_PROVIDER:-google}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash data_annotation/scripts/run_gemini_stage_annotation.sh
  latest_run="$(ls -td "${GEMINI_OUT_ROOT}"/run_* | head -1)"
  GEMINI_JSONL="${latest_run}/stage_annotations_normalized.jsonl"
fi

if [ "${RUN_FUSED}" = "1" ]; then
  "${PYTHON_BIN}" fuse_annotations.py \
    --state "${STATE_OUT}" \
    --vlm "${QWEN_OUT}" \
    --out "${FUSED_OUT}" \
    --fps "${FPS}" \
    --tol-s "${FUSE_TOL_S:-0.5}"
fi

if [ "${RUN_VIZ}" = "1" ]; then
  gemini_args=()
  if [ -n "${GEMINI_JSONL}" ] && [ -f "${GEMINI_JSONL}" ]; then
    gemini_args=(--gemini-jsonl "${GEMINI_JSONL}")
  fi
  "${PYTHON_BIN}" visualize_annotation_tracks.py \
    --data "${DATA_CHUNK}" \
    --state "${STATE_OUT}" \
    --qwen "${QWEN_OUT}" \
    --fused "${FUSED_OUT}" \
    --out "${VIZ_OUT}" \
    --fps "${FPS}" \
    "${eps_args[@]}" \
    "${gemini_args[@]}"
fi

echo "pipeline complete for ${DATASET_ID}"
