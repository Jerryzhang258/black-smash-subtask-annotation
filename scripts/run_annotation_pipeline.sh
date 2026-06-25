#!/usr/bin/env bash
# End-to-end annotation pipeline for one black_smash dataset:
# state-only -> Qwen visual check -> fused boundaries -> qwen-stage semantics
# -> multi-track visualization.
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
RUN_QWEN_STAGE="${RUN_QWEN_STAGE:-1}"
RUN_VIZ="${RUN_VIZ:-1}"
RUN_GEMINI="${RUN_GEMINI:-0}"

STATE_OUT="${STATE_OUT:-${OUT_ROOT}/annotations_state_${DATASET_ID}}"
QWEN_VERIFY_OUT="${QWEN_VERIFY_OUT:-${OUT_ROOT}/qwen_local_verify_${DATASET_ID}}"
FUSED_OUT="${FUSED_OUT:-${OUT_ROOT}/annotations_fused_${DATASET_ID}}"
QWEN_STAGE_OUT_ROOT="${QWEN_STAGE_OUT_ROOT:-${OUT_ROOT}/annotations_qwen_stage_${DATASET_ID}}"
VIZ_OUT="${VIZ_OUT:-${OUT_ROOT}/compare_tracks_${DATASET_ID}}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-${OUT_ROOT}/annotations_gemini_stage_${DATASET_ID}}"
GEMINI_JSONL="${GEMINI_JSONL:-}"
STAGE_JSONL="${STAGE_JSONL:-}"
STAGE_LABEL="${STAGE_LABEL:-qwen-stage}"

QWEN_MODEL="${QWEN_MODEL:-qwen}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://localhost:8000/v1}"
QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
QWEN_VERIFY_WINDOW_S="${QWEN_VERIFY_WINDOW_S:-2.0}"
QWEN_VERIFY_MAX_MOVE_S="${QWEN_VERIFY_MAX_MOVE_S:-0.67}"
QWEN_VERIFY_CANDIDATES="${QWEN_VERIFY_CANDIDATES:-7}"
QWEN_VERIFY_SIZE="${QWEN_VERIFY_SIZE:-192}"
QWEN_VERIFY_CROP="${QWEN_VERIFY_CROP:-0.6}"
QWEN_VERIFY_MAX_TOKENS="${QWEN_VERIFY_MAX_TOKENS:-320}"
QWEN_VERIFY_CAMERAS="${QWEN_VERIFY_CAMERAS:-observation.images.camera0,observation.images.camera1}"
QWEN_STAGE_MODEL="${QWEN_STAGE_MODEL:-${QWEN_MODEL}}"
QWEN_STAGE_FRAME_SAMPLING="${QWEN_STAGE_FRAME_SAMPLING:-uniform7}"
QWEN_STAGE_SIGNAL_DETAIL="${QWEN_STAGE_SIGNAL_DETAIL:-compact}"
QWEN_STAGE_MAX_TOKENS="${QWEN_STAGE_MAX_TOKENS:-1600}"
QWEN_STAGE_CAMERA_KEYS="${QWEN_STAGE_CAMERA_KEYS:-observation.images.camera0,observation.images.camera1}"
QWEN_STAGE_LEFT_GRIPPER_DIM="${QWEN_STAGE_LEFT_GRIPPER_DIM:-3}"
QWEN_STAGE_RIGHT_GRIPPER_DIM="${QWEN_STAGE_RIGHT_GRIPPER_DIM:-13}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-The robot should pour black powder from a test tube into a mortar, then grasp the pestle and grind the powder in the mortar.}"

echo "dataset_id=${DATASET_ID}"
echo "data_chunk=${DATA_CHUNK}"
echo "state_out=${STATE_OUT}"
echo "qwen_verify_out=${QWEN_VERIFY_OUT}"
echo "fused_out=${FUSED_OUT}"
echo "qwen_stage_out=${QWEN_STAGE_OUT_ROOT}"
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

FUSE_VLM_OUT="${QWEN_VERIFY_OUT}/verified_annotations"
FUSE_TOL_S_DEFAULT="${FUSE_TOL_S:-${QWEN_VERIFY_MAX_MOVE_S}}"

if [ "${RUN_QWEN}" = "0" ]; then
  echo "RUN_QWEN=0: reusing ${FUSE_VLM_OUT}"
else
  verifier_eps_args=()
  if [ -n "${EPS}" ]; then
    verifier_eps_args=(--eps "${EPS}")
  fi
  "${PYTHON_BIN}" qwen_local_verify.py \
    --data "${DATA_CHUNK}" \
    --state "${STATE_OUT}" \
    --out "${QWEN_VERIFY_OUT}" \
    --meta "${META_PATH}" \
    --fps "${FPS}" \
    --model "${QWEN_MODEL}" \
    --base-url "${QWEN_BASE_URL}" \
    --api-key "${QWEN_API_KEY}" \
    --cameras "${QWEN_VERIFY_CAMERAS}" \
    --window-s "${QWEN_VERIFY_WINDOW_S}" \
    --max-move-s "${QWEN_VERIFY_MAX_MOVE_S}" \
    --candidates "${QWEN_VERIFY_CANDIDATES}" \
    --size "${QWEN_VERIFY_SIZE}" \
    --crop "${QWEN_VERIFY_CROP}" \
    --max-tokens "${QWEN_VERIFY_MAX_TOKENS}" \
    --move-only-p2 \
    "${verifier_eps_args[@]}"
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
    --vlm "${FUSE_VLM_OUT}" \
    --out "${FUSED_OUT}" \
    --fps "${FPS}" \
    --tol-s "${FUSE_TOL_S_DEFAULT}"
fi

if [ "${RUN_QWEN_STAGE}" = "1" ]; then
  stage_episodes_args=()
  if [ -n "${EPS}" ]; then
    stage_episodes_args=(--episodes "${EPS}")
  fi
  LOCAL_QWEN_KEY="${LOCAL_QWEN_KEY:-EMPTY}" \
  "${PYTHON_BIN}" data_annotation/tools/qwen_stage_annotation_demo.py \
    --dataset-root "${DATASET_ROOT}" \
    --meta-root "${META_ROOT}" \
    --output-root "${QWEN_STAGE_OUT_ROOT}" \
    --provider openai \
    --model "${QWEN_STAGE_MODEL}" \
    --api-key-env LOCAL_QWEN_KEY \
    --base-url "${QWEN_BASE_URL}" \
    --num-episodes 9999 \
    "${stage_episodes_args[@]}" \
    --camera-keys "${QWEN_STAGE_CAMERA_KEYS}" \
    --frame-sampling "${QWEN_STAGE_FRAME_SAMPLING}" \
    --left-gripper-dim "${QWEN_STAGE_LEFT_GRIPPER_DIM}" \
    --right-gripper-dim "${QWEN_STAGE_RIGHT_GRIPPER_DIM}" \
    --signal-detail "${QWEN_STAGE_SIGNAL_DETAIL}" \
    --max-tokens "${QWEN_STAGE_MAX_TOKENS}" \
    --critical-ref-dir "${FUSED_OUT}" \
    --task-description "${TASK_DESCRIPTION}"
  latest_stage_run="$(ls -td "${QWEN_STAGE_OUT_ROOT}"/run_* | head -1)"
  "${PYTHON_BIN}" data_annotation/tools/postprocess_qwen_stage_results.py "${latest_stage_run}"
  STAGE_JSONL="${latest_stage_run}/stage_annotations_normalized.jsonl"
  STAGE_LABEL="qwen-stage"
fi

if [ "${RUN_VIZ}" = "1" ]; then
  stage_args=()
  if [ -n "${STAGE_JSONL}" ] && [ -f "${STAGE_JSONL}" ]; then
    stage_args=(--stage-jsonl "${STAGE_JSONL}" --stage-label "${STAGE_LABEL}")
  fi
  if [ -n "${GEMINI_JSONL}" ] && [ -f "${GEMINI_JSONL}" ]; then
    stage_args=(--stage-jsonl "${GEMINI_JSONL}" --stage-label "${GEMINI_STAGE_LABEL:-gemini-stage}")
  fi
  "${PYTHON_BIN}" visualize_annotation_tracks.py \
    --data "${DATA_CHUNK}" \
    --state "${STATE_OUT}" \
    --qwen "${FUSE_VLM_OUT}" \
    --fused "${FUSED_OUT}" \
    --out "${VIZ_OUT}" \
    --fps "${FPS}" \
    "${eps_args[@]}" \
    "${stage_args[@]}"
fi

echo "pipeline complete for ${DATASET_ID}"
