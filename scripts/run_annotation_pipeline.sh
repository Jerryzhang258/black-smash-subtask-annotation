#!/usr/bin/env bash
# End-to-end annotation pipeline for one black_smash dataset:
# state-only -> Qwen critical points -> fused boundaries -> qwen-stage semantics
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
RUN_CANDIDATES="${RUN_CANDIDATES:-1}"   # SIEVE step 1: state -> candidate windows
RUN_QWEN="${RUN_QWEN:-1}"
RUN_FUSED="${RUN_FUSED:-1}"
RUN_QWEN_STAGE="${RUN_QWEN_STAGE:-1}"
RUN_MEMORY="${RUN_MEMORY:-1}"           # SIEVE step 2: semantic keyframe memory
RUN_VLA_EXPORT="${RUN_VLA_EXPORT:-1}"   # SIEVE step 3: VLA memory training samples
RUN_VIZ="${RUN_VIZ:-1}"
RUN_GEMINI="${RUN_GEMINI:-0}"

STATE_OUT="${STATE_OUT:-${OUT_ROOT}/annotations_state_${DATASET_ID}}"
CANDIDATES_OUT="${CANDIDATES_OUT:-${OUT_ROOT}/candidates_${DATASET_ID}}"
QWEN_OUT="${QWEN_OUT:-${OUT_ROOT}/annotations_qwen_${DATASET_ID}}"
FUSED_OUT="${FUSED_OUT:-${OUT_ROOT}/annotations_fused_${DATASET_ID}}"
QWEN_STAGE_OUT_ROOT="${QWEN_STAGE_OUT_ROOT:-${OUT_ROOT}/annotations_qwen_stage_${DATASET_ID}}"
MEMORY_OUT="${MEMORY_OUT:-${OUT_ROOT}/semantic_memory_${DATASET_ID}}"
VLA_OUT="${VLA_OUT:-${OUT_ROOT}/vla_memory_${DATASET_ID}.jsonl}"
VIZ_OUT="${VIZ_OUT:-${OUT_ROOT}/compare_tracks_${DATASET_ID}}"
SCHEMA_PATH="${SCHEMA_PATH:-data_annotation/framework/schemas/black_smash.json}"
MEMORY_CAMERAS="${MEMORY_CAMERAS:-camera0,camera1}"
VLA_MODE="${VLA_MODE:-text-prefix}"
VLA_STRIDE="${VLA_STRIDE:-30}"
VLA_HORIZON="${VLA_HORIZON:-16}"
VLA_MIN_CONFIDENCE="${VLA_MIN_CONFIDENCE:-0.0}"
GEMINI_OUT_ROOT="${GEMINI_OUT_ROOT:-${OUT_ROOT}/annotations_gemini_stage_${DATASET_ID}}"
GEMINI_JSONL="${GEMINI_JSONL:-}"
STAGE_JSONL="${STAGE_JSONL:-}"
STAGE_LABEL="${STAGE_LABEL:-qwen-stage}"

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
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-384}"
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
echo "candidates_out=${CANDIDATES_OUT}"
echo "qwen_out=${QWEN_OUT}"
echo "fused_out=${FUSED_OUT}"
echo "qwen_stage_out=${QWEN_STAGE_OUT_ROOT}"
echo "memory_out=${MEMORY_OUT}"
echo "vla_out=${VLA_OUT}"
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

if [ "${RUN_CANDIDATES}" = "1" ]; then
  "${PYTHON_BIN}" candidate_propose.py \
    --state "${STATE_OUT}" \
    --out "${CANDIDATES_OUT}" \
    --schema "${SCHEMA_PATH}" \
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
    --max-new-tokens "${QWEN_MAX_NEW_TOKENS}" \
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

if [ "${RUN_MEMORY}" = "1" ]; then
  memory_stage_args=()
  if [ -n "${STAGE_JSONL}" ] && [ -f "${STAGE_JSONL}" ]; then
    memory_stage_args=(--stage-jsonl "${STAGE_JSONL}")
  fi
  "${PYTHON_BIN}" build_semantic_memory.py \
    --fused "${FUSED_OUT}" \
    --candidates "${CANDIDATES_OUT}" \
    --out "${MEMORY_OUT}" \
    --cameras "${MEMORY_CAMERAS}" \
    "${memory_stage_args[@]}" \
    "${eps_args[@]}"
fi

if [ "${RUN_VLA_EXPORT}" = "1" ]; then
  "${PYTHON_BIN}" export_vla_memory.py \
    --memory "${MEMORY_OUT}" \
    --out "${VLA_OUT}" \
    --mode "${VLA_MODE}" \
    --stride "${VLA_STRIDE}" \
    --horizon "${VLA_HORIZON}" \
    --min-confidence "${VLA_MIN_CONFIDENCE}" \
    "${eps_args[@]}"
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
    --qwen "${QWEN_OUT}" \
    --fused "${FUSED_OUT}" \
    --out "${VIZ_OUT}" \
    --fps "${FPS}" \
    "${eps_args[@]}" \
    "${stage_args[@]}"
fi

echo "pipeline complete for ${DATASET_ID}"
