#!/usr/bin/env bash
set -euo pipefail

# Run from the VB-VLA project root on the server, for example:
#   bash data_annotation/scripts/run_gemini_stage_annotation.sh

if [ -f "config/api_env.local.sh" ]; then
  source "config/api_env.local.sh"
elif [ -f "data_annotation/config/api_env.local.sh" ]; then
  source "data_annotation/config/api_env.local.sh"
fi

PROVIDER="${PROVIDER:-openai}"

if [ "${PROVIDER}" = "google" ]; then
  : "${GEMINI_API_KEY:?Please export GEMINI_API_KEY or create config/api_env.local.sh}"
  MODEL="${MODEL:-${GEMINI_MODEL:-gemini-2.5-flash}}"
  API_KEY_ENV="${API_KEY_ENV:-GEMINI_API_KEY}"
  BASE_URL_ENV="${BASE_URL_ENV:-}"
else
  : "${TTK_API_KEY:?Please export TTK_API_KEY or create config/api_env.local.sh}"
  : "${TTK_BASE_URL:?Please export TTK_BASE_URL}"
  : "${TTK_MODEL:?Please export TTK_MODEL}"
  MODEL="${MODEL:-${TTK_MODEL}}"
  API_KEY_ENV="${API_KEY_ENV:-TTK_API_KEY}"
  BASE_URL_ENV="${BASE_URL_ENV:-TTK_BASE_URL}"
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/.cache/huggingface/lerobot/chaoyi/0118_data}"
META_ROOT="${META_ROOT:-/root/autodl-tmp/VB-VLA/Data_collection/dataset_converted/meta}"
OUT_ROOT="${OUT_ROOT:-gemini_stage_annotation_results_dual_camera}"
NUM_EPISODES="${NUM_EPISODES:-270}"
EPISODES="${EPISODES:-}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
CAMERA_KEYS="${CAMERA_KEYS:-observation.images.camera0,observation.images.camera1}"
FRAME_SAMPLING="${FRAME_SAMPLING:-all}"
LEFT_GRIPPER_DIM="${LEFT_GRIPPER_DIM:-6}"
RIGHT_GRIPPER_DIM="${RIGHT_GRIPPER_DIM:-13}"
SIGNAL_DETAIL="${SIGNAL_DETAIL:-full}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-The robot should first use the left hand to open the orange lid of the red pot on the table and place the lid aside, then use the right hand to pick up the blue block on the table and put it into the red pot.}"

echo "provider=${PROVIDER}"
echo "model=${MODEL}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUT_ROOT}"
echo "num_episodes=${NUM_EPISODES}"
echo "episodes=${EPISODES:-all}"

episodes_args=()
if [ -n "${EPISODES}" ]; then
  episodes_args=(--episodes "${EPISODES}")
fi

"${PYTHON_BIN}" data_annotation/tools/qwen_stage_annotation_demo.py \
  --dataset-root "${DATASET_ROOT}" \
  --meta-root "${META_ROOT}" \
  --num-episodes "${NUM_EPISODES}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --api-key-env "${API_KEY_ENV}" \
  --base-url-env "${BASE_URL_ENV}" \
  --max-tokens "${MAX_TOKENS}" \
  --output-root "${OUT_ROOT}" \
  "${episodes_args[@]}" \
  --camera-keys "${CAMERA_KEYS}" \
  --frame-sampling "${FRAME_SAMPLING}" \
  --left-gripper-dim "${LEFT_GRIPPER_DIM}" \
  --right-gripper-dim "${RIGHT_GRIPPER_DIM}" \
  --signal-detail "${SIGNAL_DETAIL}" \
  --task-description "${TASK_DESCRIPTION}"

RUN_DIR=$(ls -td "${OUT_ROOT}"/run_* | head -1)
echo "RUN_DIR=${RUN_DIR}"

"${PYTHON_BIN}" data_annotation/tools/postprocess_qwen_stage_results.py "${RUN_DIR}"
"${PYTHON_BIN}" data_annotation/tools/batch_self_check_predictions.py "${RUN_DIR}"

echo "Annotation and self-check sample generation finished."
echo "Next optional step:"
if [ "${PROVIDER}" = "google" ]; then
  echo "  ${PYTHON_BIN} data_annotation/tools/run_future_verification.py ${RUN_DIR} --provider google --model \"${MODEL}\" --api-key-env GEMINI_API_KEY --max-episodes 50 --retries 3"
else
  echo "  ${PYTHON_BIN} data_annotation/tools/run_future_verification.py ${RUN_DIR} --model \"${MODEL}\" --api-key-env TTK_API_KEY --base-url-env TTK_BASE_URL --max-episodes 50 --retries 3"
fi
