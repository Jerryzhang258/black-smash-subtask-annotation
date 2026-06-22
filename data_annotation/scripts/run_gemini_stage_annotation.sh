#!/usr/bin/env bash
set -euo pipefail

# Run from the VB-VLA project root on the server, for example:
#   bash data_annotation/scripts/run_gemini_stage_annotation.sh

if [ -f "config/api_env.local.sh" ]; then
  source "config/api_env.local.sh"
elif [ -f "data_annotation/config/api_env.local.sh" ]; then
  source "data_annotation/config/api_env.local.sh"
fi

: "${TTK_API_KEY:?Please export TTK_API_KEY or create config/api_env.local.sh}"
: "${TTK_BASE_URL:?Please export TTK_BASE_URL}"
: "${TTK_MODEL:?Please export TTK_MODEL}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/.cache/huggingface/lerobot/chaoyi/0118_data}"
META_ROOT="${META_ROOT:-/root/autodl-tmp/VB-VLA/Data_collection/dataset_converted/meta}"
OUT_ROOT="${OUT_ROOT:-gemini_stage_annotation_results_dual_camera}"
NUM_EPISODES="${NUM_EPISODES:-270}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
CAMERA_KEYS="${CAMERA_KEYS:-observation.images.camera0,observation.images.camera1}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-The robot should first use the left hand to open the orange lid of the red pot on the table and place the lid aside, then use the right hand to pick up the blue block on the table and put it into the red pot.}"

echo "model=${TTK_MODEL}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUT_ROOT}"
echo "num_episodes=${NUM_EPISODES}"

"${PYTHON_BIN}" tools/qwen_stage_annotation_demo.py \
  --dataset-root "${DATASET_ROOT}" \
  --meta-root "${META_ROOT}" \
  --num-episodes "${NUM_EPISODES}" \
  --model "${TTK_MODEL}" \
  --api-key-env TTK_API_KEY \
  --base-url-env TTK_BASE_URL \
  --max-tokens "${MAX_TOKENS}" \
  --output-root "${OUT_ROOT}" \
  --camera-keys "${CAMERA_KEYS}" \
  --frame-sampling all \
  --task-description "${TASK_DESCRIPTION}"

RUN_DIR=$(ls -td "${OUT_ROOT}"/run_* | head -1)
echo "RUN_DIR=${RUN_DIR}"

"${PYTHON_BIN}" tools/postprocess_qwen_stage_results.py "${RUN_DIR}"
"${PYTHON_BIN}" tools/batch_self_check_predictions.py "${RUN_DIR}"

echo "Annotation and self-check sample generation finished."
echo "Next optional step:"
echo "  ${PYTHON_BIN} tools/run_future_verification.py ${RUN_DIR} --model \"${TTK_MODEL}\" --api-key-env TTK_API_KEY --base-url-env TTK_BASE_URL --max-episodes 50 --retries 3"

