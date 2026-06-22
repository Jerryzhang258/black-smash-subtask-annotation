#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   RUN_DIR=gemini_stage_annotation_results_dual_camera/run_YYYYMMDD_HHMMSS \
#   bash data_annotation/scripts/run_future_verification_first50.sh

if [ -f "config/api_env.local.sh" ]; then
  source "config/api_env.local.sh"
elif [ -f "data_annotation/config/api_env.local.sh" ]; then
  source "data_annotation/config/api_env.local.sh"
fi

: "${RUN_DIR:?Please set RUN_DIR to an annotation run directory}"
: "${TTK_API_KEY:?Please export TTK_API_KEY or create config/api_env.local.sh}"
: "${TTK_BASE_URL:?Please export TTK_BASE_URL}"
: "${TTK_MODEL:?Please export TTK_MODEL}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MAX_EPISODES="${MAX_EPISODES:-50}"
RETRIES="${RETRIES:-3}"

"${PYTHON_BIN}" tools/run_future_verification.py "${RUN_DIR}" \
  --model "${TTK_MODEL}" \
  --api-key-env TTK_API_KEY \
  --base-url-env TTK_BASE_URL \
  --retries "${RETRIES}" \
  --max-episodes "${MAX_EPISODES}"

echo "Future verification finished."
echo "Summary: ${RUN_DIR}/quality_summary.json"

