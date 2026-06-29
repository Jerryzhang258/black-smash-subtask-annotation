# data_annotation

This directory contains optional stage-annotation utilities. The repository's
main pipeline is documented in the root `README.md`:

```text
state -> Qwen local verifier -> fused -> qwen-stage -> visualization
```

The key script here is `tools/qwen_stage_annotation_demo.py`. Despite the
historical filename, it supports both:

- local/OpenAI-compatible Qwen endpoints
- official Google Gemini API

In the current pipeline, Qwen-stage should be run with
`--critical-ref-dir annotations_fused_<id>` so fused critical points remain the
fixed boundaries and the model only supplies semantic stage descriptions.

## Files

| path | role |
|---|---|
| `tools/qwen_stage_annotation_demo.py` | stage annotation with optional fixed critical-point boundaries |
| `tools/postprocess_qwen_stage_results.py` | stage-name and interval normalization |
| `tools/batch_self_check_predictions.py` | future-observation sample generation |
| `tools/run_future_verification.py` | optional Gemini future-frame verification |
| `scripts/run_gemini_stage_annotation.sh` | optional Gemini/OpenAI-compatible experiment runner |
| `config/api_env.example` | local API-key template |

## Local Qwen Stage Annotation

Normally this is called by `scripts/run_annotation_pipeline.sh`. To run it
directly:

```bash
LOCAL_QWEN_KEY=EMPTY \
python data_annotation/tools/qwen_stage_annotation_demo.py \
  --dataset-root /home/hillbot/black_smash_07 \
  --meta-root /home/hillbot/black_smash_07/meta \
  --output-root annotations_qwen_stage_07 \
  --provider openai \
  --model qwen \
  --api-key-env LOCAL_QWEN_KEY \
  --base-url http://localhost:8000/v1 \
  --camera-keys observation.images.camera0,observation.images.camera1 \
  --frame-sampling uniform7 \
  --left-gripper-dim 3 \
  --right-gripper-dim 13 \
  --signal-detail compact \
  --critical-ref-dir annotations_fused_07 \
  --task-description "The robot should pour black powder from a test tube into a mortar, then grasp the pestle and grind the powder in the mortar."
```

Postprocess:

```bash
RUN_DIR=$(ls -td annotations_qwen_stage_07/run_* | head -1)
python data_annotation/tools/postprocess_qwen_stage_results.py "$RUN_DIR"
```

The normalized stage JSONL can be drawn as the fourth visualization row:

```bash
python visualize_annotation_tracks.py \
  --data /home/hillbot/black_smash_07/data/chunk-000 \
  --state annotations_state_07 \
  --qwen qwen_local_verify_07/verified_annotations \
  --fused annotations_fused_07 \
  --stage-jsonl "$RUN_DIR/stage_annotations_normalized.jsonl" \
  --stage-label qwen-stage \
  --out compare_tracks_07
```

## Optional Gemini

Gemini is supported for experiments but is not the default pipeline. Keep keys
in the ignored local config:

```bash
cp data_annotation/config/api_env.example data_annotation/config/api_env.local.sh
```

Run:

```bash
PROVIDER=google \
DATASET_ROOT=/home/hillbot/black_smash_07 \
META_ROOT=/home/hillbot/black_smash_07/meta \
OUT_ROOT=annotations_gemini_stage_07 \
FRAME_SAMPLING=uniform7 \
SIGNAL_DETAIL=compact \
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
bash data_annotation/scripts/run_gemini_stage_annotation.sh
```

## Outputs

Each run writes:

```text
OUT_ROOT/run_xxxxxxxx_xxxxxx/
  stage_annotations.jsonl
  summary.csv
  stage_annotations_normalized.jsonl
  summary_normalized.csv
  prediction_self_check_samples.jsonl
  keyframes/
  gripper_plots/
```

Do not commit real API keys, `.env`, `api_env.local.sh`, generated keyframes, or
large experiment outputs.
