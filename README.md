# Proprioception-Grounded Subtask Segmentation

This repository labels long-horizon bimanual manipulation episodes for the
black-smash task: pour black powder from a test tube into a mortar, then grasp
the pestle and grind. Each episode is represented as 7 subtasks separated by 6
critical points.

The current pipeline is proprioception-first, with Qwen used as a visual and
semantic assistant rather than as the sole boundary source.

## Current Pipeline

The reproducible pipeline for one dataset is:

```text
state boundaries -> Qwen critical-point check -> fused boundaries
                 -> qwen-stage semantic layer -> same-image visualization
```

The four outputs are:

| output | role |
|---|---|
| `annotations_state_<id>/` | state-only boundaries from `observation.state` |
| `annotations_qwen_<id>/` | local Qwen visual critical-point proposal |
| `annotations_fused_<id>/` | per-point fused boundary labels for training/review |
| `annotations_qwen_stage_<id>/` | Qwen semantic descriptions using fused boundaries |
| `compare_tracks_<id>/` | same-image visual comparison: state / qwen / fused / qwen-stage |

Generated annotation and visualization folders are ignored by git because they
are regenerable data artifacts.

## Why This Design

State signals are reliable for gripper and motion boundaries, while camera-only
boundary localization is coarse on this low-light fisheye footage. The fused
labels therefore keep state as the stable skeleton and let Qwen help with the
visual `p2_start_pour` event and with semantic review.

`qwen-stage` is not a replacement for `fused`. It uses fused critical points as
fixed boundaries and asks Qwen to describe each interval with a stage name,
prediction prompt, expected future observation, and short evidence. This avoids
free-form VLM stage splitting, which tends to create missing stages, extra
micro-stages, or unstable boundaries.

## Subtasks

| id | subtask | boundary |
|---|---|---|
| S0 | reach for the test tube | episode start |
| S1 | lift the test tube and move it over the mortar | p1 grasp tube |
| S2 | pour the black powder into the mortar | p2 start pour |
| S3 | release the test tube and reach for the pestle | p3 release tube |
| S4 | bring the pestle over the mortar | p4 grasp pestle |
| S5 | grind the powder in the mortar | p5 start grind |
| S6 | lift the pestle and return to rest | p6 lift pestle |

Each standard annotation directory writes `ep<NNN>_subtasks.json` and
`ep<NNN>_subtask_index.npy`. Fused JSON files also include source and review
metadata such as `sources`, `disagree_frames`, `flags`, and `review_points`.

## Setup

Start the local Qwen2.5-VL-7B-AWQ server:

```bash
./scripts/start_vllm.sh /home/hillbot/models/Qwen2.5-VL-7B-Instruct-AWQ
```

The server exposes an OpenAI-compatible endpoint at:

```text
http://localhost:8000/v1
```

For Linux + RTX 5080 setup details, see
[`docs/INSTALL_QWEN_LINUX.md`](docs/INSTALL_QWEN_LINUX.md).

## Run The Full Pipeline

Run from the repository root:

```bash
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=/home/hillbot/black_smash_05 DATASET_ID=05 \
bash scripts/run_annotation_pipeline.sh

PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=/home/hillbot/black_smash_06 DATASET_ID=06 \
bash scripts/run_annotation_pipeline.sh

PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh
```

Useful switches:

```bash
# subset of episodes
EPS=0,1,2 DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh

# reuse existing annotations and only redraw visualization
RUN_STATE=0 RUN_QWEN=0 RUN_FUSED=0 RUN_QWEN_STAGE=0 RUN_VIZ=1 \
DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh

# skip qwen-stage and draw only state / qwen / fused
RUN_QWEN_STAGE=0 DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh

# disable state-guided p2 temporal refinement for a pure-Qwen baseline
QWEN_P2_HISTORY=0 DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh
```

## Visualization

`compare_tracks_<id>/index.html` shows one row each for:

```text
state / qwen / fused / qwen-stage
```

You can also call the visualizer directly:

```bash
python visualize_annotation_tracks.py \
  --data /home/hillbot/black_smash_07/data/chunk-000 \
  --state annotations_state_07 \
  --qwen annotations_qwen_07 \
  --fused annotations_fused_07 \
  --stage-jsonl annotations_qwen_stage_07/run_xxx/stage_annotations_normalized.jsonl \
  --stage-label qwen-stage \
  --out compare_tracks_07
```

## Optional Gemini Experiment

Gemini support remains available under `data_annotation/`, but it is not the
default pipeline. In local tests, Gemini free-tier quota was insufficient for
full 05/06/07 annotation, and free-form Gemini stage splitting was too coarse for
training boundaries. Treat it as an optional comparison track only.

Keep keys in the ignored local config:

```bash
cp data_annotation/config/api_env.example data_annotation/config/api_env.local.sh
```

Run a Gemini stage experiment:

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

## Repository Map

| file | role |
|---|---|
| `batch_annotate.py` | state-only segmentation |
| `vlm_annotate.py` | local Qwen critical-point annotation and p2 temporal refinement |
| `fuse_annotations.py` | fused labels plus disagreement/review metadata |
| `data_annotation/tools/qwen_stage_annotation_demo.py` | qwen-stage semantic layer, optionally fixed to fused boundaries |
| `visualize_annotation_tracks.py` | same-image comparison for state / qwen / fused / stage rows |
| `scripts/run_annotation_pipeline.sh` | current end-to-end pipeline |
| `scripts/start_vllm.sh` | local Qwen2.5-VL-7B-AWQ vLLM launcher |
| `annotate_gui.py` | manual review GUI |
| `docs/INSTALL_QWEN_LINUX.md` | current local Qwen setup notes |

## Current Local Status

The latest local pipeline is configured for Qwen2.5-VL-7B-AWQ served by vLLM as
`qwen`. The latest full regeneration completed for datasets 05, 06, and 07,
including state, qwen, fused, qwen-stage, and same-image visualizations. To
refresh the artifacts, rerun the pipeline commands above.

## Citation

```bibtex
@misc{subtask_segmentation,
  title  = {Proprioception-Grounded Subtask Segmentation for Long-Horizon Bimanual Manipulation},
  author = {Zhang, Rongxuan},
  year   = {2026},
  howpublished = {\url{https://github.com/Jerryzhang258/black-smash-subtask-annotation}}
}
```
