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

## Reproduction Quick Start

The commands below assume this repository is checked out at
`~/black-smash-subtask-annotation` and the local Qwen server exposes the model
name `qwen` at `http://localhost:8000/v1`.

The pipeline is dataset-agnostic. Any dataset can be used as long as it follows
the layout below and is passed through `DATASET_ROOT`. For convenience, create a
stable symlink such as `~/black_smash_current` that points to the dataset you
want to annotate:

```bash
ln -sfn /path/to/your/black_smash_dataset ~/black_smash_current
```

Expected dataset layout:

```text
black_smash_<id>/
  data/chunk-000/episode_*.parquet
  meta/tasks.jsonl
  meta/info.json
```

Final reproducible outputs are written under the repository root. The suffix is
controlled by `DATASET_ID`, so use a meaningful id for each dataset:

```text
annotations_state_<id>/
annotations_qwen_<id>/
annotations_fused_<id>/
annotations_qwen_stage_<id>/
compare_tracks_<id>/
```

For training, use `annotations_fused_<id>/` as the boundary labels. For semantic
stage descriptions, use `annotations_qwen_stage_<id>/`. Pure Qwen and Gemini are
kept only as comparison baselines.

## Setup

Start the local Qwen2.5-VL-7B-AWQ server:

```bash
./scripts/start_vllm.sh ~/models/Qwen2.5-VL-7B-Instruct-AWQ
```

The server exposes an OpenAI-compatible endpoint at:

```text
http://localhost:8000/v1
```

For Linux + RTX 5080 setup details, see
[`docs/INSTALL_QWEN_LINUX.md`](docs/INSTALL_QWEN_LINUX.md).

In another terminal, confirm the OpenAI-compatible vLLM endpoint is reachable:

```bash
curl http://localhost:8000/v1/models
```

## Run The Full Pipeline

First run a small smoke test on any dataset:

```bash
cd ~/black-smash-subtask-annotation

PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_current DATASET_ID=current EPS=0,1,2 \
bash scripts/run_annotation_pipeline.sh
```

Then run the full dataset:

```bash
PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_current DATASET_ID=current \
bash scripts/run_annotation_pipeline.sh
```

The local 05/06/07 datasets used for our experiments can be reproduced with:

```bash
PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_05 DATASET_ID=05 \
bash scripts/run_annotation_pipeline.sh

PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_06 DATASET_ID=06 \
bash scripts/run_annotation_pipeline.sh

PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh
```

Useful switches:

```bash
# subset of episodes
EPS=0,1,2 DATASET_ROOT=~/black_smash_current DATASET_ID=current \
bash scripts/run_annotation_pipeline.sh

# reuse existing annotations and only redraw visualization
RUN_STATE=0 RUN_QWEN=0 RUN_FUSED=0 RUN_QWEN_STAGE=0 RUN_VIZ=1 \
DATASET_ROOT=~/black_smash_current DATASET_ID=current \
bash scripts/run_annotation_pipeline.sh

# skip qwen-stage and draw only state / qwen / fused
RUN_QWEN_STAGE=0 DATASET_ROOT=~/black_smash_current DATASET_ID=current \
bash scripts/run_annotation_pipeline.sh

# disable state-guided p2 temporal refinement for a pure-Qwen baseline
QWEN_P2_HISTORY=0 DATASET_ROOT=~/black_smash_current DATASET_ID=current \
bash scripts/run_annotation_pipeline.sh
```

## Check Outputs

For any dataset id, check the generated outputs with:

```bash
id=current
echo "dataset ${id}"
ls annotations_fused_${id}/ep*_subtasks.json | wc -l
latest_stage_run=$(ls -td annotations_qwen_stage_${id}/run_* | head -1)
wc -l "${latest_stage_run}/stage_annotations_normalized.jsonl"
jq '.count' compare_tracks_${id}/summary.json
```

For the local 05/06/07 experiment, the expected counts are:

```text
05: 232 episodes
06: 50 episodes
07: 100 episodes
```

A quick count check for 05/06/07:

```bash
for id in 05 06 07; do
  echo "dataset ${id}"
  ls annotations_fused_${id}/ep*_subtasks.json | wc -l
  latest_stage_run=$(ls -td annotations_qwen_stage_${id}/run_* | head -1)
  wc -l "${latest_stage_run}/stage_annotations_normalized.jsonl"
  jq '.count' compare_tracks_${id}/summary.json
done
```

Each fused JSON should contain 6 ordered critical points and 7 subtasks. Each
qwen-stage normalized JSONL row should contain 7 semantic stages fixed to the
fused boundaries.

## Visualization

`compare_tracks_<id>/index.html` shows one row each for:

```text
state / qwen / fused / qwen-stage
```

You can also call the visualizer directly:

```bash
python visualize_annotation_tracks.py \
  --data ~/black_smash_current/data/chunk-000 \
  --state annotations_state_current \
  --qwen annotations_qwen_current \
  --fused annotations_fused_current \
  --stage-jsonl annotations_qwen_stage_current/run_xxx/stage_annotations_normalized.jsonl \
  --stage-label qwen-stage \
  --out compare_tracks_current
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
DATASET_ROOT=~/black_smash_current \
META_ROOT=~/black_smash_current/meta \
OUT_ROOT=annotations_gemini_stage_current \
FRAME_SAMPLING=uniform7 \
SIGNAL_DETAIL=compact \
PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
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
| `docs/P2_ORIENTATION_OPTIMIZATION.md` | notes on the experimental state-only p2 detector |
| `ego_wipe_annotation/` | separate subproject — ego *wipe-tube* subtask annotation for raw Quest demo folders (signal + ego/fisheye visualization); see its README |

## Current Local Status

The latest local pipeline is configured for Qwen2.5-VL-7B-AWQ served by vLLM as
`qwen`. The latest full regeneration completed for datasets 05, 06, and 07,
including state, qwen, fused, qwen-stage, and same-image visualizations. To
refresh the artifacts, rerun the pipeline commands above.

An experimental state-orientation detector for `p2_start_pour` was tested but is
not part of the production pipeline. It can detect early tube motion instead of
the true pour onset, so fused remains the recommended boundary source.

## Citation

```bibtex
@misc{subtask_segmentation,
  title  = {Proprioception-Grounded Subtask Segmentation for Long-Horizon Bimanual Manipulation},
  author = {Zhang, Rongxuan},
  year   = {2026},
  howpublished = {\url{https://github.com/Jerryzhang258/black-smash-subtask-annotation}}
}
```
