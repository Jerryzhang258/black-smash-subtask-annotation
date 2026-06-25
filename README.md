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
state boundaries -> visual p2/local checks -> fused boundaries
                 -> qwen-stage semantic layer -> same-image visualization
```

The main outputs are:

| output | role |
|---|---|
| `annotations_state_<id>/` | state-only boundaries from `observation.state` |
| `qwen_local_verify_<id>/` | Qwen local visual verifier evidence and p2 proposal |
| `annotations_fused_<id>/` | per-point fused boundary labels for training/review |
| `annotations_qwen_stage_<id>/` | Qwen semantic descriptions using fused boundaries |
| `compare_tracks_<id>/` | same-image visual comparison: state / verifier / fused / qwen-stage |

Generated annotation and visualization folders are ignored by git because they
are regenerable data artifacts.

## Why This Design

State signals are reliable for gripper and motion boundaries, while camera-only
boundary localization is coarse on this low-light fisheye footage. The fused
labels therefore keep state as the stable skeleton and let Qwen help with the
visual `p2_start_pour` event and with semantic review.

The removed Qwen raw pass asked the VLM to predict all 6 critical points over
the whole episode. It was not reliable enough to keep as a pipeline path: it
often collapsed later critical points together or drifted far down the episode.
The current approach is a local verifier: start from the state critical points,
show Qwen a small contact sheet around one event, and only allow a conservative
local correction, mainly for `p2_start_pour`.

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
qwen_local_verify_<id>/
annotations_fused_<id>/
annotations_qwen_stage_<id>/
compare_tracks_<id>/
```

For training, use `annotations_fused_<id>/` as the boundary labels. For semantic
stage descriptions, use `annotations_qwen_stage_<id>/`. Gemini is kept only as
an optional comparison baseline.

## Current Best Practice

Use this hierarchy:

```text
state cps as the skeleton
Qwen3-VL-8B local verifier for p2_start_pour only
qwen-stage for interval semantics after fused boundaries are fixed
```

The local verifier is much more stable than raw global VLM boundary prediction
because it turns the VLM task from global time localization into local visual
choice.

Recommended conservative fusion rule for the verifier:

```text
if abs(p2_delta) <= 20 frames: accept automatically
if abs(p2_delta) > 20 frames: reject and keep state
```

The verifier should keep `p1`, `p3`, `p4`, `p5`, and `p6` as state-owned by
default. Qwen evidence for those points is still useful for review, but should
not automatically overwrite state boundaries.

## Setup

The Qwen server exposes an OpenAI-compatible endpoint at:

```text
http://localhost:8000/v1
```

For the current local-verifier pipeline, Qwen3-VL-8B-Instruct-GGUF Q4_K_M was
served with llama.cpp on the RTX 5080:

```text
model: /home/hillbot/models/Qwen3-VL-8B-Instruct-GGUF-Q4_K_M/Qwen3VL-8B-Instruct-Q4_K_M.gguf
mmproj: /home/hillbot/models/Qwen3-VL-8B-Instruct-GGUF-Q4_K_M/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
endpoint: http://localhost:8000/v1
served name: qwen
ctx: 32768
```

In another terminal, confirm the OpenAI-compatible endpoint is reachable:

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

## Qwen Local Verifier Results

The recent 8B local-verifier runs used these parameters on datasets 05 and 06:

```text
model: Qwen3-VL-8B-Instruct-GGUF:Q4_K_M
window: +/- 2.0s around the state critical point
candidates: 7 frames
image size: 192
automatic move target: p2_start_pour only
max accepted move in the experiment: 1.0s
```

Dataset 07 was run after tightening the p2 prompt and reducing the automatic
move limit to 0.67s, or about 20 frames. This is the preferred setting for the
next pass because it blocks the more aggressive early-pour moves.

Outputs were written outside the repository:

```text
/tmp/qwen_local_verify_05_full
/tmp/qwen_local_verify_06_full
/tmp/qwen_local_verify_07_opt
```

Run summary:

| dataset | episodes | p2 accepted | p2 kept | p2 large rejected | accepted mean delta | accepted median delta | accepted range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 05 | 232 | 154 | 73 | 5 | -7.23 frames | -7.5 frames | -25 to +12 |
| 06 | 50 | 37 | 9 | 4 | -12.76 frames | -15 frames | -27 to +6 |
| 07 opt | 100 | 66 | 31 | 3 | -14.26 frames | -20 frames | -20 to +7 |

Compared with the removed 7B raw critical-point baseline, the local verifier is
much more stable for `p2_start_pour`:

| dataset | method | p2 mean absolute delta vs state | p2 large drift |
|---|---|---:|---:|
| 05 | Qwen2.5-VL-7B raw global cps | 55.1 frames | 52 episodes over 60 frames |
| 05 | Qwen3-VL-8B local verifier | 6.6 frames | 0 episodes over 60 frames |
| 06 | Qwen2.5-VL-7B raw global cps | 43.2 frames | 5 episodes over 60 frames |
| 06 | Qwen3-VL-8B local verifier | 10.4 frames | 0 episodes over 60 frames |

The main caveat is that the 8B verifier tends to move `p2_start_pour` earlier.
Before using verifier outputs as final training labels, review examples with
`delta <= -20` and all `move_too_large` rejections. These are the likely cases
where Qwen may be selecting early tilt or first powder trace instead of the
stable pour onset.

## Visualization

`compare_tracks_<id>/index.html` shows one row each for:

```text
state / verifier / fused / qwen-stage
```

You can also call the visualizer directly:

```bash
python visualize_annotation_tracks.py \
  --data ~/black_smash_current/data/chunk-000 \
  --state annotations_state_current \
  --qwen qwen_local_verify_current/verified_annotations \
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
| `qwen_local_verify.py` | Qwen local verifier for visual p2 review and conservative proposals |
| `fuse_annotations.py` | fused labels plus disagreement/review metadata |
| `data_annotation/tools/qwen_stage_annotation_demo.py` | qwen-stage semantic layer, optionally fixed to fused boundaries |
| `visualize_annotation_tracks.py` | same-image comparison for state / qwen / fused / stage rows |
| `scripts/run_annotation_pipeline.sh` | current end-to-end pipeline |
| `annotate_gui.py` | manual review GUI |
| `docs/P2_ORIENTATION_OPTIMIZATION.md` | notes on the experimental state-only p2 detector |
| `ego_wipe_annotation/` | separate subproject — ego *wipe-tube* subtask annotation for raw Quest demo folders (signal + ego/fisheye visualization); see its README |

## Current Local Status

Qwen3-VL-8B local verifier has completed for datasets 05, 06, and 07. Dataset
07 used the tightened p2 prompt and `max_move_s=0.67`; this should be treated as
the preferred verifier setting. The Qwen3-VL-8B server is OpenAI-compatible as
`qwen` at `http://localhost:8000/v1`.

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
