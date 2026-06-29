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

## EventVLA 关键帧训练流程

EventVLA 训练需要在每条 episode 的 `meta/episodes.jsonl` 里提供关键帧元数据：

```json
{
  "episode_index": 0,
  "length": 774,
  "keyframe_steps": [184, 215, 281, 490, 545, 680]
}
```

当前 black-smash 标注流程已经会为每条 episode 预测 6 个关键点：

```text
p1 grasp_tube
p2 start_pour
p3 release_tube
p4 grasp_pestle
p5 start_grind
p6 lift_pestle
```

这些关键点可以直接导出为 EventVLA 使用的 `keyframe_steps`。

### 第一轮弱教师标注

建议第一轮先用本地 state 标注。它只读取 `observation.state`，不需要 Qwen，也不需要解码图像，
速度快、稳定，适合作为第一轮弱监督 teacher：

```bash
cd /home/hillbot/black-smash-subtask-annotation

/home/hillbot/miniforge3/envs/qwenvl/bin/python batch_annotate.py \
  --data /home/hillbot/black_smash_05/data/chunk-000 \
  --meta /home/hillbot/black_smash_05/meta/tasks.jsonl \
  --out annotations_state_05_eventvla_export \
  --fps 30
```

然后把这些关键点写入已经转换好的 EventVLA 数据集：

```bash
/home/hillbot/miniforge3/envs/qwenvl/bin/python export_eventvla_keyframes.py \
  --annotations annotations_state_05_eventvla_export \
  --dataset /home/hillbot/black_smash_05_eventvla \
  --in-place \
  --points all
```

这会更新：

```text
/home/hillbot/black_smash_05_eventvla/meta/episodes.jsonl
```

并自动保留一份备份：

```text
/home/hillbot/black_smash_05_eventvla/meta/episodes.jsonl.bak
```

state 信号对夹爪开合、抓取、释放、研磨开始/结束这类边界比较稳定。它不是完美标注，
尤其是 `p2_start_pour` 这种视觉语义更强的点，但已经足够作为 EventVLA 第一轮训练的弱
teacher。

### EventVLA 课程学习

EventVLA 可以先使用 `episodes.jsonl` 里的 teacher keyframes，然后在训练过程中逐步切换到
模型自己的关键帧预测。第一轮推荐使用下面的配置：

```yaml
framework:
  memory_buffer:
    keyframe_loss_weight: 0.5
    keyframe_positive_weight: 7.0
    keyframe_threshold: 0.5
    keyframe_predict_mode: chunk_future

    keyframe_train_memory_source: teacher_to_predict
    keyframe_eval_memory_source: predict
    keyframe_train_memory_schedule: teacher_to_predict

    keyframe_schedule_warmup_steps: 10000
    keyframe_schedule_transition_steps: 30000
    keyframe_schedule_teacher_prob_start: 1.0
    keyframe_schedule_teacher_prob_end: 0.0
```

含义是：

```text
0 - 10k steps:
  主要使用 episodes.jsonl 里的 teacher keyframes

10k - 40k steps:
  逐渐混合 teacher keyframes 和模型自己预测的 keyframes

40k steps 之后:
  主要使用模型自己预测的 keyframes
```

这样可以避免训练一开始就用随机预测出来的关键帧污染 memory，同时又能让模型逐渐学会依赖
自己的 keyframe predictor。

`keyframe_loss_weight` 控制关键帧预测 head 的 loss 占比。如果 state 标注被当作弱标签，
建议第一轮用 `0.5`，不要直接用太强的 `1.0`。但第一轮也不要设成 `0`，否则 keyframe head
学不到有效预测，后面的 `predict` 阶段就没有意义。

`keyframe_positive_weight: 7.0` 用来处理关键帧稀疏导致的正负样本不平衡。除非训练日志显示
关键帧误报或漏报特别严重，否则先保持这个值。

这里要注意：EventVLA 配置里的 `predict` 指的是 EventVLA 模型自己的 keyframe head 预测关键帧，
不是调用本仓库里的外部 Qwen 标注脚本。Qwen local verifier 是离线标注/修正工具，可以在训练前
改进 `episodes.jsonl`，但不会在 EventVLA trainer 里面一边训练一边被调用。

### 可选 Qwen 精修

如果后续需要更好的标签，可以跑完整标注流程，让 Qwen local verifier 对 state 边界做局部视觉校验：

```bash
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=/home/hillbot/black_smash_05 \
DATASET_ID=05_eventvla_full \
RUN_QWEN_STAGE=0 \
RUN_VIZ=0 \
bash scripts/run_annotation_pipeline.sh
```

然后不要再从 `annotations_state_*` 导出，而是从 fused 标签导出：

```bash
/home/hillbot/miniforge3/envs/qwenvl/bin/python export_eventvla_keyframes.py \
  --annotations annotations_fused_05_eventvla_full \
  --dataset /home/hillbot/black_smash_05_eventvla \
  --in-place \
  --points all
```

这一步会慢很多，因为 Qwen 会逐条 episode 做局部视觉校验。建议把它当作第二阶段精修，
不要把它作为第一轮训练的前置阻塞。

第一轮 checkpoint 训练出比较合理的关键帧预测能力之后，第二轮可以更依赖模型自己的预测：

```yaml
framework:
  memory_buffer:
    keyframe_train_memory_source: predict
    keyframe_train_memory_schedule: predict
    keyframe_eval_memory_source: predict
    keyframe_loss_weight: 0.2
```

推荐总流程：

```text
1. 先把 image-only LeRobot 数据转换成 EventVLA 需要的视频版 LeRobot。
2. 用 state 信号生成第一版 critical points。
3. 把 critical points 导出为 episodes.jsonl 里的 keyframe_steps。
4. 用 teacher_to_predict 课程学习训练 EventVLA。
5. 可选：再跑 Qwen/fused 标注，对 labels 做精修后继续训练或微调。
6. 后续训练轮次逐渐更多依赖 predict，也就是模型自己的关键帧预测。
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
| `export_eventvla_keyframes.py` | export critical points to EventVLA `keyframe_steps` |
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
