# EventVLA 关键帧标注导出说明

这个文档说明如何把 black-smash 的子任务标注导出成 EventVLA 训练需要的关键帧格式。

核心脚本：

```text
export_eventvla_keyframes.py
```

它会把 `annotations_state_*` 或 `annotations_fused_*` 里的 6 个关键点写入 EventVLA/LeRobot 数据集的：

```text
meta/episodes.jsonl
```

写入字段：

```json
{
  "keyframe_steps": [184, 215, 281, 490, 545, 680],
  "inspect_keyframe_steps": [184, 215, 281, 490, 545, 680],
  "eventvla_keyframe_source": {
    "annotation_episode": 0,
    "points": "all",
    "critical_points": [184, 215, 281, 490, 545, 680],
    "critical_names": [
      "grasp_tube",
      "start_pour",
      "release_tube",
      "grasp_pestle",
      "start_grind",
      "lift_pestle"
    ]
  }
}
```

## 1. 输入输出

输入一：标注目录，例如：

```text
annotations_state_05_eventvla_export/
annotations_fused_05_eventvla_full/
```

每个 episode 一个文件：

```text
ep000_subtasks.json
ep001_subtasks.json
...
```

输入二：已经转换好的 EventVLA/LeRobot 数据集，例如：

```text
/home/hillbot/black_smash_05_eventvla
```

该目录需要包含：

```text
data/
videos/
meta/episodes.jsonl
meta/info.json
meta/modality.json
```

输出：

```text
<dataset>/meta/episodes.jsonl
<dataset>/meta/eventvla_keyframe_export_summary.json
```

如果使用 `--in-place`，脚本会默认备份原文件：

```text
<dataset>/meta/episodes.jsonl.bak
```

## 2. 推荐流程：state 标注导出

第一轮推荐用 state 标注作为弱 teacher：

```bash
cd /home/hillbot/black-smash-subtask-annotation

/home/hillbot/miniforge3/envs/qwenvl/bin/python batch_annotate.py \
  --data /home/hillbot/black_smash_05/data/chunk-000 \
  --meta /home/hillbot/black_smash_05/meta/tasks.jsonl \
  --out annotations_state_05_eventvla_export \
  --fps 30
```

导出到 EventVLA 数据集：

```bash
/home/hillbot/miniforge3/envs/qwenvl/bin/python export_eventvla_keyframes.py \
  --annotations annotations_state_05_eventvla_export \
  --dataset /home/hillbot/black_smash_05_eventvla \
  --in-place \
  --points all
```

验证：

```bash
/home/hillbot/miniforge3/envs/qwenvl/bin/python - <<'PY'
import json
from pathlib import Path

p = Path("/home/hillbot/black_smash_05_eventvla/meta/episodes.jsonl")
rows = [json.loads(line) for line in p.open() if line.strip()]
print("episodes:", len(rows))
print("episodes_with_keyframes:", sum(bool(r.get("keyframe_steps")) for r in rows))
print("first:", rows[0].get("keyframe_steps"))
PY
```

期望：

```text
episodes: 232
episodes_with_keyframes: 232
```

## 3. 可选流程：Qwen/fused 精修后导出

如果需要更好的视觉语义边界，可以先跑完整 pipeline 得到 fused 标注：

```bash
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=/home/hillbot/black_smash_05 \
DATASET_ID=05_eventvla_full \
RUN_QWEN_STAGE=0 \
RUN_VIZ=0 \
bash scripts/run_annotation_pipeline.sh
```

然后从 fused 结果导出：

```bash
/home/hillbot/miniforge3/envs/qwenvl/bin/python export_eventvla_keyframes.py \
  --annotations annotations_fused_05_eventvla_full \
  --dataset /home/hillbot/black_smash_05_eventvla \
  --in-place \
  --points all
```

## 4. points 选项

默认导出全部 6 个点：

```bash
--points all
```

6 个点分别是：

```text
1 grasp_tube
2 start_pour
3 release_tube
4 grasp_pestle
5 start_grind
6 lift_pestle
```

内置 preset：

```text
all         -> 1,2,3,4,5,6
eventvla4   -> 2,3,5,6
pour_grind  -> 2,5
```

也可以自定义 1-based 编号：

```bash
--points 2,3,5,6
```

第一轮训练建议使用：

```bash
--points all
```

因为 EventVLA 配置里的 `max_keyframe_images` 可以控制实际输入 memory 数量，保留完整 teacher 标签更方便后续调参。

## 5. 上传 Hugging Face 前检查

上传前建议确认：

```bash
du -sh /home/hillbot/black_smash_05_eventvla
find /home/hillbot/black_smash_05_eventvla/data -name '*.parquet' | wc -l
find /home/hillbot/black_smash_05_eventvla/videos -name '*.mp4' | wc -l
```

当前 `black_smash_05_eventvla` 应该是：

```text
232 parquet
464 mp4
232 episodes with keyframe_steps
```

不要上传本地缓存和备份：

```text
meta/episodes.jsonl.bak
meta/steps_data_index.pkl
```

## 6. 和服务器训练的关系

这个仓库负责：

```text
raw/state/fused annotation -> EventVLA keyframe metadata
```

服务器训练仓库负责：

```text
download HF dataset -> verify keyframe_steps -> train EventVLA
```

服务器训练分支：

```text
https://github.com/wjstx0425/EventVLA-UMI
branch: main
```

训练时不需要重新跑本仓库的标注脚本，除非你要从 raw 数据重新生成标签。
