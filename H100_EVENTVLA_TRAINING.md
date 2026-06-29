# H100 服务器一键部署 EventVLA 训练

本文档用于在一台 H100 80GB 服务器上，从零部署 EventVLA 训练环境，并使用已经上传到 Hugging Face 的 `black_smash_05_eventvla` 数据集开始训练。

推荐服务器：

- GPU: 1x H100 80GB
- CPU: 16 核以上
- RAM: 128GB 以上
- Disk: 1TB NVMe
- OS: Ubuntu 22.04
- CUDA Driver: 能支持 CUDA 12.x 或 13.x

## 0. 目标目录

建议服务器目录结构如下：

```text
/workspace/EventVLA-UMI
/workspace/data/black_smash_05_eventvla
/workspace/models/Qwen3-VL-4B-Instruct
/workspace/results
```

## 1. 重要前提：先同步本地 EventVLA 适配

本地训练代码已经做过 black-smash 适配，服务器必须同步这些改动，否则只拉官方 EventVLA 仓库会找不到：

```text
eventvla_black_smash
bimanual_cartesian_delta_20d
Qwen2.5-VL hidden_size compatibility
Qwen2.5-VL attn_implementation override
```

本地涉及文件：

```text
EventVLA/eventvla/dataloader/gr00t_lerobot/data_config.py
EventVLA/eventvla/dataloader/gr00t_lerobot/embodiment_tags.py
EventVLA/eventvla/dataloader/gr00t_lerobot/mixtures.py
EventVLA/eventvla/model/modules/vlm/QWen2_5.py
```

推荐做法：先把本地 `/home/hillbot/EventVLA-UMI` 的改动 commit 并 push 到你自己的 GitHub 分支，然后服务器从这个分支 clone。

检查本地改动：

```bash
cd /home/hillbot/EventVLA-UMI/EventVLA
git diff --name-only
git diff
```

提交到你自己的分支后，下面一键脚本里的 `EVENTVLA_REPO` 和 `EVENTVLA_BRANCH` 要指向那个分支。

## 2. 一键部署脚本

在 H100 服务器上新建脚本：

```bash
nano bootstrap_h100_eventvla.sh
```

写入下面内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKDIR/EventVLA-UMI}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data}
MODEL_ROOT=${MODEL_ROOT:-$WORKDIR/models}
RESULT_ROOT=${RESULT_ROOT:-$WORKDIR/results}

EVENTVLA_REPO=${EVENTVLA_REPO:-https://github.com/wjstx0425/EventVLA-UMI.git}
EVENTVLA_BRANCH=${EVENTVLA_BRANCH:-main}
DATASET_REPO=${DATASET_REPO:-Aether258/black_smash_05_eventvla}
BASE_VLM_REPO=${BASE_VLM_REPO:-Qwen/Qwen3-VL-4B-Instruct}

CONDA_ENV=${CONDA_ENV:-eventvla}

mkdir -p "$WORKDIR" "$DATA_ROOT" "$MODEL_ROOT" "$RESULT_ROOT"
cd "$WORKDIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda or Miniforge first."
  exit 1
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$EVENTVLA_BRANCH" "$EVENTVLA_REPO" "$REPO_DIR"
fi

if [ -d "$REPO_DIR/EventVLA" ]; then
  APP_DIR="$REPO_DIR/EventVLA"
else
  APP_DIR="$REPO_DIR"
fi

cd "$APP_DIR"

conda create -y -n "$CONDA_ENV" python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip

# H100 推荐使用官方 CUDA wheel。若服务器镜像已有合适 PyTorch，可跳过或按实际 CUDA 版本调整。
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# EventVLA 训练常用依赖。若仓库提供 requirements.txt/pyproject.toml，优先使用仓库自己的依赖。
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
pip install -e .

pip install \
  accelerate \
  deepspeed \
  datasets \
  huggingface_hub \
  transformers \
  qwen-vl-utils \
  decord \
  pyav \
  omegaconf \
  wandb \
  albumentations \
  numpydantic

# H100 上建议使用 flash-attn。如果安装失败，可以先把训练命令里的 attn_implementation 改成 eager。
pip install flash-attn --no-build-isolation || true

echo "Downloading dataset: $DATASET_REPO"
huggingface-cli download "$DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/black_smash_05_eventvla" \
  --local-dir-use-symlinks False

echo "Downloading base VLM: $BASE_VLM_REPO"
huggingface-cli download "$BASE_VLM_REPO" \
  --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --local-dir-use-symlinks False

echo "Bootstrap done."
echo "Repo:    $REPO_DIR"
echo "App:     $APP_DIR"
echo "Data:    $DATA_ROOT/black_smash_05_eventvla"
echo "Model:   $MODEL_ROOT/Qwen3-VL-4B-Instruct"
echo "Results: $RESULT_ROOT"
```

运行：

```bash
chmod +x bootstrap_h100_eventvla.sh
./bootstrap_h100_eventvla.sh
```

如果 Hugging Face 下载私有模型或私有数据，需要先登录：

```bash
huggingface-cli login
```

不要把 token 写进脚本。

## 3. 确认数据格式

下载完成后检查：

```bash
ls /workspace/data/black_smash_05_eventvla
ls /workspace/data/black_smash_05_eventvla/meta
```

应该能看到：

```text
data/
videos/
meta/info.json
meta/modality.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/stats_gr00t.json
meta/eventvla_keyframe_export_summary.json
```

确认关键帧已经写入：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("/workspace/data/black_smash_05_eventvla/meta/episodes.jsonl")
rows = [json.loads(line) for line in p.open()]
print("episodes:", len(rows))
print("episodes_with_keyframes:", sum(bool(r.get("keyframe_steps")) for r in rows))
print("first:", rows[0])
PY
```

期望：

```text
episodes: 232
episodes_with_keyframes: 232
```

## 4. 训练配置要点

当前数据集的关键配置：

```text
robot_type: bimanual_cartesian_delta_20d
action_dim: 20
state_dim: 20
views:
  observation.images.left_wrist
  observation.images.right_wrist
```

训练时建议先使用单视角 `left_wrist`：

```bash
--datasets.vla_data.keyframe_image_memory.include_names '[left_wrist]'
```

如果后面要同时用双腕相机，需要确认 EventVLA 的 keyframe memory 配置支持多视角，再把 `strict_single_view` 关掉。

关键帧课程学习第一轮推荐：

```bash
--framework.memory_buffer.keyframe_loss_weight 0.5
--framework.memory_buffer.keyframe_train_memory_source teacher_to_predict
--framework.memory_buffer.keyframe_eval_memory_source predict
--framework.memory_buffer.keyframe_train_memory_schedule teacher_to_predict
--framework.memory_buffer.keyframe_schedule_warmup_steps 10000
--framework.memory_buffer.keyframe_schedule_transition_steps 30000
--framework.memory_buffer.keyframe_schedule_teacher_prob_start 1.0
--framework.memory_buffer.keyframe_schedule_teacher_prob_end 0.0
```

含义：

```text
0 - 10k steps:
  使用 episodes.jsonl 里的 teacher keyframes

10k - 40k steps:
  teacher keyframes 和模型预测 keyframes 混合

40k steps 之后:
  主要使用模型自己的 keyframe head 预测
```

## 5. 单卡 H100 启动训练

进入仓库：

```bash
cd /workspace/EventVLA-UMI/EventVLA
conda activate eventvla
```

如果用 Qwen3-VL-4B-Instruct：

```bash
export EVENTVLA_BLACK_SMASH_DATASETS=black_smash_05_eventvla
export WANDB_PROJECT=eventvla_black_smash

accelerate launch \
  --config_file eventvla/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 38569 \
  --num_processes 1 \
  eventvla/training/train_eventvla.py \
  --config_yaml ./examples/UMI/train_files/eventvla_umi_black_smash.yaml \
  --framework.name EventVLA \
  --framework.qwenvl.base_vlm /workspace/models/Qwen3-VL-4B-Instruct \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir /workspace/data \
  --datasets.vla_data.data_mix eventvla_black_smash \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.num_workers 8 \
  --datasets.vla_data.keyframe_image_memory.include_names '[left_wrist]' \
  --framework.action_model.action_dim 20 \
  --framework.action_model.state_dim 20 \
  --framework.action_model.action_horizon 50 \
  --framework.action_model.future_action_window_size 49 \
  --framework.memory_buffer.keyframe_loss_weight 0.5 \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.keep_recent_checkpoints 2 \
  --run_root_dir /workspace/results \
  --run_id black_smash_05_eventvla_h100
```

如果 `flash_attn` 安装失败，先改成：

```bash
--framework.qwenvl.attn_implementation eager
```

但 H100 上正式训练更推荐修好 `flash-attn`。

## 6. 先跑 smoke test

正式长训前建议先跑 10 step：

```bash
accelerate launch \
  --config_file eventvla/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 38569 \
  --num_processes 1 \
  eventvla/training/train_eventvla.py \
  --config_yaml ./examples/UMI/train_files/eventvla_umi_black_smash.yaml \
  --framework.name EventVLA \
  --framework.qwenvl.base_vlm /workspace/models/Qwen3-VL-4B-Instruct \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir /workspace/data \
  --datasets.vla_data.data_mix eventvla_black_smash \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.keyframe_image_memory.include_names '[left_wrist]' \
  --framework.action_model.action_dim 20 \
  --framework.action_model.state_dim 20 \
  --framework.action_model.action_horizon 50 \
  --framework.action_model.future_action_window_size 49 \
  --framework.memory_buffer.keyframe_loss_weight 0.5 \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 10 \
  --trainer.keep_recent_checkpoints 1 \
  --run_root_dir /workspace/results \
  --run_id smoke_black_smash_05_eventvla_h100
```

smoke test 通过的标志：

- 能初始化 `black_smash_05_eventvla`
- 能读到 232 episodes
- 能打印 `Used action keys`
- 能进入 training loop
- 能产生 loss
- 能保存 checkpoint

## 7. 断点续训

假设 checkpoint 在：

```text
/workspace/results/black_smash_05_eventvla_h100/checkpoint-10000
```

续训时加上：

```bash
--trainer.is_resume true
--trainer.resume_step 10000
```

如果训练代码支持 `resume_epoch` 或 checkpoint path，也可以按仓库里的 trainer 参数继续补充。

## 8. 常见问题

### 找不到数据集 mixture

报错类似：

```text
Unknown data_mix eventvla_black_smash
```

说明 EventVLA 仓库里还没有注册 `eventvla_black_smash`。需要确认以下改动已经在服务器代码中：

- `eventvla/dataloader/gr00t_lerobot/mixtures.py`
- `eventvla/dataloader/gr00t_lerobot/data_config.py`
- `eventvla/dataloader/gr00t_lerobot/embodiment_tags.py`

本地已经为 `bimanual_cartesian_delta_20d` 和 `eventvla_black_smash` 做过适配，服务器代码要同步这些改动。

### Qwen 配置没有 hidden_size

如果使用 Qwen2.5-VL，可能报：

```text
AttributeError: 'Qwen2_5_VLConfig' object has no attribute 'hidden_size'
```

需要在 Qwen2.5 wrapper 里补：

```python
self.model.config.hidden_size = self.model.config.text_config.hidden_size
```

本地已经做过这个兼容，服务器代码也要同步。

### flash-attn 安装失败

临时方案：

```bash
--framework.qwenvl.attn_implementation eager
```

正式训练建议解决 `flash-attn`，否则速度和显存都会差一些。

### H100 仍然 OOM

优先尝试：

```bash
--datasets.vla_data.per_device_batch_size 1
--trainer.gradient_accumulation_steps 4
```

然后确认：

- 没有其它进程占 GPU
- 没有同时启动本地 Qwen server
- `max_keyframe_images` 没有设太大
- 只用一个 keyframe 视角开始训练

### 不建议用 GGUF 或 AWQ 作为训练基座

GGUF 适合 llama.cpp 推理服务，不适合作为 Transformers/EventVLA 训练基座。

AWQ 在本地 smoke test 中遇到过 kernel 兼容问题。服务器正式训练建议使用 HuggingFace Transformers 格式的 BF16 模型，例如：

```text
Qwen/Qwen3-VL-4B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct
```

## 9. 推荐训练顺序

第一阶段：

```text
H100 单卡
Qwen3-VL-4B-Instruct
left_wrist 单视角
batch_size 1 或 2
keyframe_loss_weight 0.5
max_train_steps 80000
```

第二阶段：

```text
加入 right_wrist 或多视角
调高 batch size
根据日志调 keyframe_loss_weight
观察 keyframe precision/recall 和 action loss
```

第三阶段：

```text
把 05/06/07 合并成多个 HF dataset 或同一个 data_root 下多个 dataset
用 EVENTVLA_BLACK_SMASH_DATASETS=black_smash_05_eventvla,black_smash_06_eventvla,black_smash_07_eventvla
正式长训
```

## 10. 最小检查清单

训练前确认：

- HF dataset 已上传并可下载
- 服务器代码包含 `bimanual_cartesian_delta_20d` 适配
- 服务器代码包含 `eventvla_black_smash` mixture
- `meta/episodes.jsonl` 内有 `keyframe_steps`
- 使用 HF/Transformers 格式模型，不使用 GGUF
- smoke test 先跑 10 step
- smoke test 通过后再跑 80k step
