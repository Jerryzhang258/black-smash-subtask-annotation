# 本地 Qwen2.5-VL 安装教程（用于 Stage 1 视觉标注）

本文档面向**另一台带 NVIDIA RTX 5080（16 GB 显存）的台式机**，从零搭好运行 **Qwen2.5-VL** 的环境，作为标注流水线的 **第 1 步（VLM 粗标）**。

> 流水线总览：**Stage 1 Qwen2.5-VL 视觉粗标 → Stage 2 state 信号精标（`batch_annotate.py`）→ Stage 3 人工复查（`annotate_gui.py`）**。
> 本文只负责把 Stage 1 的环境装好并验证通过；`vlm_annotate.py`（实际标注脚本）随后单独提交，CLI 见文末「第 8 步」。

适用系统：**Windows 11 + Miniconda**（与笔记本端一致）。Linux 用户见每步的注释，差异很小。

---

## 0. 显存与模型选择（重要，先读）

RTX 5080 是 **Blackwell 架构，算力 `sm_120`（compute capability 12.0）**。这点决定了 **必须装 CUDA 12.8 版本的 PyTorch（cu128）**，否则会报 `no kernel image is available for execution on the device`。

16 GB 显存放不下 7B 的 **全精度（bf16）** 权重（约 16.5 GB，再加视觉 token 的 KV cache 必爆显存）。因此：

| 模型 | 精度 | 权重显存 | 16GB 是否可行 | 说明 |
|---|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | 4-bit | ~7 GB | ✅ **推荐** | 质量最好且留足显存给多帧图像 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | bf16 | ~7 GB | ✅ 最稳 | 不依赖任何量化库，装起来零踩坑 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | bf16 | ~16.5 GB | ❌ | 16GB 会 OOM（需 ≥24GB 或 CPU offload） |

**建议路线：**
- 想要质量 → 走 **7B-AWQ**（需要 `autoawq`，见第 4 步）。
- 想要零踩坑、先跑通 → 走 **3B**（跳过第 4 步）。

两个都先下也行（见第 5 步），`vlm_annotate.py` 用 `--model` 切换。

---

## 1. 前置：驱动与 Miniconda

1. **NVIDIA 驱动**：装支持 CUDA 12.8 的驱动（Windows 驱动版本 **≥ 572.xx**）。装好后开 PowerShell 跑：
   ```powershell
   nvidia-smi
   ```
   右上角 `CUDA Version:` 应 **≥ 12.8**。看到 `RTX 5080` 和显存即正常。
   > 这里的 CUDA Version 是「驱动支持的最高版本」，不需要单独装 CUDA Toolkit——PyTorch 的 cu128 wheel 自带运行时。

2. **Miniconda**：从 https://docs.conda.io/en/latest/miniconda.html 装好。后续命令都在 **Anaconda PowerShell Prompt** 里执行。

---

## 2. 创建 conda 环境

```powershell
conda create -n qwenvl python=3.11 -y
conda activate qwenvl
```

> 国内 pip 加速（可选，强烈建议）：
> ```powershell
> pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
> ```

---

## 3. 安装 PyTorch（cu128，Blackwell 关键步骤）

⚠️ **不要** `pip install torch`（默认装到 CPU 版或低 CUDA 版，5080 跑不了）。必须指定 cu128 源：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**验证 GPU 可用 + 识别到 sm_120：**
```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('dev', torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability(0))"
```
期望输出里 `cuda True`、`dev ... RTX 5080`、`cap (12, 0)`。
如果 `cap (12, 0)` 但跑模型报 `no kernel image`，说明 torch 不是 cu128 版——卸载重装本步。

---

## 4. 安装推理依赖

**通用依赖（两种模型都要）：**
```powershell
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow pandas pyarrow numpy
```
> Qwen2.5-VL 需要 `transformers>=4.49.0`（更早版本没有 `Qwen2_5_VLForConditionalGeneration`）。装最新稳定版即可。

**仅当走 7B-AWQ 时，额外装量化库：**
```powershell
pip install autoawq
```
> `autoawq` 在 Windows + Blackwell 上偶有兼容问题。装不上 / 加载报错就改走 3B（本步跳过）。装完务必重新验证 `torch.cuda.is_available()` 仍为 `True`（autoawq 有时会把 torch 降级，若被降级就回到第 3 步重装 cu128 torch）。

**（可选）flash-attention**：Windows 上编译很麻烦，**不装**。脚本默认用 `attn_implementation="sdpa"`，速度足够。

---

## 5. 下载 Qwen2.5-VL 模型

国内直连 HuggingFace 慢，推荐 **ModelScope（魔搭，国内快）** 或 **hf-mirror 镜像**，二选一。

### 方式 A：ModelScope（推荐，国内）
```powershell
pip install modelscope
# 7B-AWQ（推荐）
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local_dir D:\models\Qwen2.5-VL-7B-Instruct-AWQ
# 或 3B（最稳）
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct      --local_dir D:\models\Qwen2.5-VL-3B-Instruct
```

### 方式 B：HuggingFace 镜像
```powershell
pip install -U "huggingface_hub[cli]"
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local-dir D:\models\Qwen2.5-VL-7B-Instruct-AWQ
```

记下本地路径（如 `D:\models\Qwen2.5-VL-7B-Instruct-AWQ`），后面 `--model` 直接传**本地目录**最稳（不再联网）。

---

## 6. 拷贝数据（硬盘搬过去）

把笔记本上的 LeRobot parquet 数据拷到台式机。脚本只需要 **parquet 数据目录** 和 **meta**：

```
<任意位置>\black_smash_07\
├── data\chunk-000\episode_000000.parquet   ← 标注读这些
├── data\chunk-000\episode_000001.parquet
├── ...
└── meta\tasks.jsonl                         ← 读任务描述（可选）
```

建议直接拷到台式机的 `C:\Intern\black_smash_07\`（与笔记本同路径，省得改参数）。也可放任意盘，运行时用 `--data D:\xxx\black_smash_07\data\chunk-000` 指定。

> 只标注前 N 集的话，只拷对应的 `episode_0000NN.parquet` 即可，不必全拷。

---

## 7. 克隆代码仓库

```powershell
cd C:\
git clone https://github.com/Jerryzhang258/black-smash-subtask-annotation.git Intern
cd Intern
```
（已有仓库就 `git pull` 拉最新。）

---

## 8. 验证安装（务必先跑通这步）

仓库里有 `test_qwen_vl.py`，会自动造一张测试图、加载模型、跑一次推理，打印显存占用与输出，用来确认环境 OK。

```powershell
conda activate qwenvl
# 用你第 5 步下载的本地模型路径
python test_qwen_vl.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ
# 3B：
python test_qwen_vl.py --model D:\models\Qwen2.5-VL-3B-Instruct
```

看到类似下面就算通过：
```
[env] torch 2.x.x+cu128 | cuda True | RTX 5080 | cap (12, 0)
[load] model loaded in 18.3s, VRAM allocated 7.1 GB
[infer] output: This image shows ...
OK ✅
```

---

## 9. 运行 Stage 1（`vlm_annotate.py` —— 随后提交）

> 说明：实际标注脚本 `vlm_annotate.py` 我会在环境验证通过后单独提交。预期用法如下（CLI 可能微调，以仓库 README 为准）：

```powershell
python vlm_annotate.py `
  --backend qwen-local `
  --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ `
  --data  C:\Intern\black_smash_07\data\chunk-000 `
  --out   C:\Intern\mvt_annotations_vlm `
  --eps   0,1,2 `
  --n-frames 32
```
输出与现有 state 标注同 schema（5 临界点 / 6 子任务），写到 `mvt_annotations_vlm/`，供 Stage 2、Stage 3 对照。

跑完后接：
```powershell
# Stage 2：state 信号精标
& "C:\Users\jerry\miniconda3\envs\vlm\python.exe" batch_annotate.py --eps 0,1,2
# Stage 3：人工复查（GUI 会同时显示 VLM 与 state 两条参考时间线）
& "C:\Users\jerry\miniconda3\envs\vlm\python.exe" annotate_gui.py --ep 0
```

---

## 10. 常见问题排查

| 现象 | 原因 / 解决 |
|---|---|
| `no kernel image is available for execution on the device` | torch 不是 cu128 版。卸载后按第 3 步用 `--index-url .../cu128` 重装。 |
| `torch.cuda.is_available()` 为 False | 驱动太旧（升到 ≥572）；或装了 CPU 版 torch；或 autoawq 把 torch 降级了——重装 cu128 torch。 |
| `KeyError: 'qwen2_5_vl'` / 找不到 `Qwen2_5_VLForConditionalGeneration` | transformers 太旧，`pip install -U "transformers>=4.49.0"`。 |
| CUDA out of memory | 别用 7B 全精度；改 7B-AWQ 或 3B；或在 `vlm_annotate.py` 里调小 `--n-frames` / 降低图像分辨率（`max_pixels`）。 |
| 模型下载很慢 / 连不上 | 用第 5 步的 ModelScope 或 `HF_ENDPOINT=https://hf-mirror.com`。 |
| autoawq 装不上 | 直接走 3B（`Qwen/Qwen2.5-VL-3B-Instruct`），跳过第 4 步的 autoawq。 |
| 首次推理卡很久 | 正常，模型加载 + CUDA 编译核函数；之后会快。 |

---

### 附：最小依赖清单（速查）
```
python 3.11
torch / torchvision      (cu128, 必须)
transformers >= 4.49.0
accelerate
qwen-vl-utils
pillow pandas pyarrow numpy
autoawq                  (仅 7B-AWQ 需要)
modelscope               (下载模型用，可选)
```
