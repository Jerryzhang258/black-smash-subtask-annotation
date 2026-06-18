# 在 Linux + RTX 5080 上跑 Qwen2.5-VL-7B(Stage 1 视觉标注)

面向 **Linux(Ubuntu 22.04/24.04)+ NVIDIA RTX 5080(16 GB)** 跑 **Qwen2.5-VL-7B**,作为标注流水线的第 1 步(VLM 粗标)。

> 为什么是 7B + Linux:在 8GB 笔记本上实测 **3B 无法定位事件**(暗光鱼眼画面,3B 直接输出等间距瞎编的帧号)。**7B 才有戏**;Linux 还能用 **vLLM**(批量吞吐远高于 transformers,适合 100 集)。
> 三阶段总览见 [README](../README.md);Windows 版见 [`INSTALL_QWEN.md`](INSTALL_QWEN.md)。

RTX 5080 是 **Blackwell(算力 `sm_120`)**,**必须 CUDA 12.8 的 PyTorch(cu128)**。16 GB 放不下 7B 全精度(~16.5 GB),用 **AWQ 4-bit(~7 GB)**。

---

## 0. 两条路线,二选一

| 路线 | 适合 | 说明 |
|---|---|---|
| **A. vLLM 服务 + openai 后端**(推荐) | 跑全部 100 集 | 吞吐高、并发好;起一个本地服务,`vlm_annotate.py --backend openai` 连它 |
| **B. transformers 直跑** | 少量集 / 想简单 | 不起服务,`--backend qwen-local`,逐集推理,慢但省事 |

---

## 1. 驱动与基础环境

```bash
nvidia-smi                       # 驱动需支持 CUDA ≥ 12.8;能看到 RTX 5080 即可
sudo apt-get update && sudo apt-get install -y git git-lfs build-essential
# Miniconda(若没有)
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
conda create -n qwenvl python=3.11 -y
conda activate qwenvl
```
> 国内 pip 加速(可选):`pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`

## 2. PyTorch(cu128,Blackwell 关键)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 期望: 2.x+cu128  True  (12, 0)
```

## 3. 拉代码 + 拷数据

```bash
git clone https://github.com/Jerryzhang258/black-smash-subtask-annotation.git
cd black-smash-subtask-annotation
# 用移动硬盘把 parquet 数据拷进来,例如:
#   black_smash_07/data/chunk-000/episode_*.parquet
#   black_smash_07/meta/tasks.jsonl
```

## 4. 下载模型(AWQ 4-bit)

国内推荐 **ModelScope**(Qwen 官方、最快):
```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --local_dir ~/models/Qwen2.5-VL-7B-Instruct-AWQ
```
或 hf 镜像:`HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local-dir ~/models/Qwen2.5-VL-7B-Instruct-AWQ`

---

## 路线 A:vLLM 服务 + openai 后端(推荐)

### A1. 装 vLLM
```bash
pip install vllm openai
# 数据预处理依赖
pip install qwen-vl-utils pillow pandas pyarrow numpy
```

### A2. 起服务(开一个终端常驻)
```bash
vllm serve ~/models/Qwen2.5-VL-7B-Instruct-AWQ \
  --served-model-name qwen \
  --quantization awq_marlin \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=40 \
  --gpu-memory-utilization 0.92
```
> `--limit-mm-per-prompt image=40` 很重要:我们一次最多发 ~32 帧,默认上限太低会报错。
> 服务起来后监听 `http://localhost:8000/v1`。

### A3. 跑 Stage 1(另一个终端)
```bash
conda activate qwenvl
python vlm_annotate.py --backend openai --model qwen \
  --base-url http://localhost:8000/v1 \
  --data black_smash_07/data/chunk-000 \
  --out mvt_annotations_vlm --eps 0,1,2 --n-frames 32
# 跑通后去掉 --eps 标全部
```

---

## 路线 B:transformers 直跑(简单)

```bash
pip install "transformers>=4.49.0" accelerate qwen-vl-utils autoawq pillow pandas pyarrow numpy
python vlm_annotate.py --backend qwen-local \
  --model ~/models/Qwen2.5-VL-7B-Instruct-AWQ \
  --data black_smash_07/data/chunk-000 \
  --out mvt_annotations_vlm --eps 0,1,2 --n-frames 32
```
> 不装 flash-attn 也行(脚本默认 `sdpa`)。`--dry-run` 可在无 GPU 时只检查抽帧/prompt。

---

## 5. 接着跑 Stage 2 / 3(任一路线之后)

```bash
python batch_annotate.py --data black_smash_07/data/chunk-000 --out mvt_annotations   # state 精标
python fuse_annotations.py --tol-s 0.5                                                 # 融合 + 标待复查点
python annotate_gui.py --ep 0                                                          # 人工复查(需图形界面)
```

## 6. 常见问题

| 现象 | 解决 |
|---|---|
| `no kernel image ... device` | torch 不是 cu128;按第 2 步重装 |
| vLLM 报图片数超限 | 调大 `--limit-mm-per-prompt image=NN`,或 `vlm_annotate.py` 调小 `--n-frames` |
| CUDA OOM | 用 AWQ(别用全精度 7B);降 `--gpu-memory-utilization`、`--max-model-len`、`--n-frames` |
| 模型下载慢/连不上 | 用 ModelScope 或 `HF_ENDPOINT=https://hf-mirror.com` |
| AWQ 加载报错(transformers 路线) | 确认装了 `autoawq`,且 torch 仍是 cu128(autoawq 有时会降级 torch) |

## 7. 调质量(7B 值得开)

- **粗→细两遍**:`vlm_annotate.py` 默认 `--fine`(每个点 ±1.5s 密集重采样精定),7B 上建议开着。
- 帧数:`--n-frames 32`(够则别更高,省显存/token)。
- 临界点归属:p2(开始倒)交给 VLM,其余以 state 为准,融合时 `|VLM−state|>容差` 的点才丢给人工(见 README)。
