# 在 Linux + RTX 5080 上跑 Qwen2.5-VL-7B-AWQ

面向 **Linux (Ubuntu 22.04/24.04) + NVIDIA RTX 5080 (16 GB)** 跑 **Qwen2.5-VL-7B-AWQ**，作为标注流水线第 1 步 (VLM 粗标)。

> 当前完整流水线见 [README](../README.md)；Windows 版见 [`INSTALL_QWEN.md`](INSTALL_QWEN.md)。

RTX 5080 是 **Blackwell (`sm_120`)**。实测 **vLLM 0.23 + FlashInfer 需要 CUDA ≥ 12.9**，请用 **PyTorch CUDA 13.0 (`cu130`)** 全栈，不要用文档旧版的 cu128。16 GB 显存放不下 7B 全精度，用 **AWQ 4-bit (~7 GB)**。

---

## 0. 两条路线

| 路线 | 适合 | 说明 |
|---|---|---|
| **A. vLLM + openai 后端** (推荐) | 跑全部 100 集 | 吞吐高；`vlm_annotate.py --backend openai` 连本地服务 |
| **B. transformers 直跑** | 少量集 | `--backend qwen-local`，慢但省事 |

---

## 1. 环境

```bash
nvidia-smi   # 驱动 ≥ 572，能看到 RTX 5080
conda create -n qwenvl python=3.11 -y
conda activate qwenvl
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/   # 国内可选
```

## 2. PyTorch (CUDA 13.0)

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 期望: 2.11.0+cu130  13.0  True  (12, 0)
```

> 若 `torchaudio` 报 CUDA 版本不匹配：`pip install torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps` 后，再重装 cu130 版 `torchaudio==2.11.0`。

## 3. vLLM + 依赖

```bash
pip install vllm qwen-vl-utils pandas pyarrow pillow numpy modelscope
```

对齐 FlashInfer JIT 用的 CUDA 工具链 (pip 会混装 13.0/13.2，需手动对齐到 13.0.88)：

```bash
pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*"
CU=~/miniforge3/envs/qwenvl/lib/python3.11/site-packages/nvidia/cu13
ln -sf libcudart.so.13 $CU/lib/libcudart.so
ln -sfn lib $CU/lib64
```

## 4. 模型与数据

```bash
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --local_dir ~/models/Qwen2.5-VL-7B-Instruct-AWQ
# 数据放 ~/black_smash_07/data/chunk-000/episode_*.parquet
```

---

## 路线 A：vLLM 服务 (推荐)

**终端 1 — 起服务：**

```bash
conda activate qwenvl
./scripts/start_vllm.sh
# 或手动:
# export CUDA_HOME=~/miniforge3/envs/qwenvl/lib/python3.11/site-packages/nvidia/cu13
# export PATH=~/miniforge3/envs/qwenvl/bin:$CUDA_HOME/bin:$PATH
# vllm serve ~/models/Qwen2.5-VL-7B-Instruct-AWQ \
#   --served-model-name qwen --quantization awq_marlin \
#   --max-model-len 32768 --limit-mm-per-prompt '{"image":40}' \
#   --gpu-memory-utilization 0.92
```

> vLLM 0.23 的 `--limit-mm-per-prompt` 用 JSON：`'{"image":40}'`（旧版 `image=40` 会报错）。
> 首次启动 FlashInfer 会 JIT 编译，约 1–3 分钟；服务监听 `http://localhost:8000/v1`。

**终端 2 — 跑标注：**

```bash
conda activate qwenvl
cd ~/black-smash-subtask-annotation
python vlm_annotate.py --backend openai --model qwen \
  --base-url http://localhost:8000/v1 \
  --data ~/black_smash_07/data/chunk-000 \
  --out mvt_annotations_vlm --eps 0,1,2 --n-frames 32
# 跑通后去掉 --eps 标全部 100 集
```

## 路线 B：transformers 直跑

```bash
pip install "transformers>=4.49.0" accelerate autoawq
python vlm_annotate.py --backend qwen-local \
  --model ~/models/Qwen2.5-VL-7B-Instruct-AWQ \
  --data ~/black_smash_07/data/chunk-000 \
  --out mvt_annotations_vlm --eps 0,1,2 --n-frames 32
```

---

## 5. 完整流水线

```bash
PYTHON_BIN=~/miniforge3/envs/qwenvl/bin/python \
DATASET_ROOT=~/black_smash_07 DATASET_ID=07 \
bash scripts/run_annotation_pipeline.sh
```

这会生成 state、qwen、fused、qwen-stage 四类标注，以及
`compare_tracks_07/index.html` 同图对比可视化。

## 6. 常见问题

| 现象 | 解决 |
|---|---|
| `SM 12.x requires CUDA >= 12.9` | 换 cu130 torch，别用 cu128 |
| `libcudart.so.13: cannot open` | `export LD_LIBRARY_PATH=.../nvidia/cu13/lib:$LD_LIBRARY_PATH` 或直接用 `start_vllm.sh` |
| `Could not find nvcc` | 设 `CUDA_HOME=.../nvidia/cu13`，`PATH` 加 `$CUDA_HOME/bin` |
| `No such file or directory: 'ninja'` | `PATH` 加 `~/miniforge3/envs/qwenvl/bin` |
| `CUDA compiler and toolkit headers are incompatible` | `pip install nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.* nvidia-cuda-crt==13.0.*` |
| `cannot find -lcudart` | 建软链：`ln -sf libcudart.so.13 $CU/lib/libcudart.so` 和 `ln -sfn lib $CU/lib64` |
| `Unsupported .version 9.2; current version is 9.0` | nvvm/crt 版本太高，降到 13.0.88 |
| vLLM 图片数超限 | 调大 `--limit-mm-per-prompt '{"image":NN}'` 或减小 `--n-frames` |
| CUDA OOM | 用 AWQ；降 `--gpu-memory-utilization` / `--max-model-len` |

## 7. 实测 (black_smash_07, ep0–2)

- 速度：~6 s/episode (vLLM, coarse+fine, 32 frames)
- 7B-AWQ 通过 vLLM 可以稳定服务本地 `qwen` 模型名
- 与 proprio 标注比仍偏粗：mean |VLM − state| ≈ 3.7 s；融合容差 0.5 s 下多数点需人工复查
- 样例见 `examples/sample_ep000_vlm_subtasks.json`
