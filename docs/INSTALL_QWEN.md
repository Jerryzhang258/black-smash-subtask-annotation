# 本地 Qwen2.5-VL-7B-AWQ 安装教程

本文档用于在 Windows 11 + RTX 5080 上安装当前流水线使用的本地
Qwen2.5-VL-7B-AWQ。Linux + vLLM 推荐路线见
[`INSTALL_QWEN_LINUX.md`](INSTALL_QWEN_LINUX.md)。

当前仓库默认使用：

```text
Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

## 1. 环境

RTX 5080 是 Blackwell 架构，算力 `sm_120`。Windows 上需要支持 CUDA
12.8 的驱动和 cu128 PyTorch。

```powershell
conda create -n qwenvl python=3.11 -y
conda activate qwenvl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow pandas pyarrow numpy autoawq modelscope
```

验证：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

期望看到 `cuda True` 和 `(12, 0)`。

## 2. 下载模型

```powershell
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local_dir D:\models\Qwen2.5-VL-7B-Instruct-AWQ
```

## 3. Smoke Test

```powershell
python test_qwen_vl.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ
```

看到模型加载、推理输出和 `OK` 即通过。

## 4. 跑标注

Windows 上可以用 transformers 直跑少量 episode：

```powershell
python vlm_annotate.py `
  --backend qwen-local `
  --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ `
  --data C:\Intern\black_smash_07\data\chunk-000 `
  --out annotations_qwen_07 `
  --eps 0,1,2 `
  --n-frames 32
```

全量推荐 Linux + vLLM，本仓库当前流水线默认连接
`http://localhost:8000/v1` 的本地服务。

## 常见问题

| 现象 | 解决 |
|---|---|
| `no kernel image is available for execution on the device` | torch 不是 cu128 版，按第 1 步重装。 |
| `torch.cuda.is_available()` 为 False | 驱动太旧，或装了 CPU 版 torch。 |
| 找不到 `Qwen2_5_VLForConditionalGeneration` | 升级 `transformers>=4.49.0`。 |
| CUDA out of memory | 使用 AWQ 模型，降低 `--n-frames` 或图片尺寸。 |
| 模型下载慢 | 使用 ModelScope 或 HuggingFace 镜像。 |
