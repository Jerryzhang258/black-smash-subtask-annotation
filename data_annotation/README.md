# data_annotation


## 1. 文件说明

```text
data_annotation/
  README.md
  requirements.txt
  config/
    api_env.example
  scripts/
    run_gemini_stage_annotation.sh
    run_future_verification_first50.sh
  tools/
    qwen_stage_annotation_demo.py
    postprocess_qwen_stage_results.py
    batch_self_check_predictions.py
    run_future_verification.py
  REFERENCE_SERVER_README.md
```

核心脚本：

- `tools/qwen_stage_annotation_demo.py`：主标注脚本。虽然文件名里有 qwen，但它使用 OpenAI-compatible client，可以通过 `--model`、`--api-key-env`、`--base-url-env` 调 Gemini 或 Qwen。
- `tools/postprocess_qwen_stage_results.py`：对 API 输出做 stage 名称和 interval 质量校验，生成 `stage_annotations_normalized.jsonl`。
- `tools/batch_self_check_predictions.py`：为每个 stage 生成后续图像自测样本，生成 `prediction_self_check_samples.jsonl`。
- `tools/run_future_verification.py`：调用 Gemini 对预测式 prompt 做 future verification，并计算 `quality_summary.json`。

## 2. API 密钥配置

不要把真实 key 写进 README 或提交到 git。建议在服务器上手动导出：

```bash
export TTK_API_KEY="你的 key"
export TTK_BASE_URL="https://api.ttk.homes/v1"
export TTK_MODEL="gemini-3.5-flash-low-反重力"
```

或者复制模板：

```bash
cp data_annotation/config/api_env.example data_annotation/config/api_env.local.sh
```

然后只在 `api_env.local.sh` 里填真实 key。这个文件不要分享。

## 3. 安装依赖

在项目虚拟环境中安装：

```bash
.venv/bin/pip install -r data_annotation/requirements.txt
```

如果原 VB-VLA 环境已经能运行 pandas、PIL、openai，一般不需要重复安装。

## 4. 标注流程

当前最终方案是：

1. 输入完整任务描述；
2. 输入 `camera0` 和 `camera1` 两个外部相机的全帧图像；
3. 输入左右夹爪宽度完整序列和速度序列；
4. Gemini 输出 stage interval、`prediction_prompt`、`expected_future_observation`；
5. 后处理做 interval/stage name 质量校验；
6. 生成自测样本；
7. 可选地调用 Gemini 做 future verification，得到质量分。

## 5. 一键运行标注

在服务器项目根目录运行：

```bash
bash data_annotation/scripts/run_gemini_stage_annotation.sh
```

默认参数：

- `NUM_EPISODES=270`
- `CAMERA_KEYS=observation.images.camera0,observation.images.camera1`
- `OUT_ROOT=gemini_stage_annotation_results_dual_camera`
- `MODEL=$TTK_MODEL`

可以临时覆盖：

```bash
NUM_EPISODES=10 OUT_ROOT=debug_stage_annotation \
bash data_annotation/scripts/run_gemini_stage_annotation.sh
```

运行结束后会得到：

```text
OUT_ROOT/run_xxxxxxxx_xxxxxx/
  stage_annotations.jsonl
  summary.csv
  stage_annotations_normalized.jsonl
  summary_normalized.csv
  prediction_self_check_samples.jsonl
  keyframes/
  gripper_plots/
```

## 6. Future verification

标注结束后，指定 `RUN_DIR` 跑前 50 个 episode 的自测：

```bash
RUN_DIR=gemini_stage_annotation_results_dual_camera/run_xxxxxxxx_xxxxxx \
bash data_annotation/scripts/run_future_verification_first50.sh
```

输出：

```text
RUN_DIR/future_verification_results.jsonl
RUN_DIR/quality_summary.json
```

## 7. 质量分计算

`quality_summary.json` 中的 confidence 不是 API 自报分，而是后处理质量分：

```text
stage_quality_score =
0.30 * interval_score
+ 0.30 * boundary_signal_score
+ 0.40 * future_verification_score
```

含义：

- `interval_score`：stage interval 是否完整覆盖 episode，有没有漏帧、重叠、越界等。
- `boundary_signal_score`：stage 和夹爪曲线是否匹配，例如 `grasp` 应有夹爪闭合趋势，`release/place` 应有打开趋势。
- `future_verification_score`：后续图像是否验证了 `expected_future_observation`。

## 8. 不要打包的内容

为了避免泄露或包太大，以下内容不要放进共享包：

- 真实 API key；
- `.env`、`api_env.local.sh`；
- `/root/.config/api-keys/`；
- 大规模 `keyframes/`、`gripper_plots/`、实验结果目录；
- 服务器私有日志。



