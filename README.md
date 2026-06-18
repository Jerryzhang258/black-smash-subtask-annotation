# black-smash 子任务标注

把一个双臂操作数据集(LeRobot v2.1,任务 `Pour the black powder into the mortar and grind.`)的每条 episode,逐帧切成 **7 个子任务**(由 **6 个临界点**划分)。

标注用 **三阶段流水线**:两种互补的「机器视角」各标一遍 → 自动融合 → 人工只复查两者不一致的点。

![ep000 故事板](mvt_annotations/ep000_storyboard.png)

---

## 三阶段流水线

```
增强相机帧 ─► Stage 1  Qwen2.5-VL(本地)  ─► mvt_annotations_vlm/    语义粗标
state 信号 ─► Stage 2  本体信号分段        ─► mvt_annotations/        时序精标
两版+置信度 ─► (融合 fuse)                  ─► mvt_annotations_fused/   + 待复查点
6 路视频   ─► Stage 3  人工复查 GUI        ─► mvt_annotations_human/   定稿
```

**核心思路:每个临界点交给最擅长它的方法,人工只看「两法不一致」的点。**

| 临界点 | 事件 | 由谁定 | 为什么 |
|---|---|---|---|
| p1 抓到试管 | 夹爪闭(dim3) | **state** | 夹爪事件,误差≈1–2 帧 |
| p2 开始倒 | 看到倾倒/粉流 | **VLM** | 视觉事件,state 只能用低漂移凑 |
| p3 放下试管 | 夹爪开(dim3) | **state** | 夹爪事件 |
| p4 抓到杵 | 夹爪闭(dim13) | **state** | 夹爪事件(双臂 +10 镜像) |
| p5 开始磨 | 原地运动起点 | **state** | 载波漂移检测 |
| p6 抬杵 | 研磨结束 | **state** | 载波漂移检测 |

VLM 对 state 主导的点做交叉校验;`|VLM − state| > 容差(默认 0.5s)` 的点标记 `review_points`,Stage 3 只调这些。三阶段输出**同一套 JSON schema**(`critical_points`(6) + `subtask_starts`(7) + 7 段 `subtasks`),仅 `annotator` 字段不同。

## 7 个子任务

| id | 标签(写入数据) | id | 标签 |
|----|----------------|----|------|
| S0 | reach for the test tube | S4 | bring the pestle over the mortar |
| S1 | lift the test tube and move it over the mortar | S5 | grind the powder in the mortar |
| S2 | pour the black powder into the mortar | S6 | lift the pestle and return to rest |
| S3 | release the test tube and reach for the pestle | | |

标签固定在 `batch_annotate.py` 顶部 `LABELS`,三阶段共用。

---

## 用法

```powershell
# Stage 1  VLM 粗标(5080 桌机 qwenvl 环境;安装见 docs/INSTALL_QWEN.md)
python vlm_annotate.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ --eps 0,1,2
python vlm_annotate.py --dry-run --eps 0          # 无模型,只检查抽帧/prompt

# Stage 2  state 信号精标(本机 vlm 环境)
& "C:\Users\jerry\miniconda3\envs\vlm\python.exe" batch_annotate.py            # 全部
& "...python.exe" batch_annotate.py --eps 0,5,9 --storyboard                   # 子集+故事板

# 融合 -> 标出待复查点
& "...python.exe" fuse_annotations.py --tol-s 0.5

# Stage 3  人工复查(GUI 显示 VLM/state 参考线,红▲=待复查点;键 1–6 打点,s 保存)
& "...python.exe" annotate_gui.py --ep 0
```

> 本机裸 `python` 是失效的 Store stub,务必用全路径 `C:\Users\jerry\miniconda3\envs\vlm\python.exe`。

## 数据集

`black_smash_07/`:LeRobot v2.1 双臂,**100 集**,每条约 1049–1290 帧,30 fps,**不含在仓库**(已 gitignore)。
6 路图像(`camera0/1` 场景 + 4 路触觉,224×224)+ 20 维 `observation.state`/`actions`。state 双臂 +10 镜像:试管夹=dim3,杵夹=dim13。
下载:`hf download EricChen06/black_smash_07 --repo-type dataset --local-dir black_smash_07`。

## 输出

每阶段一个目录,同结构:`mvt_annotations_vlm/` · `mvt_annotations/` · `mvt_annotations_fused/` · `mvt_annotations_human/`。

| 文件 | 内容 |
|------|------|
| `ep<NNN>_subtasks.json` | `critical_points`(6) + `subtask_starts`(7) + 7 段 `subtasks` + `flags`;融合版另含 `sources`/`disagree_frames`/`review_points` |
| `ep<NNN>_subtask_index.npy` | `int16`,逐帧子任务 id(0..6),可直接当训练标签 |
| `summary.csv` / `all_subtasks.jsonl` | 每集汇总 / 单集一行 |

## 脚本

| 脚本 | 阶段 | 作用 |
|------|------|------|
| `vlm_annotate.py` | 1 | 本地 Qwen2.5-VL 视觉粗标(粗→细两遍);`test_qwen_vl.py` 为环境冒烟测试 |
| `batch_annotate.py` | 2 | 本体信号自动分段(7 段),写 JSON/npy/CSV |
| `fuse_annotations.py` | 1+2 | 融合两版 + 按 `|VLM−state|` 标出待复查点 |
| `annotate_gui.py` | 3 | 人工复查 GUI(三线对照,只停在待复查点) |
| `compare_timelines.py` / `outlier_report.py` / `verify_tail.py` | QA | 时间线对比 / 离群检测 / 收尾核验 |
| `analyze_subtasks.py` / `inspect_episode.py` | 诊断 | 单集信号 / 数据集检查 |

安装教程:[`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md)。

## 状态

- Stage 2 ✅ 100/100,0 flag。
- Stage 3 ✅ GUI 就绪(6 点 / 7 段,三线对照)。
- Stage 1 🟡 代码就绪。实测 **3B(8GB 笔记本)对暗光鱼眼画面无法定位事件**(输出等间距瞎编帧号)→ 改用 **7B**。安装教程:Windows [`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md);**Linux + RTX 5080 + 7B(vLLM,推荐)** [`docs/INSTALL_QWEN_LINUX.md`](docs/INSTALL_QWEN_LINUX.md)。`vlm_annotate.py` 支持 `--backend qwen-local`(transformers)与 `--backend openai`(vLLM)。
