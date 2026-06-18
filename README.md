# black-smash 子任务标注
### 任务:`Pour the black powder into the mortar and grind.`(把黑色粉末倒入研钵并研磨)

![数据集](https://img.shields.io/badge/数据集-LeRobot%20v2.1-blue)
![机器人](https://img.shields.io/badge/机器人-双臂-orange)
![方法](https://img.shields.io/badge/方法-3%20阶段：VLM→信号→人工-purple)
![标注](https://img.shields.io/badge/子任务-6%20段%20·%205%20临界点-teal)
![Stage2](https://img.shields.io/badge/state%20自动标注-100%20%2F%20100%20集-brightgreen)

对一个双臂操作数据集(LeRobot v2.1)做**时序子任务分段**:每条 episode 都是同一个长程任务,把每一帧标注成 **6 个连续子任务**(由 **5 个临界点**切分)。

标注采用 **三阶段流水线**,让两种互补的「机器视角」先各标一遍,再由人工快速复查定稿:

| 阶段 | 方法 | 看什么 | 强 / 弱 | 状态 |
|---|---|---|---|---|
| **Stage 1** | **Qwen2.5-VL**(本地) | 增强后的相机画面(像素) | 语义强、时间糙 | 🟡 环境/教程就绪,`vlm_annotate.py` 进行中 |
| **Stage 2** | **本体信号**(`batch_annotate.py`) | 20 维 `observation.state` | 时间准、语义盲 | ✅ 完成(100/100,0 flag) |
| **Stage 3** | **人工复查**(`annotate_gui.py`) | 6 路视频 + 两条机器参考线 | 定稿金标准 | ✅ GUI 已有,正增强为双参考线 |

**为什么要三阶段?** 场景相机是**低照度鱼眼**,纯像素不可靠 → 单靠 VLM 时间不准;而本体信号时间很准、却**语义盲**且有已知盲点(如 `p3 放管` 信号回落早于人眼)。让 VLM(语义)与信号(时序)各出一版、相互印证,人工只需在两条参考线上微调即可定稿——比从零标快得多,也能互相兜住对方的失败模式。

![ep000 故事板](mvt_annotations/ep000_storyboard.png)

---

## 目录
1. [流水线](#流水线) · 2. [任务与子任务](#任务与子任务) · 3. [数据集](#数据集)
4. [Stage 1:Qwen2.5-VL 视觉粗标](#stage-1qwen25-vl-视觉粗标) · 5. [Stage 2:本体信号精标](#stage-2本体信号精标) · 6. [Stage 3:人工复查](#stage-3人工复查)
7. [质检](#质检) · 8. [环境](#环境) · 9. [输出](#输出) · 10. [结果](#结果) · 11. [脚本](#脚本) · 12. [已知局限](#已知局限)

---

## 流水线

```
                    ┌─────────────────────────┐
  增强相机帧  ─────► │ Stage 1  Qwen2.5-VL 本地 │ ─► mvt_annotations_vlm/   (语义粗标)
                    └─────────────────────────┘
                    ┌─────────────────────────┐
 observation.state ►│ Stage 2  本体信号分段    │ ─► mvt_annotations/       (时序精标)
                    └─────────────────────────┘
                    ┌─────────────────────────┐
   6 路视频 + 两参考►│ Stage 3  人工复查 GUI    │ ─► mvt_annotations_human/ (定稿金标准)
                    └─────────────────────────┘
```

三阶段输出**完全同一套 JSON schema**(`critical_points`(5) + `subtask_starts`(6) + 6 段 `subtasks`),只是 `annotator` 字段不同(`claude-vlm`/`qwen-vl` · `auto-signal` · `human-gui`),便于逐点对比。

---

## 任务与子任务

单一任务,切成 **6 个子任务**,由 **5 个临界点**划分(首=0、尾=末帧自动):

| id | 标签(写入数据的实际值) | 含义 |
|----|--------------------------|------|
| S0 | `reach for the test tube` | 伸手够试管 |
| S1 | `lift the test tube and move it over the mortar` | 端起试管移到研钵上方 |
| S2 | `pour the black powder into the mortar` | 倒粉入研钵 |
| S3 | `set down the test tube and pick up the pestle` | 放下试管、取杵 |
| S4 | `grind the powder in the mortar` | 研磨 |
| S5 | `lift the pestle and return to rest` | 抬杵收回 |

| 临界点 | 含义 | 切换 |
|---|---|---|
| p1 | 抓到试管 | S0→S1 |
| p2 | 开始倒 | S1→S2 |
| p3 | 放试管 | S2→S3 |
| p4 | 开始磨 | S3→S4 |
| p5 | 抬杵 | S4→S5 |

> 标签固定在 `batch_annotate.py` 顶部的 `LABELS`,三个阶段共用;脚本按临界点数量参数化,增减点不用大改。

---

## 数据集

`black_smash_07/` 是 LeRobot v2.1 双臂数据集,**100 条 episode**,单一任务,每条约 1049–1290 帧,30 fps。**不含在仓库**(已 gitignore)。
6 路图像(`camera0/1` 场景 + `tactile_{left,right}_{0,1}` 触觉,均 224×224)+ 20 维 `observation.state` / `actions` + `timestamp`(均匀 `frame_idx/30`)。
下载:`hf download EricChen06/black_smash_07 --repo-type dataset --local-dir black_smash_07`(数据集所有者为 `EricChen06`)。

---

## Stage 1:Qwen2.5-VL 视觉粗标

第一遍用**本地 Qwen2.5-VL** 看画面给出一版语义标注(每条 episode 均匀抽帧 → 增强对比度 → 让模型判定 5 个临界点所在帧),写入 `mvt_annotations_vlm/`,schema 与其它两阶段一致。

- **为何用本地、用 Qwen**:无需联网/密钥;用户有独立的 **RTX 5080 桌机(16 GB)** 跑它。本仓库的 `vlm` 笔记本环境(RTX 5060 8GB)只跑 Stage 2/3。
- **模型选择**(16 GB 显存):推荐 `Qwen2.5-VL-7B-Instruct-AWQ`(~7 GB);零踩坑备选 `Qwen2.5-VL-3B-Instruct`。全精度 7B(~16.5 GB)放不下 16 GB。
- **安装**:详见 **[`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md)** —— Windows + RTX 5080(Blackwell `sm_120`,必须 **cu128** 版 torch)的完整步骤、国内模型下载镜像、排错表。
- **验证**:`test_qwen_vl.py` 造一张测试图跑一次推理,确认环境可用(打印显存占用)。

```powershell
# 在 5080 桌机的 qwenvl 环境里(见安装教程)
python test_qwen_vl.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ   # 先验证环境

# 标注脚本(即将提交;预期 CLI):
python vlm_annotate.py --backend qwen-local `
  --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ `
  --data C:\Intern\black_smash_07\data\chunk-000 `
  --out  C:\Intern\mvt_annotations_vlm --eps 0,1,2 --n-frames 32
```

> 状态:Stage 1 的**环境与验证已就绪**;实际标注脚本 `vlm_annotate.py` 正在编写(后端可插拔,默认 `qwen-local`)。

---

## Stage 2:本体信号精标

`batch_annotate.py` —— 边界全部来自 20 维 `observation.state`(用**帧序号**,与帧率无关),**不看图、纯 CPU、可批量复现**。开发时只用增强 `camera1` 抽查。

- **握管窗口 `[E1,E2]`** —— 握住试管期间,状态**第 3 维**会从静息值发生**双峰态偏离**;取前 70% 内最长的偏离段。→ `p1 抓管 = E1`,`p3 放管 = E2+1`。
- **开始倒 `p2`** —— 在 `[E1,E2]` 内,手臂从「搬运(平移=高载波漂移)」切到「在研钵上方倒(原地=低漂移)」;取窗口内最长低漂移段的起点。(载波 = 位姿 ~1s 滑动均值;漂移 = 载波速度。)
- **研磨 `[Gs,Ge]`** —— 研磨 = 原地运动:`漂移 < 第40百分位` 且 `原始速度 > 0.12×最大`;取 `E2` 之后、桥接 <1.2s 短停顿后的最长连续段。→ `p4 开始磨 = Gs`,`p5 抬杵 = Ge+1`。
- **容错**:任一窗口找不到 / 临界点非递增 → 按比例回退并写入 `flags`。规模化质检只看 `flags` 非空的几条(当前 100 集 **0 个**)。

```powershell
& "C:\Users\jerry\miniconda3\envs\vlm\python.exe" batch_annotate.py                 # 全部 episode(快)
& "...python.exe" batch_annotate.py --storyboard    # 额外出 6 格质检故事板(解码 camera1,慢)
& "...python.exe" batch_annotate.py --eps 0,5,9     # 子集(不会覆盖全量 summary.csv)
```

---

## Stage 3:人工复查

`annotate_gui.py` —— **像放视频一样**复查并定稿金标准。

- **同屏 6 路画面**:上排 cam0/cam1,下排 4 路触觉(触觉最能判断接触/受力 → 抓管、倒、研磨)。
- 播放 / 单帧步进,按 **1–5** 在当前帧打 5 个临界点;时间轴实时画出 6 段。
- **复查模式(增强中)**:同屏叠加 **Stage 1(VLM)** 与 **Stage 2(信号)** 两条参考时间线,可一键吸附到任一版本再微调,大幅加快定稿。
- 存到 `mvt_annotations_human/`,**与前两阶段同字段**。

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| 空格 | 播放/暂停 | **1–5** | 打临界点(抓管/开始倒/放管/开始磨/抬杵) |
| ← → | 单帧 | Shift+1~5 | 清除该点 |
| , . | 跳 ±10 帧 | 0 | 清空 |
| Home/End | 首/尾帧 | s | 保存 |
| +/− | 播放速度 | n / p | 下/上一集(自动存) |

```powershell
& "...python.exe" annotate_gui.py --ep 0          # 默认全 6 画面
& "...python.exe" annotate_gui.py --ep 5 --layout cam1   # 只看一个相机
```

**人工 vs 自动对比** —— `compare_timelines.py` 对每个同时有人工+自动标注的 episode,画上下对齐时间线,5 个临界点连线标 **帧差 Δ**,算**各段时序 IoU** 与全集 **MAE**。

![人工 vs 自动 时间线对比](docs/ep000_compare.png)

**ep000 实测**(人工金标准 vs 信号自动):磨(S4 IoU=0.78)、抬杵(S5 IoU=0.73)**高度吻合**;分歧集中在 **p3 放管(Δ≈−5s)**——信号「握管」早于人眼判定的放下时刻回落,把「倒」判短。这正是 Stage 1 VLM 语义视角要补的盲点。

---

## 质检

- `outlier_report.py` —— 跨集一致性 / 离群检测(robust modified-z,按临界点位置与各段时长占比),挑出可疑集。当前 100 集挑出约 14 集,集中在研磨收尾(渐变所致),非错标。
- `verify_tail.py` —— 把某集「研磨→收尾」放大渲染、标出边界,人工核验。

---

## 环境

| 阶段 | 机器 / 环境 | 依赖 |
|---|---|---|
| Stage 1(VLM) | 5080 桌机,conda `qwenvl` | torch **cu128** + transformers≥4.49 + qwen-vl-utils(见 [`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md)) |
| Stage 2 / 3 | 本机,conda `vlm` | `pandas` `numpy` `Pillow`(GUI 用标准库 `tkinter`,对比图用系统中文字体)——**不需要** matplotlib/scipy/联网 |

> 本机裸 `python` 是失效的 Microsoft Store stub,务必用全路径 `C:\Users\jerry\miniconda3\envs\vlm\python.exe`。环境名「vlm」与「是否调用 VLM」无关。

---

## 输出

每阶段一个目录,**同结构**:`mvt_annotations_vlm/`(Stage 1)· `mvt_annotations/`(Stage 2)· `mvt_annotations_human/`(Stage 3 定稿)· 对比图 `mvt_compare/`(已 gitignore)。

| 文件 | 内容 |
|------|------|
| `ep<NNN>_subtasks.json` | `critical_points`(5)+ `subtask_starts`(6)+ 6 段 `subtasks`(`label`/`start_frame`/`end_frame`/`dur_s`…)+ `flags` |
| `ep<NNN>_subtask_index.npy` | `int16`,长度=帧数,每帧子任务 id(0..5),可直接当训练标签 |
| `summary.csv` | 每集一行:`p1..p5` 临界点帧 + `S0_s..S5_s` 段时长 + flags |
| `all_subtasks.jsonl` | 所有单集 JSON,一行一条 |

```json
{
  "episode_index": 0, "n_frames": 1159, "fps": 30, "annotator": "auto-signal",
  "critical_points": [187, 245, 366, 758, 1024],
  "subtask_starts": [0, 187, 245, 366, 758, 1024],
  "n_subtasks": 6,
  "subtasks": [{"subtask_id": 0, "label": "reach for the test tube",
                "start_frame": 0, "end_frame": 186, "dur_s": 6.23, "...": "..."}]
}
```

---

## 结果(Stage 2)

全部 **100 条** episode 信号自动标注完(1049–1290 帧,均值 ~1169):**0 个 pipeline flag**,各段时长跨集一致。

| 子任务 | 均值 (s) | 范围 (s) |
|------|------|------|
| S0 伸手够试管 | 6.4 | 5.4 – 7.7 |
| S1 端试管到研钵 | 2.1 | 1.4 – 3.7 |
| S2 倒 | 4.2 | 3.1 – 6.3 |
| S3 放试管+取杵 | 11.8 | 10.0 – 16.0 |
| S4 研磨 | 9.5 | 5.8 – 12.5 |
| S5 抬杵收回 | 5.0 | 3.2 – 9.0 |

---

## 脚本

| 脚本 | 阶段 | 作用 |
|------|------|------|
| `vlm_annotate.py` | Stage 1 | **🟡 进行中**——本地 Qwen2.5-VL 视觉粗标,写 `mvt_annotations_vlm/` |
| `test_qwen_vl.py` | Stage 1 | Qwen2.5-VL 环境冒烟测试(造图跑一次推理) |
| `batch_annotate.py` | Stage 2 | **主工具**——本体信号自动分段全部 episode,写 JSON/npy/CSV,标记异常 |
| `annotate_gui.py` | Stage 3 | **人工复查 GUI**——全 6 画面、按 1–5 打临界点,出金标准 |
| `compare_timelines.py` | QA | 人工 vs 自动时间线对比 + 帧误差 / IoU |
| `outlier_report.py` | QA | 跨集一致性 / 离群检测 |
| `verify_tail.py` | QA | 某集研磨→收尾放大核验 |
| `analyze_subtasks.py` | 诊断 | 单集逐秒信号时间线 + 事件 |
| `inspect_episode.py` | 诊断 | 数据集检查——schema、图像列、抽帧 |

安装教程:[`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md)。

---

## 已知局限

- **Stage 2(信号)**:最可靠的是研磨/抬杵(S4/S5,人机 IoU≈0.75+);最软的是**握管→倒→放管**那段——`p1 抓管` 比人工晚约 2s(信号需姿态变化累积才触发)、`p3 放管` 分歧最大(信号回落早于人眼判定)。这些正是 Stage 1 VLM 与 Stage 3 人工要补的。
- **Stage 1(VLM)**:相机低照度鱼眼,即便增强后时间分辨率也有限,定位偏粗 → 只作语义参考,精确边界靠信号 + 人工。
- `ENGAGE_DIM = 3` 针对**本数据集本体**验证(100 集稳定);换数据集 / 机器人需重新确认该维度。
- 分段以帧序号为准;`info.json` 的 fps 只影响输出秒数。
- 标签为单一任务固定 6 类,写死在 `LABELS`,三阶段共用。
