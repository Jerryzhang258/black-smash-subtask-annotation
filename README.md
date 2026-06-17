# black-smash 子任务标注
### 任务:`Pour the black powder into the mortar and grind.`(把黑色粉末倒入研钵并研磨)

![数据集](https://img.shields.io/badge/数据集-LeRobot%20v2.1-blue)
![机器人](https://img.shields.io/badge/机器人-双臂-orange)
![方法](https://img.shields.io/badge/方法-本体感知信号-purple)
![依赖](https://img.shields.io/badge/依赖-pandas%20%7C%20numpy%20%7C%20Pillow-green)
![进度](https://img.shields.io/badge/已标注-100%20%2F%20100%20集-brightgreen)

对一个双臂操作数据集(LeRobot v2.1)做**时序子任务分段**:数据集里每条 episode 都是同一个长程任务,本仓库把每一帧自动标注成 5 个连续子任务。分段依据是机器人的**本体感知信号**(`observation.state`),而**不是相机画面**——场景相机是低照度鱼眼、不可靠;而状态信号能干净地暴露出「抓取 / 松手」和「研磨」这些关键事件。

![ep000 故事板](mvt_annotations/ep000_storyboard.png)

---

## 目录
1. [概述](#概述)
2. [任务与子任务](#任务与子任务)
3. [数据集](#数据集)
4. [方法](#方法)
5. [环境与安装](#环境与安装)
6. [用法](#用法)
7. [输出](#输出)
8. [结果](#结果)
9. [脚本说明](#脚本说明)
10. [已知局限](#已知局限)
11. [复现步骤](#复现步骤)

---

## 概述

目标只有一个:**把每条 episode 切成有意义的子任务**,产出可以直接当作训练标签的逐帧序列。不追求亚帧级的边界精度,而是要**稳、可批量、可复现**。

整套分段**不看图**(因此又快、跨集又稳),只在开发阶段用增强后的 `camera1` 关键帧做人工抽查。值得强调的是:**最终管线完全不依赖任何 VLM / 神经网络模型**——纯本体信号处理,离线、纯 CPU。

---

## 任务与子任务

数据集只有一个任务:**`Pour the black powder into the mortar and grind.`**
被切分为 5 个连续子任务:

| id | 标签(写入数据里的实际值) | 含义 |
|----|----------------------------|------|
| S0 | `reach for and grasp the powder container` | 伸手抓取粉末容器 |
| S1 | `pour the black powder into the mortar` | 把黑色粉末倒入研钵 |
| S2 | `set down the container and bring the pestle to the mortar` | 放下容器、把杵移到研钵 |
| S3 | `grind the powder in the mortar` | 研磨研钵中的粉末 |
| S4 | `lift the pestle and return to rest` | 抬起杵、收回 |

> 标签字符串固定在 `batch_annotate.py` 顶部的 `LABELS` 里,要改措辞改那里即可。

---

## 数据集

`black_smash_07/` 是一个 LeRobot v2.1 双臂数据集,**共 100 条 episode**,单一任务,每条约 1000–1266 帧,30 fps。它**不包含在本仓库中**(约 4.3 GB,已 gitignore)——把脚本指向你本地的副本即可。

特征(来自 `meta/info.json`):

| 特征 | dtype | shape | 说明 |
|------|-------|-------|------|
| `observation.images.camera0`、`camera1` | image | 224×224×3 | 场景相机(低照度鱼眼) |
| `observation.images.tactile_{left,right}_{0,1}` | image | 224×224×3 | 4 路触觉(GelSight 类) |
| `observation.state` | float32 | (20,) | 本体状态;**分段就用它** |
| `actions` | float32 | (20,) | 动作 |
| `timestamp` | float32 | (1,) | 均匀的 `frame_idx/30` |

目录结构:`data/chunk-000/episode_{NNNNNN}.parquet`;元数据:`meta/{info,episodes,episodes_stats,tasks}.{json,jsonl}`。

---

## 方法

所有边界都来自 20 维 `observation.state` 时间序列,用**帧序号**表示(与帧率无关),开发阶段再用增强后的 `camera1` 关键帧做抽查。

定义 4 个内部边界:`b1`(抓取)、`b2`(松手)、`b3`(研磨开始)、`b4`(研磨结束)。
据此分段:`S0=[0, b1-1]`、`S1=[b1, b2]`、`S2=[b2+1, b3-1]`、`S3=[b3, b4]`、`S4=[b4+1, N-1]`。

### 1) 抓取 `b1` / 松手 `b2`

倒粉末的过程中,状态**第 3 维**会从静息值发生一次**短暂的双峰态偏离**(抓住容器 → 保持 → 放回)。做法:把第 3 维用 1/99 百分位归一化到 `[0,1]`,取「静息值」为前 `T/20` 帧的中位数;当 `|g − 静息值| > 0.5` 视为「介入态」;只在前 70% 的时间里找**最长**的那一段,它的起止就是 `b1 / b2`。这是最可靠的信号(双峰切换非常干净)。

### 2) 研磨 `b3 → b4`

研磨 = **原地运动**:末端有速度,但「载波漂移」很低。其中**载波** = 位姿的约 1 秒滑动均值,**漂移** = 载波的速度。搬运阶段是平移(高漂移)→ 排除;最后抬杵是高漂移 → 排除;最后静止是低速度 → 排除。判定规则:

```
grind_ok = (漂移 < 第40百分位) 且 (原始速度 > 0.12 × 最大原始速度)
```

只取 `b2` 之后的帧,桥接研磨过程中 < 1.2s 的短暂停顿,取**最长连续段**即 `[b3, b4]`。

信号定义(`fps=30`):位姿 = 除 `[3,4]` 外的 18 维并标准化;原始速度 = `‖Δ位姿‖` 经 0.3s 平滑;载波 = 各维 1.0s 滑动均值;漂移 = `‖Δ载波‖` 经 0.3s 平滑。

### 3) 容错与异常标记

| 情况 | 处理 |
|------|------|
| 找不到 / 过短的抓取窗口 | 按比例回退 10% / 30%,并写入 `flag` |
| 找不到研磨段(< 0.8s) | 按比例回退 62% / 92%,并写入 `flag` |
| 边界顺序不满足 `0<b1<b2<b3<b4<N-1` | 强制比例分段 16/32/62/90%,并写入 `flag` |

> 规模化质检 = **只看 `flags` 非空的那几条**。全部 100 集 **0 个 flag**。

---

## 环境与安装

只需 Python 加 `pandas`、`numpy`、`Pillow`(**不需要** matplotlib / scipy,也不需要任何模型或联网)。

本机用 conda 环境 `vlm`(注意:裸 `python` 指向的是失效的 Microsoft Store stub,务必用全路径):

```powershell
& "C:\Users\jerry\miniconda3\envs\vlm\python.exe" batch_annotate.py
```

> 「vlm」只是这个环境的名字,与「调用 VLM」无关。任何装了上述三个库的 Python 都能跑。

其它机器:`pip install pandas numpy pillow` 之后直接 `python batch_annotate.py`。

---

## 用法

```bash
# 标注 data 目录下找到的所有 episode(只读 state 列,很快)
python batch_annotate.py

# 额外为每集生成质检故事板(会解码 camera1,较慢)
python batch_annotate.py --storyboard

# 子集 / 自定义路径 / 不同帧率
python batch_annotate.py --eps 0,5,9
python batch_annotate.py --data /path/to/chunk-000 --out /path/to/out --fps 30
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `…\black_smash_07\data\chunk-000` | episode parquet 所在目录 |
| `--out` | `…\mvt_annotations` | 输出目录 |
| `--meta` | `…\meta\tasks.jsonl` | 读取任务字符串的来源 |
| `--fps` | `30` | 仅影响输出里的秒数 |
| `--eps` | (全部) | 逗号分隔的 episode 序号 |
| `--storyboard` | 关闭 | 是否生成故事板 |

---

## 输出(`mvt_annotations/`)

| 文件 | 内容 |
|------|------|
| `ep<NNN>_subtasks.json` | 单集分段:`subtasks[]`(含 `label`、`start_frame`/`end_frame`、`start_t`/`end_t`、`dur_s`)+ `boundaries` + `flags` |
| `ep<NNN>_subtask_index.npy` | `int16` 数组,长度 = 帧数,每帧的子任务 id(可直接作训练标签列) |
| `summary.csv` | 每集一行:边界帧 + 各段时长 + flags |
| `all_subtasks.jsonl` | 所有单集 JSON,一行一条 |
| `ep<NNN>_storyboard.png` | 每个子任务一张代表帧(需 `--storyboard`) |

`ep<NNN>_subtasks.json` 结构:

```json
{
  "episode_index": 0,
  "task": "Pour the black powder into the mortar and grind.",
  "n_frames": 1159, "fps": 30,
  "method": "signal-derived (state-dim3 pour deviation + carrier-drift grind)",
  "boundaries": {"b1": 187, "b2": 365, "b3": 758, "b4": 1023},
  "flags": [],
  "n_subtasks": 5,
  "subtasks": [
    {"subtask_id": 0, "label": "reach for and grasp the powder container",
     "start_frame": 0, "end_frame": 186, "start_t": 0.0, "end_t": 6.2,
     "n_frames": 187, "dur_s": 6.23}
  ]
}
```

`summary.csv` 的列:
`episode, n_frames, b1_grasp, b2_release, b3_grindStart, b4_grindEnd, S0_s, S1_s, S2_s, S3_s, S4_s, flags`

读取逐帧标签:

```python
import numpy as np
y = np.load("mvt_annotations/ep000_subtask_index.npy")  # 形状 (帧数,),int16,取值 0..4
```

---

## 结果

全部 **100 条** episode 已标注(每条 1049–1290 帧,均值约 1169;总时长约 39.0s):**0 个 pipeline flag**,各子任务时长跨集高度一致。

| 子任务 | 均值 (s) | 范围 (s) |
|------|------|------|
| S0 伸手 + 抓取 | 6.4 | 5.4 – 7.7 |
| S1 倒入 | 6.4 | 4.6 – 9.8 |
| S2 放下容器 + 取杵 | 11.8 | 10.0 – 16.0 |
| S3 研磨 | 9.5 | 5.8 – 12.5 |
| S4 抬杵 + 收回 | 5.0 | 3.2 – 9.0 |

**质检(方法3,见 `outlier_report.py`):** 对 100 集做跨集一致性 / 离群检测,挑出 14 集偏离常态,**全部集中在研磨结束边界 `b4`(偏早)、`S4` 偏长**;逐集看图(`verify_tail.py`)确认是**边界精度问题(±1~3 秒)而非错标**——五段结构 100% 正确。研磨的渐入/渐出会溢出到相邻段。

---

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `batch_annotate.py` | **主工具**——自动分段全部 episode,写 JSON / npy / CSV,并标记异常 |
| `annotate_ep.py` | 单集参考标注器(ep000 手工细化过边界) |
| `analyze_subtasks.py` | 诊断工具——打印某一集的逐秒时间线(夹爪 / 速度 / 振荡)与检测到的事件 |
| `inspect_episode.py` | 数据集检查工具——schema、图像列、抽帧 |
| `outlier_report.py` | **质检(方法3)**——跨集一致性 / 离群检测,robust z-score 挑出可疑集 |
| `verify_tail.py` | 质检——把某集「研磨→收尾」放大渲染并标出 `b4`,核验边界 |

---

## 已知局限

- 最可靠的是 `b1 / b2`(抓取 / 松手);**最软的是研磨边界 `b3 / b4`**(边定位边开始磨、磨完停一下再抬,是渐变过程)。离群检测显示约 14% 的集 `b4` 偏早 1~3 秒,研磨收尾被算进 S4(详见结果)。
- 相机是低照度鱼眼,所以标注主要靠本体感知信号、不靠像素。
- `ENGAGE_DIM = 3` 是针对**本数据集的本体**验证出来的(在 100 集上稳定);换数据集 / 换机器人需要重新确认这个维度。
- 分段以**帧序号**为准;`info.json` 里的 fps 只影响输出里的秒数(时间戳是均匀的 `frame_idx/30`)。
- 标签是单一任务的固定 5 类,写死在 `LABELS` 里。

---

## 复现步骤

```bash
# 1. 准备数据集(本仓库不含):放到 ./black_smash_07/data/chunk-000/episode_*.parquet
# 2. 安装依赖
pip install pandas numpy pillow
# 3. 运行
python batch_annotate.py --data ./black_smash_07/data/chunk-000 --out ./mvt_annotations
# 4. 查看汇总与异常:./mvt_annotations/summary.csv(运行结束时也会在末尾打印 flagged 列表)
```

单集调试:`python analyze_subtasks.py <episode.parquet>` 会打印逐秒信号时间线和事件,便于核对某一集的边界。
