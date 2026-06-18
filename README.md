# black-smash 子任务标注

双臂操作数据集(LeRobot v2.1,任务 `Pour the black powder into the mortar and grind.`),把每条 episode 逐帧切成 **7 个子任务 / 6 个临界点**。

**三阶段流水线**(两种互补的机器视角各标一遍 → 融合 → 人工只复查不一致的点):

```
Stage 1  vlm_annotate     Qwen2.5-VL 看画面    → mvt_annotations_vlm/     语义粗标
Stage 2  batch_annotate   本体 state 信号       → mvt_annotations/         时序精标
         fuse_annotations 按置信度融合两版      → mvt_annotations_fused/   + 待复查点
Stage 3  annotate_gui     人工复查 GUI          → mvt_annotations_human/   定稿
```

![标注可视化示例](docs/example_annotation.png)

> 一条 episode 的标注可视化(`visualize_annotation.py`):顶部 7 段彩色时间线(含每段时长),下面每段一张关键帧(已 center-crop + 白平衡,试管/研钵/杵清晰可见)。

**核心:每个临界点交给最擅长它的方法。** 夹爪/运动事件(p1 抓管 · p3 放管 · p4 抓杵 · p5 开始磨 · p6 抬杵)归 **state**(夹爪事件误差≈1–2 帧);视觉事件(p2 开始倒)归 **VLM**。融合时 `|VLM−state|>容差(默认 0.5s)` 的点标 `review_points`,人工只调这些。三阶段输出同一套 JSON schema,仅 `annotator` 字段不同。

## 子任务与临界点

标签固定在 `batch_annotate.py` 的 `LABELS`,三阶段共用:

| | 子任务(写入数据的标签) | 起始临界点 |
|---|---|---|
| S0 | reach for the test tube | (0) |
| S1 | lift the test tube and move it over the mortar | p1 抓到试管(夹爪闭 dim3) |
| S2 | pour the black powder into the mortar | p2 开始倒(视觉) |
| S3 | release the test tube and reach for the pestle | p3 放下试管(夹爪开 dim3) |
| S4 | bring the pestle over the mortar | p4 抓到杵(夹爪闭 dim13) |
| S5 | grind the powder in the mortar | p5 开始磨(原地运动) |
| S6 | lift the pestle and return to rest | p6 抬杵(研磨结束) |

## state 判断原理(Stage 2)

完全不看画面,只从 20 维 `observation.state` 推断(暗光鱼眼相机不可靠,本体感知干净);边界用**帧序号**,与帧率无关。先算两类信号:

- **夹爪开合**:双臂在 state 里 **+10 镜像** —— 试管夹=`dim3`、杵夹=`dim13`。归一化后"夹住"= 偏离静息(张开)值;`grip_window()` 取某夹爪"闭合"的**最长连续段**(`close_gaps` 桥接瞬时抖动),**段起点=抓、终点+1=放**。
- **运动分解(关键技巧)**:非夹爪维度做位姿 P;`raw`= 逐帧速度(动多快),`carrier`= P 的 ~1s 滑动均值(慢轨迹),`drift`= carrier 速度(平移量)。**研磨 = `raw` 高但 `drift` 低**(在研钵上方原地来回振、没走位),搬运则 drift 高 —— 用这个对比干净区分"原地磨"与"搬运"(普通速度会饱和,分不出)。

6 个临界点由此导出:

| 临界点 | 信号依据 |
|---|---|
| p1 抓管 / p3 放管 | 试管夹爪(dim3)闭合窗口 `[E1,E2]` 的 起点 / 终点+1 |
| p2 开始倒 | 握管窗口内最长"低 drift"段起点(手臂 settle 到研钵上方;*代理,偏软*) |
| p4 抓杵 | 杵夹爪(dim13,p3 之后)闭合窗口起点 |
| p5 开始磨 / p6 抬杵 | p4 之后 `drift 低 且 raw 高`(原地运动)最长段的 起点 / 终点+1 |

**稳健性**:最长段 + 形态学桥接缺口 + 1–99 百分位归一化;任一窗口缺失或 6 点非递增 → 按比例回退并写 `flags`(当前 382 集 **0 flag**)。

**强弱**(信号细查实测):夹爪事件 **p1/p3/p4 亚秒级最稳**(p3 ≈ 0.4s 误差);运动事件 **p5/p6 较稳**(p6 收尾渐变略软);**仅 p2「开始倒」真软** —— 粉末流出是视觉/手腕事件,本体信号里没有,只能代理 → 这点留给 **VLM / 人工**。

## 用法

```powershell
$PY = "C:\Users\jerry\miniconda3\envs\vlm\python.exe"   # 裸 python 是失效 stub,务必全路径
& $PY batch_annotate.py                 # Stage 2:全部 episode(state,快;--eps 子集,--storyboard 故事板)
& $PY fuse_annotations.py --tol-s 0.5   # 融合 → 标待复查点
& $PY annotate_gui.py --ep 0            # Stage 3:人工复查(键 1–6 打点,s 保存,红▲=待复查)

# Stage 1 需 GPU(安装见下);也可 --dry-run 无模型只检查抽帧/prompt
python vlm_annotate.py --backend openai --model qwen --base-url http://localhost:8000/v1 --eps 0,1,2
```

## 数据集

`black_smash_07/`:LeRobot v2.1 双臂,100 集,~1049–1290 帧/集,30 fps(已 gitignore)。6 路图像(`camera0/1` + 4 路触觉,224²)+ 20 维 `state`/`actions`。双臂 state **+10 镜像**:试管夹=dim3、杵夹=dim13。
下载:`hf download EricChen06/black_smash_07 --repo-type dataset --local-dir black_smash_07`。

## 输出

每阶段一个同结构目录。每集 `ep<NNN>_subtasks.json`(`critical_points`(6) + `subtask_starts`(7) + 7 段 `subtasks`;融合版另含 `sources`/`disagree_frames`/`review_points`)+ `ep<NNN>_subtask_index.npy`(逐帧 id 0..6,可直接当训练标签);全量另有 `summary.csv` / `all_subtasks.jsonl`。

## 脚本

| 脚本 | 作用 |
|---|---|
| `vlm_annotate.py` | Stage 1 Qwen2.5-VL 粗标(后端 `qwen-local`/`openai`,粗→细两遍);`test_qwen_vl.py` 冒烟测试 |
| `batch_annotate.py` | Stage 2 本体信号分段(7 段),写 JSON/npy/CSV |
| `fuse_annotations.py` | 融合两版 + 按 `\|VLM−state\|` 标待复查点 |
| `annotate_gui.py` | Stage 3 人工复查 GUI(三线对照,只停在待复查点) |
| `visualize_annotation.py` · `zoom_boundary.py` | 可视化:时间线+关键帧(`--tile`)/ 临界点逐帧放大 |
| `compare_timelines.py` · `outlier_report.py` · `verify_tail.py` | QA:时间线对比 / 离群检测 / 收尾核验 |
| `analyze_subtasks.py` · `inspect_episode.py` | 诊断:单集信号 / 数据集检查 |

## 安装与状态

- Stage 2 ✅ 100/100,0 flag;Stage 3 ✅ GUI 就绪;Stage 1 🟡 代码就绪,待 GPU 跑模型。
- 实测 **3B 对暗光鱼眼画面无法定位事件**(输出等间距瞎编帧号)→ 用 **7B**。
- 安装教程:Windows [`docs/INSTALL_QWEN.md`](docs/INSTALL_QWEN.md) · **Linux + RTX 5080 + 7B(vLLM,推荐)** [`docs/INSTALL_QWEN_LINUX.md`](docs/INSTALL_QWEN_LINUX.md)。
