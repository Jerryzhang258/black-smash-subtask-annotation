# ego_wipe_annotation — subtask annotation for the *wipe-the-tube* ego dataset

Signal-first (+ optional VLM fusion) subtask segmentation for the **raw teleop
demo-folder** format (Quest 3 egocentric stereo + bimanual grippers). Same design
philosophy as the black_smash `batch_annotate.py` pipeline at the repo root —
crisp proprioceptive boundaries own the geometric points, a VLM owns the visual
ones, disagreement routes to a human — but adapted to a completely different data
layout (per-frame folders, not LeRobot parquet).

## Data format (one demo folder)

```
demo_bimanual_<timestamp>/
  gripper_width_left.csv  gripper_width_right.csv   # frame,width  -> 1 row/frame
  pose_data/{left,right}_hand_trajectory.csv        # timestamp,x,y,z,q_*  (~60-70 Hz)
  tag_detection_{left,right}.pkl                     # list[{frame_idx,time,tag_dict}]
  left_hand_visual_img/  right_hand_visual_img/      # per-frame per-HAND fisheye jpgs (224^2, ~180deg)
  ego_data/<id>/...                                  # Quest 3 headset stereo mp4 (1280^2, ~73deg) + sidecars
```

> Camera note: `*_visual_img/` are per-hand **wide-angle fisheye** cameras (used
> for visualization / VLM input — no decode needed, they see both hands + the
> tube every frame). The true **headset ego** video is the `ego_data/*.mp4` (a
> Quest HEVC "spatialmp4", ~73° FOV, needs special decoding). Segmentation uses
> neither — only gripper width + hand pose.

Gripper CSVs are already one row per image frame. The pose trajectory runs on its
own unix clock at a higher rate; it is aligned to frames via the per-frame `time`
field in the tag-detection pickle (same clock). Pose is **optional** — only the
wipe sub-boundary uses it; gripper events do not.

## Taxonomy (6 subtasks, 5 critical points)

Defined in [`config.py`](config.py) (`LABELS`, `CRIT_NAMES`) — edit there.

| | subtask | start critical point | owner |
|---|---|---|---|
| S0 | reach for the test tube | (0) | — |
| S1 | grasp and lift the test tube | c1 grasp_tube (holder gripper closes) | state |
| S2 | acquire the wiper, bring it to the tube | c2 acquire_wiper (wiper gripper closes) | state |
| S3 | wipe the test tube | c3 start_wipe (wiper hand in-place oscillation) | state* |
| S4 | place the test tube back | c4 finish_wipe (wiper gripper opens) | state |
| S5 | release and retract | c5 release_tube (holder gripper opens) | state |

**Roles are assigned per demo**: the hand that stays closed longer is the *holder*
(holds the tube); the other is the *wiper*. So left/right-handed demos both work.
\* `start_wipe` is the softest boundary — set `CRIT_OWNER[2]="vlm"` in `config.py`
to let the VLM own it under fusion.

## How it works (algorithm)

Boundaries come from **proprioception, not pixels** — two frame-aligned signals on
the ~60 fps hand-frame grid: per-hand **gripper width** (already one value per frame)
and **hand 3-D pose** (interpolated onto the frame timestamps). Cameras are used only
for visualization / the optional VLM and vision cues.

**1. Gripper events → grasp & release (crisp, sub-second).**
Each gripper-width signal is percentile-normalized to `[0,1]`; the resting value
(open) is the median of the first frames. "Closed" = width drops below
`rest − GRIP_CLOSE_FRAC`. Short gaps are morphologically closed, and the **longest
closed run** gives `grasp = run start`, `release = run end`.

**2. Holder/wiper roles (handedness-robust).**
The hand whose closed run is **longer** is the *holder* (it holds the tube the whole
time); the other is the *wiper*. So the algorithm doesn't care which physical hand
does what. This yields four of the five points directly:
`c1 grasp_tube` & `c5 release_tube` (holder open/close), `c2 acquire_wiper` &
`c4 finish_wipe` (wiper close/open).

**3. Wipe onset `c3` (the one soft boundary).**
Inside the wiper's held window `[c2, c4]`, form the wiper-hand pose speed: `raw`
(per-frame speed) and `drift` (speed of a ~0.6 s rolling-mean "carrier"). **Wiping =
high `raw`, low `drift`** — the hand oscillates in place without translating. The
onset is the start of the longest such run (fallback: a proportion of the window if
pose is missing). Because it has no gripper transition, this is the softest point —
hence the optional vision/VLM help below.

**4. Optional vision cue (fisheye → wipe).** `vision_signal.py` computes
frame-difference energy on the wiper-hand fisheye (strokes make the wrist view
change a lot); its onset is fused with the pose onset, and a disagreement beyond
tolerance flags `c3` for review.

**5. Optional VLM prior + fusion.** A VLM proposes all five points from frames; each
point keeps its **owner** modality's value (`CRIT_OWNER` in `config.py` — gripper
events are owned by the signal), and any point with `|VLM − state| > tol` is added to
`review_points` so human effort goes only to ambiguous boundaries.

**6. ego ↔ fisheye timestamp alignment.** The headset (~30 fps) and robot (~60 fps)
run on **unsynchronized clocks** (observed offset ~minutes), so frames are aligned by
**elapsed seconds from each stream's first frame** using each stream's real
timestamps — correct across the rate difference; `--ego-offset-s` corrects a residual
start mismatch.

The 5 ordered critical points define the 6 subtasks; ordering is validated and falls
back to fixed proportions if a signal is missing (logged in `flags`).

### Pipeline stages

```
Stage 2  signal      gripper width (grasp/release) + hand pose (wipe onset)        always
Stage 1  VLM         frames -> 5 critical points (OpenAI-compatible)               --vlm
Stage 3  fusion      per-point owner value; |VLM-state| > tol -> review_points     --vlm
```

## Outputs (per demo, under `--out`)

- `<name>_subtasks.json` — `critical_points`(5), `subtask_starts`(6), 6 `subtasks`;
  with `--vlm` also `vlm_critical_points`, `sources`, `disagree_frames`, `review_points`.
- `<name>_subtask_index.npy` — per-frame subtask id 0–5 (policy supervision).
- `<name>_timeline.png` — timeline bar + one keyframe per subtask (`--visualize`).
- `<name>_boundaries.png` — per-critical-point zoom (±0.33 s) with the signal value
  on each frame, for verifying boundary localization (`--qa`).
- `<name>_boundaries_dual_<eye>.png` — dual-view boundary QA: an ego row **and** the
  owning hand's fisheye row per critical point (`--qa-dual`; needs pyav).
- `<name>_combined_<eye>.png` / `<name>_dashboard_<eye>.png` — ego∥fisheye keyframes
  (`--combined`) / full timeline+ego+fisheye+signals dashboard (`--dashboard`).
- `summary.csv` — one row per demo (critical points + durations + flags).

## Usage

```bash
# signal only (no API key, no GPU) — needs pandas numpy pillow
# default paths are <repo>/demos and <repo>/wipe_annotations; override with --demos/--out
python -m ego_wipe_annotation.run --demos test_clean/demos --out test_clean/wipe_annotations
python -m ego_wipe_annotation.run --visualize --qa --eps 0   # + timeline & boundary QA images

# visualize/annotate from the HEADSET EGO view instead of the hand fisheye
# (decodes ego_data/*.mp4; needs `pip install av`. mp4 must be finalized — i.e.
#  contain a moov atom; unfinalized Quest captures cannot be decoded.)
python -m ego_wipe_annotation.run --frames ego --eye left --visualize --qa
#   -> writes <name>_timeline_ego_left.png / <name>_boundaries_ego_left.png

# DUAL-VIEW boundary QA: ego row + owning-hand fisheye row per critical point
python -m ego_wipe_annotation.run --frames ego --qa-dual   # -> <name>_boundaries_dual_left.png

# COMBINED dual-view: ego (context) beside hand-fisheye (grasp detail) per subtask
python -m ego_wipe_annotation.run --combined        # -> <name>_combined_left.png

# ego<->hand TIMESTAMP alignment (~30 vs ~60 fps). Default "elapsed": match
# seconds-from-start using each stream's real timestamps (hand: tag_detection
# time; ego: camera_frames timestamp_ns). The headset & robot clocks are NOT
# synced (offset ~minutes), so absolute unix sync is not used; --align-report
# prints the durations + device-clock offset, and --ego-offset-s nudges the start.
python -m ego_wipe_annotation.run --frames ego --align-report           # check alignment
python -m ego_wipe_annotation.run --frames ego --ego-offset-s 0.3 --visualize   # nudge +0.3s

# DASHBOARD: one shared frame axis — timeline + ego row + fisheye row + signals
# (gripper L/R, pose drift, fisheye frame-diff motion) with pose/vision/fused
# wipe-onset marks. The fisheye motion is fused into start_wipe and disagreements
# are flagged for review (see vision_signal.py).
python -m ego_wipe_annotation.run --dashboard       # -> <name>_dashboard_left.png

# + VLM prior & fusion (needs `pip install openai` and an OpenAI-compatible endpoint)
python -m ego_wipe_annotation.run --vlm --base-url http://localhost:8000/v1 \
       --model qwen --api-key-env OPENAI_API_KEY --tol-s 0.5
```

Paths default to `EGO_WIPE_DEMOS` / `EGO_WIPE_OUT` env vars, else `<repo>/demos`
and `<repo>/wipe_annotations` (`config.py`). Incomplete demo folders (missing a
gripper CSV) are reported and skipped, not crashed.

**Dependencies.** Signal path: `pandas numpy pillow`. The ego-video options
(`--frames ego`, `--combined`, `--dashboard`, `--qa-dual`) additionally need
`av` (PyAV); `--vlm` needs `openai`.

```bash
# one shot: everything (signal + ego visuals + dual QA + dashboard + alignment report)
python -m ego_wipe_annotation.run --frames ego --visualize --qa --qa-dual \
       --combined --dashboard --align-report
```

## Files

| file | role |
|---|---|
| `config.py` | taxonomy + thresholds + paths + ego alignment mode (single source of truth) |
| `ego_dataio.py` | load a demo folder (gripper, pose, frame times); decode ego mp4 + timestamp-align to hand grid |
| `signal_segment.py` | Stage 2 — gripper events + holder/wiper role assignment + wipe-onset |
| `vlm_prior.py` | Stage 1 — VLM proposes 5 critical points from frames |
| `vision_signal.py` | fisheye frame-diff motion + vision/pose wipe-onset fusion (the soft boundary) |
| `fuse.py` | Stage 3 — per-point fusion + disagreement flagging |
| `visualize.py` | timeline, boundary QA (single + dual view), combined, dashboard, signal plot |
| `run.py` | batch CLI orchestrating the stages and writing outputs |
