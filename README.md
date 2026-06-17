# Subtask annotation — "Pour the black powder into the mortar and grind"

Temporal **subtask segmentation** for a bimanual manipulation dataset (LeRobot v2.1).
Every episode is the same long-horizon task; this repo auto-labels each episode's
frames into 5 contiguous subtasks, driven by the robot's **proprioceptive signal**
(`observation.state`) rather than the cameras — the scene cameras are low-light fisheye
and unreliable to read, while the state stream cleanly exposes grasp/release and grinding.

![ep000 storyboard](mvt_annotations/ep000_storyboard.png)

## The 5 subtasks

| id | label |
|----|-------|
| S0 | reach for and grasp the powder container |
| S1 | pour the black powder into the mortar |
| S2 | set down the container and bring the pestle to the mortar |
| S3 | grind the powder in the mortar |
| S4 | lift the pestle and return to rest |

## How segmentation works

All boundaries are derived from the 20-dim `observation.state` time series (rate-independent,
expressed in frame indices), then spot-checked against enhanced `camera1` keyframes:

- **Grasp (b1) / release (b2)** — during the pour, state **dim 3** makes a transient bimodal
  deviation from its resting value. The longest such run in the first 70% of the episode
  brackets the grasp→pour→release. This is the most reliable signal (clean bimodal flip).
- **Grind (b3 → b4)** — grinding is *motion in place*: nonzero raw end-effector speed but low
  **carrier drift** (speed of the ~1 s rolling-mean pose). Transport phases translate the arm
  (high drift) and are excluded; the final lift is high-drift; the final rest is low-speed.
  The dominant low-drift / still-moving block (with short mid-grind pauses bridged) is the grind.

Episodes whose pattern doesn't fit (window missing, boundaries out of order) are written with a
non-empty `flags` field and listed at the end of a batch run, so QA at scale = review only the flagged few.

## Dataset (not included)

`black_smash_07/` is a LeRobot v2.1 bimanual dataset, **100 episodes**, single task
`"Pour the black powder into the mortar and grind."`, ~1000–1266 frames/episode, 30 fps.
6 image streams (`camera0`, `camera1`, 4 tactile) + 20-dim `observation.state` / `actions`.
It is **gitignored** (~4 GB) — point the scripts at your local copy.

## Usage

Requires Python with `pandas`, `numpy`, `Pillow` (no matplotlib/scipy needed).

```bash
# annotate every episode found under the data dir (state-only, fast)
python batch_annotate.py

# also emit a per-episode QA storyboard (decodes camera1, slower)
python batch_annotate.py --storyboard

# a subset / custom paths / different fps
python batch_annotate.py --eps 0,5,9
python batch_annotate.py --data /path/to/chunk-000 --out /path/to/out --fps 30
```

## Outputs (`mvt_annotations/`)

| file | contents |
|------|----------|
| `ep<NNN>_subtasks.json` | per-episode segments: `subtasks[]` with `label`, `start_frame`/`end_frame`, `start_t`/`end_t`, `dur_s`; plus `boundaries` and `flags` |
| `ep<NNN>_subtask_index.npy` | `int16` array, length = n_frames — the subtask id at every frame (drop-in training label column) |
| `summary.csv` | one row per episode: boundary frames + per-subtask durations + flags |
| `all_subtasks.jsonl` | all per-episode JSON docs, one per line |
| `ep<NNN>_storyboard.png` | representative frame per subtask (with `--storyboard`) |

`ep<NNN>_subtasks.json` shape:

```json
{
  "episode_index": 0, "task": "...", "n_frames": 1159, "fps": 30,
  "boundaries": {"b1": 187, "b2": 365, "b3": 758, "b4": 1023}, "flags": [],
  "subtasks": [
    {"subtask_id": 0, "label": "reach for and grasp the powder container",
     "start_frame": 0, "end_frame": 186, "start_t": 0.0, "end_t": 6.2,
     "n_frames": 187, "dur_s": 6.23}
  ]
}
```

## Scripts

| script | role |
|--------|------|
| `batch_annotate.py` | **main tool** — auto-segments all episodes, writes JSON/npy/CSV, flags anomalies |
| `annotate_ep.py` | single-episode reference annotator (manually-refined boundaries for ep000) |
| `analyze_subtasks.py` | diagnostic — prints a per-second timeline (gripper / speed / oscillation) + detected events for one episode |
| `inspect_episode.py` | dataset inspector — schema, image columns, sample frame extraction |

## Notes

- The dataset's `info.json`/timestamps say 30 fps and timestamps are perfectly uniform; segmentation is in frame indices, so the rate only affects reported seconds.
- Most reliable boundary is grasp/release (b1/b2). Softest is S2↔S3 (transport→grind is gradual, ~±0.5 s).
- Validated on the 40 currently-downloaded episodes: 0 flagged, consistent per-subtask durations.
