# P2: state-based pour detector — how to validate

The authoritative plan is [`docs/P2_ORIENTATION_OPTIMIZATION.md`](../../docs/P2_ORIENTATION_OPTIMIZATION.md).
`OrientationDetector` now implements that plan (phase-gated window, directional
sustained-tilt onset, abstain, calibrated sigma). This file is just the runbook
to find the wrist dims and validate on the server, where `observation.state` lives.

The framework reproduces current fused outputs when orientation is off (verified
on 05/06/07), so orientation stays **opt-in** until it passes the protocol below.

## 1. Find the wrist dims (cross-dataset)

```bash
git fetch origin && git checkout feature/p2-orientation-detector

python data_annotation/framework/tools/probe_state_layout.py \
    --dataset 05:/home/hillbot/black_smash_05/data/chunk-000:annotations_state_05 \
    --dataset 06:/home/hillbot/black_smash_06/data/chunk-000:annotations_state_06 \
    --dataset 07:/home/hillbot/black_smash_07/data/chunk-000:annotations_state_07
```

It prints a per-dataset and a cross-dataset ranking (no hard threshold) and a
suggested `tube_wrist_dims`. Prior runs pointed at dims 2 and 5.

## 2. Set the dims in the schema

`schemas/black_smash.json` → `state_layout`:

```json
"tube_wrist_dims": [2, 5]
```

You can tune the detector without code via `state_layout.p2_orientation`
(`transport_delay_s`, `hold_frames`, `delta_threshold`, …) — defaults are in
`detectors.py:DEFAULT_ORIENTATION_CFG`.

## 3. Evaluate (no production change)

```bash
python data_annotation/framework/tools/eval_orientation_p2.py \
    --data /home/hillbot/black_smash_07/data/chunk-000 \
    --ref-fused annotations_fused_07 --wrist-dims 2,5
```

Prints the protocol metrics: fire/abstain counts, sigma split,
median/mean/max `|orientation_p2 − fused_p2|`, `p2 − p1` gap, count `p2 − p1 ≤ 10`,
count outside `[p1+30, p3−10]`, and the worst episodes.

**Acceptance** (from the plan): `p2 − p1 ≤ 10` near zero, median
`|orientation_p2 − fused_p2|` clearly below the current Qwen/state disagreement,
outliers visually explainable, abstains on uncertain episodes.

## 4. Eyeball the cases

```bash
python data_annotation/framework/tools/plot_orientation_p2.py \
    --data /home/hillbot/black_smash_07/data/chunk-000 \
    --ref-fused annotations_fused_07 --wrist-dims 2,5 \
    --eps 0,1,2,3,4 --out compare_tracks_07/orientation_debug
```

One PNG per episode: the wrist dim(s) with p1 / ref p2 / orientation p2 / p3 and
the search window. Check best matches, big disagreements, early-p2, abstains.

## 5. Decide

Only if the protocol passes: enable `--orientation` in
`scripts/run_annotation_pipeline.sh` and regenerate. Otherwise re-run the probe
(wider window / more dims) or adjust `p2_orientation` config, and tell me the
`eval_orientation_p2.py` summary — I can tune the onset logic from those numbers.

## What I still need from you

The `eval_orientation_p2.py` summary on 05/06/07 (fire/abstain counts and the
`|Δ|` / gap stats). That tells me whether to tighten the window, change the
strength calibration, or pick the dim per-dataset.
