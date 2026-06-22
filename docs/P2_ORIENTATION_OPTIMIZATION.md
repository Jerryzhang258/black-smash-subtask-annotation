# P2 Orientation Detector Optimization Plan

This note explains why the current PR #1 `OrientationDetector` should not yet
replace the fused `p2_start_pour` boundary, and how to improve it into a reliable
state-based pour detector.

## Goal

`p2_start_pour` marks the first moment black powder starts leaving the test tube.
The current production pipeline uses fused labels:

```text
state skeleton + Qwen visual p2 check -> fused -> qwen-stage
```

The long-term goal is to detect `p2_start_pour` directly from
`observation.state`, because pouring should correspond to a tube-arm wrist
rotation. If this works, the weakest VLM dependency can be removed.

## Current Test Result

PR #1 introduces a schema-driven confidence arbitration framework and an
experimental `OrientationDetector`.

The framework itself is good:

- Regression test passes against the legacy fused example.
- With orientation disabled, the framework reproduces current fused outputs:
  - dataset 05: 218/232 episodes exactly match, max difference 1 frame
  - dataset 06: 50/50 episodes exactly match
  - dataset 07: 98/100 episodes exactly match, max difference 1 frame

This confirms that the existing state segmentation logic is not the problem.

The experimental orientation detector is not ready:

- Using `tube_wrist_dims=[2]`, many detected p2 points are too early.
- Using `tube_wrist_dims=[5]` or `[2,5]` is worse on dataset 05 and 07.
- Many episodes place p2 only 1 to 10 frames after p1, which is likely the tube
  lift or transport motion, not powder flow.

The main failure mode is:

```text
detector catches the first wrist/arm motion after grasp,
instead of the actual pour-tilt onset over the mortar
```

## Observed State-Dim Signal

The probe script found consistent signal around dims 2 and 5:

| dataset | strongest dims | auto suggestion |
|---|---|---|
| 05 | 5, 2, 0 | none |
| 06 | 2, 5, 0 | `[2]` |
| 07 | 5, 2, 0 | none |

So there is useful state signal, but the current onset heuristic is too broad.
It treats generic motion spikes as pour onset.

## Optimization Strategy

### 1. Gate By Task Phase

Do not search for pour onset immediately after p1. Restrict the search window to
the phase where the tube is already transported over the mortar.

Candidate rule:

```text
start search after p1 + transport_delay
end search before p3 release_tube
```

Initial settings to test:

```text
transport_delay = 1.0s to 1.5s
search_end = p3 - 0.2s
```

This should remove the common false positive where p2 is only a few frames after
p1.

### 2. Detect Tilt, Not Motion Magnitude

The current detector uses the norm of wrist-dim changes. That catches any sharp
movement. Instead, use directional evidence:

- choose the dominant pour dimension per dataset or per episode
- estimate the sign of the pour tilt from labelled state p2 examples
- detect sustained monotonic change in that direction

Useful features:

```text
delta = x[t] - median(x[t - pre_window : t])
slope = smoothed derivative of x
zscore = normalized change relative to early held-window baseline
```

The detector should fire only when:

```text
abs(delta) > delta_threshold
and slope keeps the pour direction for at least hold_frames
```

### 3. Add Stability / Hold-Time Requirement

Pouring is not a single noisy spike. Require the tilt signal to remain changed
for a short duration.

Initial settings:

```text
hold_frames = 8 to 15
delta_threshold = 1.0 to 1.5 standard deviations
slope_threshold = percentile-based, but measured after transport begins
```

This should reject small transport corrections and short wrist jitters.

### 4. Use Existing State Boundaries As Guards

Use current state-derived points as constraints:

```text
p1 grasp_tube < p2 start_pour < p3 release_tube
```

More specifically, p2 should not be too close to p1 or p3:

```text
p2 >= p1 + min_pour_after_grasp
p2 <= p3 - min_before_release
```

Suggested defaults:

```text
min_pour_after_grasp = 30 frames
min_before_release = 10 frames
```

If no valid orientation onset is found inside this guarded window, the detector
should abstain instead of emitting a bad candidate.

### 5. Calibrate Sigma From Validation

The detector should not always report `sigma=6`. Calibrate based on validation
error against the current fused/manual reference.

Suggested rule:

```text
if onset is strong and stable: sigma = 6
if onset exists but weak: sigma = 12 to 18
if no clean onset: abstain
```

The arbiter can then prefer state only when the state signal is actually good.

## Validation Protocol

Run every candidate detector on datasets 05, 06, and 07 without changing the
production outputs.

For each dataset, report:

```text
episodes
p2 source counts
p2 abstain count
median / mean / max |orientation_p2 - fused_p2|
median / mean p2 - p1 gap
count where p2 - p1 <= 10 frames
count where p2 is outside [p1 + 30, p3 - 10]
```

Acceptance criteria before replacing fused p2:

```text
p2 - p1 <= 10 frames should be near zero
median |orientation_p2 - fused_p2| should be clearly below current Qwen/state disagreement
bad outliers should be visually explainable
detector should abstain on uncertain episodes instead of forcing a wrong p2
```

After the numeric pass, generate same-image visualizations for a sample of:

- best matches
- largest disagreements
- early-p2 failures
- abstained episodes

Only enable orientation in the full pipeline after these visual checks pass.

## Implementation Checklist

1. Update `probe_state_layout.py`
   - remove the hard `score >= 1.0` pick rule
   - report top-k dims even when absolute scores are small
   - report per-dataset and cross-dataset rankings

2. Update `OrientationDetector`
   - search only inside guarded `[p1 + delay, p3 - margin]`
   - use directional sustained tilt, not only speed magnitude
   - abstain when no clean onset is found
   - output evidence: selected dim, direction, score, search window

3. Add evaluation script
   - compare orientation p2 to current fused p2
   - print dataset-level summary
   - list worst episodes for visual inspection

4. Add visual debug output
   - plot selected state dim over time
   - mark p1, old fused p2, orientation p2, p3
   - save alongside compare-track examples

5. Re-run on 05/06/07
   - first in a temporary branch/worktree
   - then decide whether to merge into the production pipeline

## Current Recommendation

Do not merge orientation as the production p2 source yet.

Keep the current main pipeline as the reliable baseline:

```text
state -> qwen critical points -> fused -> qwen-stage -> visualization
```

Use PR #1 as the framework foundation, but treat orientation-based p2 as a P2
research task until it passes the validation protocol above.
