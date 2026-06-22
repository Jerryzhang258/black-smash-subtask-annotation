# Confidence-Arbitration Annotation Framework

A schema-driven, detector-based replacement for the hand-written fusion logic in
the root `fuse_annotations.py`. The goal is to turn "a script for the black-smash
task" into "a reusable method": the task taxonomy lives in a JSON schema, boundary
sources are pluggable detectors, and ownership of each critical point is *derived*
from confidence instead of a hardcoded table.

## The core change: kill the OWNER table

The legacy fuser hardcodes which modality owns each critical point:

```python
# fuse_annotations.py
OWNER = ["state", "vlm", "state", "state", "state", "state"]
```

That line is really a hand-tuned prior on *which detector is more reliable for
which event*. This framework makes that prior explicit and per-detector: every
detector reports a **localization uncertainty `sigma` (in frames)** with each
candidate boundary, and the arbiter derives ownership from the sigmas plus a
"state is the skeleton" safety rule.

Result: with the sigma profile in `detectors.py`, the arbiter reproduces the old
OWNER behaviour **frame-for-frame** (verified by the regression test), but adding
a better detector now changes the outcome automatically — no table edit.

## Pieces

| file | role |
|---|---|
| `core.py` | `Candidate`, `EventSpec`, `Decision`, `TaskSchema`, `load_schema` |
| `schemas/black_smash.json` | the task: events, subtask labels, state layout (replaces hardcoded `LABELS`/`CRIT_NAMES`/`OWNER`/`TUBE_GRIP`) |
| `detectors.py` | `Detector` base + file / signal / orientation detectors and their sigma profiles |
| `arbitrate.py` | `arbitrate_event` (the OWNER replacement) + `enforce_order` |
| `fuse.py` | orchestrator + CLI; writes a superset of the legacy fused JSON |
| `tests/test_arbitrate.py` | regression vs legacy OWNER fusion (runs on `examples/`, no raw data) |

## Arbitration rule (`arbitrate_event`)

For each event, over the candidates from detectors that handle its type:

1. `primary` = lowest-sigma candidate.
2. `anchor`  = lowest-sigma **state** candidate (the proprioceptive skeleton).
3. primary is state → use it. primary is vision →
   - agrees with anchor (≤ tol) → inverse-variance fuse the two;
   - disagrees → fall back to anchor (vision is coarse on this footage) + flag.
4. `needs_review` if any candidate disagrees beyond tol, **or** the resolved sigma
   exceeds `sigma_max`, **or** there was no state anchor.

Step 4's sigma clause is strictly more than the legacy rule: it can flag a point
where state and vision *agree but are both weak*, which the old `|vlm-state|>tol`
test could never catch.

## Run

```bash
# regression (no raw data needed)
python data_annotation/framework/tests/test_arbitrate.py

# reproduce legacy fusion from existing annotation dirs
python -m data_annotation.framework.fuse file \
    --state annotations_state_07 --vlm annotations_qwen_07 \
    --out annotations_fused_07

# run from raw observation.state, vlm as cross-check
python -m data_annotation.framework.fuse parquet \
    --data /data/black_smash_07/data/chunk-000 --vlm annotations_qwen_07 \
    --out annotations_fused_07
```

Output is a superset of the legacy fused doc (same keys + per-point `sigmas` and
arbiter `notes`), so `visualize_annotation_tracks.py` and `qwen-stage` consume it
unchanged.

## Adding a new task

Write a new `schemas/<task>.json` with its `events` (each a `type` the detectors
understand) and `subtasks`. No Python changes if the events reuse existing types
(`gripper_close`, `gripper_open`, `motion_regime`, `orientation_change`).

## Migration phases

| phase | status | what |
|---|---|---|
| **P0** | **done** | framework + arbiter reproduce legacy fusion (regression green); signal path reuses the proven `segment_episode` |
| **P1** | **done** | OWNER table removed; ownership derived from sigma + schema config |
| **P2** | **implemented · pending validation** | `OrientationDetector` lets the pour leave VLM (phase-gated window, directional sustained tilt, abstain, calibrated sigma — per `docs/P2_ORIENTATION_OPTIMIZATION.md`). Opt-in via `tube_wrist_dims`; validate on the server with `tools/eval_orientation_p2.py` before enabling. See `NEXT_STEPS_P2.md`. |
| **P3** | future | replace `segment_episode` internals with changepoint / HMM segmentation per detector |

## Caveats

- The signal and orientation detectors run on `observation.state`, which is **not
  present in this checkout** — validate them on the server (`black_smash_05/06/07`).
  Only the arbitration layer is covered by the local regression test.
- Sigma values in `detectors.py` are calibrated by hand to match current
  behaviour. They are the right place to fit against human-reviewed labels later.
