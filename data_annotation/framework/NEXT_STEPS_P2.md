# Next step (P2): make the pour leave the VLM

**Goal.** Today p2 `start_pour` is the only critical point localized by the VLM,
and the VLM is coarse on this low-light fisheye footage. The pour is physically a
**wrist rotation** of the tube arm, so it should be detectable from
`observation.state`. If a state detector localizes p2 tightly, the arbiter will
hand the pour to state automatically (lower sigma wins) — no code or table edit,
just a schema field — and the weakest VLM dependency disappears.

The framework is already wired for this: `OrientationDetector` exists but stays
**inert until the schema names the wrist dims** (`state_layout.tube_wrist_dims`).
So the whole task is: *find those dims, put them in the schema, validate.*

---

## Do this on the server (observation.state is not in this checkout)

### Step 1 — find the wrist dims

```bash
git fetch origin
git checkout feature/confidence-arbitration-framework

python data_annotation/framework/tools/probe_state_layout.py \
    --data       /home/hillbot/black_smash_07/data/chunk-000 \
    --info       /home/hillbot/black_smash_07/meta/info.json \
    --state-ann  annotations_state_07 \
    --eps 0,1,2,3,4,5,6,7,8,9
```

What it does (read-only, prints only):
- prints the state feature names from `info.json`. For these datasets `names` is
  usually just `["observation.state"]` (no per-dim labels) — if so, ignore it and
  use the table below;
- ranks the tube-arm, non-gripper dims by how sharply they move **right at the
  labelled pour onset (p2)** while the arm is already settled over the mortar.

It ends with a ready-to-paste line, e.g.:

```
"tube_wrist_dims": [5, 6]   <-- the dims that stood out
```

If nothing stands out, widen the search: `--window-s 0.8` and more `--eps`. If the
tube arm is not dims 0–9 on your data, set `--tube-arm-dims`.

### Step 2 — put the dims in the schema

Edit `data_annotation/framework/schemas/black_smash.json`, add the dims under
`state_layout`:

```json
"state_layout": {
  "tube_gripper": 3,
  "pestle_gripper": 13,
  "grip_dims": [3, 4, 13, 14],
  "tube_wrist_dims": [5, 6]
}
```

That single field activates `OrientationDetector`.

### Step 3 — run with the detector on and compare

```bash
python -m data_annotation.framework.fuse parquet \
    --data /home/hillbot/black_smash_07/data/chunk-000 \
    --vlm  annotations_qwen_07 \
    --out  /tmp/fused_p2 \
    --orientation \
    --eps 0,1,2,3,4,5,6,7,8,9
```

Then look at p2 in `/tmp/fused_p2/ep*_subtasks.json`:
- `sources[1]` should now read `orientation` (state won the pour), and
- `sigmas[1]` should be small (~6) instead of falling back to the state proxy (~25).

Sanity-check a few against the camera frames (or the existing qwen p2) to confirm
the state-detected pour is actually on the powder-leaving-the-tube moment.

### Step 4 — decide

- **Looks right** → set `--orientation` on in `scripts/run_annotation_pipeline.sh`,
  regenerate 05/06/07, and the pour no longer needs the VLM.
- **Noisy / wrong dims** → re-run the probe with a wider window / more episodes, or
  tell me the ranked table and I'll tune `OrientationDetector`'s onset logic.

---

## What I need from you to finish writing the detector

`OrientationDetector` currently uses a generic "first sustained motion spike in the
held window" heuristic on the named dims. To make it precise I need the **probe's
ranked table** (or just the chosen `tube_wrist_dims`). With that I can:
- pick rotation vs translation dims correctly,
- tune the onset criterion (roll/tilt threshold) to the real signal,
- set a calibrated sigma from the spread of detected vs labelled p2.

Paste the probe output here and I'll finish P2 and validate it.
