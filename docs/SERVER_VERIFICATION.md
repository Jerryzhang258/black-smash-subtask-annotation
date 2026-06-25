# Server verification checklist (what could not be validated locally)

The SIEVE memory layer was developed in a checkout that has **no raw
`observation.state`, no camera parquet, and no GPU / Qwen server**. So the
*logic* of the new scripts was verified locally, but anything that consumes real
proprioception, real frames, or the VLM still needs a run on the server
(`black_smash_05/06/07`). This file lists exactly what is still unverified and
how to close each gap.

## Already verified locally (for contrast)

- `candidate_propose.propose_windows`, `build_semantic_memory.build_memory`,
  `export_vla_memory.make_samples` — pure logic, on `examples/`.
- The **templated** memory path (no qwen-stage) and the cumulative VLA export.
- The existing arbiter regression still passes (framework untouched).
- Pipeline `bash -n`, schema JSON validity, `py_compile` of all new scripts.

Run: `python data_annotation/framework/tests/test_semantic_memory.py` and
`python data_annotation/framework/tests/test_arbitrate.py`.

## Needs the server

### 1. State boundaries → candidate windows on real episodes
**Why local fails:** `batch_annotate.py segment_episode` reads the 20-dim
`observation.state` parquet column, absent here. Locally we only fed
`candidate_propose.py` the example *annotation JSON*, so the window **math** is
verified but not whether windows actually bracket the true events on real data.

```bash
RUN_STATE=1 RUN_CANDIDATES=1 \
RUN_QWEN=0 RUN_FUSED=0 RUN_QWEN_STAGE=0 RUN_MEMORY=0 RUN_VLA_EXPORT=0 RUN_VIZ=0 \
DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 EPS=0,1,2 \
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
bash scripts/run_annotation_pipeline.sh
```
**Check:** `candidates_07/ep000_candidates.json` — each window's `[lo,hi]` should
straddle the real event; `start_pour` span wide / low `state_confidence`, gripper
events tight / high; `flagged:true` where the state doc had a "no clear window" flag.

### 2. VLM verification (Qwen2.5-VL-7B-AWQ via vLLM)
**Why local fails:** needs the GPU vLLM server (`scripts/start_vllm.sh`). The
structured-JSON outputs, the p2 window refinement picking the right frame, and the
qwen-stage semantic descriptions were never exercised here.

```bash
./scripts/start_vllm.sh /home/hillbot/models/Qwen2.5-VL-7B-Instruct-AWQ   # then, separately:
RUN_STATE=0 RUN_CANDIDATES=0 RUN_QWEN=1 RUN_FUSED=1 RUN_QWEN_STAGE=1 \
RUN_MEMORY=0 RUN_VLA_EXPORT=0 RUN_VIZ=0 \
DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 EPS=0,1,2 \
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
bash scripts/run_annotation_pipeline.sh
```
**Check:** `annotations_qwen_07` p2 lands inside the state-bracketed window;
qwen-stage `run_*/stage_annotations_normalized.jsonl` has 7 `normalized_stages`
with `prediction_prompt` / `expected_future_observation` / `reason`.

### 3. `build_semantic_memory.py` with a **real** `--stage-jsonl`
**Why local fails:** only the templated fallback ran. The branch that pulls
`memory_fact`/`future_relevance`/`evidence` from `normalized_stages` and the p2
`parsed_response.p2_reason`/`p2_confidence` was never run against a real qwen-stage
record — confirm the stage-index mapping (`_stage_after` uses stage `i+1`) and the
field names line up with the actual JSONL.

```bash
STAGE=$(ls -td annotations_qwen_stage_07/run_* | head -1)/stage_annotations_normalized.jsonl
python build_semantic_memory.py --fused annotations_fused_07 \
  --candidates candidates_07 --stage-jsonl "$STAGE" --out semantic_memory_07 --eps 0,1,2
```
**Check:** `has_semantics:true`; `semantic.future_relevance` reads like the model's
`prediction_prompt` (not the templated "Next, …"); `start_pour.evidence` == p2 reason.

### 4. `--dump-frames` (decode keyframes from parquet)
**Why local fails:** no parquet, and pandas/PIL decode path never ran. Verify the
camera column names (`observation.images.camera0/1`), the `episode_{id:06d}` /
`t{frame:04d}` file naming, and that JSON `visual_keyframe` paths are rewritten to
the saved files.

```bash
python build_semantic_memory.py --fused annotations_fused_07 --candidates candidates_07 \
  --out semantic_memory_07 --dump-frames --data /home/hillbot/black_smash_07/data/chunk-000 --eps 0
```
**Check:** `semantic_memory_07/ep000/camera0_t0234.jpg` etc. exist and open.

### 5. Full pipeline integration + no regression
**Why local fails:** the new `RUN_CANDIDATES/RUN_MEMORY/RUN_VLA_EXPORT` steps are
`bash -n`-clean but never executed in sequence with the env-var flow.

```bash
DATASET_ROOT=/home/hillbot/black_smash_07 DATASET_ID=07 EPS=0,1,2 \
PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
bash scripts/run_annotation_pipeline.sh
```
**Check:** `candidates_07/`, `semantic_memory_07/`, `vla_memory_07.jsonl` all
produced; **`compare_tracks_07` renders exactly as before** (the memory layer must
not change the state/qwen/fused/qwen-stage tracks).

### 6. The research evaluation (entirely future work)
Doc §12's comparisons — state+VLM vs state-only / VLM-only / KEMO-style; memory-fact
accuracy; VLA downstream TSR/SCR with vs without memory — are not implemented and
not in scope here. Listed so the gap is explicit.

## Optional-dependency note
The three new scripts use only the standard library, so they import anywhere.
`--dump-frames` additionally needs `pandas` + `Pillow`; the VLM steps need the
`openai` client + a running vLLM endpoint. Those optional branches are only
reachable on the server.
