# SIEVE — creative points mapped to code

**SIEVE = State-Proposed, VLM-Verified Semantic Keyframe Memory.**

> State proposes where to look. VLM verifies what matters. VLA remembers why it matters.

This repo is the data side of SIEVE: it turns long-horizon bimanual demonstrations
into semantic keyframe memory for VLA policies. The pipeline reuses the existing,
proven boundary logic and adds a thin memory layer on top — nothing in the working
state/qwen/fused path was rewritten.

```text
long-horizon trajectory
  -> state critical points        (batch_annotate.py, unchanged)
  -> candidate windows            (candidate_propose.py)            [SIEVE 1]
  -> VLM verification             (vlm_annotate.py p2-history, unchanged) [SIEVE 2/3]
  -> confidence-aware fusion      (data_annotation/framework, unchanged)  [SIEVE 5]
  -> semantic keyframe memory     (build_semantic_memory.py)        [SIEVE 4]
  -> VLA memory export            (export_vla_memory.py)            [SIEVE 7]
```

## Creative points → where they live

| # | doc creative point | code | status |
|---|---|---|---|
| 1 | State-Proposed Candidate Windows | `candidate_propose.py` (`propose_windows`); per-event `window_half_s`/`state_confidence` in `data_annotation/framework/schemas/black_smash.json` | added |
| 2 | VLM Semantic Verification | `vlm_annotate.py` (`refine_p2_with_history`) verifies inside the state-bracketed window and returns `{frame, confidence, reason}` | existing |
| 3 | Candidate Window + Local VLM Refinement | same `p2-history` window refinement; `candidate_propose.py` generalizes the window representation to every event | existing + added |
| 4 | Semantic Keyframe Memory | `build_semantic_memory.py` (`build_memory`) → `semantic_memory_<id>/ep<NNN>_memory.json` | added |
| 5 | Confidence-Aware State-VLM Fusion | `data_annotation/framework/arbitrate.py` (`arbitrate_event`) — ownership derived from per-detector `sigma`, not a hardcoded table | existing |
| 6 | Bounded VLM, not online planner | design invariant: the VLM only verifies state-proposed windows and writes structured JSON; it never free-form splits stages (see root `README.md` "Why This Design") | existing |
| 7 | Memory-Supervised VLA Data Generator | `export_vla_memory.py` (`make_samples`) → `vla_memory_<id>.jsonl` | added |

## Data contracts

**Candidate window** (`candidates_<id>/ep<NNN>_candidates.json`):

```json
{"candidate_id": 0, "event_hint": "grasp_tube", "center": 172, "lo": 163, "hi": 181,
 "state_source": "tube_gripper_close", "state_confidence": 0.95, "flagged": false}
```

**Semantic memory entry** (`semantic_memory_<id>/ep<NNN>_memory.json`):

```json
{"memory_id": 1, "frame": 234, "event_type": "start_pour",
 "visual_keyframe": {"camera0": "ep000/camera0_t0234.jpg", "camera1": "ep000/camera1_t0234.jpg"},
 "state_transition": {"summary": "lift the test tube ... -> pour the black powder ...", "source": "vlm"},
 "semantic": {"memory_fact": "Black powder has started pouring ...",
              "future_relevance": "...", "evidence": "...", "event_confidence": 0.86},
 "needs_review": true, "source": "state_proposed_vlm_verified"}
```

`event_confidence` fuses the candidate `state_confidence`, the boundary source
(state vs VLM), the VLM p2 confidence (when qwen-stage is present), and the
review flag. With no qwen-stage run, `memory_fact`/`future_relevance` are
templated from the task taxonomy carried in the boundary doc.

**VLA training sample** (`vla_memory_<id>.jsonl`, one per timestep `t`):

```json
{"episode_index": 0, "t": 250, "instruction": "...", "mode": "text-prefix",
 "memory": [{"frame": 172, "event_type": "grasp_tube", "memory_fact": "...", "confidence": 0.66},
            {"frame": 234, "event_type": "start_pour", "memory_fact": "...", "confidence": 0.32}],
 "text_prefix": "Task:\n...\n\nTask memory:\n1. ...\n2. ...\n\nCurrent instruction:\nContinue the task.",
 "action_chunk_ref": {"episode_index": 0, "start_t": 250, "horizon": 16}}
```

The memory is **cumulative**: a sample at time `t` includes exactly the keyframes
with `frame <= t`, so the prefix grows monotonically across the episode.

## Contributions (paper-ready)

1. **State-Proposed Candidate Keyframe Windows** — proprioception-grounded module
   that reduces dense long-horizon trajectories into sparse, high-recall candidate
   event windows.
2. **Bounded VLM Semantic Verification** — the VLM verifies only state-proposed
   windows and emits structured memory facts, instead of free-form stage splitting.
3. **Semantic Keyframe Memory for VLA** — verified keyframes become
   visual-state-language memory entries that condition VLA policies for
   long-horizon manipulation.

## Deferred (out of current scope)

- Full `segment_episode` rewrite from points to windows (P3 in the framework).
- A standalone per-event VLM verifier (`vlm_verify_candidates.py`) beyond p2.
- `--mode hybrid` token memory wired into a real VLA tokenizer (stubbed in
  `export_vla_memory.py`).
- Lightweight verifier distillation (doc MVP-5).
