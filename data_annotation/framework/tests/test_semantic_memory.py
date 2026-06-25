"""
Local regression for the SIEVE memory layer (candidate_propose -> build_semantic_memory
-> export_vla_memory). Runs with NO raw data and NO numpy/pandas: it uses the
committed example annotations in examples/ and exercises the pure functions.

  python data_annotation/framework/tests/test_semantic_memory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_annotation.framework.core import load_schema                      # noqa: E402
from data_annotation.framework.detectors import (                           # noqa: E402
    EpisodeContext, StateFileDetector, VLMFileDetector,
)
from data_annotation.framework.fuse import fuse_episode                     # noqa: E402

import candidate_propose                                                    # noqa: E402
import build_semantic_memory                                               # noqa: E402
import export_vla_memory                                                   # noqa: E402

SCHEMA = REPO_ROOT / "data_annotation" / "framework" / "schemas" / "black_smash.json"
EXAMPLES = REPO_ROOT / "examples"


def main() -> int:
    schema = load_schema(SCHEMA)
    state_doc = json.loads((EXAMPLES / "sample_ep000_subtasks.json").read_text())
    vlm_doc = json.loads((EXAMPLES / "sample_ep000_vlm_subtasks.json").read_text())

    # 1) verified boundaries via the existing arbiter (same call as test_arbitrate)
    tol = int(0.5 * schema.fps)
    ctx = EpisodeContext(0, state_doc["n_frames"], schema.fps,
                         docs={"state": state_doc, "vlm": vlm_doc})
    fused = fuse_episode(schema, ctx, [StateFileDetector(), VLMFileDetector()], tol, 12.0)

    # 2) state -> candidate windows (SIEVE step 1)
    event_cfg = candidate_propose.load_event_config(str(SCHEMA))
    cand_doc = candidate_propose.episode_doc(state_doc, event_cfg)

    # 3) semantic keyframe memory, no qwen-stage -> templated facts (SIEVE step 2)
    mem_doc = build_semantic_memory.build_memory(fused, cand_doc, stage_rec=None)

    # 4) VLA text-prefix samples (SIEVE step 3)
    samples = export_vla_memory.make_samples(mem_doc, stride=60)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}")

    # candidate windows: one per event, high-recall (lo<=center<=hi), conf in [0,1]
    cw = cand_doc["candidate_windows"]
    check("candidate: one window per event", len(cw) == schema.n_events)
    check("candidate: lo<=center<=hi", all(w["lo"] <= w["center"] <= w["hi"] for w in cw))
    check("candidate: confidence in [0,1]", all(0 <= w["state_confidence"] <= 1 for w in cw))
    pour = next(w for w in cw if w["event_hint"] == "start_pour")
    grasp = next(w for w in cw if w["event_hint"] == "grasp_tube")
    check("candidate: pour window wider than grasp (state is weak on pour)",
          (pour["hi"] - pour["lo"]) > (grasp["hi"] - grasp["lo"]))
    check("candidate: pour confidence < grasp confidence",
          pour["state_confidence"] < grasp["state_confidence"])

    # memory entries: one per critical point, ordered, well-formed
    mem = mem_doc["memory"]
    frames = [e["frame"] for e in mem]
    check("memory: one entry per critical point", len(mem) == schema.n_events)
    check("memory: frames strictly increasing", all(a < b for a, b in zip(frames, frames[1:])))
    check("memory: every entry has a non-empty memory_fact",
          all(e["semantic"]["memory_fact"].strip() for e in mem))
    check("memory: event_confidence in [0,1]",
          all(0 <= e["semantic"]["event_confidence"] <= 1 for e in mem))
    check("memory: visual_keyframe references both cameras",
          all({"camera0", "camera1"} <= set(e["visual_keyframe"]) for e in mem))
    check("memory: source is state_proposed_vlm_verified (fused boundaries)",
          all(e["source"] == "state_proposed_vlm_verified" for e in mem))

    # VLA samples: cumulative, monotonically growing memory prefix over t
    sizes_by_t = [(s["t"], len(s["memory"])) for s in samples]
    grew = all(b >= a for (_, a), (_, b) in zip(sizes_by_t, sizes_by_t[1:]))
    check("vla: memory prefix is non-decreasing over t", grew)
    last = samples[-1]
    check("vla: final sample carries all keyframes", len(last["memory"]) == len(mem))
    check("vla: text_prefix lists the remembered facts",
          last["memory"][0]["memory_fact"] in last["text_prefix"])
    check("vla: only past keyframes are included",
          all(all(m["frame"] <= s["t"] for m in s["memory"]) for s in samples))

    print("\nsample memory entry:")
    print(json.dumps(mem[1], indent=2, ensure_ascii=False))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
