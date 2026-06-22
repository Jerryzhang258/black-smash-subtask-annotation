"""
Orchestrator: schema + detectors + arbiter -> fused critical points.

This is the framework replacement for fuse_annotations.py. The output JSON is a
superset of the legacy fused doc (same keys: critical_points, sources,
disagree_frames, review_points, flags, subtasks ...) plus per-point `sigmas` and
arbiter `notes`, so downstream tools and visualize_annotation_tracks.py keep
working unchanged.

Two input modes:
  file    : read existing ep<NNN>_subtasks.json from --state / --vlm dirs
            (reproduces the legacy pipeline; needs no raw data).
  parquet : run signal detectors on observation.state from --data, optionally
            cross-checked by a --vlm annotation dir.

Examples:
  # reproduce legacy fusion from existing annotations
  python -m data_annotation.framework.fuse file \
      --state annotations_state_07 --vlm annotations_qwen_07 --out annotations_fused_07

  # run from raw state, with vlm cross-check
  python -m data_annotation.framework.fuse parquet \
      --data /data/black_smash_07/data/chunk-000 --vlm annotations_qwen_07 \
      --out annotations_fused_07
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from .arbitrate import arbitrate_event, enforce_order
from .core import Candidate, TaskSchema, load_schema
from .detectors import (
    EpisodeContext,
    OrientationDetector,
    SignalStateDetector,
    StateFileDetector,
    VLMFileDetector,
)

DEFAULT_SCHEMA = Path(__file__).parent / "schemas" / "black_smash.json"


def subtasks_from_cps(cps: list[int], T: int, fps: int, labels: list[str]) -> list[dict]:
    starts = [0] + list(cps)
    n = len(labels)
    out = []
    for i in range(n):
        a = starts[i]
        b = (starts[i + 1] - 1) if i < n - 1 else T - 1
        out.append({
            "subtask_id": i, "label": labels[i], "start_frame": a, "end_frame": b,
            "start_t": round(a / fps, 2), "end_t": round(b / fps, 2),
            "n_frames": b - a + 1, "dur_s": round((b - a + 1) / fps, 2),
        })
    return out


def fuse_episode(
    schema: TaskSchema,
    ctx: EpisodeContext,
    detectors: list,
    tol_frames: int,
    sigma_max: float,
) -> dict:
    # gather candidates per event index
    per_event: dict[int, list[Candidate]] = {ev.index: [] for ev in schema.events}
    for det in detectors:
        proposals = det.propose(schema, ctx)
        for idx, cand in proposals.items():
            ev = schema.events[idx]
            if det.handles(ev.type):
                per_event[idx].append(cand)

    T, fps = ctx.n_frames, ctx.fps
    frames: list[int] = []
    sources: list[str] = []
    sigmas: list[float] = []
    state_cps: list[int | None] = []
    vlm_cps: list[int | None] = []
    disagree: list[int | None] = []
    review: list[int] = []
    flags: list[str] = []

    for ev in schema.events:
        cands = per_event[ev.index]
        dec = arbitrate_event(ev, cands, tol_frames, sigma_max)
        frames.append(int(dec.frame) if dec.frame is not None else 0)
        sources.append(dec.source)
        sigmas.append(round(dec.sigma, 2) if dec.sigma != float("inf") else None)
        if dec.note:
            flags.append(f"{ev.id}: {dec.note}")
        if dec.needs_review:
            review.append(ev.index + 1)

        s_cand = next((c.frame for c in cands if c.modality == "state"), None)
        v_cand = next((c.frame for c in cands if c.modality == "vision"), None)
        state_cps.append(s_cand)
        vlm_cps.append(v_cand)
        disagree.append(abs(v_cand - s_cand) if (s_cand is not None and v_cand is not None) else None)

    frames = enforce_order(frames, T, flags)
    doc = {
        "episode_index": ctx.episode_index,
        "task": schema.task_description or schema.task,
        "n_frames": T, "fps": fps,
        "annotator": "fused-arbitrated", "tol_frames": tol_frames, "sigma_max": sigma_max,
        "critical_points": frames, "critical_names": schema.crit_names,
        "subtask_starts": [0] + frames,
        "sources": sources, "sigmas": sigmas,
        "state_cps": state_cps, "vlm_cps": vlm_cps,
        "disagree_frames": disagree, "review_points": review, "flags": flags,
        "n_subtasks": len(schema.subtasks),
        "subtasks": subtasks_from_cps(frames, T, fps, schema.subtasks),
    }
    return doc


def _write(out_dir: Path, doc: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = doc["episode_index"]
    (out_dir / f"ep{ep:03d}_subtasks.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    try:
        import numpy as np
        idx = np.zeros(doc["n_frames"], dtype=np.int16)
        for s in doc["subtasks"]:
            idx[s["start_frame"]:s["end_frame"] + 1] = s["subtask_id"]
        np.save(out_dir / f"ep{ep:03d}_subtask_index.npy", idx)
    except Exception:
        pass  # npy is optional; json is the source of truth


def _load_doc(d: str | None, ep: int) -> dict | None:
    if not d:
        return None
    fp = Path(d) / f"ep{ep:03d}_subtasks.json"
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None


def _eps_from_state_dir(state_dir: str) -> list[int]:
    return sorted(
        int(os.path.basename(f).split("_")[0][2:])
        for f in glob.glob(os.path.join(state_dir, "ep*_subtasks.json"))
    )


def build_detectors(mode: str, use_orientation: bool) -> list:
    if mode == "file":
        dets = [StateFileDetector(), VLMFileDetector()]
    else:
        dets = [SignalStateDetector(), VLMFileDetector()]
    if use_orientation:
        dets.append(OrientationDetector())
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["file", "parquet"])
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--state", help="dir with state ep<NNN>_subtasks.json (file mode)")
    ap.add_argument("--vlm", help="dir with vlm ep<NNN>_subtasks.json (cross-check)")
    ap.add_argument("--data", help="chunk dir with episode_*.parquet (parquet mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-s", type=float, default=0.5, dest="tol_s")
    ap.add_argument("--sigma-max", type=float, default=12.0, dest="sigma_max")
    ap.add_argument("--orientation", action="store_true", help="enable experimental state orientation detector")
    ap.add_argument("--eps", default="", help="comma list of episode indices; default = all")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    fps = schema.fps
    tol = int(args.tol_s * fps)
    out_dir = Path(args.out)
    detectors = build_detectors(args.mode, args.orientation)

    want = [int(x) for x in args.eps.split(",") if x.strip().isdigit()] if args.eps else None

    if args.mode == "file":
        if not args.state:
            ap.error("file mode needs --state")
        eps = want or _eps_from_state_dir(args.state)
        n_review = 0
        for ep in eps:
            state_doc = _load_doc(args.state, ep)
            if state_doc is None:
                print(f"ep{ep:03d}: no state annotation, skipped")
                continue
            ctx = EpisodeContext(ep, state_doc["n_frames"], fps,
                                 docs={"state": state_doc, "vlm": _load_doc(args.vlm, ep)})
            doc = fuse_episode(schema, ctx, detectors, tol, args.sigma_max)
            _write(out_dir, doc)
            n_review += len(doc["review_points"])
            print(f"ep{ep:03d}  fused={doc['critical_points']}  review={doc['review_points']}"
                  + ("  " + ";".join(doc["flags"]) if doc["flags"] else ""))
        print(f"\nfused -> {out_dir}  (total points to review: {n_review})")
    else:
        if not args.data:
            ap.error("parquet mode needs --data")
        import numpy as np
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.data, "episode_*.parquet")))
        for fp in files:
            ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
            if want is not None and ep not in want:
                continue
            S = np.stack([np.asarray(x, dtype=np.float64)
                          for x in pd.read_parquet(fp, columns=["observation.state"])["observation.state"].values])
            ctx = EpisodeContext(ep, len(S), fps, state=S, docs={"vlm": _load_doc(args.vlm, ep)})
            doc = fuse_episode(schema, ctx, detectors, tol, args.sigma_max)
            _write(out_dir, doc)
            print(f"ep{ep:03d}  fused={doc['critical_points']}  review={doc['review_points']}")
        print(f"\nfused -> {out_dir}")


if __name__ == "__main__":
    main()
