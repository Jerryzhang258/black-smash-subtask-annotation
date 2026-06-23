"""
Ego wipe-tube subtask annotation — batch runner (signal-first + optional VLM fusion).

For every demo folder under --demos it:
  1. Stage 2 (always): proprioceptive segmentation -> 5 critical points, 6 subtasks.
  2. Stage 1 (--vlm):   VLM prior over egocentric frames (OpenAI-compatible API).
  3. Stage 3 (--vlm):   per-point fusion; |VLM - state| > tol -> review_points.

Writes, per demo:  <out>/<name>_subtasks.json  +  <name>_subtask_index.npy
Dataset-level:     <out>/summary.csv  +  flagged list (printed)

Usage:
  python -m ego_wipe_annotation.run                          # signal only, all demos
  python -m ego_wipe_annotation.run --demos D:\test_clean\demos --eps 0
  python -m ego_wipe_annotation.run --visualize --qa          # timeline + boundary QA images
  python -m ego_wipe_annotation.run --vlm --base-url http://localhost:8000/v1 \
         --model qwen --tol-s 0.5 --visualize
"""
from __future__ import annotations
import os, csv, json, argparse
import numpy as np

from . import config as C
from . import ego_dataio as io
from . import signal_segment as seg


def _index_npy(subtasks: list[dict], n: int) -> np.ndarray:
    idx = np.zeros(n, dtype=np.int16)
    for s in subtasks:
        idx[s["start_frame"]:s["end_frame"] + 1] = s["subtask_id"]
    return idx


def annotate_demo(demo, args, vlm_backend=None) -> dict:
    cps, subtasks, flags = seg.segment(demo)
    doc = {
        "demo": demo.name, "task": "wipe the test tube", "n_frames": demo.n_frames,
        "fps": round(demo.fps, 3), "annotator": "auto-signal",
        "method": "ego signal-derived (gripper width grasp/release + pose wipe-onset)",
        "critical_points": cps, "critical_names": C.CRIT_NAMES,
        "subtask_starts": [0] + cps, "flags": flags,
        "n_subtasks": len(subtasks), "subtasks": subtasks,
    }

    if vlm_backend is not None:
        from . import vlm_prior, fuse
        vlm_cps, vlm_flags = vlm_prior.propose(vlm_backend, demo, args)
        doc["vlm_critical_points"] = vlm_cps
        fused, sources, disagree, review = fuse.fuse(cps, vlm_cps, demo.fps, args.tol_s)
        doc.update({
            "annotator": "signal+vlm-fused", "critical_points": fused,
            "subtask_starts": [0] + fused, "subtasks": seg._subtasks_from_cps(fused, demo.n_frames, demo.fps),
            "sources": sources, "disagree_frames": disagree, "review_points": review,
            "flags": flags + vlm_flags + ([f"review {len(review)} points"] if review else []),
        })

    json.dump(doc, open(os.path.join(args.out, f"{demo.name}_subtasks.json"), "w"), indent=2)
    np.save(os.path.join(args.out, f"{demo.name}_subtask_index.npy"),
            _index_npy(doc["subtasks"], demo.n_frames))
    if args.visualize or args.qa or args.qa_dual or args.combined or args.dashboard:
        from . import visualize
        sfx = "" if C.FRAME_SOURCE == "hand" else f"_{C.FRAME_SOURCE}_{C.EGO_EYE}"
        if args.visualize:
            visualize.render(demo, doc, os.path.join(args.out, f"{demo.name}_timeline{sfx}.png"))
        if args.qa:
            visualize.render_boundaries(demo, doc, os.path.join(args.out, f"{demo.name}_boundaries{sfx}.png"))
        if args.qa_dual:
            visualize.render_boundaries_dual(demo, doc, os.path.join(args.out, f"{demo.name}_boundaries_dual_{C.EGO_EYE}.png"))
        if args.combined:
            visualize.render_combined(demo, doc, os.path.join(args.out, f"{demo.name}_combined_{C.EGO_EYE}.png"))
        if args.dashboard:
            _render_dashboard(demo, doc, visualize, os.path.join(args.out, f"{demo.name}_dashboard_{C.EGO_EYE}.png"))
    return doc


def _render_dashboard(demo, doc, visualize, out_png):
    """Compose timeline + ego + fisheye + proprio/vision signals (one frame axis)."""
    from . import signal_segment as seg, vision_signal as vs
    flags = doc["flags"]
    wiper = next((f.split("=")[1] for f in flags if f.startswith("wiper=")), "right")
    cps = doc["critical_points"]
    gl = seg.norm01(demo.grip["left"]); gr = seg.norm01(demo.grip["right"])
    traces = [("L grip", (90, 150, 255), gl), ("R grip", (90, 230, 230), gr)]
    marks = []
    if demo.pose.get(wiper) is not None:
        _, drift = seg._speeds(demo.pose[wiper], demo.fps)
        energy = vs.frame_diff_energy(demo, wiper)
        traces += [("pose drift", (245, 170, 70), drift), ("fisheye motion", (235, 90, 210), energy)]
        vis_w, _ = vs.wipe_onset_vision(energy, demo.fps, cps[1], cps[3])
        fused, _, _ = vs.fuse_wipe_onset(cps[2], vis_w, demo.fps)
        marks = [("pose", (90, 245, 120), cps[2]), ("vis", (245, 80, 80), vis_w), ("fused", (255, 220, 0), fused)]
    visualize.render_dashboard(demo, doc, traces, out_png, marks=marks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=C.DEMOS_ROOT)
    ap.add_argument("--out", default=C.OUT_ROOT)
    ap.add_argument("--eps", default="", help="comma list of demo indices (order in --demos); default all")
    ap.add_argument("--vlm", action="store_true", help="run VLM prior + fusion")
    ap.add_argument("--backend", default="openai", choices=["openai"])
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", dest="base_url")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", dest="api_key_env")
    ap.add_argument("--n-frames", type=int, default=24, dest="n_frames", help="VLM coarse frames")
    ap.add_argument("--tol-s", type=float, default=0.5, dest="tol_s", help="fusion disagreement tol (s)")
    ap.add_argument("--visualize", action="store_true", help="write <name>_timeline.png")
    ap.add_argument("--qa", action="store_true", help="write <name>_boundaries.png (per-critical-point zoom)")
    ap.add_argument("--qa-dual", action="store_true", dest="qa_dual", help="write <name>_boundaries_dual.png (ego + fisheye rows per critical point; needs pyav)")
    ap.add_argument("--combined", action="store_true", help="write <name>_combined.png (ego + fisheye side by side; needs pyav)")
    ap.add_argument("--dashboard", action="store_true", help="write <name>_dashboard.png (timeline + ego + fisheye + signals; needs pyav)")
    ap.add_argument("--frames", choices=["hand", "ego"], default=C.FRAME_SOURCE,
                    help="visual source for viz/VLM: hand-fisheye jpgs or headset ego mp4 (needs pyav)")
    ap.add_argument("--eye", choices=["left", "right"], default=C.EGO_EYE, help="ego stereo eye")
    ap.add_argument("--ego-align", choices=["elapsed", "proportional"], default=C.EGO_ALIGN,
                    dest="ego_align", help="ego<->hand timestamp alignment mode")
    ap.add_argument("--ego-offset-s", type=float, default=C.EGO_TIME_OFFSET_S, dest="ego_offset_s",
                    help="shift ego later(+)/earlier(-) vs hand, seconds")
    ap.add_argument("--align-report", action="store_true", dest="align_report",
                    help="print ego<->hand alignment diagnostics per demo")
    args = ap.parse_args()
    C.FRAME_SOURCE, C.EGO_EYE = args.frames, args.eye
    C.EGO_ALIGN, C.EGO_TIME_OFFSET_S = args.ego_align, args.ego_offset_s
    os.makedirs(args.out, exist_ok=True)

    demos, skipped = io.scan_demos(args.demos)
    for path, why in skipped:
        print(f"[skip] {os.path.basename(path)}: {why}")
    if not demos:
        raise SystemExit(f"no complete demo folders under {args.demos}")
    if args.eps.strip():
        want = {int(x) for x in args.eps.split(",") if x.strip().isdigit()}
        demos = [d for i, d in enumerate(demos) if i in want]

    vlm_backend = None
    if args.vlm:
        from . import vlm_prior
        vlm_backend = vlm_prior.make_backend(args)

    rows, flagged = [], []
    for path in demos:
        demo = io.load_demo(path)
        doc = annotate_demo(demo, args, vlm_backend)
        cps = doc["critical_points"]
        durs = [s["dur_s"] for s in doc["subtasks"]]
        rows.append([demo.name, demo.n_frames, round(demo.fps, 1)] + cps + durs + ["|".join(doc["flags"])])
        review = doc.get("review_points")
        tag = f"  REVIEW{review}" if review else ""
        print(f"{demo.name}  N={demo.n_frames} fps={demo.fps:.1f}  cps={cps}{tag}")
        if args.align_report:
            print(f"  align: {io.alignment_report(demo)}")
        if any(f for f in doc["flags"] if not f.startswith(("holder=", "wiper="))):
            flagged.append((demo.name, "; ".join(doc["flags"])))

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demo", "n_frames", "fps"]
                   + [f"c{i+1}_{n}" for i, n in enumerate(C.CRIT_NAMES)]
                   + [f"S{i}_dur" for i in range(len(C.LABELS))] + ["flags"])
        w.writerows(rows)

    print(f"\n{len(rows)} demos annotated -> {args.out}")
    print(f"flagged: {len(flagged)}")
    for name, why in flagged:
        print(f"  {name}: {why}")


if __name__ == "__main__":
    main()
