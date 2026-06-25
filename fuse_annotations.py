"""
Stage 1+2 fusion -> targeted human review.

Merges the VLM annotation (mvt_annotations_vlm/) and the state-signal annotation
(mvt_annotations/) into a single fused annotation (mvt_annotations_fused/), giving
each critical point to the method that detects it best, and flagging the points
where the two methods disagree so Stage 3 only has to review those.

Per-point ownership (value taken from):
  p1 grasp_tube   -> state   (gripper close: crisp in proprioception)
  p2 start_pour   -> vlm     (visible powder/tilt; state only has a proxy)
  p3 release_tube -> state   (gripper open)
  p4 grasp_pestle -> state   (gripper close)
  p5 start_grind  -> state   (in-place motion onset)
  p6 lift_pestle  -> state   (end of grind)
VLM cross-checks the state-owned points; any point where |vlm - state| exceeds
--tol-s is marked for human review (review_points). Missing VLM -> all-state + flag.

Usage:
  python fuse_annotations.py
  python fuse_annotations.py --tol-s 0.5 --ep 0
"""
import os, json, glob, argparse
import numpy as np
from batch_annotate import CRIT_NAMES, LABELS

OWNER = ["state", "vlm", "state", "state", "state", "state"]   # value source per point


def enforce_order(cps, T, flags):
    cps = [max(1, min(T - 2, int(c))) for c in cps]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
            flags.append("nudged p%d for ordering" % (i + 1))
    if cps[-1] > T - 2:
        cps[-1] = T - 2
        for i in range(len(cps) - 2, -1, -1):
            if cps[i] >= cps[i + 1]:
                cps[i] = cps[i + 1] - 1
        if cps[0] < 1:
            flags.append("could not fit ordered points in episode")
    return cps


def subtasks_from_cps(cps, T, fps):
    starts = [0] + list(cps)
    out = []
    for i, label in enumerate(LABELS):
        start = starts[i]
        end = starts[i + 1] - 1 if i < len(LABELS) - 1 else T - 1
        out.append({
            "subtask_id": i,
            "label": label,
            "start_frame": start,
            "end_frame": end,
            "start_t": round(start / fps, 2),
            "end_t": round(end / fps, 2),
            "n_frames": end - start + 1,
            "dur_s": round((end - start + 1) / fps, 2),
        })
    return out


def load(d, ep):
    fp = os.path.join(d, f"ep{ep:03d}_subtasks.json")
    return json.load(open(fp)) if os.path.exists(fp) else None


def fuse_ep(ep, state_dir, vlm_dir, out_dir, tol_frames):
    A = load(state_dir, ep)
    if A is None:
        return None, "no state annotation"
    V = load(vlm_dir, ep)
    T, fps = A["n_frames"], A["fps"]
    scp = A["critical_points"]
    vcp = V["critical_points"] if V else None
    flags = []

    fused, sources, disagree, review = [], [], [], []
    for i in range(6):
        if vcp is None:
            fused.append(scp[i]); sources.append("state"); disagree.append(None)
            continue
        d = abs(vcp[i] - scp[i])
        disagree.append(int(d))
        val = vcp[i] if OWNER[i] == "vlm" else scp[i]
        # if the owner is VLM but VLM is way off vs state, fall back to state and review
        if OWNER[i] == "vlm" and d > tol_frames:
            val = scp[i]; flags.append("p%d: vlm-owned but disagrees, used state" % (i + 1))
        fused.append(val); sources.append(OWNER[i])
        if d > tol_frames:
            review.append(i + 1)            # 1-based point id
    if vcp is None:
        flags.append("no vlm annotation -> all points from state")
        review = [1, 2, 3, 4, 5, 6]

    fused = enforce_order(fused, T, flags)
    doc = {"episode_index": ep, "task": A["task"], "n_frames": T, "fps": fps,
           "annotator": "fused", "tol_frames": tol_frames,
           "critical_points": fused, "critical_names": CRIT_NAMES, "subtask_starts": [0] + fused,
           "sources": sources, "state_cps": scp, "vlm_cps": vcp,
           "disagree_frames": disagree, "review_points": review, "flags": flags,
           "n_subtasks": 7, "subtasks": subtasks_from_cps(fused, T, fps)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(doc, open(os.path.join(out_dir, f"ep{ep:03d}_subtasks.json"), "w"), indent=2)
    idx = np.zeros(T, dtype=np.int16)
    for s in doc["subtasks"]:
        idx[s["start_frame"]:s["end_frame"] + 1] = s["subtask_id"]
    np.save(os.path.join(out_dir, f"ep{ep:03d}_subtask_index.npy"), idx)
    return doc, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=r"C:\Intern\mvt_annotations")
    ap.add_argument("--vlm",   default=r"C:\Intern\mvt_annotations_vlm")
    ap.add_argument("--out",   default=r"C:\Intern\mvt_annotations_fused")
    ap.add_argument("--fps",   type=int, default=30)
    ap.add_argument("--tol-s", type=float, default=0.5, dest="tol_s",
                    help="|vlm-state| above this (seconds) flags a point for review")
    ap.add_argument("--ep", type=int, default=None)
    args = ap.parse_args()
    tol = int(args.tol_s * args.fps)

    if args.ep is not None:
        eps = [args.ep]
    else:
        eps = sorted(int(os.path.basename(f).split("_")[0][2:])
                     for f in glob.glob(os.path.join(args.state, "ep*_subtasks.json")))

    docs, all_dis = [], []
    for ep in eps:
        doc, err = fuse_ep(ep, args.state, args.vlm, args.out, tol)
        if err:
            print(f"ep{ep:03d}: {err}"); continue
        docs.append(doc)
        if doc["vlm_cps"] is not None:
            all_dis.append(doc["disagree_frames"])
        rp = doc["review_points"]
        print(f"ep{ep:03d}  fused={doc['critical_points']}  review_points={rp}"
              + ("  FLAG:" + ";".join(doc["flags"]) if doc["flags"] else ""))

    n_review = sum(len(d["review_points"]) for d in docs)
    print(f"\n{len(docs)} episodes fused -> {args.out}")
    print(f"total points needing human review: {n_review} "
          f"(of {len(docs)*6}) across {sum(1 for d in docs if d['review_points'])} episodes")
    if all_dis:
        D = np.array(all_dis, dtype=float)
        print("vlm-vs-state mean |Δ| per point (frames / s):")
        for i, nm in enumerate(CRIT_NAMES):
            print(f"  p{i+1} {nm:12s} {D[:,i].mean():5.1f}f / {D[:,i].mean()/args.fps:.2f}s")


if __name__ == "__main__":
    main()
