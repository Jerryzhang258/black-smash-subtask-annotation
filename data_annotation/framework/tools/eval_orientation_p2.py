"""
Evaluate the state-based pour detector (OrientationDetector) against a reference
fused p2, implementing the validation protocol in docs/P2_ORIENTATION_OPTIMIZATION.md.
Does not change production outputs -- it only runs the detector and prints metrics.

  python data_annotation/framework/tools/eval_orientation_p2.py \
      --data /home/hillbot/black_smash_07/data/chunk-000 \
      --ref-fused annotations_fused_07 \
      --wrist-dims 2,5 --eps 0,1,2,3,4

Per the protocol, prints: episodes, fire/abstain counts, sigma split,
median/mean/max |orientation_p2 - fused_p2|, p2-p1 gap stats, count p2-p1<=10,
count p2 outside [p1+30, p3-10], and the worst episodes for visual inspection.

Acceptance (before enabling in production): p2-p1<=10 near zero, median
|orientation_p2 - fused_p2| clearly below the current Qwen/state disagreement,
outliers visually explainable, and the detector abstains on uncertain episodes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_annotation.framework.core import load_schema           # noqa: E402
from data_annotation.framework.detectors import (                # noqa: E402
    EpisodeContext, OrientationDetector,
)

DEFAULT_SCHEMA = REPO_ROOT / "data_annotation" / "framework" / "schemas" / "black_smash.json"


def load_state(parquet: str) -> np.ndarray:
    col = pd.read_parquet(parquet, columns=["observation.state"])["observation.state"].values
    return np.stack([np.asarray(x, dtype=np.float64) for x in col])


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return (float(np.median(a)), float(a.mean()), float(a.max())) if len(a) else (0.0, 0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="chunk dir with episode_*.parquet")
    ap.add_argument("--ref-fused", required=True, dest="ref_fused",
                    help="reference fused dir for p1/p2/p3 and comparison")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--wrist-dims", default="", dest="wrist_dims",
                    help="override schema tube_wrist_dims, e.g. 2,5")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--eps", default="")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    schema = load_schema(args.schema)
    if args.wrist_dims:
        schema.state_layout["tube_wrist_dims"] = [int(x) for x in args.wrist_dims.split(",") if x.strip().isdigit()]
    if not schema.state_layout.get("tube_wrist_dims"):
        ap.error("no tube_wrist_dims in schema; pass --wrist-dims, e.g. --wrist-dims 2,5")

    pour_idx = next(e.index for e in schema.events if e.type == "orientation_change")
    det = OrientationDetector()
    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None

    n = fired = abstained = 0
    abs_err, gaps, early, outside = [], [], 0, 0
    strong = weak = 0
    rows = []  # (ep, orient_p2, ref_p2, err, gap, sigma)

    for fp in sorted(glob.glob(os.path.join(args.data, "episode_*.parquet"))):
        ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if want is not None and ep not in want:
            continue
        ref_fp = os.path.join(args.ref_fused, f"ep{ep:03d}_subtasks.json")
        if not os.path.exists(ref_fp):
            continue
        ref = json.load(open(ref_fp))["critical_points"]
        p1, ref_p2, p3 = ref[0], ref[1], ref[2]
        S = load_state(fp)
        n += 1
        cand = det.propose(schema, EpisodeContext(ep, len(S), args.fps, state=S)).get(pour_idx)
        if cand is None:
            abstained += 1
            continue
        fired += 1
        err = abs(cand.frame - ref_p2)
        gap = cand.frame - p1
        abs_err.append(err)
        gaps.append(gap)
        if gap <= 10:
            early += 1
        if not (p1 + 30 <= cand.frame <= p3 - 10):
            outside += 1
        if cand.sigma <= 6.0:
            strong += 1
        else:
            weak += 1
        rows.append((ep, cand.frame, ref_p2, err, gap, cand.sigma))

    print(f"episodes evaluated: {n}")
    print(f"orientation fired : {fired}    abstained: {abstained}")
    if fired:
        em, ea, ex = stats(abs_err)
        gm, ga, gx = stats(gaps)
        fps = args.fps
        print(f"sigma split       : strong(<=6) {strong}   weak(>6) {weak}")
        print(f"|orient_p2-ref_p2|: median {em:.0f}f/{em/fps:.2f}s  mean {ea:.0f}f  max {ex:.0f}f")
        print(f"p2 - p1 gap       : median {gm:.0f}f  mean {ga:.0f}f  min {gm and min(gaps)}")
        print(f"p2 - p1 <= 10     : {early}  (should be near zero)")
        print(f"p2 outside guard  : {outside}  (outside [p1+30, p3-10])")
        print(f"\nworst {min(args.worst, len(rows))} by |orient_p2 - ref_p2| (inspect visually):")
        for ep, op2, rp2, err, gap, sig in sorted(rows, key=lambda r: -r[3])[:args.worst]:
            print(f"  ep{ep:03d}  orient_p2={op2:>4}  ref_p2={rp2:>4}  |Δ|={err:>4}  gap={gap:>4}  σ={sig}")
    else:
        print("orientation never fired -- widen window or check tube_wrist_dims.")


if __name__ == "__main__":
    main()
