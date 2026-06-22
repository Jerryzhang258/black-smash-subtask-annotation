"""
Visual debug for the pour detector (docs/P2_ORIENTATION_OPTIMIZATION.md, item 4).
For each episode it plots the selected wrist state dim(s) over time and marks
p1 (grasp), the reference fused p2, the detected orientation p2, and p3 (release),
shading the guarded search window. Use it to eyeball best matches, big
disagreements, early-p2 failures, and abstained episodes.

  python data_annotation/framework/tools/plot_orientation_p2.py \
      --data /home/hillbot/black_smash_07/data/chunk-000 \
      --ref-fused annotations_fused_07 --wrist-dims 2,5 \
      --eps 0,1,2 --out compare_tracks_07/orientation_debug

Writes one PNG per episode. Read-only w.r.t. annotations.
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref-fused", required=True, dest="ref_fused")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--wrist-dims", default="", dest="wrist_dims")
    ap.add_argument("--eps", default="0,1,2")
    ap.add_argument("--out", default="orientation_debug")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = load_schema(args.schema)
    if args.wrist_dims:
        schema.state_layout["tube_wrist_dims"] = [int(x) for x in args.wrist_dims.split(",") if x.strip().isdigit()]
    dims = schema.state_layout.get("tube_wrist_dims") or []
    if not dims:
        ap.error("no tube_wrist_dims; pass --wrist-dims, e.g. --wrist-dims 2,5")

    pour_idx = next(e.index for e in schema.events if e.type == "orientation_change")
    det = OrientationDetector()
    os.makedirs(args.out, exist_ok=True)
    eps = [int(x) for x in args.eps.split(",") if x.strip().isdigit()]

    for ep in eps:
        ref_fp = os.path.join(args.ref_fused, f"ep{ep:03d}_subtasks.json")
        data_fp = os.path.join(args.data, f"episode_{ep:06d}.parquet")
        if not (os.path.exists(ref_fp) and os.path.exists(data_fp)):
            print(f"ep{ep:03d}: missing ref or parquet, skipped")
            continue
        ref = json.load(open(ref_fp))["critical_points"]
        p1, ref_p2, p3 = ref[0], ref[1], ref[2]
        S = load_state(data_fp)
        cand = det.propose(schema, EpisodeContext(ep, len(S), args.fps, state=S)).get(pour_idx)

        fig, axptr = plt.subplots(figsize=(11, 4))
        t = np.arange(len(S))
        for d in dims:
            axptr.plot(t, S[:, d], lw=1.0, label=f"state dim {d}")
        if cand is not None:
            lo, hi = cand.evidence["search_window"]
            axptr.axvspan(lo, hi, color="tab:blue", alpha=0.07, label="search window")
        axptr.axvline(p1, color="gray", ls="--", lw=1, label="p1 grasp")
        axptr.axvline(ref_p2, color="tab:orange", lw=1.5, label="ref fused p2")
        if cand is not None:
            axptr.axvline(cand.frame, color="tab:green", lw=1.5,
                          label=f"orientation p2 (σ={cand.sigma})")
        axptr.axvline(p3, color="gray", ls=":", lw=1, label="p3 release")
        title = f"ep{ep:03d}  " + ("abstained" if cand is None
                                    else f"orient_p2={cand.frame} ref_p2={ref_p2} |Δ|={abs(cand.frame-ref_p2)}")
        axptr.set_title(title)
        axptr.set_xlabel("frame")
        axptr.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out_png = os.path.join(args.out, f"ep{ep:03d}_orientation_p2.png")
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"ep{ep:03d}: wrote {out_png}  ({'abstained' if cand is None else 'fired'})")


if __name__ == "__main__":
    main()
