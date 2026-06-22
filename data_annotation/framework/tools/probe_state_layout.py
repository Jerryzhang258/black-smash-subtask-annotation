"""
Probe which observation.state dimensions carry the tube-arm wrist rotation, so
the pour (p2) can be detected from proprioception instead of from the VLM
(framework phase P2 / OrientationDetector).

Two things it does:
  1. Print the state feature names from meta/info.json (path 1). For these
     datasets `names` is usually just ["observation.state"] -- i.e. no per-dim
     labels -- so step 2 is what actually finds the dims.
  2. Empirically rank the tube-arm, non-gripper dims by how sharply they move
     right at the labelled pour onset (p2) while the arm is already settled over
     the mortar (path 2). The top dims are the wrist-rotation candidates.

Run on the server, where observation.state exists:

  python data_annotation/framework/tools/probe_state_layout.py \
      --data   /home/hillbot/black_smash_07/data/chunk-000 \
      --info   /home/hillbot/black_smash_07/meta/info.json \
      --state-ann annotations_state_07 \
      --eps 0,1,2,3,4,5,6,7,8,9

It prints a ranked table and a ready-to-paste `"tube_wrist_dims": [...]` snippet
for schemas/black_smash.json. NOTHING is written; it only reads + prints.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def load_state(parquet: str) -> np.ndarray:
    col = pd.read_parquet(parquet, columns=["observation.state"])["observation.state"].values
    return np.stack([np.asarray(x, dtype=np.float64) for x in col])


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip().lstrip("-").isdigit()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="chunk dir with episode_*.parquet")
    ap.add_argument("--info", help="meta/info.json (path 1: print state feature names)")
    ap.add_argument("--state-ann", required=True, dest="state_ann",
                    help="dir with state ep<NNN>_subtasks.json (provides the labelled p2)")
    ap.add_argument("--p2-index", type=int, default=1, help="0-based index of the pour point in critical_points")
    ap.add_argument("--tube-arm-dims", default="0,1,2,5,6,7,8,9", dest="tube_arm_dims",
                    help="tube-arm dims to consider (gripper dims excluded by default)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window-s", type=float, default=0.5, dest="window_s",
                    help="half-window around p2 used to measure the displacement")
    ap.add_argument("--eps", default="", help="comma list of episode ids; default = all found")
    args = ap.parse_args()

    # ---- path 1: feature names from meta ----
    if args.info and os.path.exists(args.info):
        info = json.load(open(args.info))
        st = info.get("features", {}).get("observation.state", {})
        print("=== meta/info.json: observation.state ===")
        print("  shape:", st.get("shape"))
        print("  names:", json.dumps(st.get("names"), ensure_ascii=False))
        names = st.get("names")
        if names and len(names) > 1:
            print("  -> per-dim names present; the wrist dims may be readable directly above.")
        else:
            print("  -> no per-dim names; using the empirical probe below.\n")

    # ---- path 2: empirical displacement at the pour ----
    dims = parse_ints(args.tube_arm_dims)
    w = int(args.window_s * args.fps)
    want = set(parse_ints(args.eps)) if args.eps else None

    files = sorted(glob.glob(os.path.join(args.data, "episode_*.parquet")))
    scores: dict[int, list[float]] = defaultdict(list)
    n_used = 0
    for fp in files:
        ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if want is not None and ep not in want:
            continue
        ann_fp = os.path.join(args.state_ann, f"ep{ep:03d}_subtasks.json")
        if not os.path.exists(ann_fp):
            continue
        p2 = json.load(open(ann_fp))["critical_points"][args.p2_index]
        S = load_state(fp)
        T = len(S)
        if not (w < p2 < T - w):
            continue
        n_used += 1
        for d in dims:
            x = S[:, d]
            sd = x.std() + 1e-9
            before = np.median(x[p2 - w:p2])
            after = np.median(x[p2:p2 + w])
            step = abs(after - before) / sd                 # normalized displacement at p2
            # contrast: is the move concentrated at p2, or is this dim just noisy everywhere?
            dx = np.abs(np.diff(x)) / sd
            local = dx[max(0, p2 - w):p2 + w].mean()
            glob_ = dx.mean() + 1e-9
            scores[d].append(step * (local / glob_))         # displacement weighted by local sharpness

    if n_used == 0:
        print("No usable episodes (need both parquet and a state annotation with p2). Check paths/--eps.")
        return

    ranked = sorted(((d, float(np.mean(v))) for d, v in scores.items()), key=lambda t: -t[1])
    med = np.median([s for _, s in ranked])
    print(f"=== pour-onset displacement per dim (mean over {n_used} episodes) ===")
    print("  dim   score   (relative to median)")
    picks = []
    for d, s in ranked:
        flag = "  <== wrist candidate" if s >= max(1.0, 1.5 * med) else ""
        if flag:
            picks.append(d)
        print(f"  {d:>3}  {s:6.2f}   x{s / (med + 1e-9):4.1f}{flag}")

    print("\n=== suggested schema edit (schemas/black_smash.json -> state_layout) ===")
    print(f'  "tube_wrist_dims": {sorted(picks) if picks else "[]  # no dim stood out; widen --window-s or --eps"}')
    print("\nThen enable the detector:")
    print("  python -m data_annotation.framework.fuse parquet \\")
    print(f"      --data {args.data} --vlm <qwen_dir> --out /tmp/fused_p2 --orientation --eps {args.eps or '0,1,2'}")
    print("  # compare p2 in /tmp/fused_p2 against the VLM p2 to see if state now wins cleanly.")


if __name__ == "__main__":
    main()
