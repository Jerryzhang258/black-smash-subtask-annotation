"""
Probe which observation.state dimensions carry the tube-arm wrist rotation, so
the pour (p2) can be detected from proprioception instead of from the VLM
(framework phase P2 / OrientationDetector). See docs/P2_ORIENTATION_OPTIMIZATION.md.

It does NOT hard-pick dims with a fixed threshold (that masked the real signal).
Instead it ranks every tube-arm, non-gripper dim by how sharply it moves at the
labelled pour onset (p2) while the arm is settled over the mortar, and reports the
top-k per dataset plus a cross-dataset ranking, so you choose tube_wrist_dims from
evidence.

Single dataset:
  python data_annotation/framework/tools/probe_state_layout.py \
      --data /home/hillbot/black_smash_07/data/chunk-000 \
      --info /home/hillbot/black_smash_07/meta/info.json \
      --state-ann annotations_state_07 --eps 0,1,2,3,4

Several datasets (cross-dataset ranking, recommended):
  python data_annotation/framework/tools/probe_state_layout.py \
      --dataset 05:/home/hillbot/black_smash_05/data/chunk-000:annotations_state_05 \
      --dataset 06:/home/hillbot/black_smash_06/data/chunk-000:annotations_state_06 \
      --dataset 07:/home/hillbot/black_smash_07/data/chunk-000:annotations_state_07

Read-only: it only prints.
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


def score_dataset(data_dir, state_ann, dims, p2_index, fps, window_s, eps):
    """Mean pour-onset displacement score per dim over a dataset."""
    w = int(window_s * fps)
    want = set(eps) if eps else None
    scores: dict[int, list[float]] = defaultdict(list)
    n_used = 0
    for fp in sorted(glob.glob(os.path.join(data_dir, "episode_*.parquet"))):
        ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if want is not None and ep not in want:
            continue
        ann_fp = os.path.join(state_ann, f"ep{ep:03d}_subtasks.json")
        if not os.path.exists(ann_fp):
            continue
        p2 = json.load(open(ann_fp))["critical_points"][p2_index]
        S = load_state(fp)
        T = len(S)
        if not (w < p2 < T - w):
            continue
        n_used += 1
        for d in dims:
            x = S[:, d]
            sd = x.std() + 1e-9
            step = abs(np.median(x[p2:p2 + w]) - np.median(x[p2 - w:p2])) / sd
            dx = np.abs(np.diff(x)) / sd
            local = dx[max(0, p2 - w):p2 + w].mean()
            glob_ = dx.mean() + 1e-9
            scores[d].append(step * (local / glob_))
    return {d: float(np.mean(v)) for d, v in scores.items()}, n_used


def print_ranked(title, scores, topk):
    print(f"\n=== {title} ===")
    if not scores:
        print("  (no usable episodes -- check paths / --eps)")
        return []
    ranked = sorted(scores.items(), key=lambda t: -t[1])
    for rank, (d, s) in enumerate(ranked[:topk], 1):
        print(f"  #{rank}  dim {d:>3}   score {s:6.2f}")
    return [d for d, _ in ranked[:topk]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[],
                    help="repeatable 'name:data_dir:state_ann_dir' for cross-dataset ranking")
    ap.add_argument("--data", help="single-dataset chunk dir with episode_*.parquet")
    ap.add_argument("--info", help="meta/info.json (prints state feature names)")
    ap.add_argument("--state-ann", dest="state_ann", help="single-dataset state annotation dir")
    ap.add_argument("--p2-index", type=int, default=1, dest="p2_index")
    ap.add_argument("--tube-arm-dims", default="0,1,2,5,6,7,8,9", dest="tube_arm_dims")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window-s", type=float, default=0.5, dest="window_s")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--eps", default="")
    args = ap.parse_args()

    dims = parse_ints(args.tube_arm_dims)

    if args.info and os.path.exists(args.info):
        st = json.load(open(args.info)).get("features", {}).get("observation.state", {})
        print("=== meta/info.json: observation.state ===")
        print("  shape:", st.get("shape"), "| names:", json.dumps(st.get("names"), ensure_ascii=False))
        if not (st.get("names") and len(st["names"]) > 1):
            print("  -> no per-dim names; use the empirical ranking below.")

    datasets = []
    for spec in args.dataset:
        parts = spec.split(":")
        if len(parts) != 3:
            ap.error(f"--dataset must be name:data_dir:state_ann_dir, got '{spec}'")
        datasets.append(tuple(parts))
    if not datasets:
        if not (args.data and args.state_ann):
            ap.error("give --dataset ... (repeatable) or both --data and --state-ann")
        datasets = [("dataset", args.data, args.state_ann)]

    eps = parse_ints(args.eps) if args.eps else None
    combined: dict[int, list[float]] = defaultdict(list)
    for name, data_dir, ann_dir in datasets:
        scores, n = score_dataset(data_dir, ann_dir, dims, args.p2_index, args.fps, args.window_s, eps)
        print_ranked(f"{name}: pour-onset score per dim ({n} episodes)", scores, args.topk)
        for d, s in scores.items():
            combined[d].append(s)

    if len(datasets) > 1:
        cross = {d: float(np.mean(v)) for d, v in combined.items()}
        picks = print_ranked("cross-dataset (mean over datasets)", cross, args.topk)
    else:
        picks = sorted(combined, key=lambda d: -np.mean(combined[d]))[:3]

    print("\n=== suggested schema edit (schemas/black_smash.json -> state_layout) ===")
    print(f'  "tube_wrist_dims": {sorted(picks[:3])}   # top candidates; validate with eval_orientation_p2.py')
    print("\nThen validate (does not change production):")
    print("  python -m data_annotation.framework.fuse parquet --data <chunk> --vlm <qwen_dir> \\")
    print("      --out /tmp/fused_p2 --orientation")
    print("  python data_annotation/framework/tools/eval_orientation_p2.py \\")
    print("      --orientation-fused /tmp/fused_p2 --ref-fused <annotations_fused_dir>")


if __name__ == "__main__":
    main()
