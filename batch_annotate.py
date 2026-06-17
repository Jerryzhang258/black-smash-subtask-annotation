"""
Batch subtask annotation for the black_smash "pour the black powder into the mortar
and grind" dataset (LeRobot v2.1, bimanual, 20-dim observation.state).

Auto-segments EVERY episode parquet into 5 fixed subtasks, using only the state signal
(fast: image columns are never loaded unless --storyboard is set):

  S0  reach for and grasp the powder container
  S1  pour the black powder into the mortar
  S2  set down the container and bring the pestle to the mortar
  S3  grind the powder in the mortar
  S4  lift the pestle and return to rest

Boundary heuristics (all rate-independent, expressed in frame indices):
  * grasp(b1)/release(b2): the "pour" is a transient bimodal deviation of state dim 3
    away from its resting value -> longest such run in the first 70% of the episode.
  * grind(b3..b4): late, sustained IN-PLACE motion = nonzero raw speed but low
    "carrier drift" (speed of the ~1s rolling-mean pose). Transport has high drift;
    grinding stays over the mortar -> raw/drift ratio spikes.

Per-episode output:  <out>/ep<NNN>_subtasks.json  +  ep<NNN>_subtask_index.npy
Dataset output:      <out>/summary.csv  +  <out>/all_subtasks.jsonl  +  flagged list
Optional:            --storyboard  -> ep<NNN>_storyboard.png (decodes camera1, slower)

Usage:
  python batch_annotate.py                       # all episodes in default data dir
  python batch_annotate.py --eps 0,1,2 --storyboard
  python batch_annotate.py --data D:\path\chunk-000 --out D:\ann --fps 30
"""
import os, io, json, glob, argparse, traceback
import numpy as np, pandas as pd

LABELS = [   # 6 subtasks, split by 5 critical points (matches the manual annotator)
    "reach for the test tube",
    "lift the test tube and move it over the mortar",
    "pour the black powder into the mortar",
    "set down the test tube and pick up the pestle",
    "grind the powder in the mortar",
    "lift the pestle and return to rest",
]
ENGAGE_DIM = 3        # state dim that deviates while the tube is held (validated on ep000)
GRIP_DIMS  = [3, 4]   # excluded from the "pose" used for motion/drift


def smooth(x, w):
    w = max(1, int(w) | 1)  # odd
    return np.convolve(x, np.ones(w) / w, mode="same")


def norm01(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def longest_run(mask):
    best = (0, 0); s = None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best  # [start, end)  half-open


def close_gaps(mask, g):
    """Fill False gaps shorter than g frames between True runs (morphological close)."""
    m = mask.copy()
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < i and j < n and (j - i) < g:   # interior gap shorter than g
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def segment_episode(S, fps=30):
    """6 subtasks from 5 critical points: p1 grasp tube, p2 start pour, p3 set down
    tube, p4 start grind, p5 lift pestle. Returns (segments, flags, critical_points)."""
    T, D = S.shape
    flags = []

    # ---- tube-held window [E1,E2]: transient deviation of ENGAGE_DIM from resting ----
    g = norm01(S[:, ENGAGE_DIM])
    rest = np.median(g[: max(5, T // 20)])
    engaged = np.abs(g - rest) > 0.5
    engaged[int(0.70 * T):] = False
    E1, e2e = longest_run(engaged); E2 = e2e - 1
    if E2 <= E1 or (E2 - E1) < 0.3 * fps:
        flags.append("no clear tube-held window (state dim%d)" % ENGAGE_DIM)
        E1, E2 = int(0.10 * T), int(0.34 * T)

    # ---- motion / carrier drift ----
    pos = [d for d in range(D) if d not in GRIP_DIMS]
    P = (S[:, pos] - S[:, pos].mean(0)) / (S[:, pos].std(0) + 1e-9)
    raw = smooth(np.linalg.norm(np.diff(P, axis=0, prepend=P[:1]), axis=1), 0.3 * fps)
    carrier = np.vstack([smooth(P[:, j], 1.0 * fps) for j in range(P.shape[1])]).T
    drift = smooth(np.linalg.norm(np.diff(carrier, axis=0, prepend=carrier[:1]), axis=1), 0.3 * fps)

    # ---- pour start (p2) inside [E1,E2]: arm settles over mortar = start of low-drift run ----
    sd = drift[E1:E2 + 1]
    if len(sd) >= 3:
        low = close_gaps(sd < np.percentile(sd, 45), int(0.5 * fps))
        r0, _ = longest_run(low); p2 = E1 + int(r0)
    else:
        p2 = E1 + int(0.4 * (E2 - E1))
    p2 = min(max(p2, E1 + 1), E2 - 1)

    # ---- grind window [Gs,Ge]: late in-place motion (low drift, still moving) ----
    grind_ok = (drift < np.percentile(drift, 40)) & (raw > 0.12 * raw.max())
    grind_ok[: E2 + 1] = False
    grind_ok = close_gaps(grind_ok, int(1.2 * fps))
    gs, ge = longest_run(grind_ok)
    if ge - gs > 0.8 * fps:
        Gs, Ge = int(gs), int(ge - 1)
    else:
        flags.append("no clear grind window")
        Gs, Ge = int(0.62 * T), int(0.92 * T)

    # ---- 5 critical points = start frames of S1..S5 ----
    cps = [E1, p2, E2 + 1, Gs, Ge + 1]
    if not (0 < cps[0] < cps[1] < cps[2] < cps[3] < cps[4] < T - 1):
        flags.append("ordering off: cps=%s (T=%d)" % (cps, T))
        cps = [int(p * T) for p in (0.14, 0.24, 0.34, 0.60, 0.90)]

    starts = [0] + cps
    segs = [(starts[i], (starts[i + 1] - 1 if i < 5 else T - 1), LABELS[i]) for i in range(6)]
    return segs, flags, cps


# ---------- optional storyboard ----------
def make_storyboard(parquet, segs, fps, out_png):
    from PIL import Image, ImageDraw, ImageEnhance
    df = pd.read_parquet(parquet, columns=["observation.images.camera1"])
    def dec(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    def enh(im):
        a = np.asarray(im).astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
        o = Image.fromarray(a.astype(np.uint8))
        return ImageEnhance.Contrast(ImageEnhance.Color(o).enhance(1.4)).enhance(1.2)
    TILE, LAB = 260, 44
    cv = Image.new("RGB", (len(segs) * TILE, TILE + LAB), (12, 12, 12)); d = ImageDraw.Draw(cv)
    for sid, (a, b, lab) in enumerate(segs):
        im = enh(dec(df["observation.images.camera1"].iloc[(a + b) // 2])).resize((TILE, TILE))
        cv.paste(im, (sid * TILE, LAB))
        d.text((sid * TILE + 4, 3), f"S{sid} f{a}-{b} ({a/fps:.1f}-{b/fps:.1f}s)", fill=(255, 220, 0))
        words, lines, cur = lab.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 30: lines.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        lines.append(cur)
        d.text((sid * TILE + 4, 19), "\n".join(lines[:2]), fill=(0, 255, 120))
    cv.save(out_png)


def annotate_one(parquet, out_dir, fps, ep_idx, task, storyboard=False):
    S = np.stack([np.asarray(x, dtype=np.float64)
                  for x in pd.read_parquet(parquet, columns=["observation.state"])["observation.state"].values])
    T = len(S)
    segs, flags, cps = segment_episode(S, fps)
    subtasks = [{"subtask_id": i, "label": lab, "start_frame": a, "end_frame": b,
                 "start_t": round(a / fps, 2), "end_t": round(b / fps, 2),
                 "n_frames": b - a + 1, "dur_s": round((b - a + 1) / fps, 2)}
                for i, (a, b, lab) in enumerate(segs)]
    doc = {"episode_index": ep_idx, "task": task, "n_frames": T, "fps": fps,
           "annotator": "auto-signal",
           "method": "signal-derived (state-dim%d tube-held window + carrier-drift pour/grind)" % ENGAGE_DIM,
           "critical_points": cps, "subtask_starts": [0] + cps,
           "flags": flags, "n_subtasks": len(subtasks), "subtasks": subtasks}
    json.dump(doc, open(os.path.join(out_dir, f"ep{ep_idx:03d}_subtasks.json"), "w"), indent=2)
    idx = np.zeros(T, dtype=np.int16)
    for i, (a, b, _) in enumerate(segs):
        idx[a:b + 1] = i
    np.save(os.path.join(out_dir, f"ep{ep_idx:03d}_subtask_index.npy"), idx)
    if storyboard:
        make_storyboard(parquet, segs, fps, os.path.join(out_dir, f"ep{ep_idx:03d}_storyboard.png"))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Intern\black_smash_07\data\chunk-000")
    ap.add_argument("--out",  default=r"C:\Intern\mvt_annotations")
    ap.add_argument("--meta", default=r"C:\Intern\black_smash_07\meta\tasks.jsonl")
    ap.add_argument("--fps",  type=int, default=30)
    ap.add_argument("--eps",  default="", help="comma list of episode indices; default = all found")
    ap.add_argument("--storyboard", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    task = "Pour the black powder into the mortar and grind."
    try:
        task = json.loads(open(args.meta).readline())["task"]
    except Exception:
        pass

    files = sorted(glob.glob(os.path.join(args.data, "episode_*.parquet")))
    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None

    full = want is None        # only rewrite aggregates on a full run, never on a subset
    rows, flagged = [], []
    jl = open(os.path.join(args.out, "all_subtasks.jsonl"), "w") if full else None
    for fp in files:
        ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if want is not None and ep not in want:
            continue
        try:
            doc = annotate_one(fp, args.out, args.fps, ep, task, args.storyboard)
        except Exception as e:
            flagged.append((ep, "EXCEPTION: " + str(e)))
            traceback.print_exc()
            continue
        if jl: jl.write(json.dumps(doc) + "\n")
        durs = [s["dur_s"] for s in doc["subtasks"]]
        cps = doc["critical_points"]
        rows.append([ep, doc["n_frames"]] + cps + durs + ["|".join(doc["flags"])])
        if doc["flags"]:
            flagged.append((ep, "; ".join(doc["flags"])))
        tag = "  FLAG" if doc["flags"] else ""
        print(f"ep{ep:03d}  N={doc['n_frames']:4d}  cps={cps}  durs={durs}{tag}")
    if jl: jl.close()

    if full:
        import csv
        with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "n_frames", "p1_grasp", "p2_pour", "p3_setdown", "p4_grind", "p5_lift",
                        "S0_s", "S1_s", "S2_s", "S3_s", "S4_s", "S5_s", "flags"])
            w.writerows(rows)
    else:
        print("(subset run — summary.csv / all_subtasks.jsonl left unchanged)")

    print(f"\n{len(rows)} episodes annotated -> {args.out}")
    print(f"flagged for review: {len(flagged)}")
    for ep, why in flagged:
        print(f"  ep{ep:03d}: {why}")


if __name__ == "__main__":
    main()
