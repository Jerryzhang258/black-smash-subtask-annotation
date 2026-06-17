"""
Human vs auto subtask timeline comparison (QA method 4, visual).
For every episode that has BOTH a human (mvt_annotations_human/) and an auto
(mvt_annotations/) annotation, render two aligned timelines with the 5 critical
points connected, per-point frame delta, and per-subtask temporal IoU.
Also prints a metrics table + dataset-wide mean abs error per critical point.

Run:  python compare_timelines.py            # all episodes with both
      python compare_timelines.py --ep 0
"""
import os, json, glob, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HUMAN = r"C:\Intern\mvt_annotations_human"
AUTO  = r"C:\Intern\mvt_annotations"
OUT   = r"C:\Intern\mvt_compare"
FPS   = 30
COLORS = [(255, 85, 85), (255, 176, 0), (255, 122, 0), (160, 112, 255), (0, 220, 90), (80, 160, 255)]
SUB_CN = ["S0 伸手", "S1 端管", "S2 倒", "S3 放管+取杵", "S4 磨", "S5 抬杵"]
PT_CN  = ["p1 抓管", "p2 开始倒", "p3 放管", "p4 开始磨", "p5 抬杵"]
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load(d, ep):
    fp = os.path.join(d, f"ep{ep:03d}_subtasks.json")
    return json.load(open(fp)) if os.path.exists(fp) else None


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    uni = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / uni if uni > 0 else 0.0


def intervals(doc):
    return [(s["start_frame"], s["end_frame"]) for s in doc["subtasks"]]


def render(ep):
    H, A = load(HUMAN, ep), load(AUTO, ep)
    if H is None or A is None:
        return None
    N = H["n_frames"]
    hcp, acp = H["critical_points"], A["critical_points"]
    hst, ast = H["subtask_starts"], A["subtask_starts"]
    deltas = [acp[i] - hcp[i] for i in range(len(hcp))]
    ious = [iou(h, a) for h, a in zip(intervals(H), intervals(A))]
    mae = float(np.mean(np.abs(deltas)))

    W, L, R = 1120, 70, 30
    bw = W - L - R
    yH, yA, bh = 78, 168, 40
    Hh = 320
    im = Image.new("RGB", (W, Hh), (18, 18, 22)); d = ImageDraw.Draw(im)
    f20, f14, f12, f11 = font(20), font(15), font(13), font(12)

    def X(fr): return int(L + fr / N * bw)

    d.text((L, 12), f"ep{ep:03d}  人工 vs 自动 时间线对比   N={N}  {N/FPS:.1f}s   "
                    f"平均|Δ|={mae:.0f}帧/{mae/FPS:.2f}s", fill=(240, 240, 240), font=f20)

    def bar(y, starts, tag):
        d.text((6, y + bh / 2 - 8), tag, fill=(230, 230, 230), font=f14)
        for i in range(len(starts)):
            a = starts[i]; b = (starts[i + 1] - 1) if i + 1 < len(starts) else N - 1
            d.rectangle([X(a), y, X(b), y + bh], fill=COLORS[i], outline=(20, 20, 20))
            if X(b) - X(a) > 22:
                d.text((X(a) + 3, y + 3), f"S{i}", fill=(0, 0, 0), font=f11)
    bar(yH, hst, "人工")
    bar(yA, ast, "自动")

    # connectors + delta labels for the 5 critical points
    for i in range(len(hcp)):
        hx, ax = X(hcp[i]), X(acp[i])
        d.line([hx, yH, hx, yH - 10], fill=(255, 255, 255), width=2)
        d.line([ax, yA + bh, ax, yA + bh + 10], fill=(255, 255, 255), width=2)
        col = (255, 90, 90) if abs(deltas[i]) > 30 else (170, 170, 170)
        d.line([hx, yH + bh, ax, yA], fill=col, width=2)
        my = (yH + bh + yA) // 2
        ly_off = -16 if i % 2 == 0 else 2   # stagger to avoid overlap on close points
        d.text(((hx + ax) // 2 + 3, my + ly_off), f"{PT_CN[i]} Δ{deltas[i]:+d}f/{deltas[i]/FPS:+.1f}s",
               fill=col, font=f12)

    # legend
    ly = yA + bh + 36
    for i in range(6):
        d.rectangle([L + i * 175, ly, L + i * 175 + 14, ly + 14], fill=COLORS[i])
        d.text((L + i * 175 + 18, ly), f"{SUB_CN[i]}", fill=(220, 220, 220), font=f11)
    # per-subtask IoU
    d.text((L, ly + 28), "各段时序 IoU:  " + "   ".join(f"S{i}={ious[i]:.2f}" for i in range(6)),
           fill=(200, 220, 200), font=f14)
    # axis
    for fr in [0, N - 1]:
        d.text((X(fr) - 6, Hh - 22), f"{fr/FPS:.0f}s", fill=(150, 150, 150), font=f11)

    fp = os.path.join(OUT, f"ep{ep:03d}_compare.png"); im.save(fp)
    return {"ep": ep, "deltas": deltas, "mae": mae, "ious": ious, "png": fp}


ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=None)
args = ap.parse_args()

if args.ep is not None:
    eps = [args.ep]
else:
    eps = sorted(int(os.path.basename(f).split("_")[0][2:])
                 for f in glob.glob(os.path.join(HUMAN, "ep*_subtasks.json")))

results = []
for ep in eps:
    r = render(ep)
    if r is None:
        print(f"ep{ep:03d}: missing human or auto annotation — skipped"); continue
    results.append(r)
    print(f"ep{ep:03d}  Δ(frames)={r['deltas']}  mean|Δ|={r['mae']:.0f}f/{r['mae']/FPS:.2f}s  "
          f"IoU={[round(x,2) for x in r['ious']]}  -> {os.path.basename(r['png'])}")

if results:
    D = np.array([r["deltas"] for r in results])
    print("\n=== dataset-wide (n=%d) mean abs error per critical point ===" % len(results))
    for i, nm in enumerate(PT_CN):
        print(f"  {nm:10s}  MAE={np.abs(D[:, i]).mean():5.1f}f / {np.abs(D[:, i]).mean()/FPS:.2f}s   "
              f"bias={D[:, i].mean():+.1f}f")
    I = np.array([r["ious"] for r in results])
    print("  mean per-subtask IoU:", [f"S{i}={I[:, i].mean():.2f}" for i in range(6)])
print("DONE")
