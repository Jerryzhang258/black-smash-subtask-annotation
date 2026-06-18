"""
Visualize a subtask annotation: a timeline of the 7 colored segments + one
readable keyframe (center-crop + white-balance) per segment.

Run:
  python visualize_annotation.py --ann mvt_annotations --data black_smash_07\data\chunk-000 --ep 0
  python visualize_annotation.py --ann mvt_annotations_smash_05 --data black_smash_05\data\chunk-000 --eps 0,1,2
"""
import io, os, json, glob, argparse
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from batch_annotate import LABELS
from vlm_annotate import enh   # center-crop + gray-world WB + gamma

COLORS = [(255, 85, 85), (255, 176, 0), (255, 122, 0), (160, 112, 255),
          (0, 220, 90), (80, 160, 255), (255, 102, 204)]


def font(sz):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def viz(ann_dir, data_dir, ep, out_png, cam, tile=420):
    doc = json.load(open(os.path.join(ann_dir, f"ep{ep:03d}_subtasks.json")))
    subs, N, fps = doc["subtasks"], doc["n_frames"], doc["fps"]
    df = pd.read_parquet(os.path.join(data_dir, f"episode_{ep:06d}.parquet"), columns=[cam])

    def frame(i):
        return enh(Image.open(io.BytesIO(df[cam].iloc[int(i)]["bytes"])).convert("RGB"))

    n = len(subs); TILE = tile; W = TILE * n
    barY, barH, gap = 34, 56, 22
    tileY = barY + barH + gap + 22
    H = tileY + TILE + 66
    im = Image.new("RGB", (W, H), (18, 18, 22)); d = ImageDraw.Draw(im)
    ft, fb, fl = font(max(20, tile // 18)), font(max(15, tile // 26)), font(max(14, tile // 28))
    wrap = max(20, tile // 13)

    d.text((8, 6), f"{os.path.basename(ann_dir)}  ep{ep:03d}   N={N}  {N/fps:.1f}s   "
                   f"7 段 / 6 临界点", fill=(235, 235, 235), font=ft)

    # timeline bar (full width = whole episode)
    for i, s in enumerate(subs):
        x0 = int(s["start_frame"] / N * W); x1 = int((s["end_frame"] + 1) / N * W)
        d.rectangle([x0, barY, x1, barY + barH], fill=COLORS[i], outline=(15, 15, 15))
        if x1 - x0 > 34:
            d.text((x0 + 4, barY + 4), f"S{i}", fill=(0, 0, 0), font=fb)
            d.text((x0 + 4, barY + barH // 2 + 2), f"{s['dur_s']}s", fill=(0, 0, 0), font=fl)

    # one keyframe per segment (segment midpoint)
    for i, s in enumerate(subs):
        mid = (s["start_frame"] + s["end_frame"]) // 2
        im.paste(frame(mid).resize((TILE, TILE)), (i * TILE, tileY))
        d.rectangle([i * TILE, tileY, i * TILE + TILE - 1, tileY + TILE - 1], outline=COLORS[i], width=2)
        d.text((i * TILE + 4, tileY - 20), f"S{i}  f{s['start_frame']}-{s['end_frame']}",
               fill=COLORS[i], font=fb)
        words, lines, cur = LABELS[i].split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > wrap: lines.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        lines.append(cur)
        d.text((i * TILE + 4, tileY + TILE + 4), "\n".join(lines[:3]), fill=(205, 205, 205), font=fl)

    im.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="annotation dir (e.g. mvt_annotations)")
    ap.add_argument("--data", required=True, help="<dataset>\\data\\chunk-000")
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--eps", default="")
    ap.add_argument("--cam", default="observation.images.camera1")
    ap.add_argument("--tile", type=int, default=420, help="per-keyframe pixel size")
    ap.add_argument("--out", default=r"C:\Intern\mvt_viz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.eps:
        eps = [int(x) for x in args.eps.split(",") if x.strip().isdigit()]
    elif args.ep is not None:
        eps = [args.ep]
    else:
        eps = sorted(int(os.path.basename(f)[2:5]) for f in glob.glob(os.path.join(args.ann, "ep*_subtasks.json")))

    for ep in eps:
        p = viz(args.ann, args.data, ep, os.path.join(args.out, f"{os.path.basename(args.ann)}_ep{ep:03d}.png"), args.cam, args.tile)
        print("saved", p)


if __name__ == "__main__":
    main()
