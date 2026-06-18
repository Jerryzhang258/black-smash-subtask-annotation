"""
Zoom in on a critical-point boundary: render a strip of consecutive (center-crop +
white-balance) frames around the state-chosen frame, so you can read off the true
moment of the event and see how far the auto boundary is off.

Run:
  python zoom_boundary.py --ann mvt_annotations --data black_smash_07\data\chunk-000 --ep 0 --points 2,3
"""
import io, os, json, argparse
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from batch_annotate import CRIT_NAMES
from vlm_annotate import enh

def font(sz):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def strip(ann_dir, data_dir, ep, point, out_png, cam, window_s, stride, tile, cols):
    doc = json.load(open(os.path.join(ann_dir, f"ep{ep:03d}_subtasks.json")))
    cps, N, fps = doc["critical_points"], doc["n_frames"], doc["fps"]
    center = cps[point - 1]
    df = pd.read_parquet(os.path.join(data_dir, f"episode_{ep:06d}.parquet"), columns=[cam])
    w = int(window_s * fps)
    idxs = list(range(max(0, center - w), min(N - 1, center + w) + 1, stride))
    rows = (len(idxs) + cols - 1) // cols
    lab = 18
    im = Image.new("RGB", (cols * tile, rows * (tile + lab) + 26), (18, 18, 22))
    d = ImageDraw.Draw(im); f = font(13); ft = font(16)
    d.text((6, 4), f"ep{ep:03d}  p{point} {CRIT_NAMES[point-1]}  state={center} ({center/fps:.2f}s)  "
                   f"window ±{window_s}s stride {stride}", fill=(235, 235, 235), font=ft)
    for k, i in enumerate(idxs):
        x = (k % cols) * tile; y = 26 + (k // cols) * (tile + lab)
        im.paste(enh(Image.open(io.BytesIO(df[cam].iloc[i]["bytes"])).convert("RGB")).resize((tile, tile)), (x, y + lab))
        hit = (i == center)
        d.text((x + 3, y), f"f{i} {i/fps:.2f}s" + ("  <= state" if hit else ""),
               fill=(255, 80, 80) if hit else (180, 180, 180), font=f)
        if hit:
            d.rectangle([x, y + lab, x + tile - 1, y + tile + lab - 1], outline=(255, 60, 60), width=3)
    im.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--points", default="2,3")
    ap.add_argument("--cam", default="observation.images.camera1")
    ap.add_argument("--window-s", type=float, default=1.3, dest="window_s")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tile", type=int, default=180)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--out", default=r"C:\Intern\mvt_zoom")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for p in [int(x) for x in args.points.split(",") if x.strip().isdigit()]:
        out = os.path.join(args.out, f"{os.path.basename(args.ann)}_ep{args.ep:03d}_p{p}.png")
        print("saved", strip(args.ann, args.data, args.ep, p, out, args.cam, args.window_s, args.stride, args.tile, args.cols))


if __name__ == "__main__":
    main()
