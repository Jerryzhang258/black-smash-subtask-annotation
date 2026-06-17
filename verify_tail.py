"""Render the late portion (grind -> lift -> rest) of one episode at higher res,
to eyeball whether grinding actually stops at b4. Center-crop + enhance camera1.
Tiles are border-colored by the labeled subtask; the b4 boundary is marked.
Usage: python verify_tail.py <ep> [start_frac]   (default start at b3-ish)"""
import sys, io, os, json, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageEnhance

EP = int(sys.argv[1])
PARQUET = rf"C:\Intern\black_smash_07\data\chunk-000\episode_{EP:06d}.parquet"
ANN = rf"C:\Intern\mvt_annotations\ep{EP:03d}_subtasks.json"
OUT = rf"C:\Intern\mvt_verify"
os.makedirs(OUT, exist_ok=True)
d = json.load(open(ANN)); b = d["boundaries"]; N = d["n_frames"]; fps = d["fps"]
b1, b2, b3, b4 = b["b1"], b["b2"], b["b3"], b["b4"]

# sample the grind+lift+rest span: from a bit before b3 to the end
start = int(sys.argv[2]) if len(sys.argv) > 2 else max(0, b3 - 30)
NT = 10
frames = [int(round(start + i * (N - 1 - start) / (NT - 1))) for i in range(NT)]

df = pd.read_parquet(PARQUET, columns=["observation.images.camera1"])
def dec(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def enh(im):
    a = np.asarray(im).astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    o = Image.fromarray(a.astype(np.uint8))
    return ImageEnhance.Contrast(ImageEnhance.Color(o).enhance(1.4)).enhance(1.2)

def subtask_of(f):
    return 3 if b3 <= f <= b4 else (4 if f > b4 else 2)
COL = {2: (255, 140, 0), 3: (0, 220, 90), 4: (80, 160, 255)}  # S2 orange, S3 grind green, S4 blue

TILE, LAB, BD = 300, 20, 6
cv = Image.new("RGB", (NT * TILE, TILE + LAB), (10, 10, 10)); dr = ImageDraw.Draw(cv)
for n, f in enumerate(frames):
    im = enh(dec(df["observation.images.camera1"].iloc[f]))
    w, h = im.size
    crop = im.crop((int(w*0.18), int(h*0.30), int(w*0.82), int(h*0.98))).resize((TILE - 2*BD, TILE - 2*BD))
    s = subtask_of(f); x = n * TILE
    dr.rectangle([x, LAB, x + TILE - 1, LAB + TILE - 1], fill=COL[s])
    cv.paste(crop, (x + BD, LAB + BD))
    mk = "  b4!" if (frames[n-1] if n else -1) < b4 <= f else ""
    dr.text((x + 3, 3), f"f{f} {f/fps:.1f}s S{s}{mk}", fill=COL[s])
fp = os.path.join(OUT, f"tail_ep{EP:03d}.png"); cv.save(fp)
print(f"ep{EP} N={N} b3={b3}({b3/fps:.1f}s) b4={b4}({b4/fps:.1f}s)  grind=S3 green / rest=S4 blue")
print("frames:", frames); print("saved", fp)
