"""
Inspect one LeRobot-format episode parquet:
- print schema (columns, dtypes)
- detect image columns and decode a few frames
- print states / actions / language instruction
- extract N evenly-spaced frames per camera -> PNG on disk
"""
import sys, os, io, json
import pandas as pd
import numpy as np
from PIL import Image

PARQUET = sys.argv[1] if len(sys.argv) > 1 else r"C:\Intern\black_smash_07\data\chunk-000\episode_000000.parquet"
OUTDIR  = sys.argv[2] if len(sys.argv) > 2 else r"C:\Intern\mvt_frames\ep000"
N_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 12

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("FILE:", PARQUET)
print("size: %.1f MB" % (os.path.getsize(PARQUET) / 1e6))
print("=" * 70)

df = pd.read_parquet(PARQUET)
print("rows (frames):", len(df))
print("columns:")
for c in df.columns:
    v = df[c].iloc[0]
    t = type(v).__name__
    extra = ""
    if isinstance(v, (np.ndarray, list)):
        arr = np.asarray(v)
        extra = f" shape={arr.shape} dtype={arr.dtype}"
    elif isinstance(v, dict):
        extra = f" dict_keys={list(v.keys())}"
    elif isinstance(v, (bytes, bytearray)):
        extra = f" bytes_len={len(v)}"
    elif isinstance(v, str):
        extra = f" = {v[:60]!r}"
    elif isinstance(v, (int, float, np.integer, np.floating)):
        extra = f" = {v}"
    print(f"   {c:40s} {t}{extra}")

print("-" * 70)

# language instruction
for cand in ["task", "language_instruction", "instruction", "task_index"]:
    if cand in df.columns:
        vals = df[cand].unique()[:5]
        print(f"[{cand}] unique(<=5): {vals}")

# state / action ranges
for cand in ["observation.state", "action", "state"]:
    if cand in df.columns:
        arr = np.stack([np.asarray(x) for x in df[cand].values])
        print(f"[{cand}] stack shape={arr.shape}  min={arr.min():.3f} max={arr.max():.3f}")

print("-" * 70)

# detect image columns
def decode_image(v):
    """Return PIL.Image from various LeRobot storage forms, else None."""
    try:
        if isinstance(v, dict):
            b = v.get("bytes")
            if b is not None:
                return Image.open(io.BytesIO(b))
            p = v.get("path")
            if p and os.path.exists(p):
                return Image.open(p)
            return None
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(v)))
        if isinstance(v, np.ndarray):
            a = v
            if a.dtype != np.uint8:
                a = (255 * (a - a.min()) / (a.ptp() + 1e-9)).astype(np.uint8)
            if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
                a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
            return Image.fromarray(a)
        if isinstance(v, list):
            return decode_image(np.asarray(v))
    except Exception as e:
        print("   decode error:", e)
    return None

img_cols = []
for c in df.columns:
    v = df[c].iloc[0]
    img = decode_image(v)
    if img is not None:
        img_cols.append(c)
        print(f"[IMAGE COL] {c}  -> first frame size={img.size} mode={img.mode}")

if not img_cols:
    print("!! no image columns decoded. Dump first-row raw types above to debug.")
    sys.exit(0)

# extract N evenly spaced frames per image column
n = len(df)
idxs = np.linspace(0, n - 1, min(N_FRAMES, n)).round().astype(int)
print("-" * 70)
print("extracting frame indices:", list(idxs))
manifest = []
for c in img_cols:
    safe = c.replace(".", "_")
    for k, i in enumerate(idxs):
        img = decode_image(df[c].iloc[int(i)])
        if img is None:
            continue
        fn = f"{safe}__f{k:02d}_idx{int(i):04d}.png"
        fp = os.path.join(OUTDIR, fn)
        img.convert("RGB").save(fp)
        manifest.append({"cam": c, "k": k, "frame_index": int(i), "file": fp,
                         "size": img.size})

with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
    json.dump({"parquet": PARQUET, "n_frames_total": n, "cameras": img_cols,
               "sampled_idxs": [int(x) for x in idxs], "frames": manifest}, f, indent=2)

print(f"saved {len(manifest)} PNGs to {OUTDIR}")
print("DONE")
