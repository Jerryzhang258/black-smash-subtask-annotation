"""
First-pass subtask annotation for ONE episode.
Boundaries derived from signal analysis (gripper-state flips + motion/oscillation
profile), cross-checked against enhanced scene keyframes. Emits:
  - ep000_subtasks.json    (segment list, LeRobot-friendly)
  - ep000_subtask_index.npy (per-frame subtask id, len = n_frames)
  - ep000_storyboard.png   (one labeled representative frame per subtask)
"""
import os, io, json, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageEnhance

EP = 0
PARQUET = rf"C:\Intern\black_smash_07\data\chunk-000\episode_{EP:06d}.parquet"
OUT = r"C:\Intern\mvt_annotations"
FPS = 30  # dataset rate; frame-index segmentation is rate-independent regardless
TASK = "Pour the black powder into the mortar and grind."
os.makedirs(OUT, exist_ok=True)

# --- subtask segmentation (start/end inclusive, contiguous) ---
# Boundaries (refined):
#   f191  = gripper-state flip -> grasp container        (clean bimodal flip)
#   f374  = flip back -> release/handoff container        (clean bimodal flip)
#   f744  = grind onset: arm settles over mortar, raw motion w/ low carrier drift
#           (raw/drift ratio jumps ~1 -> 2.5-4.5); f375-744 is set-down + carry pestle over
#   f1036 = grind end / pestle lifted out (drift rises, f1050+ is the lift-out move)
SEGMENTS = [
    (0,    190,  "reach for and grasp the powder container"),
    (191,  374,  "pour the black powder into the mortar"),
    (375,  743,  "set down the container and bring the pestle to the mortar"),
    (744,  1036, "grind the powder in the mortar"),
    (1037, 1158, "lift the pestle and return to rest"),
]

df = pd.read_parquet(PARQUET, columns=["observation.images.camera1"])
N = len(df)
assert SEGMENTS[-1][1] == N - 1, f"last end {SEGMENTS[-1][1]} != N-1 {N-1}"

subtasks = []
for sid, (a, b, lab) in enumerate(SEGMENTS):
    subtasks.append({
        "subtask_id": sid, "label": lab,
        "start_frame": a, "end_frame": b,
        "start_t": round(a / FPS, 2), "end_t": round(b / FPS, 2),
        "n_frames": b - a + 1, "dur_s": round((b - a + 1) / FPS, 2),
    })

doc = {
    "episode_index": EP, "task": TASK, "n_frames": N, "fps": FPS,
    "method": "signal-derived boundaries (gripper-state flips + motion/oscillation profile) "
              "cross-checked against enhanced scene keyframes (camera1)",
    "n_subtasks": len(subtasks), "subtasks": subtasks,
}
json.dump(doc, open(os.path.join(OUT, f"ep{EP:03d}_subtasks.json"), "w"), indent=2)

# per-frame subtask index
idx = np.zeros(N, dtype=np.int16)
for sid, (a, b, _) in enumerate(SEGMENTS):
    idx[a:b + 1] = sid
np.save(os.path.join(OUT, f"ep{EP:03d}_subtask_index.npy"), idx)

# --- storyboard: representative (middle) frame per subtask ---
def dec(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def enh(im):
    a = np.asarray(im).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    out = Image.fromarray(a.astype(np.uint8))
    return ImageEnhance.Contrast(ImageEnhance.Color(out).enhance(1.4)).enhance(1.2)

TILE, LAB = 300, 46
cols = len(SEGMENTS)
canvas = Image.new("RGB", (cols * TILE, TILE + LAB), (12, 12, 12))
d = ImageDraw.Draw(canvas)
for sid, (a, b, lab) in enumerate(SEGMENTS):
    mid = (a + b) // 2
    im = enh(dec(df["observation.images.camera1"].iloc[mid])).resize((TILE, TILE))
    x = sid * TILE
    canvas.paste(im, (x, LAB))
    d.text((x + 4, 3),  f"S{sid}: f{a}-{b} ({a/FPS:.1f}-{b/FPS:.1f}s)", fill=(255, 220, 0))
    # wrap label to ~2 lines
    words, lines, cur = lab.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 34: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    lines.append(cur)
    d.text((x + 4, 19), "\n".join(lines[:2]), fill=(0, 255, 120))
canvas.save(os.path.join(OUT, f"ep{EP:03d}_storyboard.png"))

print(f"wrote ep{EP:03d}_subtasks.json, ep{EP:03d}_subtask_index.npy, ep{EP:03d}_storyboard.png")
for s in subtasks:
    print(f"  S{s['subtask_id']}  f{s['start_frame']:4d}-{s['end_frame']:4d}  "
          f"{s['start_t']:5.1f}-{s['end_t']:5.1f}s  {s['dur_s']:4.1f}s  {s['label']}")
print("DONE")
