"""
Render multiple annotation tracks for one episode in a single image.

Rows:
  - keyframes from the episode video
  - state timeline
  - qwen timeline
  - fused timeline
  - optional Gemini stage timeline (reach/grasp/transport/place/release/adjust/done)

Examples:
  python visualize_annotation_tracks.py \
    --data /home/hillbot/black_smash_07/data/chunk-000 \
    --state annotations_state_07 \
    --qwen annotations_qwen_07 \
    --fused annotations_fused_07 \
    --out compare_tracks_07

  python visualize_annotation_tracks.py ... \
    --gemini-jsonl gemini_stage_annotation_results/run_xxx/stage_annotations_normalized.jsonl
"""
import argparse
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vlm_annotate import enh


SUBTASK_COLORS = [
    (255, 85, 85),
    (255, 176, 0),
    (255, 122, 0),
    (160, 112, 255),
    (0, 220, 90),
    (80, 160, 255),
    (255, 102, 204),
]

GEMINI_COLORS = {
    "reach": (90, 170, 255),
    "grasp": (255, 190, 70),
    "transport": (120, 220, 135),
    "place": (180, 130, 255),
    "release": (255, 115, 115),
    "adjust": (170, 170, 170),
    "done": (90, 210, 210),
    "global": (210, 210, 210),
}


def font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_doc(directory, ep):
    if not directory:
        return None
    path = os.path.join(directory, f"ep{ep:03d}_subtasks.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_gemini(jsonl_path):
    if not jsonl_path or not os.path.exists(jsonl_path):
        return {}
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec.get("episode_index", rec.get("episode", -1)))
            stages = rec.get("normalized_stages")
            if stages is None:
                parsed = rec.get("parsed_response") or {}
                stages = parsed.get("stages") or []
            out[ep] = {
                "episode_length": int(rec.get("episode_length", rec.get("n_frames", 0))),
                "stages": stages,
            }
    return out


def standard_segments(doc):
    if not doc:
        return None
    return [
        {
            "label": f"S{int(s['subtask_id'])}",
            "name": s.get("label", ""),
            "start": int(s["start_frame"]),
            "end": int(s["end_frame"]),
            "color": SUBTASK_COLORS[int(s["subtask_id"]) % len(SUBTASK_COLORS)],
        }
        for s in doc["subtasks"]
    ]


def gemini_segments(gemini_record, n_frames):
    if not gemini_record:
        return None
    segs = []
    for st in gemini_record.get("stages", []):
        name = str(st.get("name", "adjust")).strip().lower() or "adjust"
        start = int(st.get("start_t", st.get("start_frame", 0)))
        end = int(st.get("end_t", st.get("end_frame", start)))
        start = max(0, min(n_frames - 1, start))
        end = max(start, min(n_frames - 1, end))
        segs.append(
            {
                "label": name,
                "name": name,
                "start": start,
                "end": end,
                "color": GEMINI_COLORS.get(name, GEMINI_COLORS["adjust"]),
            }
        )
    return segs or None


def draw_timeline(draw, x0, y, width, n_frames, name, segs, label_font, small_font):
    row_h = 38
    draw.text((16, y + 10), name, fill=(235, 235, 235), font=label_font)
    if not segs:
        draw.rectangle([x0, y, x0 + width, y + row_h], outline=(80, 80, 85), fill=(35, 35, 40))
        draw.text((x0 + 8, y + 10), "missing", fill=(180, 180, 180), font=small_font)
        return

    for seg in segs:
        sx = int(x0 + seg["start"] / n_frames * width)
        ex = int(x0 + (seg["end"] + 1) / n_frames * width)
        ex = max(sx + 1, ex)
        draw.rectangle([sx, y, ex, y + row_h], fill=seg["color"], outline=(18, 18, 20))
        if ex - sx > 24:
            draw.text((sx + 4, y + 4), seg["label"], fill=(0, 0, 0), font=small_font)


def episode_list(state_dir, eps_arg):
    if eps_arg:
        return [int(x) for x in eps_arg.split(",") if x.strip().isdigit()]
    return sorted(
        int(os.path.basename(p).split("_")[0][2:])
        for p in glob.glob(os.path.join(state_dir, "ep*_subtasks.json"))
    )


def render_one(args, ep, gemini_by_ep):
    state = load_doc(args.state, ep)
    qwen = load_doc(args.qwen, ep)
    fused = load_doc(args.fused, ep)
    base_doc = state or qwen or fused
    if base_doc is None:
        return None

    n_frames = int(base_doc["n_frames"])
    fps = float(base_doc.get("fps", args.fps))
    parquet = os.path.join(args.data, f"episode_{ep:06d}.parquet")
    df = pd.read_parquet(parquet, columns=[args.cam])

    def frame(idx):
        cell = df[args.cam].iloc[int(idx)]
        return enh(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    W = args.width
    margin_l, margin_r = 110, 24
    bar_w = W - margin_l - margin_r
    thumb_n = args.keyframes
    thumb_gap = 6
    thumb_w = (bar_w - thumb_gap * (thumb_n - 1)) // thumb_n
    thumb_h = thumb_w
    top = 46
    key_y = top
    row0 = key_y + thumb_h + 34
    row_gap = 48
    tracks = [
        ("state", standard_segments(state)),
        ("qwen", standard_segments(qwen)),
        ("fused", standard_segments(fused)),
        ("gemini", gemini_segments(gemini_by_ep.get(ep), n_frames)),
    ]
    H = row0 + row_gap * len(tracks) + 42
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    title_font = font(22)
    label_font = font(15)
    small_font = font(12)

    d.text(
        (18, 12),
        f"ep{ep:03d}  N={n_frames}  {n_frames / fps:.1f}s  state / qwen / fused / gemini-stage",
        fill=(245, 245, 245),
        font=title_font,
    )

    idxs = np.linspace(0, n_frames - 1, thumb_n).round().astype(int).tolist()
    d.text((16, key_y + thumb_h // 2 - 8), "frames", fill=(235, 235, 235), font=label_font)
    for k, idx in enumerate(idxs):
        x = margin_l + k * (thumb_w + thumb_gap)
        tile = frame(idx).resize((thumb_w, thumb_h))
        im.paste(tile, (x, key_y))
        d.rectangle([x, key_y, x + thumb_w - 1, key_y + thumb_h - 1], outline=(55, 55, 60))
        d.rectangle([x, key_y, x + 66, key_y + 17], fill=(0, 0, 0))
        d.text((x + 3, key_y + 2), f"f{idx}", fill=(255, 230, 120), font=small_font)

    for i, (name, segs) in enumerate(tracks):
        draw_timeline(d, margin_l, row0 + i * row_gap, bar_w, n_frames, name, segs, label_font, small_font)

    out = os.path.join(args.out, f"ep{ep:03d}_tracks.png")
    im.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--gemini-jsonl", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="")
    ap.add_argument("--cam", default="observation.images.camera1")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--keyframes", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gemini_by_ep = load_gemini(args.gemini_jsonl)
    made = []
    for ep in episode_list(args.state, args.eps):
        out = render_one(args, ep, gemini_by_ep)
        if out:
            made.append(out)
            print("saved", out)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(made),
                "state": args.state,
                "qwen": args.qwen,
                "fused": args.fused,
                "gemini_jsonl": args.gemini_jsonl,
                "images": [os.path.basename(p) for p in made],
            },
            f,
            indent=2,
        )
    index_path = os.path.join(args.out, "index.html")
    cards = "\n".join(
        f'<section><h2>{Path(p).stem}</h2><img src="{os.path.basename(p)}" alt="{Path(p).stem}"></section>'
        for p in made
    )
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Annotation Track Comparison</title>"
            "<style>body{margin:24px;background:#111;color:#eee;font-family:Arial,sans-serif}"
            "section{margin:0 0 28px}img{max-width:100%;height:auto;border:1px solid #333}"
            "h1,h2{font-weight:600}</style></head><body>"
            f"<h1>Annotation Track Comparison ({len(made)} episodes)</h1>{cards}</body></html>"
        )
    print(f"{len(made)} images -> {args.out}")
    print(f"index -> {index_path}")


if __name__ == "__main__":
    main()
