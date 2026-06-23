"""
Per-demo visualization: a subtask timeline bar (top) + one enhanced egocentric
keyframe per subtask (the segment midpoint). Mirrors the black_smash
`visualize_annotation.py` style. No GPU / API needed.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from . import ego_dataio as io

_PALETTE = [(66, 135, 245), (66, 245, 156), (245, 206, 66), (245, 132, 66),
            (245, 66, 105), (160, 66, 245), (66, 233, 245)]


def _enh(im: Image.Image, crop: float = 1.0) -> Image.Image:
    """Center-crop to the action, then white-balance + percentile stretch + gamma
    to make the dark fisheye legible."""
    if crop < 1.0:
        w, h = im.size
        cw, ch = int(w * crop), int(h * crop)
        im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    a = np.asarray(im).astype(np.float32)
    m = a.reshape(-1, 3).mean(0) + 1e-6
    a = np.clip(a * (m.mean() / m), 0, 255)
    lo, hi = np.percentile(a, 2), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    a = (a ** 0.55) * 255                       # stronger gamma lift for dark frames
    o = Image.fromarray(a.astype(np.uint8))
    return ImageEnhance.Contrast(ImageEnhance.Color(o).enhance(1.3)).enhance(1.4)


def render(demo, doc: dict, out_png: str, tile: int = 380, crop: float = 0.82) -> str:
    subs = doc["subtasks"]
    N = demo.n_frames
    BAR, LAB, PAD = 26, 46, 6
    W = max(len(subs) * tile, 720)
    cv = Image.new("RGB", (W, BAR + LAB + tile + 2 * PAD), (18, 18, 18))
    d = ImageDraw.Draw(cv)

    # timeline bar (proportional to frame span)
    for s in subs:
        x0 = PAD + int((W - 2 * PAD) * s["start_frame"] / N)
        x1 = PAD + int((W - 2 * PAD) * (s["end_frame"] + 1) / N)
        d.rectangle([x0, PAD, x1, PAD + BAR], fill=_PALETTE[s["subtask_id"] % len(_PALETTE)])
        d.text((x0 + 2, PAD + 6), f"S{s['subtask_id']}", fill=(0, 0, 0))

    # keyframes
    review = set(doc.get("review_points", []))
    for k, s in enumerate(subs):
        mid = (s["start_frame"] + s["end_frame"]) // 2
        try:
            im = _enh(io.frame_image(demo, mid), crop).resize((tile, tile))
        except Exception:
            im = Image.new("RGB", (tile, tile), (40, 40, 40))
        y = PAD + BAR + LAB
        cv.paste(im, (k * tile, y))
        col = _PALETTE[s["subtask_id"] % len(_PALETTE)]
        d.rectangle([k * tile, y, k * tile + tile - 1, y + tile - 1], outline=col, width=3)
        # label (wrap)
        head = f"S{s['subtask_id']} f{s['start_frame']}-{s['end_frame']} ({s['dur_s']}s)"
        if (k in review) or (k + 1 in review):
            head += "  *REVIEW*"
        d.rectangle([k * tile, BAR + PAD, k * tile + tile, BAR + PAD + LAB], fill=(18, 18, 18))
        d.text((k * tile + 3, BAR + PAD + 2), head, fill=(255, 220, 0))
        words, lines, cur = s["label"].split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 30:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        d.text((k * tile + 3, BAR + PAD + 18), "\n".join(lines[:2]), fill=(150, 230, 150))

    cv.save(out_png)
    return out_png


def _norm(x):
    x = np.asarray(x, float)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


PLOT_L, PLOT_R = 56, 16          # x-margins shared by the plot and the dashboard


def _make_plot(demo, doc: dict, traces: list, marks: list | None,
               W: int, H: int, label_cps: bool = True) -> Image.Image:
    """Image of normalized per-frame signals vs frame index, with the 5 critical
    points as vertical lines. x-axis spans [PLOT_L, W-PLOT_R] = frame 0..N-1, so a
    timeline bar drawn over the same x-range lines up with it."""
    N = demo.n_frames
    L, R, T, B = PLOT_L, PLOT_R, 16, 56
    pw, ph = W - L - R, H - T - B
    cv = Image.new("RGB", (W, H), (20, 20, 24)); d = ImageDraw.Draw(cv)
    d.rectangle([L, T, L + pw, T + ph], outline=(80, 80, 90))
    for g in range(5):                                    # y gridlines
        y = T + ph - int(ph * g / 4)
        d.line([L, y, L + pw, y], fill=(40, 40, 48))
        d.text((6, y - 6), f"{g/4:.2f}", fill=(120, 120, 130))
    def fx(f): return L + int(pw * f / max(1, N - 1))
    for i, (c, nm) in enumerate(zip(doc["critical_points"], doc["critical_names"])):
        x = fx(c)
        d.line([x, T, x, T + ph], fill=(150, 150, 160))
        if label_cps:
            d.text((x + 2, T + 2 + (i % 3) * 12), f"c{i+1} {nm}@{c}", fill=(200, 200, 210))
    for k, (name, color, arr) in enumerate(traces):       # traces
        y = _norm(arr)
        pts = [(fx(f), T + ph - int(ph * y[f])) for f in range(N)]
        d.line(pts, fill=color, width=2)
        d.rectangle([L + 8 + k * 190, T + ph + 16, L + 20 + k * 190, T + ph + 28], fill=color)
        d.text((L + 24 + k * 190, T + ph + 16), name, fill=(220, 220, 230))
    for lab, color, f in (marks or []):                   # event marks
        x = fx(f)
        d.line([x, T, x, T + ph], fill=color)
        for dy in (T + 14, T + 26):
            d.text((x + 3, dy), lab, fill=color)
    return cv


def plot_signals(demo, doc: dict, traces: list, out_png: str,
                 marks: list | None = None, W: int = 1180, H: int = 460) -> str:
    _make_plot(demo, doc, traces, marks, W, H).save(out_png)
    return out_png


def render_dashboard(demo, doc: dict, traces: list, out_png: str,
                     marks: list | None = None, W: int = 1320, plot_h: int = 360) -> str:
    """Everything on one shared frame axis: timeline bar, a row of ego keyframes,
    a row of fisheye keyframes, and the signal plot below — so the critical-point
    lines in the plot sit directly under the timeline transitions."""
    subs = doc["subtasks"]; N = demo.n_frames; nc = len(subs)
    L, R = PLOT_L, PLOT_R
    pw = W - L - R
    tile = pw // nc
    BAR, LAB, PAD = 24, 18, 8
    y_bar = PAD
    y_ego = y_bar + BAR + LAB
    y_fish = y_ego + tile + LAB
    y_plot = y_fish + tile + PAD
    H = y_plot + plot_h
    cv = Image.new("RGB", (W, H), (18, 18, 18)); d = ImageDraw.Draw(cv)
    def fx(f): return L + int(pw * f / max(1, N - 1))

    for s in subs:                                        # timeline bar (proportional)
        x0, x1 = fx(s["start_frame"]), fx(s["end_frame"] + 1)
        col = _PALETTE[s["subtask_id"] % len(_PALETTE)]
        d.rectangle([x0, y_bar, x1, y_bar + BAR], fill=col)
        d.text((x0 + 3, y_bar + 6), f"S{s['subtask_id']}", fill=(0, 0, 0))

    review = set(doc.get("review_points", []))
    for k, s in enumerate(subs):                          # ego + fisheye keyframe rows
        mid = (s["start_frame"] + s["end_frame"]) // 2
        col = _PALETTE[s["subtask_id"] % len(_PALETTE)]
        x = L + k * tile
        for (yrow, getter, crop, tag) in [(y_ego, io.ego_image, 1.0, "EGO"),
                                          (y_fish, io.hand_image, 0.82, "FISHEYE")]:
            try:
                src = getter(demo, mid)
                im = (_enh(src, crop) if src is not None else Image.new("RGB", (tile, tile), (40, 40, 40))).resize((tile, tile))
            except Exception:
                im = Image.new("RGB", (tile, tile), (40, 40, 40))
            cv.paste(im, (x, yrow))
            d.rectangle([x, yrow, x + tile - 1, yrow + tile - 1], outline=col, width=2)
            d.text((x + 3, yrow + 2), tag, fill=(255, 220, 0))
        head = f"S{s['subtask_id']} f{s['start_frame']}-{s['end_frame']}"
        if (k in review) or (k + 1 in review):
            head += " *REVIEW*"
        d.text((x + 3, y_ego - LAB + 2), head, fill=(255, 220, 0))
        d.text((x + 3, y_fish - LAB + 2), s["label"][:30], fill=(150, 230, 150))

    plot = _make_plot(demo, doc, traces, marks, W, plot_h, label_cps=True)
    cv.paste(plot, (0, y_plot))
    cv.save(out_png)
    return out_png


def render_combined(demo, doc: dict, out_png: str, tile: int = 300,
                    crop_ego: float = 1.0, crop_hand: float = 0.82) -> str:
    """Dual-view timeline: per subtask, the headset-ego keyframe (left, context)
    beside the hand-fisheye keyframe (right, grasp/contact detail). Ego gives the
    legible whole-scene view; fisheye shows what the gripper is doing."""
    subs = doc["subtasks"]
    N = demo.n_frames
    BAR, LAB, PAD, GAP = 26, 46, 6, 4
    cell = 2 * tile + GAP
    W = max(len(subs) * cell, 720)
    cv = Image.new("RGB", (W, BAR + LAB + tile + 2 * PAD), (18, 18, 18))
    d = ImageDraw.Draw(cv)

    for s in subs:                                  # timeline bar
        x0 = PAD + int((W - 2 * PAD) * s["start_frame"] / N)
        x1 = PAD + int((W - 2 * PAD) * (s["end_frame"] + 1) / N)
        d.rectangle([x0, PAD, x1, PAD + BAR], fill=_PALETTE[s["subtask_id"] % len(_PALETTE)])
        d.text((x0 + 2, PAD + 6), f"S{s['subtask_id']}", fill=(0, 0, 0))

    review = set(doc.get("review_points", []))
    for k, s in enumerate(subs):
        mid = (s["start_frame"] + s["end_frame"]) // 2
        col = _PALETTE[s["subtask_id"] % len(_PALETTE)]
        y = PAD + BAR + LAB
        x = k * cell
        for j, (getter, crop, tag) in enumerate(
                [(io.ego_image, crop_ego, "EGO"), (io.hand_image, crop_hand, "FISHEYE")]):
            try:
                src = getter(demo, mid)
                im = (_enh(src, crop) if src is not None else Image.new("RGB", (tile, tile), (40, 40, 40))).resize((tile, tile))
            except Exception:
                im = Image.new("RGB", (tile, tile), (40, 40, 40))
            xx = x + j * (tile + GAP)
            cv.paste(im, (xx, y))
            d.rectangle([xx, y, xx + tile - 1, y + tile - 1], outline=col, width=3)
            d.text((xx + 4, y + 3), tag, fill=(255, 220, 0))
        head = f"S{s['subtask_id']} f{s['start_frame']}-{s['end_frame']} ({s['dur_s']}s)"
        if (k in review) or (k + 1 in review):
            head += "  *REVIEW*"
        d.rectangle([x, BAR + PAD, x + cell, BAR + PAD + LAB], fill=(18, 18, 18))
        d.text((x + 4, BAR + PAD + 2), head, fill=(255, 220, 0))
        d.text((x + 4, BAR + PAD + 18), s["label"][:54], fill=(150, 230, 150))
    cv.save(out_png)
    return out_png


def render_boundaries(demo, doc: dict, out_png: str,
                      offsets=(-20, -10, 0, 10, 20), tile: int = 300, crop: float = 0.82) -> str:
    """Boundary QA: one row per critical point, frames spanning the transition.
    The center (boundary) frame is outlined red; each frame is annotated with the
    owning signal's value (gripper width 0..1, or wiper-hand speed x1e3 for the
    wipe onset) so the localization can be eyeballed against the signal."""
    from . import signal_segment as seg
    cps, names, N = doc["critical_points"], doc["critical_names"], demo.n_frames
    flags = doc.get("flags", [])
    holder = next((f.split("=")[1] for f in flags if f.startswith("holder=")), "left")
    wiper = next((f.split("=")[1] for f in flags if f.startswith("wiper=")), "right")
    hand_for = [holder, wiper, wiper, wiper, holder]   # whose gripper owns each point

    gnorm = {s: seg.norm01(demo.grip[s]) for s in ("left", "right")}
    wiper_speed = None
    if demo.pose.get(wiper) is not None:
        wiper_speed, _ = seg._speeds(demo.pose[wiper], demo.fps)

    LAB, COLS, ROWS = 34, len(offsets), len(cps)
    cv = Image.new("RGB", (COLS * tile, ROWS * (tile + LAB)), (18, 18, 18))
    d = ImageDraw.Draw(cv)
    for r, (c, nm) in enumerate(zip(cps, names)):
        hand = hand_for[r] if r < len(hand_for) else holder
        y0 = r * (tile + LAB)
        d.rectangle([0, y0, COLS * tile, y0 + LAB], fill=(30, 30, 30))
        d.text((6, y0 + 4), f"c{r+1} {nm}  @f{c} ({c/demo.fps:.2f}s)   signal: {hand} hand  "
                            f"(w=gripper width 0..1, lower=closed)", fill=(120, 200, 255))
        for k, off in enumerate(offsets):
            fi = int(np.clip(c + off, 0, N - 1))
            try:
                im = _enh(io.frame_image(demo, fi), crop).resize((tile, tile))
            except Exception:
                im = Image.new("RGB", (tile, tile), (40, 40, 40))
            x = k * tile
            cv.paste(im, (x, y0 + LAB))
            is_c = (off == 0)
            d.rectangle([x, y0 + LAB, x + tile - 1, y0 + LAB + tile - 1],
                        outline=(255, 60, 60) if is_c else (90, 90, 90), width=6 if is_c else 2)
            if nm == "start_wipe" and wiper_speed is not None:
                val = f"spd {wiper_speed[fi]*1e3:.1f}"
            else:
                val = f"w {gnorm[hand][fi]:.2f}"
            tag = f"f{fi}{'  <-CP' if is_c else ''}  {val}"
            d.rectangle([x, y0 + LAB, x + int(8.5 * len(tag)) + 6, y0 + LAB + 16], fill=(0, 0, 0))
            d.text((x + 3, y0 + LAB + 2), tag, fill=(255, 220, 0) if is_c else (200, 200, 200))
    cv.save(out_png)
    return out_png


def render_boundaries_dual(demo, doc: dict, out_png: str,
                           offsets=(-20, -10, 0, 10, 20), tile: int = 232,
                           crop_ego: float = 1.0, crop_hand: float = 0.82) -> str:
    """Dual-view boundary QA: per critical point, an EGO row AND a FISHEYE row
    spanning the transition (red = the boundary frame). Lets you compare alignment
    in both views — the fisheye (wrist cam) changes most at the grasp/contact."""
    from . import signal_segment as seg
    cps, names, N = doc["critical_points"], doc["critical_names"], demo.n_frames
    flags = doc.get("flags", [])
    holder = next((f.split("=")[1] for f in flags if f.startswith("holder=")), "left")
    wiper = next((f.split("=")[1] for f in flags if f.startswith("wiper=")), "right")
    hand_for = [holder, wiper, wiper, wiper, holder]
    gnorm = {s: seg.norm01(demo.grip[s]) for s in ("left", "right")}
    wiper_speed = None
    if demo.pose.get(wiper) is not None:
        wiper_speed, _ = seg._speeds(demo.pose[wiper], demo.fps)

    HDR, COLS = 20, len(offsets)
    block = HDR + 2 * tile
    W, H = COLS * tile, len(cps) * block
    cv = Image.new("RGB", (W, H), (18, 18, 18)); d = ImageDraw.Draw(cv)
    for r, (c, nm) in enumerate(zip(cps, names)):
        hand = hand_for[r] if r < len(hand_for) else holder
        y0 = r * block
        d.rectangle([0, y0, W, y0 + HDR], fill=(30, 30, 30))
        d.text((6, y0 + 4), f"c{r+1} {nm} @f{c} ({c/demo.fps:.2f}s)  signal:{hand} (w 0..1, lower=closed)",
               fill=(120, 200, 255))
        def ego_g(dm, i): return io.ego_image(dm, i)
        def fish_g(dm, i, h=hand): return io.fisheye_image(dm, h, i)   # the owning hand's wrist cam
        for vi, (getter, crop, tag) in enumerate([(ego_g, crop_ego, "EGO"),
                                                  (fish_g, crop_hand, f"FISH:{hand}")]):
            yrow = y0 + HDR + vi * tile
            for k, off in enumerate(offsets):
                fi = int(np.clip(c + off, 0, N - 1))
                try:
                    src = getter(demo, fi)
                    im = (_enh(src, crop) if src is not None else Image.new("RGB", (tile, tile), (40, 40, 40))).resize((tile, tile))
                except Exception:
                    im = Image.new("RGB", (tile, tile), (40, 40, 40))
                x = k * tile
                cv.paste(im, (x, yrow))
                is_c = (off == 0)
                d.rectangle([x, yrow, x + tile - 1, yrow + tile - 1],
                            outline=(255, 60, 60) if is_c else (90, 90, 90), width=5 if is_c else 2)
                if nm == "start_wipe" and wiper_speed is not None:
                    val = f"spd {wiper_speed[fi]*1e3:.1f}"
                else:
                    val = f"w {gnorm[hand][fi]:.2f}"
                lab = f"{tag} f{fi}{' <CP' if is_c else ''} {val}"
                d.rectangle([x, yrow, x + int(7.2 * len(lab)) + 4, yrow + 14], fill=(0, 0, 0))
                d.text((x + 2, yrow + 1), lab, fill=(255, 220, 0) if is_c else (200, 200, 200))
    cv.save(out_png)
    return out_png
