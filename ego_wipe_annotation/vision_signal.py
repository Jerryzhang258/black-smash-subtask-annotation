"""
Vision motion signal — complements the proprioceptive wipe-onset (the one soft
boundary). During wiping the wiper hand makes repeated in-place strokes, so its
hand-mounted fisheye sees sustained frame-to-frame change. We measure that as
per-frame frame-difference energy and use it to refine / confirm `start_wipe`.

This is the fusion of the fisheye view INTO the signal stage (vs just showing it).
Frame-diff is cheap and needs no model; everything stays frame-aligned.
"""
from __future__ import annotations
import numpy as np

from . import ego_dataio as io, signal_segment as seg


def frame_diff_energy(demo, hand: str, max_side: int = 96, smooth_s: float = 0.15) -> np.ndarray:
    """Per-frame mean |Δ| between consecutive grayscale fisheye frames of `hand`,
    aligned to the hand-frame grid (length n_frames), smoothed."""
    e = np.zeros(demo.n_frames, dtype=np.float64)
    prev = None
    for i in range(demo.n_frames):
        try:
            im = io.fisheye_image(demo, hand, i).convert("L")
        except Exception:
            prev = None
            continue
        im.thumbnail((max_side, max_side))
        a = np.asarray(im, dtype=np.float32)
        if prev is not None and a.shape == prev.shape:
            e[i] = float(np.abs(a - prev).mean())
        prev = a
    return seg.smooth(e, smooth_s * demo.fps)


def wipe_onset_vision(energy: np.ndarray, fps: float, lo: int, hi: int,
                      frac: float = 0.5, gap_s: float = 0.5) -> tuple[int, bool]:
    """Onset of sustained high frame-diff energy inside [lo, hi). Returns
    (frame, ok); ok=False means the window was too short / flat to call."""
    win = energy[lo:hi]
    if len(win) < 3 or win.max() <= 0:
        return lo + int(0.25 * (hi - lo)), False
    thr = win.min() + frac * (np.percentile(win, 90) - win.min())
    mask = seg.close_gaps(win > thr, int(gap_s * fps))
    a, bexcl = seg.longest_run(mask)
    if bexcl - a < int(0.4 * fps):
        return lo + int(0.25 * (hi - lo)), False
    return lo + int(a), True


def fuse_wipe_onset(pose_onset: int, vision_onset: int, fps: float,
                    tol_s: float = 0.5) -> tuple[int, str, int]:
    """Combine the pose- and vision-derived wipe onsets. They agree -> average;
    they disagree beyond tol -> prefer vision (the visual stroke is the event) and
    flag for review. Returns (fused_frame, source, disagree_frames)."""
    d = abs(pose_onset - vision_onset)
    if d <= tol_s * fps:
        return int(round((pose_onset + vision_onset) / 2)), "pose+vision", d
    return int(vision_onset), "vision(review)", d
