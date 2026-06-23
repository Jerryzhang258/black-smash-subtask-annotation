"""
Stage 2 — proprioceptive segmentation for the ego wipe-tube task.

Boundaries come from frame-aligned signals, never pixels:

  * Gripper width (per hand). "Closed" = width well below its resting (open)
    value; the longest closed run gives grasp (start) and release (end). The hand
    that stays closed LONGER is the *holder* (holds the tube); the other is the
    *wiper*. -> c1 grasp_tube, c2 acquire_wiper, c4 release_tube, c3 finish_wipe.
  * Hand pose (wiper hand). The wipe itself is in-place oscillation: high raw
    speed, low carrier drift. The onset of that run inside [c2, c3] is c3's
    sibling start point -> c5... here named start_wipe (w). Falls back to a
    proportion of [c2, c3] if pose is unavailable.

Returns (critical_points[5], subtasks, flags).
"""
from __future__ import annotations
import numpy as np

from . import config as C


# ----------------------------- signal helpers --------------------------------
def smooth(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w) | 1)  # force odd
    return np.convolve(x, np.ones(w) / w, mode="same")


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Longest run of True. Returns [start, end) half-open."""
    best = (0, 0); s = None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best


def close_gaps(mask: np.ndarray, g: int) -> np.ndarray:
    """Fill interior False gaps shorter than g frames (morphological close)."""
    m = mask.copy(); n = len(m); i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < i and j < n and (j - i) < g:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


# ----------------------------- gripper events --------------------------------
def grip_closed_run(width: np.ndarray, fps: float) -> tuple[int, int, int]:
    """Longest 'closed' run on a gripper-width signal. Returns
    (grasp_frame, release_frame, run_len). release = last_closed + 1."""
    g = norm01(width)
    rest = np.median(g[: max(5, len(g) // 20)])          # resting (open) value
    closed = g < (rest - C.GRIP_CLOSE_FRAC)              # closing drives width DOWN
    closed = close_gaps(closed, int(C.GRIP_GAP_S * fps))
    a, bexcl = longest_run(closed)
    return a, bexcl, bexcl - a


# ----------------------------- motion / wipe ---------------------------------
def _speeds(P: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """raw per-frame speed and slow 'carrier' drift speed of a 3-D trajectory."""
    raw = smooth(np.linalg.norm(np.diff(P, axis=0, prepend=P[:1]), axis=1),
                 C.RAW_SMOOTH_S * fps)
    carrier = np.vstack([smooth(P[:, j], C.CARRIER_SMOOTH_S * fps)
                         for j in range(P.shape[1])]).T
    drift = smooth(np.linalg.norm(np.diff(carrier, axis=0, prepend=carrier[:1]), axis=1),
                   C.RAW_SMOOTH_S * fps)
    return raw, drift


def wipe_onset(pose_wiper, fps: float, lo: int, hi: int) -> tuple[int, bool]:
    """Start of sustained in-place oscillation (wipe) inside [lo, hi).
    Returns (frame, ok). ok=False -> caller used the fallback proportion."""
    if pose_wiper is None or hi - lo < int(0.4 * fps):
        return lo + int(0.25 * (hi - lo)), False
    raw, drift = _speeds(pose_wiper, fps)
    seg_raw, seg_drift = raw[lo:hi], drift[lo:hi]
    if len(seg_raw) < 3 or seg_raw.max() <= 0:
        return lo + int(0.25 * (hi - lo)), False
    wipe = (seg_raw > C.WIPE_RAW_FRAC * seg_raw.max()) & \
           (seg_drift < np.percentile(seg_drift, C.WIPE_DRIFT_PCT))
    wipe = close_gaps(wipe, int(C.WIPE_GAP_S * fps))
    a, bexcl = longest_run(wipe)
    if bexcl - a < int(0.4 * fps):
        return lo + int(0.25 * (hi - lo)), False
    return lo + int(a), True


# ----------------------------- main entry ------------------------------------
def segment(demo) -> tuple[list[int], list[dict], list[str]]:
    N, fps = demo.n_frames, demo.fps
    flags: list[str] = []

    # gripper events for both hands
    ev = {}
    for side in ("left", "right"):
        a, b, ln = grip_closed_run(demo.grip[side], fps)
        if ln < C.GRIP_MIN_HOLD_S * fps:
            flags.append(f"no clear {side} grasp (held {ln} frames)")
        ev[side] = (a, b, ln)

    # role assignment: longer hold = holder (tube); other = wiper
    holder = "left" if ev["left"][2] >= ev["right"][2] else "right"
    wiper = "right" if holder == "left" else "left"

    grasp_tube    = ev[holder][0]   # holder closes on the tube
    acquire_wiper = ev[wiper][0]    # wiper hand closes on the cloth
    finish_wipe   = ev[wiper][1]    # wiper hand opens (done wiping)
    release_tube  = ev[holder][1]   # holder opens (tube set back)
    start_wipe, ok = wipe_onset(demo.pose.get(wiper), fps, acquire_wiper, finish_wipe)
    if not ok:
        flags.append("wipe onset = fallback proportion (no/short pose)")

    # in CRIT_NAMES order: grasp_tube, acquire_wiper, start_wipe, finish_wipe, release_tube
    cps = [grasp_tube, acquire_wiper, start_wipe, finish_wipe, release_tube]
    if not (0 < cps[0] < cps[1] < cps[2] < cps[3] < cps[4] < N - 1):
        flags.append(f"ordering off: cps={cps} (N={N}); using fallback proportions")
        cps = [int(p * N) for p in (0.20, 0.37, 0.45, 0.77, 0.91)]

    subtasks = _subtasks_from_cps(cps, N, fps)
    return cps, subtasks, flags + [f"holder={holder}", f"wiper={wiper}"]


def _subtasks_from_cps(cps: list[int], N: int, fps: float) -> list[dict]:
    starts = [0] + list(cps)
    out = []
    for i, lab in enumerate(C.LABELS):
        a = starts[i]
        b = (starts[i + 1] - 1) if i < len(C.LABELS) - 1 else N - 1
        out.append({"subtask_id": i, "label": lab, "start_frame": a, "end_frame": b,
                    "start_t": round(a / fps, 2), "end_t": round(b / fps, 2),
                    "n_frames": b - a + 1, "dur_s": round((b - a + 1) / fps, 2)})
    return out
