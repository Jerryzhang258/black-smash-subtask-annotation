"""
Load one *demo folder* of the ego wipe-tube dataset into frame-aligned arrays.

A demo folder (e.g. ``demo_bimanual_2026.06.20_20.05.05.755022/``) contains:
  gripper_width_left.csv / gripper_width_right.csv   (frame,width)  -> per-frame
  pose_data/{left,right}_hand_trajectory.csv         (timestamp,x,y,z,q_*) ~60-70 Hz
  tag_detection_{left,right}.pkl                      list[ {frame_idx,time,tag_dict} ]
  left_hand_visual_img/  right_hand_visual_img/       per-frame egocentric jpgs

The gripper CSVs are already one row per image frame. The pose trajectory runs on
its own unix clock at a higher rate, so we align it to frames using the per-frame
``time`` recorded in the tag-detection pickle (same clock as the pose CSV). Pose
is OPTIONAL — only the wipe sub-boundary needs it; gripper events do not.
"""
from __future__ import annotations
import os, glob, pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Demo:
    name: str
    path: str
    n_frames: int
    fps: float
    grip: dict            # {"left": np.ndarray[N], "right": np.ndarray[N]}
    pose: dict            # {"left": np.ndarray[N,3] | None, "right": ...}
    frame_time: np.ndarray | None   # per-frame unix seconds, or None
    frame_dir: str        # absolute dir of per-frame hand-fisheye frames (see config)


def _read_grip(path: str) -> np.ndarray:
    return pd.read_csv(path)["width"].to_numpy(dtype=np.float64)


def _frame_times(demo_path: str, n: int) -> np.ndarray | None:
    """Per-frame unix time from the tag-detection pickle (same clock as pose)."""
    for side in ("left", "right"):
        pkl = os.path.join(demo_path, f"tag_detection_{side}.pkl")
        if not os.path.exists(pkl):
            continue
        try:
            recs = pickle.load(open(pkl, "rb"))
            t = np.array([r["time"] for r in recs], dtype=np.float64)
            if len(t) >= n:
                return t[:n]
        except Exception:
            pass
    return None


def _pose_on_frames(demo_path: str, side: str, frame_time: np.ndarray | None):
    """Interpolate a hand's xyz trajectory onto the frame timestamps."""
    if frame_time is None:
        return None
    csv = os.path.join(demo_path, "pose_data", f"{side}_hand_trajectory.csv")
    if not os.path.exists(csv):
        return None
    tr = pd.read_csv(csv)
    tt = tr["timestamp"].to_numpy(dtype=np.float64)
    return np.stack([np.interp(frame_time, tt, tr[c].to_numpy(dtype=np.float64))
                     for c in ("x", "y", "z")], axis=1)


def load_demo(demo_path: str) -> Demo:
    name = os.path.basename(demo_path.rstrip("/\\"))
    gl = _read_grip(os.path.join(demo_path, "gripper_width_left.csv"))
    gr = _read_grip(os.path.join(demo_path, "gripper_width_right.csv"))
    n = min(len(gl), len(gr))
    grip = {"left": gl[:n], "right": gr[:n]}

    ftime = _frame_times(demo_path, n)
    if ftime is not None and len(ftime) >= 2:
        fps = float((len(ftime) - 1) / (ftime[-1] - ftime[0]))
    else:
        fps = C.DEFAULT_FPS
    pose = {s: _pose_on_frames(demo_path, s, ftime) for s in ("left", "right")}

    return Demo(name=name, path=demo_path, n_frames=n, fps=fps,
                grip=grip, pose=pose, frame_time=ftime,
                frame_dir=os.path.join(demo_path, C.FRAME_DIR))


REQUIRED = ("gripper_width_left.csv", "gripper_width_right.csv")


def scan_demos(demos_root: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (valid_demo_paths, [(skipped_path, reason)]). A folder is a demo
    candidate if it has either hand-timestamp CSV; it is valid only if both
    gripper-width CSVs are present (an incomplete capture is skipped, not crashed)."""
    valid, skipped = [], []
    for p in sorted(glob.glob(os.path.join(demos_root, "*"))):
        if not os.path.isdir(p):
            continue
        is_candidate = any(os.path.exists(os.path.join(p, f))
                           for f in ("left_hand_timestamps.csv", "right_hand_timestamps.csv"))
        missing = [f for f in REQUIRED if not os.path.exists(os.path.join(p, f))]
        if not missing:
            valid.append(p)
        elif is_candidate:
            skipped.append((p, "missing " + ", ".join(missing)))
    return valid, skipped


def list_demos(demos_root: str) -> list[str]:
    return scan_demos(demos_root)[0]


def frame_path(demo: Demo, idx: int) -> str:
    return os.path.join(demo.frame_dir, C.FRAME_FMT.format(i=int(idx)))


# --------------------------- headset ego mp4 frames --------------------------
# Decoded lazily and cached: the whole episode is short (~480 frames). pyav is
# imported only here so the "hand" path never needs it.
_EGO_FRAMES: dict = {}   # (mp4, eye, max_side) -> list[PIL.Image]
_EGO_MAP: dict = {}      # demo.path -> np.ndarray[n_frames] of ego indices
_EGO_WARNED: set = set() # demo.path already warned about missing ego video


def find_ego_video(demo_path: str) -> str | None:
    # mp4 may sit directly in ego_data/ or inside a session subfolder
    for pat in (C.EGO_VIDEO_GLOB, "ego_data/*.mp4", "ego_data/**/*.mp4"):
        hits = sorted(glob.glob(os.path.join(demo_path, pat), recursive=True))
        if hits:
            return hits[0]
    return None


def _ego_frame_times(demo_path: str, eye: str) -> np.ndarray | None:
    import json
    hits = glob.glob(os.path.join(demo_path, "ego_data", "*", f"{eye}_camera_frames.jsonl"))
    if not hits:
        return None
    ts = []
    for line in open(hits[0], encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                ts.append(json.loads(line)["timestamp_ns"])
            except Exception:
                pass
    return np.asarray(ts, dtype=np.float64) if ts else None


def _decode_ego(mp4: str, eye: str, max_side: int) -> list:
    import av
    from PIL import Image as _Im
    out = []
    c = av.open(mp4)
    try:
        for frame in c.decode(video=0):
            im = frame.to_image()                 # stereo side-by-side, e.g. 2560x1280
            w, h = im.size
            half = w // 2
            im = im.crop((0, 0, half, h)) if eye == "left" else im.crop((half, 0, w, h))
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), _Im.LANCZOS)
            out.append(im)
    finally:
        c.close()
    return out


def ego_frames(demo: Demo, max_side: int = 512) -> list | None:
    mp4 = find_ego_video(demo.path)
    if mp4 is None:
        return None
    key = (mp4, C.EGO_EYE, max_side)
    if key not in _EGO_FRAMES:
        _EGO_FRAMES[key] = _decode_ego(mp4, C.EGO_EYE, max_side)
    return _EGO_FRAMES[key]


def ego_manifest(demo_path: str) -> dict | None:
    import json
    hits = glob.glob(os.path.join(demo_path, "ego_data", "*", "manifest.json"))
    if not hits:
        return None
    try:
        return json.load(open(hits[0], encoding="utf-8"))
    except Exception:
        return None


def _hand_elapsed(demo: Demo) -> np.ndarray:
    """Seconds elapsed from the first hand frame (real timestamps if available)."""
    ft = demo.frame_time
    if ft is not None and len(ft) >= 2 and ft[-1] > ft[0]:
        return ft[:demo.n_frames] - ft[0]
    return np.arange(demo.n_frames) / C.DEFAULT_FPS


def _ego_elapsed(demo_path: str, ne: int) -> np.ndarray | None:
    """Seconds elapsed from the first ego frame, length ne (None if unavailable)."""
    et = _ego_frame_times(demo_path, C.EGO_EYE)
    if et is None or len(et) != ne or et[-1] <= et[0]:
        return None
    return (et - et[0]) / 1e9            # timestamp_ns -> seconds


def _hand_to_ego_index(demo: Demo, ne: int) -> np.ndarray:
    """Map each hand-frame index to an ego-frame index by TIMESTAMP.

    Default ("elapsed"): match seconds-elapsed-from-start of each stream (+ a
    manual EGO_TIME_OFFSET_S), using both streams' real per-frame timestamps —
    this is correct across the 60/30 fps rate difference and only assumes a common
    physical start. Falls back to proportional [0,1] matching when timestamps are
    missing or counts disagree."""
    if demo.path in _EGO_MAP and len(_EGO_MAP[demo.path]) == demo.n_frames:
        return _EGO_MAP[demo.path]
    he = _hand_elapsed(demo)
    ee = _ego_elapsed(demo.path, ne)
    if ee is not None and C.EGO_ALIGN == "elapsed":
        target = he + C.EGO_TIME_OFFSET_S               # ego elapsed we want per hand frame
        pos = np.clip(np.searchsorted(ee, target), 1, ne - 1)
        idx = np.where(np.abs(target - ee[pos - 1]) <= np.abs(target - ee[pos]), pos - 1, pos)
    else:                                                # proportional fallback
        rel = (he - he[0]) / (he[-1] - he[0]) if he[-1] > he[0] else np.linspace(0, 1, demo.n_frames)
        if ee is not None:
            er = (ee - ee[0]) / (ee[-1] - ee[0])
            pos = np.clip(np.searchsorted(er, rel), 1, ne - 1)
            idx = np.where(np.abs(rel - er[pos - 1]) <= np.abs(rel - er[pos]), pos - 1, pos)
        else:
            idx = np.round(rel * (ne - 1)).astype(int)
    idx = np.clip(idx, 0, ne - 1).astype(int)
    _EGO_MAP[demo.path] = idx
    return idx


def alignment_report(demo: Demo) -> dict:
    """Diagnostics for the ego<->hand timestamp alignment of one demo."""
    frames = ego_frames(demo)
    if not frames:
        return {"ok": False, "reason": "no ego video"}
    ne = len(frames)
    he, ee = _hand_elapsed(demo), _ego_elapsed(demo.path, ne)
    rep = {"ok": True, "mode": C.EGO_ALIGN, "offset_s": C.EGO_TIME_OFFSET_S,
           "hand_frames": demo.n_frames, "hand_dur_s": round(float(he[-1] - he[0]), 3),
           "ego_frames": ne, "ego_dur_s": round(float(ee[-1] - ee[0]), 3) if ee is not None else None}
    man = ego_manifest(demo.path)
    if man and ee is not None and demo.frame_time is not None:
        # absolute device-clock offset (manifest converts ego ticks -> unix)
        ss_u = man.get("session_start_unix_us"); ss_t = man.get("session_start_ticks_us")
        et = _ego_frame_times(demo.path, C.EGO_EYE)
        if ss_u and ss_t and et is not None:
            ego_unix0 = ss_u / 1e6 + (et[0] / 1e9 - ss_t / 1e6)
            rep["device_clock_offset_s"] = round(float(ego_unix0 - demo.frame_time[0]), 1)
    idx = _hand_to_ego_index(demo, ne)
    rep["ego_idx_range"] = [int(idx.min()), int(idx.max())]
    return rep


def hand_image(demo: Demo, idx: int):
    """Hand-fisheye jpg for a frame index (always, regardless of FRAME_SOURCE)."""
    from PIL import Image
    return Image.open(frame_path(demo, int(idx))).convert("RGB")


def fisheye_image(demo: Demo, hand: str, idx: int):
    """Fisheye jpg of a specific hand ('left'/'right')."""
    from PIL import Image
    p = os.path.join(demo.path, f"{hand}_hand_visual_img", f"{hand}_hand_{int(idx)}.jpg")
    return Image.open(p).convert("RGB")


def ego_image(demo: Demo, idx: int):
    """Headset-ego frame for a HAND-frame index, or None if no ego video."""
    frames = ego_frames(demo)
    if not frames:
        return None
    ego_idx = _hand_to_ego_index(demo, len(frames))[int(idx)]
    return frames[int(ego_idx)]


def frame_image(demo: Demo, idx: int):
    """PIL.Image for a HAND-frame index, from the configured FRAME_SOURCE.
    Falls back to the hand-fisheye jpg if ego frames are unavailable."""
    if C.FRAME_SOURCE == "ego":
        im = ego_image(demo, idx)
        if im is not None:
            return im
        if demo.path not in _EGO_WARNED:       # warn once per demo
            _EGO_WARNED.add(demo.path)
            print(f"[warn] FRAME_SOURCE=ego but no ego video under {demo.name}/ego_data "
                  f"-> falling back to hand-fisheye frames")
    return hand_image(demo, idx)
