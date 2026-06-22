"""
Detectors: each turns episode signals into per-event `Candidate` boundaries with
a localization sigma (frames). Detectors do NOT decide ownership -- the arbiter
does, from their sigmas. To add a new boundary source you add a detector here;
nothing else in the pipeline changes.

Two families:
  * file detectors    -- read existing ep<NNN>_subtasks.json (state / vlm). Used
                         to reproduce the legacy fuse_annotations.py output and to
                         run the regression test with no raw data present.
  * signal detectors  -- run on raw observation.state. SignalStateDetector reuses
                         the proven batch_annotate.segment_episode (so P0 stays
                         faithful by construction); OrientationDetector is the new
                         state-based pour detector (experimental, off until the
                         schema names the wrist dims).

Per-event sigma profiles below encode how good each modality is at each event
type. They are the only knob that used to live in the hand-written OWNER table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import DEFAULT_SIGMA, Candidate, EventSpec, TaskSchema


@dataclass
class EpisodeContext:
    """Everything a detector might need for one episode."""

    episode_index: int
    n_frames: int
    fps: int
    state: Any = None                       # (T, D) numpy array of observation.state, or None
    docs: dict[str, dict] = field(default_factory=dict)   # source name -> legacy annotation doc


# crisp in proprioception (gripper), coarse for the visual pour proxy.
SIGMA_STATE = {"gripper_close": 2.0, "gripper_open": 2.0, "motion_regime": 8.0, "orientation_change": 25.0}
# vision nails the visible pour, coarse on exact gripper/motion timing.
SIGMA_VLM = {"gripper_close": 15.0, "gripper_open": 15.0, "motion_regime": 12.0, "orientation_change": 6.0}


class Detector:
    """Base detector. Subclasses set name/modality and implement propose()."""

    name: str = "detector"
    modality: str = "state"            # "state" | "vision"

    def handles(self, event_type: str) -> bool:
        return True

    def propose(self, schema: TaskSchema, ctx: EpisodeContext) -> dict[int, Candidate]:
        """Return {event.index: Candidate} for the events this detector handles."""
        raise NotImplementedError


class CriticalPointFileDetector(Detector):
    """Emit candidates from a legacy ep<NNN>_subtasks.json `critical_points` list."""

    def __init__(self, name: str, modality: str, doc_key: str, sigma_profile: dict[str, float]):
        self.name = name
        self.modality = modality
        self.doc_key = doc_key
        self.sigma_profile = sigma_profile

    def propose(self, schema: TaskSchema, ctx: EpisodeContext) -> dict[int, Candidate]:
        doc = ctx.docs.get(self.doc_key)
        if not doc:
            return {}
        cps = doc["critical_points"]
        out: dict[int, Candidate] = {}
        for ev in schema.events:
            if ev.index >= len(cps):
                continue
            out[ev.index] = Candidate(
                frame=int(cps[ev.index]),
                sigma=self.sigma_profile.get(ev.type, DEFAULT_SIGMA),
                detector=self.name,
                modality=self.modality,
                evidence={"source": "file", "doc": self.doc_key},
            )
        return out


def StateFileDetector() -> CriticalPointFileDetector:
    return CriticalPointFileDetector("state", "state", "state", SIGMA_STATE)


def VLMFileDetector() -> CriticalPointFileDetector:
    return CriticalPointFileDetector("vlm", "vision", "vlm", SIGMA_VLM)


class SignalStateDetector(Detector):
    """Proprioceptive boundaries by reusing batch_annotate.segment_episode.

    P0 keeps detection identical to the legacy pipeline; only the fusion layer
    changes. (Splitting segment_episode into independent per-event detectors is
    phase P3 and must be validated against real data.)"""

    name = "state-signal"
    modality = "state"

    def propose(self, schema: TaskSchema, ctx: EpisodeContext) -> dict[int, Candidate]:
        if ctx.state is None:
            return {}
        from batch_annotate import segment_episode  # proven detector, reused verbatim

        _segs, flags, cps = segment_episode(ctx.state, ctx.fps)
        out: dict[int, Candidate] = {}
        for ev in schema.events:
            if ev.index >= len(cps):
                continue
            out[ev.index] = Candidate(
                frame=int(cps[ev.index]),
                sigma=SIGMA_STATE.get(ev.type, DEFAULT_SIGMA),
                detector=self.name,
                modality=self.modality,
                evidence={"flags": flags},
            )
        return out


# OrientationDetector tunables (override per task via schema.state_layout["p2_orientation"]).
DEFAULT_ORIENTATION_CFG = {
    "transport_delay_s": 1.0,    # don't search until the tube is over the mortar
    "release_margin_s": 0.2,     # stop searching before release
    "min_pour_after_grasp": 30,  # hard guard: p2 >= p1 + this (frames)
    "min_before_release": 10,    # hard guard: p2 <= p3 - this (frames)
    "smooth_s": 0.3,             # low-pass window; kills jitter so noise has no long runs
    "hold_frames": 10,           # the tilt must keep one direction at least this long
    "delta_threshold": 1.3,      # net tilt over the run, in std units, to count as a pour
    "strong_strength": 2.0,      # sustained tilt (std) at/above this -> sigma 6
    "weak_strength": 1.0,        # at/below this after a hit -> weakest (sigma 18)
}


def _scan_tilt(sm, slope, sd, lo, hi, cfg, np):
    """Strongest sustained, directional tilt onset inside [lo, hi).

    Pour = a monotonic rotation, so we scan for maximal constant-slope-sign runs
    (on the smoothed signal) of at least hold_frames whose net displacement clears
    delta_threshold standard deviations. Transport jitter has no such long run, so
    the detector abstains -- returns (None, 0.0, 0) -- instead of emitting it.
    Returns (onset, strength, direction) for the run with the largest displacement."""
    hold = cfg["hold_frames"]
    thr = cfg["delta_threshold"]
    sgn = np.sign(slope)
    best = (None, 0.0, 0)
    t = lo
    while t < hi:
        s = sgn[t]
        if s == 0:
            t += 1
            continue
        e = t
        while e < hi and sgn[e] == s:
            e += 1
        if e - t >= hold:
            disp = abs(sm[e - 1] - sm[t]) / sd
            if disp >= thr and disp > best[1]:
                best = (t, float(disp), int(s))
        t = e
    return best


def _sigma_from_strength(strength, cfg):
    """Calibrated sigma: a strong, clean tilt is trustworthy (6); a weak one is
    not (up to 18), so the arbiter prefers state only when the signal is good."""
    if strength >= cfg["strong_strength"]:
        return 6.0
    span = max(cfg["strong_strength"] - cfg["weak_strength"], 1e-6)
    frac = min(max((strength - cfg["weak_strength"]) / span, 0.0), 1.0)   # 0..1
    return 18.0 - 6.0 * frac


class OrientationDetector(Detector):
    """Phase P2: detect the pour onset (p2) from tube-arm wrist rotation in
    observation.state, so the pour can leave the VLM. Implements the plan in
    docs/P2_ORIENTATION_OPTIMIZATION.md:

      * gate the search to [p1 + transport_delay, p3 - margin] (skip lift/transport);
      * fire only on a sustained, directional tilt -- not any motion spike;
      * abstain (emit nothing) when no clean onset exists, vs forcing a bad p2;
      * report a calibrated sigma so the arbiter prefers state only when it is good.

    Inert until the schema names state_layout.tube_wrist_dims. Validate on real
    episodes (tools/eval_orientation_p2.py) before enabling in production."""

    name = "orientation"
    modality = "state"

    def handles(self, event_type: str) -> bool:
        return event_type == "orientation_change"

    def propose(self, schema: TaskSchema, ctx: EpisodeContext) -> dict[int, Candidate]:
        wrist_dims = schema.state_layout.get("tube_wrist_dims")
        if ctx.state is None or not wrist_dims:
            return {}
        pour = next((e for e in schema.events if e.type == "orientation_change"), None)
        if pour is None:
            return {}

        import numpy as np
        from batch_annotate import close_gaps, longest_run, smooth

        cfg = {**DEFAULT_ORIENTATION_CFG, **schema.state_layout.get("p2_orientation", {})}
        S = ctx.state
        fps = ctx.fps
        grip = schema.state_layout.get("tube_gripper", 3)

        # tube-held window -> p1 (grasp) and p3 (release) proxies
        g = S[:, grip]
        rest = np.median(g[: max(5, len(g) // 20)])
        rng = np.percentile(g, 99) - np.percentile(g, 1) + 1e-9
        held = close_gaps(np.abs(g - rest) > 0.35 * rng, int(0.4 * fps))
        a, b = longest_run(held)
        if b - a < int(0.3 * fps):
            return {}                                   # no clear held window -> abstain
        p1, p3 = a, b + 1

        # guarded search window: skip transport after grasp, stop before release
        lo = p1 + max(int(cfg["transport_delay_s"] * fps), cfg["min_pour_after_grasp"])
        hi = p3 - max(int(cfg["release_margin_s"] * fps), cfg["min_before_release"])
        lo, hi = max(lo, 1), min(hi, len(S) - 2)
        if hi - lo < cfg["hold_frames"] + 2:
            return {}                                   # window too small -> abstain

        # pick the wrist dim with the strongest sustained directional tilt in-window
        best = None
        for d in wrist_dims:
            x = S[:, d].astype(float)
            sm = smooth(x, cfg["smooth_s"] * fps)
            sd = sm.std() + 1e-9
            slope = np.diff(sm, prepend=sm[:1])
            onset, strength, direction = _scan_tilt(sm, slope, sd, lo, hi, cfg, np)
            if onset is not None and (best is None or strength > best[2]):
                best = (d, onset, strength, direction)

        if best is None:
            return {}                                   # no clean onset -> abstain

        d, onset, strength, direction = best
        sigma = _sigma_from_strength(strength, cfg)
        return {pour.index: Candidate(
            int(onset), sigma, self.name, "state",
            {"dim": int(d), "direction": int(direction), "strength": round(float(strength), 3),
             "search_window": [int(lo), int(hi)], "held_window": [int(a), int(b)]},
        )}
