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


class OrientationDetector(Detector):
    """EXPERIMENTAL (phase P2): detect the pour onset from wrist rotation in
    observation.state instead of from vision.

    Inert until the schema's state_layout names the held arm's wrist dimensions,
    e.g. "tube_wrist_dims": [..]. Once enabled it competes with the VLM purely on
    sigma -- if its localization is tight it will win the pour automatically, with
    no OWNER-table edit. Quality must be validated on real episodes before trusting
    it (no raw data is available in this checkout)."""

    name = "orientation"
    modality = "state"

    def handles(self, event_type: str) -> bool:
        return event_type == "orientation_change"

    def propose(self, schema: TaskSchema, ctx: EpisodeContext) -> dict[int, Candidate]:
        wrist_dims = schema.state_layout.get("tube_wrist_dims")
        if ctx.state is None or not wrist_dims:
            return {}
        import numpy as np
        from batch_annotate import close_gaps, longest_run, smooth

        S = ctx.state
        fps = ctx.fps
        layout = schema.state_layout
        grip = layout.get("tube_gripper", 3)

        # tube-held window: gripper deviates from its resting (open) value
        g = S[:, grip]
        rest = np.median(g[: max(5, len(g) // 20)])
        held = np.abs(g - rest) > 0.35 * (np.percentile(g, 99) - np.percentile(g, 1) + 1e-9)
        held = close_gaps(held, int(0.4 * fps))
        a, b = longest_run(held)
        if b - a < int(0.3 * fps):
            return {}

        # angular speed proxy = motion of the wrist dims; pour onset = first sustained spike
        W = S[:, wrist_dims].astype(float)
        W = (W - W.mean(0)) / (W.std(0) + 1e-9)
        ang = smooth(np.linalg.norm(np.diff(W, axis=0, prepend=W[:1]), axis=1), 0.2 * fps)
        seg = ang[a:b]
        if len(seg) < 3:
            return {}
        spike = close_gaps(seg > np.percentile(seg, 70), int(0.3 * fps))
        r0, _ = longest_run(spike)
        frame = int(a + r0)

        pour = next((e for e in schema.events if e.type == "orientation_change"), None)
        if pour is None:
            return {}
        return {pour.index: Candidate(frame, 6.0, self.name, "state", {"window": [int(a), int(b)]})}
