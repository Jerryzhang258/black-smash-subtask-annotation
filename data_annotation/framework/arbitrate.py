"""
Confidence arbitration: turn per-detector candidates into resolved critical
points, WITHOUT a hardcoded ownership table.

Rule (per event):
  1. Collect every candidate from detectors that handle the event type.
  2. primary  = the lowest-sigma candidate (most trustworthy timing).
  3. anchor   = the lowest-sigma *state* candidate, treated as the stable
               proprioceptive skeleton (the design doc keeps state as skeleton).
  4. If primary is a state candidate           -> use it.
     If primary is a vision candidate:
        - agrees with the state anchor (|d| <= tol)  -> inverse-variance fuse the two.
        - disagrees                                  -> fall back to the state anchor
                                                        (vision is coarse on this footage),
                                                        and flag it.
  5. needs_review if: any candidate disagrees with primary beyond tol, OR the
     resolved sigma is above sigma_max, OR there was no state anchor to lean on.

With the per-event sigma profile in detectors.py this reproduces the old
`OWNER = ["state","vlm","state","state","state","state"]` behaviour frame for
frame (gripper/motion events: state wins; pour: vision is primary but falls back
to the state proxy on disagreement) -- but ownership is now derived, so adding a
better detector (e.g. an orientation detector for the pour) changes the outcome
automatically with no table edit.
"""
from __future__ import annotations

import math

from .core import Candidate, Decision, EventSpec


def inverse_variance_fuse(cands: list[Candidate]) -> tuple[int, float]:
    """Combine agreeing candidates weighted by 1/sigma^2."""
    weights = [1.0 / max(c.sigma, 1e-6) ** 2 for c in cands]
    wsum = sum(weights)
    frame = sum(w * c.frame for w, c in zip(weights, cands)) / wsum
    sigma = math.sqrt(1.0 / wsum)
    return int(round(frame)), sigma


def arbitrate_event(
    event: EventSpec,
    candidates: list[Candidate],
    tol_frames: int,
    sigma_max: float,
) -> Decision:
    if not candidates:
        return Decision(event, None, math.inf, "none", True, [], "no detector fired")

    cands = sorted(candidates, key=lambda c: c.sigma)
    primary = cands[0]
    state_cands = [c for c in cands if c.modality == "state"]
    anchor = state_cands[0] if state_cands else None

    max_disagree = max((abs(c.frame - primary.frame) for c in cands[1:]), default=0)
    note = ""

    if primary.modality == "state":
        frame, sigma, source = primary.frame, primary.sigma, primary.detector
    else:
        if anchor is not None and abs(primary.frame - anchor.frame) > tol_frames:
            # vision primary disagrees with the proprioceptive skeleton -> trust state
            frame, sigma, source = anchor.frame, anchor.sigma, primary.detector
            note = f"{primary.detector} primary disagrees with state skeleton, used state"
        elif anchor is not None:
            frame, sigma = inverse_variance_fuse([primary, anchor])
            source = primary.detector
        else:
            frame, sigma, source = primary.frame, primary.sigma, primary.detector

    needs_review = (
        max_disagree > tol_frames
        or sigma > sigma_max
        or (anchor is None and primary.modality != "state")
    )
    return Decision(event, frame, sigma, source, needs_review, cands, note)


def enforce_order(cps: list[int], T: int, flags: list[str]) -> list[int]:
    """Clamp into [1, T-2] and force strictly increasing.

    Ported verbatim from vlm_annotate.enforce_order so fused boundaries keep the
    same ordering guarantees as the legacy pipeline."""
    cps = [max(1, min(T - 2, int(c))) for c in cps]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
            flags.append("nudged p%d for ordering" % (i + 1))
    if cps[-1] > T - 2:
        cps[-1] = T - 2
        for i in range(len(cps) - 2, -1, -1):
            if cps[i] >= cps[i + 1]:
                cps[i] = cps[i + 1] - 1
        if cps[0] < 1:
            flags.append("could not fit ordered points in episode")
    return cps
