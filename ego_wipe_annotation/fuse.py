"""
Stage 3 — per-point fusion of the signal (Stage 2) and VLM (Stage 1) critical
points, plus disagreement-triggered human review.

Each critical point takes its OWNER modality's value (config.CRIT_OWNER): the four
gripper events are owned by the state signal (crisp), and any point whose owner is
"vlm" takes the VLM value. Wherever the two modalities disagree by more than
``tol_s`` seconds, the point is flagged into ``review_points`` so a human only
looks at the few ambiguous boundaries.
"""
from __future__ import annotations

from . import config as C


def fuse(state_cps, vlm_cps, fps, tol_s):
    """Returns (fused_cps, sources, disagree_frames, review_points)."""
    n = len(state_cps)
    tol = tol_s * fps
    fused, sources, disagree, review = [], [], [], []
    for i in range(n):
        s = state_cps[i]
        v = vlm_cps[i] if vlm_cps and i < len(vlm_cps) and vlm_cps[i] is not None else None
        owner = C.CRIT_OWNER[i] if i < len(C.CRIT_OWNER) else "state"
        pick = v if (owner == "vlm" and v is not None) else s
        fused.append(int(pick))
        sources.append(owner if (owner == "vlm" and v is not None) else "state")
        d = abs(s - v) if v is not None else None
        disagree.append(int(d) if d is not None else None)
        if d is not None and d > tol:
            review.append(i)

    # keep the fused points strictly increasing (a VLM-owned point must not cross
    # its crisp neighbours)
    for i in range(1, n):
        if fused[i] <= fused[i - 1]:
            fused[i] = fused[i - 1] + 1
    return fused, sources, disagree, review
