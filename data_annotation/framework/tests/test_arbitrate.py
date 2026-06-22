"""
Regression test: the new sigma-arbiter must reproduce the legacy OWNER-table
fusion (fuse_annotations.py) frame-for-frame, including the p2 -> state fallback.

Runs with no raw data: it uses the committed example annotations in examples/.
No numpy / pandas / pillow needed.

  python data_annotation/framework/tests/test_arbitrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_annotation.framework.core import load_schema           # noqa: E402
from data_annotation.framework.detectors import (                # noqa: E402
    EpisodeContext, StateFileDetector, VLMFileDetector,
)
from data_annotation.framework.fuse import fuse_episode          # noqa: E402

SCHEMA = REPO_ROOT / "data_annotation" / "framework" / "schemas" / "black_smash.json"
EXAMPLES = REPO_ROOT / "examples"

# legacy ownership table from fuse_annotations.py, kept here only as the oracle.
OWNER = ["state", "vlm", "state", "state", "state", "state"]


def legacy_fuse(scp, vcp, tol):
    """Faithful inline re-implementation of fuse_annotations.fuse_ep's core."""
    fused, sources, review = [], [], []
    for i in range(6):
        d = abs(vcp[i] - scp[i])
        val = vcp[i] if OWNER[i] == "vlm" else scp[i]
        if OWNER[i] == "vlm" and d > tol:
            val = scp[i]
        fused.append(val)
        sources.append(OWNER[i])
        if d > tol:
            review.append(i + 1)
    return fused, sources, review


def main() -> int:
    schema = load_schema(SCHEMA)
    state_doc = json.loads((EXAMPLES / "sample_ep000_subtasks.json").read_text())
    vlm_doc = json.loads((EXAMPLES / "sample_ep000_vlm_subtasks.json").read_text())

    tol = int(0.5 * schema.fps)  # 15 frames, the legacy default
    ctx = EpisodeContext(
        episode_index=0, n_frames=state_doc["n_frames"], fps=schema.fps,
        docs={"state": state_doc, "vlm": vlm_doc},
    )
    doc = fuse_episode(schema, ctx, [StateFileDetector(), VLMFileDetector()], tol, sigma_max=12.0)

    exp_cps, exp_sources, exp_review = legacy_fuse(
        state_doc["critical_points"], vlm_doc["critical_points"], tol
    )

    ok = True

    def check(name, got, want):
        nonlocal ok
        status = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{status}] {name}\n        got : {got}\n        want: {want}")

    print("regression: sigma-arbiter vs legacy OWNER fusion (ep000)")
    check("critical_points", doc["critical_points"], exp_cps)
    check("sources",         doc["sources"],         exp_sources)
    check("review_points",   doc["review_points"],   exp_review)

    # the derived behaviour we care about: ownership matches OWNER WITHOUT a table,
    # and p2 falls back to state with a note.
    p2_fallback = any(f.startswith("p2:") and "used state" in f for f in doc["flags"])
    check("p2 fell back to state (derived, not hardcoded)", p2_fallback, True)

    print(f"\nsigmas (per point): {doc['sigmas']}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
