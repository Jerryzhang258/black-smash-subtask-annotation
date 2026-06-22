"""
Synthetic unit test for the improved OrientationDetector (phase P2).

No raw dataset is available in this checkout, so we build a state array with a
known held window and a planted pour tilt, and check the three behaviours from
docs/P2_ORIENTATION_OPTIMIZATION.md:

  1. detects a sustained directional tilt inside the guarded window;
  2. abstains when the wrist signal is flat;
  3. rejects an early transport spike right after grasp (gating + hold-time).

  python data_annotation/framework/tests/test_orientation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_annotation.framework.core import load_schema           # noqa: E402
from data_annotation.framework.detectors import (                # noqa: E402
    EpisodeContext, OrientationDetector,
)

SCHEMA_PATH = REPO_ROOT / "data_annotation" / "framework" / "schemas" / "black_smash.json"
T, FPS = 400, 30
GRASP, RELEASE = 100, 300          # tube-held window
WRIST_DIM = 5
POUR = 200                         # planted pour onset


def base_state() -> np.ndarray:
    """20-dim state: tube gripper (dim 3) closed over the held window, all else 0."""
    rng = np.random.default_rng(0)
    S = rng.normal(0, 0.01, size=(T, 20))
    S[GRASP:RELEASE, 3] = 1.0       # tube gripper closed
    return S


def with_tilt(S: np.ndarray) -> np.ndarray:
    """Realistic pour: tube tilts over the mortar, holds, then returns upright
    before release -- a sustained directional run, not a spike."""
    S = S.copy()
    S[POUR:POUR + 15, WRIST_DIM] += np.linspace(0, 4.0, 15)   # tilt down to pour
    S[POUR + 15:POUR + 60, WRIST_DIM] += 4.0                  # hold while pouring
    S[POUR + 60:POUR + 75, WRIST_DIM] += np.linspace(4.0, 0, 15)  # return upright
    return S


def with_early_spike(S: np.ndarray) -> np.ndarray:
    """A short transport spike just after grasp, no real pour tilt -> must be rejected."""
    S = S.copy()
    S[GRASP + 5:GRASP + 9, WRIST_DIM] += 5.0
    return S


def run(schema, S):
    ctx = EpisodeContext(episode_index=0, n_frames=T, fps=FPS, state=S)
    return OrientationDetector().propose(schema, ctx)


def main() -> int:
    schema = load_schema(SCHEMA_PATH)
    schema.state_layout["tube_wrist_dims"] = [WRIST_DIM]   # activate the detector

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    print("OrientationDetector synthetic tests")

    # 1. detects the sustained tilt, well after grasp, within the guarded window
    res = run(schema, with_tilt(base_state()))
    pour_idx = next(e.index for e in schema.events if e.type == "orientation_change")
    got = res.get(pour_idx)
    check("detects a candidate", got is not None)
    if got is not None:
        check("onset near planted pour", POUR - 5 <= got.frame <= POUR + 25, f"frame={got.frame}")
        check("onset after grasp (gating works)", got.frame > GRASP + 25, f"frame={got.frame}")
        check("picked wrist dim", got.evidence["dim"] == WRIST_DIM, f"dim={got.evidence['dim']}")
        check("upward tilt direction", got.evidence["direction"] == 1)
        check("sigma calibrated in [6,18]", 6.0 <= got.sigma <= 18.0, f"sigma={got.sigma}")

    # 2. abstains on a flat wrist signal
    check("abstains when flat", run(schema, base_state()).get(pour_idx) is None)

    # 3. rejects an early transport spike (no sustained tilt)
    check("rejects early transport spike", run(schema, with_early_spike(base_state())).get(pour_idx) is None)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
