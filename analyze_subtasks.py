"""
Signal-based subtask boundary analysis for one LeRobot episode.
Uses observation.state / actions (20-dim bimanual) to find:
  - gripper open/close (grasp / release) events
  - motion vs. dwell
  - oscillatory 'grind' phase
Prints a per-second timeline + auto-detected event list. numpy/pandas only.
"""
import sys, numpy as np, pandas as pd

PARQUET = sys.argv[1] if len(sys.argv) > 1 else r"C:\Intern\black_smash_07\data\chunk-000\episode_000000.parquet"
FPS = 30

df = pd.read_parquet(PARQUET, columns=["observation.state", "actions", "timestamp"])
S = np.stack([np.asarray(x, dtype=np.float64) for x in df["observation.state"].values])
A = np.stack([np.asarray(x, dtype=np.float64) for x in df["actions"].values])
T, D = S.shape
print(f"FILE {PARQUET}")
print(f"T={T} frames  D={D}  dur={T/FPS:.2f}s")

# --- identify gripper-like dims: strongly bimodal (time spent near the two extremes) ---
def bimodality(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-9:
        return 0.0, 0.0
    xn = (x - lo) / (hi - lo)
    mid = np.mean((xn > 0.35) & (xn < 0.65))   # low => bimodal
    extr = np.mean((xn < 0.15) | (xn > 0.85))  # high => bimodal
    return extr, mid

print("\n[dim bimodality]  extr(hi=gripper-like) mid(lo=gripper-like)  range")
scores = []
for d in range(D):
    extr, mid = bimodality(S[:, d])
    scores.append(extr - mid)
    flag = "  <== gripper?" if (extr > 0.8 and mid < 0.15) else ""
    print(f"  s{d:02d} extr={extr:.2f} mid={mid:.2f} range=[{S[:,d].min():7.3f},{S[:,d].max():7.3f}]{flag}")

grip_dims = [d for d in range(D) if bimodality(S[:, d])[0] > 0.8 and bimodality(S[:, d])[1] < 0.15]
print("\nAuto gripper dims:", grip_dims)

# fall back to convention 10 dims/arm -> gripper at idx 9 and 19
if len(grip_dims) < 2:
    grip_dims = [9, 19]
    print("fallback gripper dims:", grip_dims)

# normalize gripper signals 0..1 (0=closed assumption resolved later)
def norm01(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

gripL = norm01(S[:, grip_dims[0]])
gripR = norm01(S[:, grip_dims[1]]) if len(grip_dims) > 1 else gripL

# --- position dims = everything except gripper dims; speed = ||d/dt|| ---
pos_dims = [d for d in range(D) if d not in grip_dims]
P = S[:, pos_dims]
P = (P - P.mean(0)) / (P.std(0) + 1e-9)         # standardize so dims comparable
vel = np.linalg.norm(np.diff(P, axis=0, prepend=P[:1]), axis=1)
# smooth speed (moving avg ~0.3s)
w = 9
kern = np.ones(w) / w
speed = np.convolve(vel, kern, mode="same")

# --- oscillation score: zero-crossings of per-dim velocity in a sliding window ---
def osc_score(P, win=45):
    dv = np.diff(P, axis=0, prepend=P[:1])
    sign_changes = (np.diff(np.sign(dv), axis=0, prepend=np.sign(dv[:1])) != 0)
    sc = sign_changes.sum(axis=1).astype(float)   # how many dims reversed direction this frame
    return np.convolve(sc, np.ones(win) / win, mode="same")
osc = osc_score(P)

# --- per-second timeline ---
print("\n=== per-second timeline (t | frame | gripL gripR | speed | osc) ===")
print("    gripL/gripR: 1.0=open-ish 0.0=closed-ish (relative)")
for sec in range(0, T // FPS + 1):
    i = min(sec * FPS, T - 1)
    bar = "#" * int(speed[i] / (speed.max() + 1e-9) * 20)
    ob = "*" * int(osc[i] / (osc.max() + 1e-9) * 12)
    print(f"  t={sec:4d}s f={i:4d}  gL={gripL[i]:.2f} gR={gripR[i]:.2f}  spd={speed[i]:5.2f} |{bar:<20}| osc={osc[i]:4.1f} |{ob}")

# --- detect gripper transition events (open<->close) ---
def transitions(g, name, hi=0.6, lo=0.4):
    state = g[0] > 0.5
    evs = []
    for i in range(1, len(g)):
        if state and g[i] < lo:
            evs.append((i, f"{name} CLOSE")); state = False
        elif (not state) and g[i] > hi:
            evs.append((i, f"{name} OPEN")); state = True
    return evs

events = transitions(gripL, "L") + transitions(gripR, "R")
events.sort()
print("\n=== gripper transition events ===")
for i, lab in events:
    print(f"  f={i:4d} t={i/FPS:5.2f}s  {lab}")

# --- detect sustained grinding window (high osc & moderate sustained speed) ---
oth = np.percentile(osc, 75)
grind_mask = osc > max(oth, osc.mean() + 0.5 * osc.std())
# longest contiguous run
runs, s = [], None
for i, m in enumerate(grind_mask):
    if m and s is None: s = i
    elif not m and s is not None: runs.append((s, i)); s = None
if s is not None: runs.append((s, len(grind_mask)))
runs = [r for r in runs if r[1] - r[0] > FPS]  # >1s
print("\n=== candidate sustained-oscillation (grind) windows ===")
for a, b in sorted(runs, key=lambda r: r[1]-r[0], reverse=True)[:5]:
    print(f"  f=[{a:4d},{b:4d}]  t=[{a/FPS:5.2f},{b/FPS:5.2f}]s  dur={ (b-a)/FPS:4.1f}s  meanOsc={osc[a:b].mean():.1f}")
print("DONE")
