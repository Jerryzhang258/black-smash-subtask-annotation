"""Consistency / outlier detection across all annotated episodes (QA method 3).
Since every episode is the same task, the segmentation 'shape' should be consistent.
Features are scale-invariant (boundary positions & subtask durations as fractions of
episode length). Flags episodes whose shape deviates via a robust modified z-score
(median + MAD). Prints per-feature spread + a ranked list of the most anomalous episodes."""
import glob, json, os, argparse, numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--ann", default=r"C:\Intern\mvt_annotations")
_ap.add_argument("--thresh", type=float, default=3.5)
_args, _ = _ap.parse_known_args()
ANN = _args.ann
THRESH = _args.thresh   # |modified z| above this = outlier on that feature

docs = []
for fp in sorted(glob.glob(os.path.join(ANN, "ep*_subtasks.json"))):
    d = json.load(open(fp))
    N = d["n_frames"]; st = d["subtask_starts"]               # [0, p1..p5]
    durs = [s["n_frames"] / N for s in d["subtasks"]]          # S0.. as fraction of N
    feat = {f"p{i}/N": st[i] / N for i in range(1, len(st))}   # critical-point positions
    feat.update({f"S{i}%": durs[i] for i in range(len(durs))})  # subtask duration fractions
    docs.append({"ep": d["episode_index"], "N": N, "flags": d.get("flags", []), "feat": feat})

names = list(docs[0]["feat"].keys())
X = np.array([[e["feat"][n] for n in names] for e in docs])   # (E, F)
med = np.median(X, 0)
mad = np.median(np.abs(X - med), 0)
mad_eff = np.where(mad < 1e-9, X.std(0) + 1e-9, mad)
modz = 0.6745 * (X - med) / mad_eff                            # robust z

print(f"episodes analyzed: {len(docs)}    outlier threshold |z|>{THRESH}\n")
print("per-feature spread (median  MAD   [min, max]):")
for j, n in enumerate(names):
    print(f"  {n:6s}  med={med[j]:.3f}  mad={mad[j]:.3f}  range=[{X[:,j].min():.3f}, {X[:,j].max():.3f}]")

score = np.abs(modz).max(1)
order = np.argsort(-score)
print("\nmost anomalous episodes (ranked):")
print("  ep    score   deviating features (z)")
flagged = []
for i in order:
    e = docs[i]
    devs = [(names[j], modz[i, j]) for j in range(len(names)) if abs(modz[i, j]) > THRESH]
    tag = "  <-- OUTLIER" if devs else ""
    if devs:
        flagged.append(e["ep"])
    devstr = ", ".join(f"{n}={z:+.1f}" for n, z in sorted(devs, key=lambda t: -abs(t[1]))) or "-"
    if score[i] > 2.0 or devs:
        print(f"  {e['ep']:3d}   {score[i]:5.1f}   {devstr}{tag}")

print(f"\nOUTLIERS (|z|>{THRESH}): {flagged if flagged else 'none'}")
print("episodes with pipeline flags:", [e["ep"] for e in docs if e["flags"]] or "none")
# emit the outlier list on its own line for easy copy into --eps
print("EPS=" + ",".join(str(x) for x in flagged))
