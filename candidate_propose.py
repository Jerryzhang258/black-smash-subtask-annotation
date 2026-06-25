"""
SIEVE step 1 — State-Proposed Candidate Windows.

`State proposes where to look.` Instead of treating the proprioceptive critical
points as final answers, we turn each one into a *high-recall candidate window*
that a VLM later verifies (see `build_semantic_memory.py`). A window is wide and
honest about its uncertainty: crisp gripper events get a tight, high-confidence
window; the visual `start_pour` event gets a wide, low-confidence one (the state
signal only has a proxy there), exactly the prior the legacy `OWNER` table and
the framework's per-detector `sigma` already encode.

This script consumes the state annotation directory the pipeline already emits
(`annotations_state_<id>/ep<NNN>_subtasks.json`) — it does NOT need raw
`observation.state`, so it runs anywhere the JSON exists. Per-event window width
and state confidence come from the task schema if present
(`data_annotation/framework/schemas/black_smash.json`, optional `window_half_s` /
`state_confidence` per event), otherwise from event-type defaults below.

Output per episode: <out>/ep<NNN>_candidates.json
Dataset output:     <out>/all_candidates.jsonl

Usage:
  python candidate_propose.py --state annotations_state_07 --out candidates_07
  python candidate_propose.py --state annotations_state_07 --out candidates_07 \
      --schema data_annotation/framework/schemas/black_smash.json
"""
import os
import re
import json
import glob
import argparse

# Event-type priors (window half-width in seconds, base state confidence).
# These mirror the sigma/OWNER prior already in the repo: gripper events are
# crisp in proprioception; orientation_change (start_pour) is a visual event the
# state signal can only bracket; motion_regime sits in between.
TYPE_DEFAULTS = {
    "gripper_close":      {"window_half_s": 0.30, "state_confidence": 0.95},
    "gripper_open":       {"window_half_s": 0.30, "state_confidence": 0.93},
    "orientation_change": {"window_half_s": 1.50, "state_confidence": 0.45},
    "motion_regime":      {"window_half_s": 1.00, "state_confidence": 0.70},
}
FALLBACK_DEFAULT = {"window_half_s": 1.00, "state_confidence": 0.50}

# Human-readable state source per event name (falls back to the schema type).
STATE_SOURCE = {
    "grasp_tube":    "tube_gripper_close",
    "release_tube":  "tube_gripper_open",
    "grasp_pestle":  "pestle_gripper_close",
    "start_pour":    "low_drift_while_tube_held",
    "start_grind":   "in_place_motion_onset",
    "lift_pestle":   "in_place_motion_end",
}

# Which events a state flag casts doubt on (substring match against the flag text).
FLAG_AFFECTS = {
    "tube-held":   {"grasp_tube", "start_pour", "release_tube"},
    "pestle-held": {"grasp_pestle", "lift_pestle"},
    "grind":       {"start_grind", "lift_pestle"},
    "ordering":    {"grasp_tube", "start_pour", "release_tube",
                    "grasp_pestle", "start_grind", "lift_pestle"},
}
FLAG_PENALTY = 0.5   # multiply a candidate's confidence when its region is flagged


def load_event_config(schema_path):
    """name -> {type, window_half_s, state_confidence} from the task schema.

    Reads the schema JSON directly (no framework import) so this stays a thin,
    dependency-free script. Per-event `window_half_s` / `state_confidence` are
    optional overrides; otherwise the event `type` default is used."""
    cfg = {}
    if not schema_path or not os.path.exists(schema_path):
        return cfg
    raw = json.loads(open(schema_path, encoding="utf-8").read())
    for ev in raw.get("events", []):
        name, etype = ev.get("name"), ev.get("type")
        base = dict(TYPE_DEFAULTS.get(etype, FALLBACK_DEFAULT))
        if "window_half_s" in ev:
            base["window_half_s"] = float(ev["window_half_s"])
        if "state_confidence" in ev:
            base["state_confidence"] = float(ev["state_confidence"])
        base["type"] = etype
        cfg[name] = base
    return cfg


def _affected_by_flags(event_name, flags):
    for key, names in FLAG_AFFECTS.items():
        if event_name in names and any(key in str(f).lower() for f in flags):
            return True
    return False


def propose_windows(state_doc, event_cfg=None):
    """Turn one state annotation doc into a list of candidate-window dicts.

    Pure function (no I/O) so it can be unit-tested without raw data."""
    event_cfg = event_cfg or {}
    T = int(state_doc["n_frames"])
    fps = int(state_doc.get("fps", 30))
    cps = state_doc["critical_points"]
    names = state_doc.get("critical_names", [])
    flags = state_doc.get("flags", []) or []

    windows = []
    for i, center in enumerate(cps):
        name = names[i] if i < len(names) else f"event_{i}"
        cfg = event_cfg.get(name)
        if cfg is None:
            cfg = dict(FALLBACK_DEFAULT)
            cfg["type"] = None
        half = int(round(cfg["window_half_s"] * fps))
        conf = float(cfg["state_confidence"])
        if _affected_by_flags(name, flags):
            conf = round(conf * FLAG_PENALTY, 3)
        lo = max(0, int(center) - half)
        hi = min(T - 1, int(center) + half)
        windows.append({
            "candidate_id": i,
            "event_hint": name,
            "center": int(center),
            "lo": lo,
            "hi": hi,
            "state_source": STATE_SOURCE.get(name, cfg.get("type") or "state"),
            "state_confidence": conf,
            "flagged": _affected_by_flags(name, flags),
        })
    return windows


def episode_doc(state_doc, event_cfg=None):
    return {
        "episode_index": state_doc.get("episode_index"),
        "task": state_doc.get("task"),
        "n_frames": int(state_doc["n_frames"]),
        "fps": int(state_doc.get("fps", 30)),
        "source": "state_proposed",
        "candidate_windows": propose_windows(state_doc, event_cfg),
    }


def _ep_id(path):
    m = re.search(r"ep(\d+)_subtasks\.json$", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="state annotation dir (ep<NNN>_subtasks.json)")
    ap.add_argument("--out", required=True, help="output dir for ep<NNN>_candidates.json")
    ap.add_argument("--schema", default="data_annotation/framework/schemas/black_smash.json")
    ap.add_argument("--eps", default="", help="comma list of episode indices; default = all found")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    event_cfg = load_event_config(args.schema)
    files = sorted(glob.glob(os.path.join(args.state, "*ep*_subtasks.json")))
    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None

    jl = open(os.path.join(args.out, "all_candidates.jsonl"), "w", encoding="utf-8")
    n = 0
    for fp in files:
        ep = _ep_id(fp)
        if ep is None or (want is not None and ep not in want):
            continue
        state_doc = json.loads(open(fp, encoding="utf-8").read())
        doc = episode_doc(state_doc, event_cfg)
        json.dump(doc, open(os.path.join(args.out, f"ep{ep:03d}_candidates.json"), "w"), indent=2)
        jl.write(json.dumps(doc) + "\n")
        n += 1
        confs = [round(w["state_confidence"], 2) for w in doc["candidate_windows"]]
        spans = [w["hi"] - w["lo"] for w in doc["candidate_windows"]]
        print(f"ep{ep:03d}  windows={len(doc['candidate_windows'])}  conf={confs}  span_frames={spans}")
    jl.close()
    print(f"\n{n} episodes -> {args.out}")
    if not event_cfg:
        print("(no schema found — used event-type defaults)")


if __name__ == "__main__":
    main()
