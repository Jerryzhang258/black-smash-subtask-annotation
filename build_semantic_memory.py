"""
SIEVE step 2 — Semantic Keyframe Memory.

`VLM verifies what matters; VLA remembers why it matters.` This script turns the
*verified* critical points (the fused boundaries) into per-keyframe **memory
entries**: not just "a boundary at frame k", but what happened there, why it
matters for the future, and how confident we are. It is the doc's
`fuse_semantic_keyframes.py` step.

Each memory entry (see doc 6.2) carries:
  - frame / event_type            (the verified keyframe)
  - visual_keyframe               (referenced camera paths; decoded only with --dump-frames)
  - state_transition.summary      (before -> after subtask, from the taxonomy)
  - semantic.memory_fact          (what is now true)
  - semantic.future_relevance     (what the policy should do next)
  - semantic.event_confidence     (fused from source / state / VLM confidence)
  - source                        (provenance)

Inputs (all optional except a boundary source):
  --fused      annotations_fused_<id>/   (verified boundaries; preferred)
  --state      annotations_state_<id>/   (fallback boundary source if no fused)
  --candidates candidates_<id>/          (state-proposed windows -> state_confidence)
  --stage-jsonl <run>/stage_annotations_normalized.jsonl  (qwen-stage semantics)

When qwen-stage semantics are absent, `memory_fact` / `future_relevance` fall
back to templates derived from the task taxonomy carried in the boundary doc, so
this runs on the committed examples/ with no model and no raw data.

Output per episode: <out>/ep<NNN>_memory.json

Usage:
  python build_semantic_memory.py --fused annotations_fused_07 \
      --candidates candidates_07 \
      --stage-jsonl annotations_qwen_stage_07/run_xxx/stage_annotations_normalized.jsonl \
      --out semantic_memory_07
"""
import os
import re
import json
import glob
import argparse

# What is true once each event fires (doc-style memory facts).
MEMORY_FACT = {
    "grasp_tube":   "The robot has grasped the test tube.",
    "start_pour":   "Black powder has started pouring from the test tube into the mortar.",
    "release_tube": "The test tube has been released.",
    "grasp_pestle": "The robot has grasped the pestle.",
    "start_grind":  "The robot has started grinding the powder in the mortar.",
    "lift_pestle":  "Grinding is complete and the pestle has been lifted.",
}
P2_CONF = {"high": 0.85, "medium": 0.60, "med": 0.60, "low": 0.40}


def load_jsonl_by_episode(path):
    """episode_index -> normalized qwen-stage record."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["episode_index"])] = rec
    return out


def load_dir_by_episode(dirpath, suffix):
    """episode_index -> parsed json doc, for ep<NNN><suffix> files in a dir."""
    out = {}
    if not dirpath or not os.path.isdir(dirpath):
        return out
    for fp in glob.glob(os.path.join(dirpath, f"*ep*{suffix}")):
        m = re.search(r"ep(\d+)" + re.escape(suffix) + r"$", os.path.basename(fp))
        if m:
            out[int(m.group(1))] = json.loads(open(fp, encoding="utf-8").read())
    return out


def _stage_after(stage_rec, event_index):
    """The stage that begins right after critical point `event_index` (0-based).

    qwen-stage emits 7 stages split by the 6 critical points; stage (i+1) is the
    interval the policy enters once event i has fired — its prediction_prompt is
    the natural `future_relevance`."""
    if not stage_rec:
        return None
    stages = stage_rec.get("normalized_stages") or \
        (stage_rec.get("parsed_response") or {}).get("stages") or []
    idx = event_index + 1
    return stages[idx] if 0 <= idx < len(stages) else None


def _confidence(event_name, source, cand_conf, p2_conf, needs_review):
    """Fuse a single event_confidence in [0, 1] from the available signals."""
    base = cand_conf if cand_conf is not None else (0.90 if source == "state" else 0.60)
    if event_name == "start_pour" and p2_conf is not None:
        # the pour is VLM-owned: weight the visual verifier over the state proxy.
        conf = 0.4 * base + 0.6 * p2_conf
    else:
        conf = base
    if needs_review:
        conf *= 0.70
    return round(max(0.0, min(1.0, conf)), 2)


def build_memory(boundary_doc, candidates=None, stage_rec=None, cameras=("camera0", "camera1")):
    """Pure function: boundary doc (+ optional candidates / qwen-stage) -> memory doc."""
    ep = boundary_doc.get("episode_index")
    cps = boundary_doc["critical_points"]
    names = boundary_doc.get("critical_names", [])
    subtasks = boundary_doc.get("subtasks", [])
    sources = boundary_doc.get("sources") or ["state"] * len(cps)
    review_points = set(boundary_doc.get("review_points", []) or [])

    cand_by_name = {}
    if candidates:
        for w in candidates.get("candidate_windows", []):
            cand_by_name[w["event_hint"]] = w

    parsed = (stage_rec or {}).get("parsed_response") or {}
    p2_conf_raw = str(parsed.get("p2_confidence", "")).lower()
    p2_conf = P2_CONF.get(p2_conf_raw)

    memory = []
    for i, frame in enumerate(cps):
        name = names[i] if i < len(names) else f"event_{i}"
        before = subtasks[i]["label"] if i < len(subtasks) else ""
        after = subtasks[i + 1]["label"] if (i + 1) < len(subtasks) else ""

        stage = _stage_after(stage_rec, i)
        future = (stage.get("prediction_prompt") if stage else None) or \
            (f"Next, {after}." if after else "Continue the task.")
        evidence = (stage.get("reason") if stage else "") or ""
        if name == "start_pour":
            evidence = parsed.get("p2_reason") or evidence

        cand = cand_by_name.get(name)
        cand_conf = cand["state_confidence"] if cand else None
        needs_review = (i + 1) in review_points
        conf = _confidence(name, sources[i], cand_conf, p2_conf, needs_review)

        annotator = boundary_doc.get("annotator", "state")
        src = "state_proposed_vlm_verified" if "fused" in annotator \
            else f"{annotator}_only"

        entry = {
            "memory_id": i,
            "frame": int(frame),
            "event_type": name,
            "visual_keyframe": {
                cam: f"ep{ep:03d}/{cam}_t{int(frame):04d}.jpg" for cam in cameras
            },
            "state_transition": {
                "summary": f"{before} -> {after}" if (before or after) else "",
                "source": sources[i],
            },
            "semantic": {
                "memory_fact": MEMORY_FACT.get(name, f"Event '{name}' has occurred."),
                "future_relevance": future,
                "evidence": evidence,
                "event_confidence": conf,
            },
            "needs_review": needs_review,
            "source": src,
        }
        if cand is not None:
            entry["candidate_window"] = {"lo": cand["lo"], "hi": cand["hi"],
                                         "state_confidence": cand["state_confidence"]}
        memory.append(entry)

    return {
        "episode_index": ep,
        "task": boundary_doc.get("task"),
        "n_frames": int(boundary_doc["n_frames"]),
        "fps": int(boundary_doc.get("fps", 30)),
        "boundary_source": boundary_doc.get("annotator", "state"),
        "has_semantics": stage_rec is not None,
        "memory": memory,
    }


def maybe_dump_frames(doc, data_chunk, out_dir, cameras):
    """Decode and save the referenced keyframes (opt-in; needs pandas + PIL + parquet)."""
    import io
    import pandas as pd
    from PIL import Image
    ep = doc["episode_index"]
    parquet = os.path.join(data_chunk, f"episode_{ep:06d}.parquet")
    cols = [f"observation.images.{c}" for c in cameras]
    df = pd.read_parquet(parquet, columns=cols)
    frame_dir = os.path.join(out_dir, f"ep{ep:03d}")
    os.makedirs(frame_dir, exist_ok=True)
    for entry in doc["memory"]:
        t = entry["frame"]
        for cam in cameras:
            col = f"observation.images.{cam}"
            img = Image.open(io.BytesIO(df[col].iloc[t]["bytes"])).convert("RGB")
            img.thumbnail((640, 640))
            rel = f"ep{ep:03d}/{cam}_t{int(t):04d}.jpg"
            img.save(os.path.join(out_dir, rel), quality=90)
            entry["visual_keyframe"][cam] = rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="", help="fused (verified) annotation dir; preferred boundary source")
    ap.add_argument("--state", default="", help="state annotation dir; fallback boundary source")
    ap.add_argument("--candidates", default="", help="candidate-window dir from candidate_propose.py")
    ap.add_argument("--stage-jsonl", default="", dest="stage_jsonl",
                    help="qwen-stage stage_annotations_normalized.jsonl (semantic text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="", help="comma list; default = all found")
    ap.add_argument("--cameras", default="camera0,camera1")
    ap.add_argument("--dump-frames", action="store_true", help="decode keyframes (needs --data)")
    ap.add_argument("--data", default="", help="data/chunk-000 dir, for --dump-frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())

    boundary_dir = args.fused or args.state
    if not boundary_dir:
        ap.error("need --fused or --state")
    boundary_docs = load_dir_by_episode(boundary_dir, "_subtasks.json")
    cand_docs = load_dir_by_episode(args.candidates, "_candidates.json")
    stage_recs = load_jsonl_by_episode(args.stage_jsonl)

    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None
    n = 0
    for ep in sorted(boundary_docs):
        if want is not None and ep not in want:
            continue
        doc = build_memory(boundary_docs[ep], cand_docs.get(ep), stage_recs.get(ep), cameras)
        if args.dump_frames:
            if not args.data:
                ap.error("--dump-frames needs --data")
            maybe_dump_frames(doc, args.data, args.out, cameras)
        json.dump(doc, open(os.path.join(args.out, f"ep{ep:03d}_memory.json"), "w"),
                  indent=2, ensure_ascii=False)
        n += 1
        confs = [e["semantic"]["event_confidence"] for e in doc["memory"]]
        sem = "qwen-stage" if doc["has_semantics"] else "templated"
        print(f"ep{ep:03d}  entries={len(doc['memory'])}  semantics={sem}  conf={confs}")
    print(f"\n{n} episodes -> {args.out}")
    if not stage_recs:
        print("(no --stage-jsonl — memory_fact/future_relevance from taxonomy templates)")


if __name__ == "__main__":
    main()
