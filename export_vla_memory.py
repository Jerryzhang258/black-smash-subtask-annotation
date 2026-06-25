"""
SIEVE step 3 — Memory-Supervised VLA Training Data Generator.

`VLA remembers why it matters.` Turns the per-episode semantic keyframe memory
(`semantic_memory_<id>/ep<NNN>_memory.json`) into VLA training samples: at sampled
timesteps `t`, the policy is conditioned on the *cumulative* memory of everything
verified so far (entries with frame <= t). This is the doc's MVP-3 (text-prefix
memory); a `hybrid` mode (visual + state-transition + semantic tokens, MVP-4) is
stubbed for later.

Each sample (doc 9.2):
  {episode_index, t, instruction, memory:[{frame,event_type,memory_fact,confidence}],
   text_prefix, action_chunk_ref, mode}

Output: <out>.jsonl  (one sample per line)

Usage:
  python export_vla_memory.py --memory semantic_memory_07 --out vla_memory_07.jsonl
  python export_vla_memory.py --memory semantic_memory_07 --out vla_memory_07.jsonl \
      --stride 30 --horizon 16 --min-confidence 0.5
"""
import os
import re
import json
import glob
import argparse


def sample_timesteps(n_frames, memory, stride):
    """Timesteps to emit a sample at: a regular grid plus each keyframe boundary,
    so every memory transition is represented. Sorted, de-duplicated, in-range."""
    ts = set(range(0, n_frames, max(1, stride)))
    ts.add(n_frames - 1)
    for e in memory:
        f = int(e["frame"])
        ts.add(min(n_frames - 1, f))            # at the event
        ts.add(min(n_frames - 1, f + 1))        # just after it (memory now includes it)
    return sorted(t for t in ts if 0 <= t < n_frames)


def render_text_prefix(task, facts):
    lines = [f"Task:\n{task}\n"]
    if facts:
        lines.append("Task memory:")
        lines.extend(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
        lines.append("")
    else:
        lines.append("Task memory:\n(none yet)\n")
    lines.append("Current instruction:\nContinue the task.")
    return "\n".join(lines)


def make_samples(mem_doc, stride=30, horizon=16, min_confidence=0.0, mode="text-prefix"):
    task = mem_doc.get("task") or "Complete the task."
    n_frames = int(mem_doc["n_frames"])
    memory = sorted(mem_doc.get("memory", []), key=lambda e: e["frame"])

    samples = []
    for t in sample_timesteps(n_frames, memory, stride):
        active = [e for e in memory
                  if e["frame"] <= t and e["semantic"]["event_confidence"] >= min_confidence]
        facts = [e["semantic"]["memory_fact"] for e in active]
        sample = {
            "episode_index": mem_doc.get("episode_index"),
            "t": t,
            "instruction": task,
            "mode": mode,
            "memory": [{
                "frame": e["frame"],
                "event_type": e["event_type"],
                "memory_fact": e["semantic"]["memory_fact"],
                "confidence": e["semantic"]["event_confidence"],
            } for e in active],
            "action_chunk_ref": {"episode_index": mem_doc.get("episode_index"),
                                 "start_t": t, "horizon": horizon},
        }
        if mode == "text-prefix":
            sample["text_prefix"] = render_text_prefix(task, facts)
        elif mode == "hybrid":
            # MVP-4 stub: a real hybrid sample would also emit visual keyframe
            # tokens + state-transition tokens. We carry the references so a model
            # tokenizer can consume them later; tokenization itself is out of scope.
            sample["hybrid_memory"] = [{
                "frame": e["frame"],
                "visual_keyframe": e["visual_keyframe"],
                "state_transition": e["state_transition"]["summary"],
                "memory_fact": e["semantic"]["memory_fact"],
                "confidence": e["semantic"]["event_confidence"],
            } for e in active]
            sample["text_prefix"] = render_text_prefix(task, facts)
        samples.append(sample)
    return samples


def _ep_id(path):
    m = re.search(r"ep(\d+)_memory\.json$", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="semantic_memory_<id> dir (ep<NNN>_memory.json)")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--mode", choices=["text-prefix", "hybrid"], default="text-prefix")
    ap.add_argument("--stride", type=int, default=30, help="emit a sample every N frames")
    ap.add_argument("--horizon", type=int, default=16, help="action_chunk_ref horizon")
    ap.add_argument("--min-confidence", type=float, default=0.0, dest="min_confidence",
                    help="drop memory entries below this event_confidence")
    ap.add_argument("--eps", default="", help="comma list; default = all found")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.memory, "*ep*_memory.json")))
    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None

    n_eps = n_samples = 0
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for fp in files:
            ep = _ep_id(fp)
            if ep is None or (want is not None and ep not in want):
                continue
            mem_doc = json.loads(open(fp, encoding="utf-8").read())
            samples = make_samples(mem_doc, args.stride, args.horizon,
                                   args.min_confidence, args.mode)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_eps += 1
            n_samples += len(samples)
            print(f"ep{ep:03d}  samples={len(samples)}")
    print(f"\n{n_eps} episodes, {n_samples} samples -> {args.out}  (mode={args.mode})")


if __name__ == "__main__":
    main()
