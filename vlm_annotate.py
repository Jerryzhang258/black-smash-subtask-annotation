"""
Stage 1 (VLM) subtask annotation — local Qwen2.5-VL.

First pass of the 3-stage pipeline: a vision model watches enhanced camera frames
and proposes the 6 critical points (same schema as the state stage). It is the
SEMANTIC view — it owns the visually-obvious events (esp. p2 "start pour") and
cross-checks the rest. Precise gripper timing is the state stage's job; here we
accept coarser timing, then optionally refine it coarse->fine.

Pipeline per episode:
  1. coarse: sample N frames evenly over the episode, enhance contrast, ask the
     model for the 6 critical-point frame indices (JSON).
  2. fine (optional, --fine): for each coarse point, sample frames densely in a
     +/- window and ask the model to pin the exact frame of that one event.
  3. validate ordering, write <out>/ep<NNN>_subtasks.json (+ _subtask_index.npy),
     annotator="qwen-vl".

Backends: qwen-local (transformers). The frame-sampling / prompt logic is backend
independent; --dry-run exercises everything except the model (writes a debug
montage + the prompt) so it can be checked on a machine with no GPU.

Setup on the 5080 desktop: see docs/INSTALL_QWEN.md.

Usage:
  python vlm_annotate.py --model D:\models\Qwen2.5-VL-7B-Instruct-AWQ --eps 0,1,2
  python vlm_annotate.py --dry-run --eps 0          # no model; check framing/prompt
"""
import os, io, json, glob, argparse, re
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageEnhance

from batch_annotate import LABELS, CRIT_NAMES   # single source of truth for taxonomy

CRIT_DESC = [   # human-facing description of each critical point, fed to the model
    "p1 grasp_tube  - the moment the gripper first grabs the test tube",
    "p2 start_pour  - the moment black powder starts pouring out of the tube into the mortar",
    "p3 release_tube - the moment the tube is set down and the gripper lets go of it",
    "p4 grasp_pestle - the moment the other gripper grabs the pestle",
    "p5 start_grind - the moment the pestle starts grinding (in-place) inside the mortar",
    "p6 lift_pestle - the moment grinding stops and the pestle is lifted up",
]
TASK_DEFAULT = "Pour the black powder into the mortar and grind."


# ---------------- frame I/O ----------------
def enh(im):
    a = np.asarray(im).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    o = Image.fromarray(a.astype(np.uint8))
    return ImageEnhance.Contrast(ImageEnhance.Color(o).enhance(1.4)).enhance(1.2)


def label_frame(im, idx, t, size):
    im = enh(im).resize((size, size))
    d = ImageDraw.Draw(im)
    tag = f"#{idx} {t:.1f}s"
    d.rectangle([0, 0, 8 * len(tag) + 6, 16], fill=(0, 0, 0))
    d.text((3, 2), tag, fill=(255, 220, 0))
    return im


class FrameStore:
    """Decode-on-demand access to one camera stream of one episode."""
    def __init__(self, parquet, cam):
        self.col = pd.read_parquet(parquet, columns=[cam])[cam].values
        self.n = len(self.col)

    def get(self, idx):
        return Image.open(io.BytesIO(self.col[int(idx)]["bytes"])).convert("RGB")


def sample_idxs(lo, hi, n):
    return [int(round(x)) for x in np.linspace(lo, hi, min(n, hi - lo + 1))]


# ---------------- prompt building ----------------
def build_messages(task, store, idxs, fps, size, head):
    items = [{"type": "text", "text": head}]
    for i in idxs:
        items.append({"type": "image", "image": label_frame(store.get(i), i, i / fps, size)})
    return [{"role": "user", "content": items}]


COARSE_HEAD = (
    "You are labeling a bimanual robot manipulation episode.\n"
    "Task: \"{task}\"\n"
    "The full episode has {T} frames at {fps} fps. Below are {k} frames sampled "
    "evenly across the episode; each is captioned with its frame index (#idx) and "
    "time. Images are low-light/fisheye and contrast-enhanced.\n\n"
    "Find these 6 critical points (transition frames). Each must be an integer "
    "frame index in [0, {T1}], and they MUST be strictly increasing:\n  "
    + "\n  ".join(CRIT_DESC) +
    "\n\nReturn ONLY a JSON object, no prose:\n"
    '{{"p1_grasp_tube": int, "p2_start_pour": int, "p3_release_tube": int, '
    '"p4_grasp_pestle": int, "p5_start_grind": int, "p6_lift_pestle": int, '
    '"notes": "one short sentence"}}'
)

FINE_HEAD = (
    "Same bimanual robot episode, task: \"{task}\".\n"
    "Below are densely sampled frames (captioned with frame index) around one event:\n"
    "  {desc}\n"
    "Pick the SINGLE frame index where this event happens. The answer must be one "
    "of the shown indices, between {lo} and {hi}.\n"
    'Return ONLY JSON: {{"frame": int}}'
)


# ---------------- parsing ----------------
def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_coarse(text, T):
    obj = extract_json(text) or {}
    keys = ["p1_grasp_tube", "p2_start_pour", "p3_release_tube",
            "p4_grasp_pestle", "p5_start_grind", "p6_lift_pestle"]
    try:
        cps = [int(round(float(obj[k]))) for k in keys]
    except Exception:
        return None, obj.get("notes", "")
    return cps, obj.get("notes", "")


def parse_fine(text, default):
    obj = extract_json(text) or {}
    try:
        return int(round(float(obj["frame"])))
    except Exception:
        return default


def enforce_order(cps, T, flags):
    """Clamp into [1, T-2] and force strictly increasing by nudging."""
    cps = [max(1, min(T - 2, c)) for c in cps]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
            flags.append("nudged p%d for ordering" % (i + 1))
    if cps[-1] >= T - 1:
        flags.append("points hit end of episode")
    return cps


# ---------------- doc builder ----------------
def subtasks_from_cps(cps, T, fps):
    starts = [0] + list(cps)
    n = len(LABELS)
    out = []
    for i in range(n):
        a = starts[i]
        b = (starts[i + 1] - 1) if i < n - 1 else T - 1
        out.append({"subtask_id": i, "label": LABELS[i], "start_frame": a, "end_frame": b,
                    "start_t": round(a / fps, 2), "end_t": round(b / fps, 2),
                    "n_frames": b - a + 1, "dur_s": round((b - a + 1) / fps, 2)})
    return out


# ---------------- backend ----------------
class QwenLocal:
    def __init__(self, model_id, max_new_tokens=256):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        try:
            from qwen_vl_utils import process_vision_info
        except Exception:
            process_vision_info = None
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = max_new_tokens
        print(f"[load] {model_id} ...", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", attn_implementation="sdpa")
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("[load] done", flush=True)

    def ask(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.process_vision_info is not None:
            imgs, vids = self.process_vision_info(messages)
        else:
            imgs = [c["image"] for m in messages for c in m["content"] if c.get("type") == "image"]
            vids = None
        inputs = self.processor(text=[text], images=imgs, videos=vids,
                                padding=True, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]


# ---------------- per-episode ----------------
def annotate_one(backend, parquet, out_dir, ep, task, args):
    store = FrameStore(parquet, args.cam)
    T, fps = store.n, args.fps
    flags = []

    # coarse
    idxs = sample_idxs(0, T - 1, args.n_frames)
    head = COARSE_HEAD.format(task=task, T=T, T1=T - 1, fps=fps, k=len(idxs))
    if args.dry_run:
        montage(store, idxs, fps, os.path.join(out_dir, f"ep{ep:03d}_coarse_montage.png"))
        open(os.path.join(out_dir, f"ep{ep:03d}_prompt.txt"), "w", encoding="utf-8").write(head)
        print(f"ep{ep:03d}: dry-run wrote montage + prompt ({len(idxs)} frames)")
        return None
    cps, notes = parse_coarse(backend.ask(build_messages(task, store, idxs, fps, args.size, head)), T)
    if cps is None:
        flags.append("coarse parse failed -> fallback proportions")
        cps = [int(p * T) for p in (0.14, 0.22, 0.34, 0.55, 0.62, 0.90)]
    cps = enforce_order(cps, T, flags)

    # fine
    if args.fine:
        w = int(args.fine_window_s * fps)
        for i in range(6):
            lo, hi = max(0, cps[i] - w), min(T - 1, cps[i] + w)
            fidx = sample_idxs(lo, hi, args.fine_frames)
            fhead = FINE_HEAD.format(task=task, desc=CRIT_DESC[i], lo=lo, hi=hi)
            cps[i] = parse_fine(backend.ask(build_messages(task, store, fidx, fps, args.size, fhead)), cps[i])
        cps = enforce_order(cps, T, flags)

    doc = {"episode_index": ep, "task": task, "n_frames": T, "fps": fps,
           "annotator": "qwen-vl", "model": args.model,
           "method": "qwen2.5-vl coarse(%d)%s, cam=%s" % (
               args.n_frames, "+fine" if args.fine else "", args.cam),
           "notes": notes, "critical_points": cps, "critical_names": CRIT_NAMES,
           "subtask_starts": [0] + cps, "flags": flags,
           "n_subtasks": len(LABELS), "subtasks": subtasks_from_cps(cps, T, fps)}
    json.dump(doc, open(os.path.join(out_dir, f"ep{ep:03d}_subtasks.json"), "w"), indent=2)
    idx = np.zeros(T, dtype=np.int16)
    for s in doc["subtasks"]:
        idx[s["start_frame"]:s["end_frame"] + 1] = s["subtask_id"]
    np.save(os.path.join(out_dir, f"ep{ep:03d}_subtask_index.npy"), idx)
    tag = "  FLAG" if flags else ""
    print(f"ep{ep:03d}  cps={cps}{tag}  notes={notes[:60]}")
    return doc


def montage(store, idxs, fps, out_png, cols=8, size=200):
    rows = (len(idxs) + cols - 1) // cols
    cv = Image.new("RGB", (cols * size, rows * size), (12, 12, 12))
    for k, i in enumerate(idxs):
        cv.paste(label_frame(store.get(i), i, i / fps, size), ((k % cols) * size, (k // cols) * size))
    cv.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen-local", choices=["qwen-local"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    ap.add_argument("--data", default=r"C:\Intern\black_smash_07\data\chunk-000")
    ap.add_argument("--out",  default=r"C:\Intern\mvt_annotations_vlm")
    ap.add_argument("--meta", default=r"C:\Intern\black_smash_07\meta\tasks.jsonl")
    ap.add_argument("--fps",  type=int, default=30)
    ap.add_argument("--eps",  default="", help="comma list; default = all")
    ap.add_argument("--cam",  default="observation.images.camera1")
    ap.add_argument("--n-frames", type=int, default=32, dest="n_frames")
    ap.add_argument("--size", type=int, default=256, help="per-frame px sent to the model")
    ap.add_argument("--fine", dest="fine", action="store_true", default=True)
    ap.add_argument("--no-fine", dest="fine", action="store_false")
    ap.add_argument("--fine-window-s", type=float, default=1.5, dest="fine_window_s")
    ap.add_argument("--fine-frames", type=int, default=16, dest="fine_frames")
    ap.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    task = TASK_DEFAULT
    try:
        task = json.loads(open(args.meta).readline())["task"]
    except Exception:
        pass

    backend = None if args.dry_run else QwenLocal(args.model, args.max_new_tokens)

    files = sorted(glob.glob(os.path.join(args.data, "episode_*.parquet")))
    want = set(int(x) for x in args.eps.split(",") if x.strip().isdigit()) if args.eps else None
    done = 0
    for fp in files:
        ep = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if want is not None and ep not in want:
            continue
        annotate_one(backend, fp, args.out, ep, task, args)
        done += 1
    print(f"\n{done} episodes -> {args.out}")


if __name__ == "__main__":
    main()
