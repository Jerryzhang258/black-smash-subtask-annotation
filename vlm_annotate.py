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
CROP = 0.6   # center-crop fraction: drop the useless distorted fisheye periphery (set by --crop)


def enh(im):
    """Make the dark/fisheye/golden frames legible: center-crop to the action,
    gray-world white-balance (kills the orange cast), percentile stretch + gamma."""
    if CROP < 1.0:
        w, h = im.size; cw, ch = int(w * CROP), int(h * CROP)
        im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    a = np.asarray(im).astype(np.float32)
    m = a.reshape(-1, 3).mean(0) + 1e-6
    a = np.clip(a * (m.mean() / m), 0, 255)            # gray-world white balance
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    a = (a ** 0.8) * 255                                # gamma lift
    return ImageEnhance.Contrast(Image.fromarray(a.astype(np.uint8))).enhance(1.25)


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
    "\n\nThink step by step: FIRST, for each shown frame write one short line "
    "'#idx: <what the robot/tube/pestle is doing>'. Use that to locate the 6 events. "
    "THEN, on the LAST line, output ONLY the JSON object (nothing after it):\n"
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

P2_HISTORY_HEAD = (
    "Same bimanual robot episode, task: \"{task}\".\n"
    "We are refining ONLY p2_start_pour.\n\n"
    "The proprioceptive state signal already narrowed the search to frames {lo}-{hi}. "
    "Below are chronological frames from that local window, each captioned with its "
    "frame index (#idx) and time. Use before/after evidence to pick the first frame "
    "where black powder visibly starts leaving the test tube or newly appears in the "
    "mortar.\n\n"
    "Do not choose the first frame where the tube is merely above the mortar, a "
    "pre-pour tilt with no visible powder, the middle of the pour, or the tube "
    "release. Frames just before p2 should still show no visible powder flow; frames "
    "just after p2 should show visible flow or more powder in the mortar. If powder "
    "is too hard to see, choose the earliest sustained pouring tilt over the mortar "
    "and mark confidence low.\n\n"
    "Return ONLY JSON:\n"
    '{{"frame": int, "confidence": "high|medium|low", "reason": "short before/after evidence"}}'
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


def load_state_cps(state_ref, ep):
    if not state_ref:
        return None
    path = os.path.join(state_ref, f"ep{ep:03d}_subtasks.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        cps = doc.get("critical_points")
        return [int(x) for x in cps] if isinstance(cps, list) and len(cps) == 6 else None
    except Exception:
        return None


def refine_p2_with_history(backend, store, task, cps, state_cps, fps, args, flags):
    T = store.n
    state_p1, state_p2, state_p3 = state_cps[0], state_cps[1], state_cps[2]
    w = int(args.p2_history_window_s * fps)
    lo = max(0, state_p1 + 1, state_p2 - w)
    hi = min(T - 1, state_p3 - 1, state_p2 + w)
    if hi <= lo:
        flags.append("p2 history refine skipped: invalid state window")
        return cps[1], None

    idxs = sample_idxs(lo, hi, args.p2_history_frames)
    idxs = sorted(set(idxs + [max(lo, min(hi, state_p2)), max(lo, min(hi, cps[1]))]))
    head = P2_HISTORY_HEAD.format(task=task, lo=lo, hi=hi)
    raw = backend.ask(build_messages(task, store, idxs, fps, args.size, head))
    obj = extract_json(raw) or {}
    frame = parse_fine(raw, cps[1])
    frame = max(lo, min(hi, frame))
    meta = {
        "old_frame": int(cps[1]),
        "new_frame": int(frame),
        "state_frame": int(state_p2),
        "window": [int(lo), int(hi)],
        "sampled_frames": [int(x) for x in idxs],
        "confidence": obj.get("confidence"),
        "reason": obj.get("reason", ""),
    }
    if os.environ.get("VLM_DEBUG"):
        print("=== RAW p2 history response ===\n%s\n=== end ===" % raw, flush=True)
    return frame, meta


def enforce_order(cps, T, flags):
    """Clamp into [1, T-2] and force strictly increasing (both directions so a
    forward nudge can't push points past the end of the episode)."""
    cps = [max(1, min(T - 2, int(c))) for c in cps]
    for i in range(1, len(cps)):                 # forward: strictly increasing
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
            flags.append("nudged p%d for ordering" % (i + 1))
    if cps[-1] > T - 2:                           # overflowed end -> pull down backward
        cps[-1] = T - 2
        for i in range(len(cps) - 2, -1, -1):
            if cps[i] >= cps[i + 1]:
                cps[i] = cps[i + 1] - 1
        if cps[0] < 1:
            flags.append("could not fit ordered points in episode")
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


def _to_openai(messages):
    """Convert the Qwen-native messages (PIL images) to OpenAI chat format (base64)."""
    import base64
    out = []
    for m in messages:
        content = []
        for c in m["content"]:
            if c["type"] == "text":
                content.append({"type": "text", "text": c["text"]})
            else:
                buf = io.BytesIO(); c["image"].save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + b64}})
        out.append({"role": m["role"], "content": content})
    return out


class OpenAIBackend:
    """OpenAI-compatible chat API — e.g. a local vLLM server (recommended for 7B on
    Linux): `vllm serve <model> --served-model-name qwen` then --base-url .../v1."""
    def __init__(self, model, base_url, api_key, max_new_tokens=512):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.max_new_tokens = max_new_tokens
        print(f"[openai] {model} @ {base_url}", flush=True)

    def ask(self, messages):
        r = self.client.chat.completions.create(
            model=self.model, messages=_to_openai(messages),
            max_tokens=self.max_new_tokens, temperature=0)
        return r.choices[0].message.content


# ---------------- per-episode ----------------
def annotate_one(backend, parquet, out_dir, ep, task, args):
    store = FrameStore(parquet, args.cam)
    T, fps = store.n, args.fps
    flags = []
    p2_refinement = None

    # coarse
    idxs = sample_idxs(0, T - 1, args.n_frames)
    head = COARSE_HEAD.format(task=task, T=T, T1=T - 1, fps=fps, k=len(idxs))
    if args.dry_run:
        montage(store, idxs, fps, os.path.join(out_dir, f"ep{ep:03d}_coarse_montage.png"))
        open(os.path.join(out_dir, f"ep{ep:03d}_prompt.txt"), "w", encoding="utf-8").write(head)
        print(f"ep{ep:03d}: dry-run wrote montage + prompt ({len(idxs)} frames)")
        return None
    raw = backend.ask(build_messages(task, store, idxs, fps, args.size, head))
    if os.environ.get("VLM_DEBUG"):
        print("=== RAW coarse response ep%03d ===\n%s\n=== end ===" % (ep, raw), flush=True)
    cps, notes = parse_coarse(raw, T)
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

    if args.p2_history and args.state_ref:
        state_cps = load_state_cps(args.state_ref, ep)
        if state_cps is None:
            flags.append("p2 history refine skipped: missing state reference")
        else:
            cps[1], p2_refinement = refine_p2_with_history(
                backend, store, task, cps, state_cps, fps, args, flags
            )
            cps = enforce_order(cps, T, flags)

    doc = {"episode_index": ep, "task": task, "n_frames": T, "fps": fps,
           "annotator": "qwen-vl", "model": args.model,
           "method": "qwen2.5-vl coarse(%d)%s%s, cam=%s" % (
               args.n_frames,
               "+fine" if args.fine else "",
               "+state-guided-p2-history" if args.p2_history and args.state_ref else "",
               args.cam),
           "notes": notes, "critical_points": cps, "critical_names": CRIT_NAMES,
           "p2_history_refinement": p2_refinement,
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
    ap.add_argument("--backend", default="qwen-local", choices=["qwen-local", "openai"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", dest="base_url",
                    help="openai backend: vLLM/OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="EMPTY", dest="api_key")
    ap.add_argument("--data", default=r"C:\Intern\black_smash_07\data\chunk-000")
    ap.add_argument("--out",  default=r"C:\Intern\mvt_annotations_vlm")
    ap.add_argument("--meta", default=r"C:\Intern\black_smash_07\meta\tasks.jsonl")
    ap.add_argument("--fps",  type=int, default=30)
    ap.add_argument("--eps",  default="", help="comma list; default = all")
    ap.add_argument("--cam",  default="observation.images.camera1")
    ap.add_argument("--n-frames", type=int, default=32, dest="n_frames")
    ap.add_argument("--size", type=int, default=256, help="per-frame px sent to the model")
    ap.add_argument("--crop", type=float, default=0.6, help="center-crop fraction (1.0 = no crop)")
    ap.add_argument("--fine", dest="fine", action="store_true", default=True)
    ap.add_argument("--no-fine", dest="fine", action="store_false")
    ap.add_argument("--fine-window-s", type=float, default=1.5, dest="fine_window_s")
    ap.add_argument("--fine-frames", type=int, default=16, dest="fine_frames")
    ap.add_argument("--state-ref", default="", help="state annotation directory used to guide p2 history refinement")
    ap.add_argument("--p2-history", action="store_true",
                    help="refine p2_start_pour with chronological frames from the state-guided local window")
    ap.add_argument("--p2-history-window-s", type=float, default=3.0, dest="p2_history_window_s")
    ap.add_argument("--p2-history-frames", type=int, default=15, dest="p2_history_frames")
    ap.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global CROP; CROP = args.crop

    task = TASK_DEFAULT
    try:
        task = json.loads(open(args.meta).readline())["task"]
    except Exception:
        pass

    if args.dry_run:
        backend = None
    elif args.backend == "openai":
        backend = OpenAIBackend(args.model, args.base_url, args.api_key, args.max_new_tokens)
    else:
        backend = QwenLocal(args.model, args.max_new_tokens)

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
