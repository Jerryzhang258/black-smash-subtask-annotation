"""
Stage 1 — VLM prior over egocentric frames (OpenAI-compatible backend).

A vision model watches enhanced ego frames sampled across the demo and proposes
the 5 critical-point frame indices (same schema as the signal stage). It is the
SEMANTIC view; precise gripper timing is the signal stage's job, so here we accept
coarser timing and let fusion (fuse.py) decide ownership per point.

Backend: any OpenAI-compatible chat endpoint (local vLLM, DashScope, Gemini proxy,
…). Set --base-url and the api-key env var. Frame enhancement matches visualize.py.
Requires the `openai` package only when --vlm is used.
"""
from __future__ import annotations
import os, io as _io, json, re, base64
import numpy as np
from PIL import Image, ImageEnhance

from . import config as C
from . import ego_dataio as io

CRIT_DESC = [
    "c1 grasp_tube    - the holder hand first grips the test tube",
    "c2 acquire_wiper - the other hand grips the wiper/cloth used to clean the tube",
    "c3 start_wipe    - wiping begins: the wiper makes repeated in-place strokes on the tube",
    "c4 finish_wipe   - wiping stops and the wiper hand lets go of the cloth",
    "c5 release_tube  - the tube is set back down and the holder hand lets go",
]
KEYS = ["c1_grasp_tube", "c2_acquire_wiper", "c3_start_wipe", "c4_finish_wipe", "c5_release_tube"]

HEAD = (
    "You are labeling a bimanual robot manipulation episode seen from a hand-mounted"
    " wide-angle (fisheye) camera.\n"
    'Task: "wipe the test tube" (one hand holds a test tube, the other wipes it clean).\n'
    "The episode has {T} frames at {fps:.0f} fps. Below are {k} frames sampled evenly,"
    " each captioned with its frame index (#idx) and time. Views are fisheye and"
    " contrast-enhanced.\n\n"
    "Find these 5 critical points (transition frames), each an integer in [0, {T1}],"
    " STRICTLY increasing:\n  " + "\n  ".join(CRIT_DESC) +
    "\n\nFirst, for each shown frame write one short line '#idx: <what the hands/tube"
    " are doing>'. Then on the LAST line output ONLY the JSON object:\n"
    '{{"c1_grasp_tube": int, "c2_acquire_wiper": int, "c3_start_wipe": int,'
    ' "c4_finish_wipe": int, "c5_release_tube": int, "notes": "one short sentence"}}'
)


def _enh_b64(im: Image.Image, size: int = 256) -> str:
    im = im.convert("RGB")
    a = np.asarray(im).astype(np.float32)
    m = a.reshape(-1, 3).mean(0) + 1e-6
    a = np.clip(a * (m.mean() / m), 0, 255)
    lo, hi = np.percentile(a, 2), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    im = ImageEnhance.Contrast(Image.fromarray(a.astype(np.uint8))).enhance(1.3).resize((size, size))
    buf = _io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAIBackend:
    def __init__(self, model, base_url, api_key, max_tokens=400):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model, self.max_tokens = model, max_tokens

    def ask(self, text, image_urls):
        content = [{"type": "text", "text": text}]
        for u in image_urls:
            content.append({"type": "image_url", "image_url": {"url": u}})
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            temperature=0, max_tokens=self.max_tokens)
        return r.choices[0].message.content or ""


def make_backend(args):
    return OpenAIBackend(args.model, args.base_url, os.environ.get(args.api_key_env, "EMPTY"))


def _extract(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def _enforce(cps, T):
    cps = [max(1, min(T - 2, int(c))) for c in cps]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
    return cps


def propose(backend, demo, args):
    """Return (vlm_critical_points[5], flags)."""
    T, fps, flags = demo.n_frames, demo.fps, []
    idxs = [int(round(x)) for x in np.linspace(0, T - 1, min(args.n_frames, T))]
    urls, capt = [], []
    for i in idxs:
        try:
            urls.append(_enh_b64(io.frame_image(demo, i)))
            capt.append(f"#{i} {i / fps:.1f}s")
        except Exception:
            pass
    head = HEAD.format(T=T, T1=T - 1, fps=fps, k=len(urls))
    head += "\nFrames: " + ", ".join(capt)
    raw = backend.ask(head, urls)
    if os.environ.get("VLM_DEBUG"):
        print("=== RAW vlm ===\n", raw, "\n=== end ===")
    obj = _extract(raw) or {}
    try:
        cps = [int(round(float(obj[k]))) for k in KEYS]
    except Exception:
        flags.append("vlm parse failed -> fallback proportions")
        cps = [int(p * T) for p in (0.20, 0.37, 0.50, 0.77, 0.91)]
    return _enforce(cps, T), flags
