"""
Local VLM verifier for state-derived critical points.

This is the preferred visual check for the current pipeline. It asks an
OpenAI-compatible VLM to choose one frame from a small local window around each
state critical point, then writes both the per-point evidence and a conservative
proposal. By default, only p2_start_pour can move automatically.

Example:
  PYTHON_BIN=/home/hillbot/miniforge3/envs/qwenvl/bin/python \
  python qwen_local_verify.py \
    --data /home/hillbot/black_smash_05/data/chunk-000 \
    --state annotations_state_05 \
    --out /tmp/qwen_local_verify_05 \
    --eps 0,1,2
"""
import argparse
import base64
import csv
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance

from batch_annotate import CRIT_NAMES, LABELS


CRIT_PROMPTS = [
    {
        "name": "p1_grasp_tube",
        "desc": "the first frame where the gripper is actually closed on and holding the test tube",
        "before": "the gripper is still reaching, open, or not yet holding the tube",
        "after": "the tube moves together with the closed gripper",
    },
    {
        "name": "p2_start_pour",
        "desc": "the first frame where powder pouring is visually confirmed, not merely the first tilt",
        "before": "the tube may be above the mortar or tilted, but there is no clear powder flow and no visible change in the powder pile",
        "after": "powder flow is visible or the black powder pile in the mortar has clearly started growing",
    },
    {
        "name": "p3_release_tube",
        "desc": "the first frame where the tube has been set down and the gripper has let go",
        "before": "the gripper is still holding or guiding the tube",
        "after": "the gripper is separated from the tube and starts moving away",
    },
    {
        "name": "p4_grasp_pestle",
        "desc": "the first frame where the gripper is actually closed on and holding the pestle",
        "before": "the gripper is approaching the pestle or still open",
        "after": "the pestle moves together with the closed gripper",
    },
    {
        "name": "p5_start_grind",
        "desc": "the first frame where the pestle is in the mortar and sustained grinding starts",
        "before": "the pestle is being positioned or lowered, not yet grinding in place",
        "after": "the pestle is inside the mortar with repeated rubbing or circular in-place motion",
    },
    {
        "name": "p6_lift_pestle",
        "desc": "the first frame where grinding has stopped and the pestle starts lifting away from the mortar",
        "before": "the pestle is still in contact with the powder/mortar during grinding",
        "after": "the pestle is visibly lifted or leaving the mortar",
    },
]


def sample_idxs(lo: int, hi: int, n: int) -> list[int]:
    return [int(round(x)) for x in np.linspace(lo, hi, min(n, hi - lo + 1))]


def enforce_order(cps: list[int], n_frames: int, flags: list[str]) -> list[int]:
    cps = [max(1, min(n_frames - 2, int(c))) for c in cps]
    for i in range(1, len(cps)):
        if cps[i] <= cps[i - 1]:
            cps[i] = cps[i - 1] + 1
            flags.append("nudged p%d for ordering" % (i + 1))
    if cps[-1] > n_frames - 2:
        cps[-1] = n_frames - 2
        for i in range(len(cps) - 2, -1, -1):
            if cps[i] >= cps[i + 1]:
                cps[i] = cps[i + 1] - 1
        if cps[0] < 1:
            flags.append("could not fit ordered points in episode")
    return cps


def subtasks_from_cps(cps: list[int], n_frames: int, fps: int) -> list[dict]:
    starts = [0] + list(cps)
    out = []
    for i, label in enumerate(LABELS):
        start = starts[i]
        end = starts[i + 1] - 1 if i < len(LABELS) - 1 else n_frames - 1
        out.append({
            "subtask_id": i,
            "label": label,
            "start_frame": start,
            "end_frame": end,
            "start_t": round(start / fps, 2),
            "end_t": round(end / fps, 2),
            "n_frames": end - start + 1,
            "dur_s": round((end - start + 1) / fps, 2),
        })
    return out


def parse_eps(text: str) -> list[int] | None:
    if not text:
        return None
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def enhance(im: Image.Image, crop: float) -> Image.Image:
    if crop < 1.0:
        w, h = im.size
        cw, ch = int(w * crop), int(h * crop)
        im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    a = np.asarray(im).astype(np.float32)
    m = a.reshape(-1, 3).mean(0) + 1e-6
    a = np.clip(a * (m.mean() / m), 0, 255)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    a = (a ** 0.8) * 255
    return ImageEnhance.Contrast(Image.fromarray(a.astype(np.uint8))).enhance(1.25)


def image_from_cell(cell) -> Image.Image:
    if isinstance(cell, dict) and "bytes" in cell:
        data = cell["bytes"]
    elif hasattr(cell, "as_py"):
        obj = cell.as_py()
        data = obj["bytes"] if isinstance(obj, dict) else obj
    else:
        data = cell
    return Image.open(io.BytesIO(data)).convert("RGB")


class MultiCameraStore:
    def __init__(self, parquet: Path, cameras: list[str]):
        self.cameras = cameras
        self.df = pd.read_parquet(parquet, columns=cameras)
        self.n = len(self.df)

    def get(self, idx: int, camera: str) -> Image.Image:
        return image_from_cell(self.df.iloc[int(idx)][camera])


def make_contact_sheet(store: MultiCameraStore, frames: list[int], fps: int, size: int, crop: float) -> Image.Image:
    cams = store.cameras
    label_h = 22
    tile_w = size
    tile_h = size + label_h
    sheet = Image.new("RGB", (tile_w * len(frames), tile_h * len(cams)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for r, cam in enumerate(cams):
        cam_name = cam.split(".")[-1]
        for c, t in enumerate(frames):
            im = enhance(store.get(t, cam), crop).resize((size, size))
            x, y = c * tile_w, r * tile_h
            sheet.paste(im, (x, y + label_h))
            draw.rectangle([x, y, x + tile_w - 1, y + label_h - 1], fill=(0, 0, 0))
            draw.text((x + 4, y + 4), f"{cam_name}  frame={t}  t={t / fps:.2f}s", fill=(255, 220, 0))
    return sheet


def data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class OpenAIVLM:
    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.max_tokens = max_tokens
        print(f"[openai] {model} @ {base_url}", flush=True)

    def ask(self, prompt: str, image: Image.Image) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url(image)}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""


def candidate_window(cps: list[int], point_idx: int, n_frames: int, window: int) -> tuple[int, int]:
    center = cps[point_idx]
    lo = max(0, center - window)
    hi = min(n_frames - 1, center + window)
    if point_idx > 0:
        lo = max(lo, cps[point_idx - 1] + 1)
    if point_idx < len(cps) - 1:
        hi = min(hi, cps[point_idx + 1] - 1)
    if hi <= lo:
        lo = max(0, center - max(2, window // 2))
        hi = min(n_frames - 1, center + max(2, window // 2))
    return lo, hi


def build_prompt(task: str, ep: int, point_idx: int, state_frame: int, frames: list[int], fps: int, show_state: bool) -> str:
    spec = CRIT_PROMPTS[point_idx]
    frame_list = ", ".join(str(x) for x in frames)
    state_line = (
        f"State signal estimate: frame {state_frame} ({state_frame / fps:.2f}s).\n"
        if show_state
        else "These candidates are from a local window around an automatic signal estimate. Choose independently from the image evidence.\n"
    )
    p2_extra = ""
    if point_idx == 1:
        p2_extra = (
            "\nFor p2_start_pour, be conservative: do NOT select a frame just because "
            "the tube is tilted, positioned over the mortar, or about to pour. Select "
            "the earliest candidate where powder flow is visible, or where the black "
            "powder pile in the mortar has visibly changed. If the earliest candidates "
            "are ambiguous, choose the later candidate with unmistakable pouring evidence.\n"
        )
    return (
        "You are a careful visual verifier for a robot manipulation dataset.\n"
        f"Episode: {ep:06d}\n"
        f"Task: {task}\n\n"
        f"Verify {spec['name']}: {spec['desc']}.\n"
        f"{state_line}"
        f"The image is a contact sheet. Each tile is labeled with camera, frame, and time.\n"
        f"You MUST choose exactly one of these candidate frames: {frame_list}.\n\n"
        f"Before the event: {spec['before']}.\n"
        f"After the event: {spec['after']}.\n\n"
        f"{p2_extra}\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "selected_frame": int,\n'
        '  "confidence": "high|medium|low",\n'
        '  "visual_evidence": "specific visible evidence",\n'
        '  "reject_previous": "why earlier candidates are not the event yet",\n'
        '  "reject_next": "why later candidates are already after the event or less exact"\n'
        "}\n"
    )


def normalize_conf(value) -> str:
    text = str(value or "").lower()
    if "high" in text:
        return "high"
    if "medium" in text or "med" in text:
        return "medium"
    return "low"


def verify_point(vlm: OpenAIVLM, store: MultiCameraStore, task: str, ep: int, cps: list[int], point_idx: int, args):
    lo, hi = candidate_window(cps, point_idx, store.n, int(args.window_s * args.fps))
    frames = sorted(set(sample_idxs(lo, hi, args.candidates) + [cps[point_idx]]))
    frames = [int(max(lo, min(hi, x))) for x in frames]
    sheet = make_contact_sheet(store, frames, args.fps, args.size, args.crop)
    prompt = build_prompt(task, ep, point_idx, cps[point_idx], frames, args.fps, args.show_state_estimate)
    raw = vlm.ask(prompt, sheet)
    obj = extract_json(raw)
    selected = obj.get("selected_frame", cps[point_idx])
    try:
        selected = int(round(float(selected)))
    except Exception:
        selected = cps[point_idx]
    if selected not in frames:
        selected = min(frames, key=lambda x: abs(x - selected))
    conf = normalize_conf(obj.get("confidence"))
    evidence = str(obj.get("visual_evidence", "")).strip()
    return {
        "point_index": point_idx + 1,
        "point_name": CRIT_NAMES[point_idx],
        "state_frame": int(cps[point_idx]),
        "selected_frame": int(selected),
        "delta_frames": int(selected - cps[point_idx]),
        "confidence": conf,
        "visual_evidence": evidence,
        "reject_previous": str(obj.get("reject_previous", "")).strip(),
        "reject_next": str(obj.get("reject_next", "")).strip(),
        "window": [int(lo), int(hi)],
        "candidate_frames": frames,
        "raw_response": raw,
    }


def should_move(result: dict, point_idx: int, args) -> tuple[bool, str]:
    delta = abs(int(result["delta_frames"]))
    if result["confidence"] != "high":
        return False, "confidence_not_high"
    if delta > int(args.max_move_s * args.fps):
        return False, "move_too_large"
    if not result["visual_evidence"]:
        return False, "missing_evidence"
    if args.move_only_p2 and point_idx != 1:
        return False, "record_only_non_p2"
    if delta < args.min_move_frames:
        return False, "move_too_small"
    return True, "accepted"


def load_task(meta: Path) -> str:
    if not meta.exists():
        return "Pour the black powder into the mortar and grind."
    with meta.open(encoding="utf-8") as f:
        row = json.loads(next(f))
    return row.get("task", "Pour the black powder into the mortar and grind.")


def write_index(path: Path, doc: dict):
    idx = np.zeros(int(doc["n_frames"]), dtype=np.int16)
    for subtask in doc["subtasks"]:
        idx[int(subtask["start_frame"]): int(subtask["end_frame"]) + 1] = int(subtask["subtask_id"])
    np.save(path, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset data/chunk-000 directory")
    ap.add_argument("--state", required=True, help="Directory containing state epNNN_subtasks.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--eps", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen"))
    ap.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY"))
    ap.add_argument("--cameras", default="observation.images.camera0,observation.images.camera1")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--max-move-s", type=float, default=0.67)
    ap.add_argument("--min-move-frames", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=7)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--crop", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--move-only-p2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--show-state-estimate", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    data_dir = Path(args.data)
    state_dir = Path(args.state)
    out_dir = Path(args.out)
    verified_dir = out_dir / "verified_annotations"
    sheet_dir = out_dir / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    verified_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    meta = Path(args.meta) if args.meta else data_dir.parent.parent / "meta" / "tasks.jsonl"
    task = load_task(meta)
    eps = parse_eps(args.eps)
    if eps is None:
        eps = sorted(int(p.stem.split("_")[1]) for p in data_dir.glob("episode_*.parquet"))
    cameras = [x.strip() for x in args.cameras.split(",") if x.strip()]
    vlm = OpenAIVLM(args.model, args.base_url, args.api_key, args.max_tokens)

    jsonl_path = out_dir / "local_verify.jsonl"
    summary_path = out_dir / "summary.csv"
    rows = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for ep in eps:
            parquet = data_dir / f"episode_{ep:06d}.parquet"
            state_path = state_dir / f"ep{ep:03d}_subtasks.json"
            if not parquet.exists() or not state_path.exists():
                print(f"ep{ep:03d}: missing parquet or state annotation", flush=True)
                continue
            state_doc = json.loads(state_path.read_text(encoding="utf-8"))
            cps = [int(x) for x in state_doc["critical_points"]]
            store = MultiCameraStore(parquet, cameras)
            results = []
            proposal = list(cps)
            decisions = []
            for i in range(6):
                result = verify_point(vlm, store, task, ep, cps, i, args)
                move, reason = should_move(result, i, args)
                result["decision"] = reason
                if move:
                    proposal[i] = int(result["selected_frame"])
                decisions.append(reason)
                results.append(result)
            flags = []
            proposal = enforce_order(proposal, int(state_doc["n_frames"]), flags)
            doc = {
                "episode_index": ep,
                "task": task,
                "n_frames": int(state_doc["n_frames"]),
                "fps": args.fps,
                "annotator": "qwen-local-verifier",
                "method": "state-local candidate verification with contact sheets",
                "state_cps": cps,
                "critical_points": proposal,
                "critical_names": CRIT_NAMES,
                "subtask_starts": [0] + proposal,
                "point_results": results,
                "flags": flags,
                "n_subtasks": 7,
                "subtasks": subtasks_from_cps(proposal, int(state_doc["n_frames"]), args.fps),
            }
            out_json = verified_dir / f"ep{ep:03d}_subtasks.json"
            out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            write_index(verified_dir / f"ep{ep:03d}_subtask_index.npy", doc)
            jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")
            moved = sum(1 for a, b in zip(cps, proposal) if a != b)
            high = sum(1 for r in results if r["confidence"] == "high")
            max_abs_delta = max(abs(int(r["delta_frames"])) for r in results)
            rows.append({
                "episode_index": ep,
                "moved_points": moved,
                "high_conf_points": high,
                "max_abs_delta": max_abs_delta,
                "state_cps": " ".join(map(str, cps)),
                "proposal_cps": " ".join(map(str, proposal)),
                "decisions": ";".join(decisions),
            })
            print(
                f"ep{ep:03d} moved={moved} high={high}/6 "
                f"state={cps} proposal={proposal} decisions={decisions}",
                flush=True,
            )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "moved_points",
                "high_conf_points",
                "max_abs_delta",
                "state_cps",
                "proposal_cps",
                "decisions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSONL={jsonl_path}")
    print(f"SUMMARY={summary_path}")
    print(f"VERIFIED_DIR={verified_dir}")


if __name__ == "__main__":
    main()
