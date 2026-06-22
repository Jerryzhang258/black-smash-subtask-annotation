#!/usr/bin/env python3
"""Run a small Qwen-VL stage annotation experiment on LeRobot parquet episodes."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image


DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/.cache/huggingface/lerobot/chaoyi/0118_data")
DEFAULT_META_ROOT = Path("/root/autodl-tmp/VB-VLA/Data_collection/dataset_converted/meta")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/VB-VLA/qwen_stage_annotation_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--base-url-env", default="")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--episodes", default="", help="Comma-separated episode ids, e.g. 0,1,2")
    parser.add_argument(
        "--camera-keys",
        default="observation.images.camera0,observation.images.camera1",
        help="Comma-separated image columns to send, e.g. observation.images.camera0,observation.images.camera1",
    )
    parser.add_argument("--frame-sampling", choices=["uniform3", "all"], default="all")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--task-description",
        default="",
        help="Optional natural-language task description that overrides sparse task metadata.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_task_map(meta_root: Path) -> dict[int, str]:
    path = meta_root / "tasks.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    task_map: dict[int, str] = {}
    for index, row in df.iterrows():
        task_index = int(row.get("task_index", len(task_map)))
        task_map[task_index] = str(index)
    return task_map


def list_episode_files(dataset_root: Path, episodes: str, num_episodes: int) -> list[Path]:
    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if episodes.strip():
        wanted = {int(x.strip()) for x in episodes.split(",") if x.strip()}
        files = [p for p in files if episode_id_from_path(p) in wanted]
    return files[:num_episodes]


def episode_id_from_path(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.astype(float)
    kernel = np.ones(window) / window
    return np.convolve(values.astype(float), kernel, mode="same")


def full_points(values: np.ndarray) -> list[list[float]]:
    return [[int(i), round(float(v), 5)] for i, v in enumerate(values)]


def hand_summary(values: np.ndarray) -> dict[str, Any]:
    smoothed = smooth(values)
    diffs = np.diff(smoothed)
    value_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
    summary = {
        "start": round(float(values[0]), 5),
        "min": round(float(np.min(values)), 5),
        "max": round(float(np.max(values)), 5),
        "end": round(float(values[-1]), 5),
        "range": round(value_range, 5),
    }
    summary["full_points"] = full_points(values)
    summary["full_velocity"] = full_points(diffs) if len(diffs) else []
    return summary


def select_frame_indices(num_frames: int, frame_sampling: str) -> list[int]:
    if frame_sampling == "all":
        return list(range(num_frames))
    return sorted(set([0, int(round((num_frames - 1) / 2)), num_frames - 1]))


def parse_camera_keys(camera_keys: str) -> list[str]:
    keys = [key.strip() for key in camera_keys.split(",") if key.strip()]
    if not keys:
        raise ValueError("--camera-keys must contain at least one camera column")
    return keys


def resolve_task_description(df: pd.DataFrame, task_map: dict[int, str], task_description: str) -> str:
    if task_description.strip():
        return task_description.strip()
    task_index = int(df["task_index"].iloc[0])
    return task_map.get(task_index, f"task_index_{task_index}")


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    states = np.stack(df["observation.state"].to_numpy())
    episode_id = int(df["episode_index"].iloc[0])
    left = states[:, 6]
    right = states[:, 13]
    left_summary = hand_summary(left)
    right_summary = hand_summary(right)
    n = len(df)
    return {
        "episode_index": episode_id,
        "episode_length": n,
        "left_gripper": left_summary,
        "right_gripper": right_summary,
    }


def image_bytes_from_cell(cell: Any) -> bytes:
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return cell["bytes"]
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def make_jpeg_data_url(image_bytes: bytes, max_side: int = 448) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=82)
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def save_keyframes(
    df: pd.DataFrame,
    summary: dict[str, Any],
    camera_keys: list[str],
    out_dir: Path,
    frame_indices: list[int],
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    episode_id = summary["episode_index"]
    for camera_key in camera_keys:
        if camera_key not in df.columns:
            raise KeyError(f"Camera column not found: {camera_key}")
        camera_name = camera_key.split(".")[-1]
        for t in frame_indices:
            image_bytes = image_bytes_from_cell(df.iloc[t][camera_key])
            path = out_dir / f"episode_{episode_id:06d}_{camera_name}_t{t:04d}.jpg"
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.thumbnail((640, 640))
            image.save(path, quality=90)
            saved.append({"camera": camera_name, "t": int(t), "path": str(path)})
    return saved


def save_gripper_plot(summary: dict[str, Any], df: pd.DataFrame, out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    states = np.stack(df["observation.state"].to_numpy())
    episode_id = summary["episode_index"]
    xs = np.arange(len(states))
    plt.figure(figsize=(8, 3.2))
    plt.plot(xs, states[:, 6], label="left_gripper")
    plt.plot(xs, states[:, 13], label="right_gripper")
    plt.title(f"Episode {episode_id:06d} gripper width")
    plt.xlabel("frame")
    plt.ylabel("width")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"episode_{episode_id:06d}_gripper.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def build_prompt(summary: dict[str, Any], task_description: str) -> str:
    compact = {k: v for k, v in summary.items() if k not in {"keyframe_files", "gripper_plot"}}
    return (
        "You are a robotics dataset annotation assistant. "
        "Given a task description, full gripper width/velocity sequences, and camera frames, split this episode into semantic stages.\n\n"
        f"Important task context: {task_description}. "
        "When the visual evidence is consistent with this task, produce predictive stage prompts that tell the robot what should happen next.\n\n"
        "Requirements:\n"
        "1. Output strict JSON only, no markdown.\n"
        "2. Use stage names only from this exact set: reach, grasp, transport, place, release, adjust, done. Do not invent labels such as lift, open, move, or carry.\n"
        "3. Stages should cover the whole episode without overlap.\n"
        "4. prediction_prompt must be a predictive command for the next few frames, not a description of the current frame. Use future-oriented language such as 'move', 'grasp', 'lift', 'place', or 'release'.\n"
        "5. expected_future_observation should state what should become visible several frames after the start of this stage if the prediction is correct. This will be checked later in a separate batch self-test after all episodes are labeled.\n\n"
        "Episode summary:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Return schema:\n"
        "{\n"
        '  "episode_index": int,\n'
        '  "stages": [\n'
        "    {\n"
        '      "name": "reach",\n'
        '      "start_t": 0,\n'
        '      "end_t": 10,\n'
        '      "prediction_prompt": "Move the left gripper toward the orange lid.",\n'
        '      "expected_future_observation": "Within the next few frames, the left gripper should be closer to the orange lid.",\n'
        '      "reason": "short evidence"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped)
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def call_qwen(client: OpenAI, model: str, prompt: str, frame_paths: list[dict[str, Any]], max_tokens: int) -> tuple[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in frame_paths:
        data_url = make_jpeg_data_url(Path(item["path"]).read_bytes())
        content.append({"type": "text", "text": f"{item.get('camera', 'camera')} frame t={item['t']}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or "", response


def validate_result(parsed: dict[str, Any] | None, episode_length: int) -> dict[str, Any]:
    if not parsed or not isinstance(parsed.get("stages"), list):
        return {"json_valid": False, "coverage_ok": False, "stage_count": 0}
    stages = parsed["stages"]
    coverage_ok = bool(stages)
    expected_start = 0
    for stage in stages:
        try:
            start = int(stage["start_t"])
            end = int(stage["end_t"])
        except Exception:
            coverage_ok = False
            continue
        if start != expected_start or end < start or end >= episode_length:
            coverage_ok = False
        expected_start = end + 1
    if expected_start != episode_length:
        coverage_ok = False
    return {
        "json_valid": True,
        "coverage_ok": coverage_ok,
        "stage_count": len(stages),
    }


def write_report(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok = sum(1 for r in rows if r["json_valid"])
    coverage = sum(1 for r in rows if r["coverage_ok"])
    report = [
        "# Qwen Stage Annotation Pilot",
        "",
        f"- time: {datetime.now().isoformat(timespec='seconds')}",
        f"- model: `{args.model}`",
        f"- dataset_root: `{args.dataset_root}`",
        f"- task_description: {args.task_description or 'metadata only'}",
        f"- episodes: {len(rows)}",
        f"- valid_json: {ok}/{len(rows)}",
        f"- full_coverage: {coverage}/{len(rows)}",
        "",
        "## Episode Summary",
        "",
        "| episode | frames | stages | json | coverage | latency_s |",
        "|---:|---:|---:|---|---|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['episode_index']} | {r['episode_length']} | {r['stage_count']} | "
            f"{r['json_valid']} | {r['coverage_ok']} | {r['latency_s']} |"
        )
    report.extend(
        [
            "",
            "## Notes for Meeting",
            "",
            "- Input to API: task description, episode length, full gripper width/velocity sequences, and camera frames from the configured external cameras.",
            "- Output from API: stage intervals, stage names, predictive stage prompts, expected future observations, and short evidence.",
            "- Next step: after all episodes are labeled, run a separate batch self-test that checks whether later frames match each stage prediction.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    keyframe_dir = run_dir / "keyframes"
    plot_dir = run_dir / "gripper_plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    task_map = load_task_map(args.meta_root)
    episode_files = list_episode_files(args.dataset_root, args.episodes, args.num_episodes)
    if not episode_files:
        raise RuntimeError(f"No episode parquet files found under {args.dataset_root}")
    client = None
    if not args.dry_run:
        base_url = args.base_url
        if args.base_url_env:
            base_url = os.environ[args.base_url_env]
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client = OpenAI(
            api_key=os.environ[args.api_key_env],
            base_url=base_url,
        )

    jsonl_path = run_dir / "stage_annotations.jsonl"
    csv_path = run_dir / "summary.csv"
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for ep_path in episode_files:
            df = pd.read_parquet(ep_path)
            task_description = resolve_task_description(df, task_map, args.task_description)
            camera_keys = parse_camera_keys(args.camera_keys)
            summary = build_summary(df)
            frame_indices = select_frame_indices(summary["episode_length"], args.frame_sampling)
            keyframes = save_keyframes(df, summary, camera_keys, keyframe_dir, frame_indices)
            plot_path = save_gripper_plot(summary, df, plot_dir)
            summary["keyframe_files"] = keyframes
            summary["gripper_plot"] = plot_path
            prompt = build_prompt(summary, task_description)
            started = time.time()
            raw_text = ""
            response_id = None
            usage = None
            error = None
            if args.dry_run:
                parsed = None
            else:
                try:
                    assert client is not None
                    raw_text, response = call_qwen(client, args.model, prompt, keyframes, args.max_tokens)
                    response_id = getattr(response, "id", None)
                    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
                    parsed = extract_json(raw_text)
                except Exception as exc:
                    parsed = None
                    error = repr(exc)
            latency_s = round(time.time() - started, 2)
            validation = validate_result(parsed, summary["episode_length"])
            record = {
                **summary,
                "model": args.model,
                "response_id": response_id,
                "usage": usage,
                "latency_s": latency_s,
                "raw_response": raw_text,
                "parsed_response": parsed,
                "error": error,
                "validation": validation,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            row = {
                "episode_index": summary["episode_index"],
                "episode_length": summary["episode_length"],
                "json_valid": validation["json_valid"],
                "coverage_ok": validation["coverage_ok"],
                "stage_count": validation["stage_count"],
                "latency_s": latency_s,
                "error": error or "",
                "keyframes": ";".join(str(k["path"]) for k in keyframes),
                "gripper_plot": plot_path or "",
            }
            rows.append(row)
            print(
                f"episode={row['episode_index']:06d} json={row['json_valid']} "
                f"coverage={row['coverage_ok']} stages={row['stage_count']} "
                f"latency={latency_s}s"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(run_dir, rows, args)
    print(f"RUN_DIR={run_dir}")
    print(f"JSONL={jsonl_path}")
    print(f"CSV={csv_path}")
    print(f"REPORT={run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
