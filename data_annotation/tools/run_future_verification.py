#!/usr/bin/env python3
"""Run Gemini future-frame verification and compute annotation quality scores."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default=os.environ.get("TTK_MODEL", "gemini-3.5-flash-low-反重力"))
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        default="openai",
        help="API provider. 'openai' means OpenAI-compatible endpoint; 'google' means official Gemini API via google-genai.",
    )
    parser.add_argument("--api-key-env", default="TTK_API_KEY")
    parser.add_argument("--base-url-env", default="TTK_BASE_URL")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=0, help="Only verify the first N episode ids in the run.")
    return parser.parse_args()


def make_jpeg_data_url(path: Path, max_side: int = 448) -> str:
    data = make_jpeg_bytes(path, max_side=max_side)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def make_jpeg_bytes(path: Path, max_side: int = 448) -> bytes:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=82)
    return out.getvalue()


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def gripper_values(record: dict[str, Any], key: str) -> np.ndarray:
    points = record[key]["full_points"]
    return np.asarray([float(v) for _, v in points], dtype=float)


def choose_gripper(stage: dict[str, Any], left: np.ndarray, right: np.ndarray) -> np.ndarray:
    text = f"{stage.get('prediction_prompt', '')} {stage.get('expected_future_observation', '')}".lower()
    if "left" in text:
        return left
    if "right" in text:
        return right
    start = int(stage["start_t"])
    end = int(stage["end_t"])
    left_delta = abs(float(left[end] - left[start]))
    right_delta = abs(float(right[end] - right[start]))
    return left if left_delta >= right_delta else right


def boundary_stage_score(stage: dict[str, Any], left: np.ndarray, right: np.ndarray) -> float:
    values = choose_gripper(stage, left, right)
    start = int(stage["start_t"])
    end = int(stage["end_t"])
    stage_name = str(stage.get("name", "")).lower()
    global_range = max(float(np.max(values) - np.min(values)), 1e-6)
    delta = float(values[end] - values[start])
    segment = values[start : end + 1]
    fluctuation = float(np.max(segment) - np.min(segment)) if len(segment) else 0.0

    if stage_name == "grasp":
        if delta < -0.3 * global_range:
            return 1.0
        if delta < -0.1 * global_range:
            return 0.7
        if abs(delta) <= 0.1 * global_range:
            return 0.4
        return 0.0
    if stage_name == "release":
        if delta > 0.3 * global_range:
            return 1.0
        if delta > 0.1 * global_range:
            return 0.7
        if abs(delta) <= 0.1 * global_range:
            return 0.4
        return 0.0
    if stage_name == "transport":
        ratio = fluctuation / global_range
        if ratio <= 0.1:
            return 1.0
        if ratio <= 0.2:
            return 0.7
        if ratio <= 0.4:
            return 0.4
        return 0.0
    if stage_name in {"reach", "done"}:
        return 1.0 if fluctuation / global_range <= 0.1 else 0.7
    return 0.7


def interval_score(record: dict[str, Any]) -> float:
    episode_length = int(record["episode_length"])
    counts = [0] * episode_length
    out_of_range = 0
    stages = (record.get("parsed_response") or {}).get("stages") or record.get("normalized_stages", [])
    for stage in stages:
        start = int(stage.get("start_t", 0))
        end = int(stage.get("end_t", start))
        for t in range(start, end + 1):
            if 0 <= t < episode_length:
                counts[t] += 1
            else:
                out_of_range += 1
    missing = sum(1 for c in counts if c == 0)
    overlap = sum(max(0, c - 1) for c in counts)
    bad = missing + overlap + out_of_range
    return round(max(0.0, 1.0 - bad / episode_length), 4)


def stage_image_paths(run_dir: Path, episode_index: int, frames: list[int]) -> list[tuple[str, int, Path]]:
    paths: list[tuple[str, int, Path]] = []
    for camera in ["camera0", "camera1"]:
        for t in frames:
            path = run_dir / "keyframes" / f"episode_{episode_index:06d}_{camera}_t{t:04d}.jpg"
            if path.exists():
                paths.append((camera, t, path))
    return paths


def verify_stage_once(client: OpenAI, model: str, sample: dict[str, Any], image_paths: list[tuple[str, int, Path]], max_tokens: int) -> tuple[dict[str, Any] | None, str, float]:
    prompt = (
        "You are verifying a robot stage annotation. "
        "Compare the stage start, middle, end, and after-stage frames from two external cameras. "
        "Decide whether the expected future observation is visually supported.\n\n"
        f"Stage name: {sample.get('stage_name')}\n"
        f"Stage interval: {sample.get('stage_interval')}\n"
        f"Prediction prompt: {sample.get('prediction_prompt')}\n"
        f"Expected future observation: {sample.get('expected_future_observation')}\n\n"
        "Return strict JSON only:\n"
        '{"pass": true, "score": 0.0, "reason": "short visual evidence"}\n'
        "Score scale: 1.0 clear support, 0.75 mostly supported, 0.5 partial, 0.25 weak, 0.0 unsupported or opposite."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for camera, t, path in image_paths:
        content.append({"type": "text", "text": f"{camera} frame t={t}:"})
        content.append({"type": "image_url", "image_url": {"url": make_jpeg_data_url(path)}})
    started = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    latency = round(time.time() - started, 2)
    raw_text = response.choices[0].message.content or ""
    return extract_json(raw_text), raw_text, latency


def verify_stage_once_google(client: Any, model: str, sample: dict[str, Any], image_paths: list[tuple[str, int, Path]], max_tokens: int) -> tuple[dict[str, Any] | None, str, float]:
    from google.genai import types

    prompt = (
        "You are verifying a robot stage annotation. "
        "Compare the stage start, middle, end, and after-stage frames from two external cameras. "
        "Decide whether the expected future observation is visually supported.\n\n"
        f"Stage name: {sample.get('stage_name')}\n"
        f"Stage interval: {sample.get('stage_interval')}\n"
        f"Prediction prompt: {sample.get('prediction_prompt')}\n"
        f"Expected future observation: {sample.get('expected_future_observation')}\n\n"
        "Return strict JSON only:\n"
        '{"pass": true, "score": 0.0, "reason": "short visual evidence"}\n'
        "Score scale: 1.0 clear support, 0.75 mostly supported, 0.5 partial, 0.25 weak, 0.0 unsupported or opposite."
    )
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for camera, t, path in image_paths:
        parts.append(types.Part.from_text(text=f"{camera} frame t={t}:"))
        parts.append(types.Part.from_bytes(data=make_jpeg_bytes(path), mime_type="image/jpeg"))
    started = time.time()
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    latency = round(time.time() - started, 2)
    raw_text = response.text or ""
    return extract_json(raw_text), raw_text, latency


def verify_stage(
    client: Any,
    model: str,
    sample: dict[str, Any],
    image_paths: list[tuple[str, int, Path]],
    max_tokens: int,
    retries: int,
    provider: str,
) -> tuple[dict[str, Any] | None, str, float, str | None]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if provider == "google":
                parsed, raw_text, latency = verify_stage_once_google(client, model, sample, image_paths, max_tokens)
            else:
                parsed, raw_text, latency = verify_stage_once(client, model, sample, image_paths, max_tokens)
            return parsed, raw_text, latency, None
        except Exception as exc:
            last_error = repr(exc)
            print(
                f"retry={attempt}/{retries} episode={sample['episode_index']} "
                f"stage={sample['stage_index']} error={last_error}",
                flush=True,
            )
            time.sleep(min(10 * attempt, 30))
    return None, "", 0.0, last_error


def main() -> None:
    args = parse_args()
    if args.provider == "google":
        from google import genai

        client = genai.Client(api_key=os.environ[args.api_key_env])
    else:
        client = OpenAI(api_key=os.environ[args.api_key_env], base_url=os.environ[args.base_url_env])
    normalized_path = args.run_dir / "stage_annotations_normalized.jsonl"
    samples_path = args.run_dir / "prediction_self_check_samples.jsonl"
    out_path = args.run_dir / "future_verification_results.jsonl"
    summary_path = args.run_dir / "quality_summary.json"

    records = load_jsonl(normalized_path)
    if args.max_episodes > 0:
        records = [record for record in records if int(record["episode_index"]) < args.max_episodes]
    records_by_episode = {int(record["episode_index"]): record for record in records}
    episode_quality: dict[int, dict[str, Any]] = {}
    for episode_index, record in records_by_episode.items():
        left = gripper_values(record, "left_gripper")
        right = gripper_values(record, "right_gripper")
        interval = interval_score(record)
        boundary_scores = [boundary_stage_score(stage, left, right) for stage in record.get("normalized_stages", [])]
        episode_quality[episode_index] = {
            "episode_index": episode_index,
            "interval_score": interval,
            "boundary_signal_score": round(float(np.mean(boundary_scores)), 4) if boundary_scores else 0.0,
            "stage_count": len(record.get("normalized_stages", [])),
            "future_scores": [],
            "latencies": [],
        }

    existing_rows = []
    done_keys = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("verification_score") is None:
                    continue
                existing_rows.append(row)
                done_keys.add((int(row["episode_index"]), int(row["stage_index"])))

    for row in existing_rows:
        episode_index = int(row["episode_index"])
        if episode_index not in episode_quality:
            continue
        quality = episode_quality[episode_index]
        try:
            quality["future_scores"].append(float(row["verification_score"]))
        except Exception:
            pass
        try:
            quality["latencies"].append(float(row["latency_s"]))
        except Exception:
            pass

    rows = list(existing_rows)
    with samples_path.open(encoding="utf-8") as f, out_path.open("a", encoding="utf-8") as out:
        for line in f:
            sample = json.loads(line)
            if int(sample["episode_index"]) not in records_by_episode:
                continue
            key = (int(sample["episode_index"]), int(sample["stage_index"]))
            if key in done_keys:
                continue
            paths = stage_image_paths(args.run_dir, int(sample["episode_index"]), list(sample["check_frames"]))
            parsed, raw_text, latency, error = verify_stage(
                client, args.model, sample, paths, args.max_tokens, args.retries, args.provider
            )
            score = None
            if parsed is not None:
                try:
                    score = float(parsed.get("score"))
                except Exception:
                    score = None
            row = {
                **sample,
                "image_count": len(paths),
                "verification": parsed,
                "raw_response": raw_text,
                "verification_score": score,
                "latency_s": latency,
                "error": error,
            }
            rows.append(row)
            quality = episode_quality[int(sample["episode_index"])]
            if score is not None:
                quality["future_scores"].append(float(score))
            if latency:
                quality["latencies"].append(latency)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"episode={sample['episode_index']} stage={sample['stage_index']} {sample['stage_name']} score={score} latency={latency}s")

    episode_summaries = []
    for quality in episode_quality.values():
        future_scores = quality.pop("future_scores")
        latencies = quality.pop("latencies")
        future = round(float(np.mean(future_scores)), 4) if future_scores else 0.0
        stage_quality = round(
            0.30 * float(quality["interval_score"])
            + 0.30 * float(quality["boundary_signal_score"])
            + 0.40 * future,
            4,
        )
        episode_summaries.append(
            {
                **quality,
                "future_verification_score": future,
                "stage_quality_score": stage_quality,
                "verified_stage_count": len(future_scores),
                "mean_verification_latency_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            }
        )
    stage_quality_scores = [row["stage_quality_score"] for row in episode_summaries]
    interval_scores = [row["interval_score"] for row in episode_summaries]
    boundary_scores = [row["boundary_signal_score"] for row in episode_summaries]
    future_scores = [row["future_verification_score"] for row in episode_summaries]
    summary = {
        "model": args.model,
        "episode_count": len(episode_summaries),
        "stage_count": sum(row["stage_count"] for row in episode_summaries),
        "verified_stage_count": sum(row["verified_stage_count"] for row in episode_summaries),
        "mean_interval_score": round(float(np.mean(interval_scores)), 4) if interval_scores else 0.0,
        "mean_boundary_signal_score": round(float(np.mean(boundary_scores)), 4) if boundary_scores else 0.0,
        "mean_future_verification_score": round(float(np.mean(future_scores)), 4) if future_scores else 0.0,
        "mean_stage_quality_score": round(float(np.mean(stage_quality_scores)), 4) if stage_quality_scores else 0.0,
        "episode_summaries": sorted(episode_summaries, key=lambda row: row["episode_index"]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FUTURE_VERIFICATION_RESULTS={out_path}")
    print(f"QUALITY_SUMMARY={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
