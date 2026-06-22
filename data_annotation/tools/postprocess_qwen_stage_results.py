#!/usr/bin/env python3
"""Normalize Qwen stage JSON results so intervals can be consumed by data conversion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED_STAGE_NAMES = {"reach", "grasp", "transport", "place", "release", "adjust", "done", "global"}
STAGE_NAME_MAP = {
    "lift": "transport",
    "move": "transport",
    "moving": "transport",
    "carry": "transport",
    "open": "release",
    "finish": "done",
    "complete": "done",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def normalize_stages(stages: list[dict], episode_length: int) -> tuple[list[dict], dict]:
    fixed: list[dict] = []
    prev_end = -1
    changed = False
    stage_name_changed = 0
    for i, stage in enumerate(stages):
        item = dict(stage)
        raw_name = str(item.get("name", "adjust")).strip().lower()
        normalized_name = STAGE_NAME_MAP.get(raw_name, raw_name)
        if normalized_name not in ALLOWED_STAGE_NAMES:
            normalized_name = "adjust"
        if normalized_name != raw_name or item.get("name") != normalized_name:
            item["original_name"] = item.get("name")
            item["name"] = normalized_name
            stage_name_changed += 1
            changed = True
        try:
            start = int(item.get("start_t", prev_end + 1))
            end = int(item.get("end_t", start))
        except Exception:
            start = prev_end + 1
            end = start
            changed = True
        desired_start = prev_end + 1
        if start != desired_start:
            start = desired_start
            changed = True
        if i == len(stages) - 1:
            desired_end = episode_length - 1
            if end != desired_end:
                end = desired_end
                changed = True
        else:
            end = max(start, min(end, episode_length - 1))
        if start >= episode_length:
            changed = True
            break
        if end < start:
            end = start
            changed = True
        item["start_t"] = start
        item["end_t"] = end
        fixed.append(item)
        prev_end = end
    if not fixed:
        fixed = [
            {
                "name": "global",
                "start_t": 0,
                "end_t": episode_length - 1,
                "instruction": "Follow the original global task instruction.",
                "reason": "Fallback because no valid model stages were parsed.",
            }
        ]
        changed = True
    elif fixed[-1]["end_t"] != episode_length - 1:
        fixed[-1]["end_t"] = episode_length - 1
        changed = True

    coverage_ok = bool(fixed and fixed[0]["start_t"] == 0 and fixed[-1]["end_t"] == episode_length - 1)
    for left, right in zip(fixed, fixed[1:]):
        coverage_ok = coverage_ok and int(right["start_t"]) == int(left["end_t"]) + 1
    return fixed, {
        "changed": changed,
        "coverage_ok": coverage_ok,
        "stage_count": len(fixed),
        "stage_name_changed": stage_name_changed,
    }


def main() -> None:
    args = parse_args()
    input_path = args.run_dir / "stage_annotations.jsonl"
    output_path = args.run_dir / "stage_annotations_normalized.jsonl"
    csv_path = args.run_dir / "summary_normalized.csv"
    records = []
    with input_path.open(encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as out:
        for line in f:
            record = json.loads(line)
            parsed = record.get("parsed_response") or {}
            stages = parsed.get("stages") or []
            normalized, info = normalize_stages(stages, int(record["episode_length"]))
            record["normalized_stages"] = normalized
            record["normalized_validation"] = info
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(
                {
                    "episode_index": record["episode_index"],
                    "episode_length": record["episode_length"],
                    "raw_stage_count": len(stages),
                    "normalized_stage_count": info["stage_count"],
                    "normalized_coverage_ok": info["coverage_ok"],
                    "changed": info["changed"],
                    "stage_name_changed": info["stage_name_changed"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"NORMALIZED_JSONL={output_path}")
    print(f"NORMALIZED_CSV={csv_path}")


if __name__ == "__main__":
    main()
