#!/usr/bin/env python3
"""Prepare batch self-check samples for predictive stage annotations.

This script does not call any API. It creates a JSONL file describing which
future frames should be checked after all episodes have been labeled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-name", default="prediction_self_check_samples.jsonl")
    return parser.parse_args()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def main() -> None:
    args = parse_args()
    input_path = args.run_dir / "stage_annotations_normalized.jsonl"
    output_path = args.run_dir / args.output_name
    rows = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            episode_length = int(record["episode_length"])
            for stage_index, stage in enumerate(record.get("normalized_stages", [])):
                start = int(stage["start_t"])
                end = int(stage["end_t"])
                mid = (start + end) // 2
                after = clamp(end + 5, 0, episode_length - 1)
                check_frames = []
                for t in [start, mid, end, after]:
                    t = clamp(t, 0, episode_length - 1)
                    if t not in check_frames:
                        check_frames.append(t)
                rows.append(
                    {
                        "episode_index": record["episode_index"],
                        "stage_index": stage_index,
                        "stage_name": stage.get("name"),
                        "stage_interval": [start, end],
                        "prediction_prompt": stage.get("prediction_prompt") or stage.get("instruction"),
                        "expected_future_observation": stage.get("expected_future_observation"),
                        "check_frames": check_frames,
                        "self_check_question": (
                            "Compare the stage start, middle, end, and after-stage frames. "
                            "Do they show that the expected_future_observation happened? "
                            "Return JSON with pass=true/false, score=0..1, and a short reason."
                        ),
                    }
                )

    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"SELF_CHECK_SAMPLES={output_path}")
    print(f"SAMPLES={len(rows)}")


if __name__ == "__main__":
    main()
