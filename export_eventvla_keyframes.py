#!/usr/bin/env python3
"""Export black-smash subtask boundaries as EventVLA keyframe metadata.

EventVLA reads per-episode keyframes from ``meta/episodes.jsonl`` using either
``keyframe_steps`` or ``inspect_keyframe_steps``.  This script bridges the
annotation pipeline in this repository to that format by copying the fused
critical points into those episode metadata fields.

Typical usage:

  python export_eventvla_keyframes.py \
    --annotations annotations_fused_05 \
    --dataset /home/hillbot/black_smash_05_eventvla \
    --in-place

By default all six critical points are exported.  That is usually the right
training target for EventVLA event supervision; the model config can still cap
visible memory images with max_keyframes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_POINT_NAMES = (
    "grasp_tube",
    "start_pour",
    "release_tube",
    "grasp_pestle",
    "start_grind",
    "lift_pestle",
)

PRESETS = {
    "all": tuple(range(6)),
    # A compact set of visually meaningful memory events if you want <=4 labels.
    "eventvla4": (1, 2, 4, 5),
    "pour_grind": (1, 4),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_episode_id(path: Path) -> int:
    name = path.name
    if not name.startswith("ep") or "_subtasks" not in name:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(name.split("_", 1)[0][2:])


def parse_point_indices(raw: str) -> tuple[int, ...]:
    key = raw.strip().lower()
    if key in PRESETS:
        return PRESETS[key]
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 1 or idx > 6:
            raise ValueError("--points custom indices must be 1..6")
        values.append(idx - 1)
    if not values:
        raise ValueError("--points produced an empty selection")
    return tuple(values)


def annotation_keyframes(doc: dict[str, Any], point_indices: tuple[int, ...]) -> list[int]:
    cps = doc.get("critical_points")
    if not isinstance(cps, list) or len(cps) < 6:
        raise ValueError(f"annotation episode {doc.get('episode_index')} has no 6 critical_points")
    keyframes = sorted({int(cps[idx]) for idx in point_indices})
    return keyframes


def clamp_and_validate(
    *,
    episode_index: int,
    keyframes: list[int],
    length: int,
    strict: bool,
) -> tuple[list[int], list[str]]:
    warnings: list[str] = []
    cleaned: list[int] = []
    for step in keyframes:
        if 0 <= step < length:
            cleaned.append(step)
            continue
        message = f"episode {episode_index}: keyframe {step} outside [0,{length})"
        if strict:
            raise ValueError(message)
        clipped = min(max(int(step), 0), max(length - 1, 0))
        warnings.append(f"{message}; clipped to {clipped}")
        cleaned.append(clipped)
    cleaned = sorted(set(cleaned))
    if not cleaned:
        message = f"episode {episode_index}: no keyframes after validation"
        if strict:
            raise ValueError(message)
        warnings.append(message)
    return cleaned, warnings


def build_annotation_map(annotation_dir: Path, point_indices: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    annotation_map: dict[int, dict[str, Any]] = {}
    paths = sorted(annotation_dir.glob("ep*_subtasks.json"))
    if not paths:
        raise FileNotFoundError(f"No ep*_subtasks.json files found under {annotation_dir}")
    for path in paths:
        doc = load_json(path)
        ep = int(doc.get("episode_index", parse_episode_id(path)))
        keyframes = annotation_keyframes(doc, point_indices)
        annotation_map[ep] = {
            "keyframe_steps": keyframes,
            "critical_points": [int(x) for x in doc["critical_points"]],
            "critical_names": doc.get("critical_names", list(DEFAULT_POINT_NAMES)),
            "source_file": str(path),
        }
    return annotation_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="annotations_fused_* or annotations_state_* dir")
    parser.add_argument("--dataset", type=Path, required=True, help="EventVLA LeRobot dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output episodes.jsonl path. Defaults to <dataset>/meta/episodes_eventvla_keyframes.jsonl.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite <dataset>/meta/episodes.jsonl")
    parser.add_argument("--backup", action="store_true", default=True, help="Backup episodes.jsonl before --in-place")
    parser.add_argument("--no-backup", action="store_false", dest="backup")
    parser.add_argument(
        "--points",
        default="all",
        help="Keyframe selection: all, eventvla4, pour_grind, or 1-based comma list like 2,3,5,6.",
    )
    parser.add_argument(
        "--annotation-offset",
        type=int,
        default=0,
        help="Add this offset when mapping dataset episode_index -> annotation episode id.",
    )
    parser.add_argument(
        "--write-inspect-alias",
        action="store_true",
        default=True,
        help="Also write inspect_keyframe_steps for compatibility.",
    )
    parser.add_argument("--no-inspect-alias", action="store_false", dest="write_inspect_alias")
    parser.add_argument("--strict", action="store_true", help="Fail on missing annotations or out-of-range steps")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    annotation_dir = args.annotations.expanduser().resolve()
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing EventVLA episodes file: {episodes_path}")
    point_indices = parse_point_indices(args.points)
    annotation_map = build_annotation_map(annotation_dir, point_indices)
    rows = read_jsonl(episodes_path)

    updated: list[dict[str, Any]] = []
    missing: list[int] = []
    warnings: list[str] = []
    nonempty = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        annotation_episode = episode_index + int(args.annotation_offset)
        annotation = annotation_map.get(annotation_episode)
        out = dict(row)
        if annotation is None:
            missing.append(episode_index)
            if args.strict:
                raise ValueError(f"episode {episode_index}: missing annotation ep{annotation_episode:03d}")
            keyframes: list[int] = []
        else:
            length = int(out.get("length", 0))
            keyframes, row_warnings = clamp_and_validate(
                episode_index=episode_index,
                keyframes=annotation["keyframe_steps"],
                length=length,
                strict=args.strict,
            )
            warnings.extend(row_warnings)
            out["eventvla_keyframe_source"] = {
                "annotation_episode": annotation_episode,
                "points": args.points,
                "critical_points": annotation["critical_points"],
                "critical_names": annotation["critical_names"],
            }
        out["keyframe_steps"] = keyframes
        if args.write_inspect_alias:
            out["inspect_keyframe_steps"] = keyframes
        if keyframes:
            nonempty += 1
        updated.append(out)

    if args.in_place:
        output_path = episodes_path
        if args.backup:
            backup_path = episodes_path.with_suffix(episodes_path.suffix + ".bak")
            shutil.copy2(episodes_path, backup_path)
    else:
        output_path = args.output or (dataset / "meta" / "episodes_eventvla_keyframes.jsonl")
    write_jsonl(output_path, updated)

    summary = {
        "dataset": str(dataset),
        "annotations": str(annotation_dir),
        "output": str(output_path),
        "episodes": len(updated),
        "episodes_with_keyframes": nonempty,
        "missing_annotations": missing,
        "point_selection": args.points,
        "point_indices_zero_based": list(point_indices),
        "warnings": warnings,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
            file.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing and not args.strict:
        print("warning: some episodes had no annotation; they received empty keyframe_steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
