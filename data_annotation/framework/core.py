"""
Core types for the confidence-arbitration annotation framework.

This replaces the hardcoded `OWNER = ["state","vlm",...]` table in
`fuse_annotations.py` with data: every detector reports a *candidate* boundary
together with its own localization uncertainty (sigma, in frames). The arbiter
(see `arbitrate.py`) derives which source "owns" each critical point from those
sigmas plus a state-skeleton safety rule, instead of a per-task hand-written
ownership list. A task is described by a `TaskSchema` (JSON), so the taxonomy is
no longer baked into Python.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Localization uncertainty, in frames, used when a detector does not override it.
# Smaller sigma == more trustworthy timing for that event type.
DEFAULT_SIGMA = 10.0


@dataclass
class Candidate:
    """One detector's proposal for a single critical point."""

    frame: int
    sigma: float                 # localization uncertainty in FRAMES (smaller = better)
    detector: str                # detector name, e.g. "gripper", "vlm", "orientation"
    modality: str                # "state" | "vision"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSpec:
    """One critical point declared by the task schema.

    `type` selects which detectors are eligible; it intentionally does NOT name
    a modality (that is what the arbiter derives)."""

    id: str                      # "p1"
    index: int                   # 0-based position among the critical points
    type: str                    # "gripper_close" | "gripper_open" | "orientation_change" | "motion_regime"
    name: str                    # "grasp_tube"
    params: dict[str, Any] = field(default_factory=dict)   # e.g. {"arm": "tube"}


@dataclass
class Decision:
    """The arbiter's resolved value for one event."""

    event: EventSpec
    frame: int | None
    sigma: float
    source: str                  # detector that owns the value (lowest-sigma handler)
    needs_review: bool
    candidates: list[Candidate] = field(default_factory=list)
    note: str = ""


@dataclass
class TaskSchema:
    """Declarative task description loaded from JSON (see schemas/black_smash.json)."""

    task: str
    task_description: str
    fps: int
    subtasks: list[str]               # one label per subtask (len == n_events + 1)
    events: list[EventSpec]
    state_layout: dict[str, Any]      # gripper dims etc., consumed by signal detectors

    @property
    def crit_names(self) -> list[str]:
        return [e.name for e in self.events]

    @property
    def n_events(self) -> int:
        return len(self.events)


def load_schema(path: str | Path) -> TaskSchema:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    events = [
        EventSpec(
            id=e["id"],
            index=i,
            type=e["type"],
            name=e["name"],
            params=e.get("params", {}),
        )
        for i, e in enumerate(raw["events"])
    ]
    subtasks = raw["subtasks"]
    if len(subtasks) != len(events) + 1:
        raise ValueError(
            f"schema '{raw['task']}': {len(subtasks)} subtasks but {len(events)} events "
            f"(expected {len(events) + 1} subtasks = n_events + 1)"
        )
    return TaskSchema(
        task=raw["task"],
        task_description=raw.get("task_description", ""),
        fps=int(raw.get("fps", 30)),
        subtasks=subtasks,
        events=events,
        state_layout=raw.get("state_layout", {}),
    )
