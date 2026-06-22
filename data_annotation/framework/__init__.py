"""Confidence-arbitration annotation framework.

Schema-driven, detector-based subtask boundary fusion. Replaces the per-task
hardcoded OWNER table in fuse_annotations.py with derived, sigma-weighted
arbitration. See README.md in this directory."""

from .core import Candidate, Decision, EventSpec, TaskSchema, load_schema

__all__ = ["Candidate", "Decision", "EventSpec", "TaskSchema", "load_schema"]
