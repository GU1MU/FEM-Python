"""Detached values shared by GUI authoring ports and tool handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class AgentProposalPreview:
    """GUI-only detached tessellation bound to one proven proposal Recipe."""

    dimension: int
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    recipe_digest: str
    proof_digest: str


class AgentPreflightState(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"
