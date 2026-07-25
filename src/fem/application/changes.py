"""Domain deltas emitted by the headless application session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .revisions import TokenStatus


class ChangeKind(str, Enum):
    """One observable category changed by a session transition."""

    SESSION = "session"
    SOURCE = "source"
    PROJECT_INPUTS = "project_inputs"
    GEOMETRY = "geometry"
    MESH_SETTINGS = "mesh_settings"
    NAMED_REGIONS = "named_regions"
    MODEL = "model"
    DEFINITIONS = "definitions"
    VALIDATIONS = "validations"
    RUNS = "runs"
    DISPLAYED_RESULT = "displayed_result"
    SAVED_STATE = "saved_state"


class ArtifactKind(str, Enum):
    """Derived state invalidated by a session transition."""

    MODEL = "model"
    VALIDATIONS = "validations"
    RUNS = "runs"
    RESULTS = "results"
    DISPLAYED_RESULT = "displayed_result"
    TASKS = "tasks"


@dataclass(frozen=True, slots=True)
class SessionDelta:
    """Ordered, UI-agnostic description of one accepted transition.

    Stale asynchronous callbacks also return a delta.  Such a delta has
    ``accepted=False``, carries the unchanged session revision, and has empty
    change/invalidation sets.
    """

    session_revision: int
    changed: frozenset[ChangeKind] = frozenset()
    invalidated: frozenset[ArtifactKind] = frozenset()
    reason: str = ""
    accepted: bool = True
    token_status: TokenStatus | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_revision", int(self.session_revision))
        object.__setattr__(self, "changed", frozenset(self.changed))
        object.__setattr__(self, "invalidated", frozenset(self.invalidated))
        object.__setattr__(self, "reason", str(self.reason))

