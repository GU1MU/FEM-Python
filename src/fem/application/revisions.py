"""Revision-bound artifacts and asynchronous task tokens."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .results.data import ResultMaterializationSnapshot
    from .results.fields import FieldMaterializationKey
    from .runs import ResultRecord


class TokenStatus(str, Enum):
    """Result of checking an asynchronous task token."""

    CURRENT = "current"
    STALE_SESSION = "stale_session"
    STALE_REVISION = "stale_revision"
    STALE_ARTIFACT = "stale_artifact"
    STALE_STEP = "stale_step"
    STALE_RUN = "stale_run"
    STALE_RESULT = "stale_result"
    WRONG_KIND = "wrong_kind"
    UNKNOWN_TASK = "unknown_task"
    ALREADY_COMPLETED = "already_completed"
    INVALID_STATE = "invalid_state"

    @property
    def is_current(self) -> bool:
        return self is TokenStatus.CURRENT


def new_identity(prefix: str) -> str:
    """Return a readable globally unique identity."""

    return f"{prefix}-{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class TaskToken:
    """Immutable proof of the state on which a background task depends."""

    session_id: str
    task_id: str
    task_kind: str
    dependency_revisions: tuple[tuple[str, int], ...] = ()
    artifact_id: str | None = None
    step_name: str | None = None
    run_id: str | None = None
    result_id: str | None = None

    def __post_init__(self) -> None:
        dependencies = tuple(
            sorted(
                (
                    (str(name), int(revision))
                    for name, revision in self.dependency_revisions
                ),
                key=lambda item: item[0],
            )
        )
        if len({name for name, _ in dependencies}) != len(dependencies):
            raise ValueError("task dependency revision names must be unique")
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "task_kind", str(self.task_kind))
        object.__setattr__(self, "dependency_revisions", dependencies)
        if self.artifact_id is not None:
            object.__setattr__(self, "artifact_id", str(self.artifact_id))
        if self.step_name is not None:
            object.__setattr__(self, "step_name", str(self.step_name))
        if self.run_id is not None:
            object.__setattr__(self, "run_id", str(self.run_id))
        if self.result_id is not None:
            object.__setattr__(self, "result_id", str(self.result_id))


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One model owned by a session and bound to its input revisions."""

    session_id: str
    artifact_id: str
    model_revision: int
    mesh_input_revision: int | None
    source_kind: str
    model: Any


@dataclass(frozen=True, slots=True)
class ImportTaskSnapshot:
    token: TaskToken
    source_path: Path


@dataclass(frozen=True, slots=True)
class MeshTaskSnapshot:
    token: TaskToken
    model_name: str
    geometry_recipe: Any
    mesh_settings: Any
    parts: tuple[Any, ...]
    feature_history: tuple[Any, ...]
    named_regions: tuple[Any, ...]
    material_definitions: tuple[Any, ...]
    section_definitions: tuple[Any, ...]
    region_assignments: tuple[Any, ...]
    analysis_definitions: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ValidationTaskSnapshot:
    token: TaskToken
    model: Any
    step_name: str


@dataclass(frozen=True, slots=True)
class SolveTaskSnapshot:
    token: TaskToken
    model: Any
    step_name: str
    run_name: str
    run_id: str
    result_id: str
    delta: Any | None = None
    prepared_system: Any | None = None


@dataclass(frozen=True, slots=True)
class ResultTaskSnapshot:
    token: TaskToken
    run_id: str
    record: Any


@dataclass(frozen=True, slots=True)
class ResultMaterializationTaskSnapshot:
    """Detached provider input for one generation-bound field recovery."""

    token: TaskToken
    run_id: str
    record: ResultRecord
    field_keys: tuple[FieldMaterializationKey, ...]

    @property
    def materialization(self) -> ResultMaterializationSnapshot:
        return self.record.materialization

