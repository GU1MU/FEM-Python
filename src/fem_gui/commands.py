"""Typed values shared by GUI actions and public workflow commands."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
from numbers import Real
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from typing import Any, overload

from PySide6.QtCore import QCoreApplication, QThread

from fem.application import NativePart, SessionDelta, UNSET, Unset
from fem.application.results import ResultSourceKey, ScalarFieldSelection
from fem.geometry import NATIVE_GEOMETRY_TYPES, NativeGeometry
from fem.mesh.settings import MeshSettings

from .task_controller import BackgroundTaskState, TaskCompletion


class GuiCommandStatus(str, Enum):
    """Immediate disposition of one public GUI command."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class GuiCommandDiagnostic:
    """Small, presentation-neutral reason for rejecting a public command."""

    code: str
    message: str
    remediation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_text(self.code, "code"))
        object.__setattr__(
            self,
            "message",
            _required_text(self.message, "message"),
        )
        object.__setattr__(
            self,
            "remediation",
            _optional_text(self.remediation, "remediation"),
        )


@dataclass(frozen=True, slots=True)
class GuiCommandOutcome:
    """Detached metadata describing one accepted non-Session operation."""

    output_path: Path | None = None
    source: ResultSourceKey | None = None
    materialization_generation: int | None = None
    selection: ScalarFieldSelection | None = None
    record_count: int | None = None
    diagnostic_summary: str = ""

    def __post_init__(self) -> None:
        output_path = _optional_output_path(self.output_path)
        result_values = (
            self.source,
            self.materialization_generation,
            self.selection,
        )
        result_value_count = sum(value is not None for value in result_values)
        if result_value_count not in {0, len(result_values)}:
            raise ValueError(
                "source, materialization_generation, and selection "
                "must be supplied together"
            )
        if result_value_count:
            source = _owned_result_source(self.source)
            generation = _materialization_generation(self.materialization_generation)
            selection = _owned_scalar_selection(self.selection)
        else:
            source = None
            generation = None
            selection = None
        record_count = _optional_record_count(self.record_count)
        if record_count is not None and source is None:
            raise ValueError("record_count requires complete result provenance")
        if output_path is None and source is None:
            raise ValueError(
                "an outcome requires an output path or complete result provenance"
            )

        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "materialization_generation",
            generation,
        )
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "record_count", record_count)
        object.__setattr__(
            self,
            "diagnostic_summary",
            _stable_diagnostic_summary(self.diagnostic_summary),
        )


@dataclass(frozen=True, slots=True)
class ResultCsvExportSpec:
    """Exact accepted result identity for one scalar CSV export."""

    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _owned_result_source(self.source),
        )
        object.__setattr__(
            self,
            "materialization_generation",
            _materialization_generation(self.materialization_generation),
        )
        object.__setattr__(
            self,
            "selection",
            _owned_scalar_selection(self.selection),
        )


@dataclass(frozen=True, slots=True)
class ResultVtkExportSpec:
    """Exact accepted result identity and deformation for one VTK export."""

    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection
    deformation_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _owned_result_source(self.source),
        )
        object.__setattr__(
            self,
            "materialization_generation",
            _materialization_generation(self.materialization_generation),
        )
        object.__setattr__(
            self,
            "selection",
            _owned_scalar_selection(self.selection),
        )
        object.__setattr__(
            self,
            "deformation_scale",
            _finite_deformation_scale(self.deformation_scale),
        )


TaskCompletionObserver = Callable[[TaskCompletion], None]


class GuiCommandCompletion:
    """Thread-safe, observable terminal handle for one pending command.

    The handle retains a metadata-only copy of :class:`TaskCompletion` and,
    for successful commands, an independent sanitized outcome. Worker
    payloads are deliberately removed so the public command layer never
    becomes an owner of a Session snapshot, model, or result.
    """

    __slots__ = (
        "_callbacks",
        "_command_id",
        "_event",
        "_lock",
        "_outcome",
        "_task_id",
        "_terminal",
    )

    def __init__(
        self,
        command_id: int,
        *,
        task_id: int | None = None,
    ) -> None:
        self._command_id = _command_id(command_id)
        self._task_id = None if task_id is None else _task_id(task_id)
        self._event = Event()
        self._lock = Lock()
        self._terminal: TaskCompletion | None = None
        self._outcome: GuiCommandOutcome | None = None
        self._callbacks: list[TaskCompletionObserver] = []

    @property
    def command_id(self) -> int:
        return self._command_id

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def task_id(self) -> int | None:
        with self._lock:
            return self._task_id

    @property
    def terminal(self) -> TaskCompletion | None:
        with self._lock:
            return self._terminal

    @property
    def outcome(self) -> GuiCommandOutcome | None:
        with self._lock:
            return self._outcome

    def observe(self, callback: TaskCompletionObserver) -> None:
        """Observe the terminal value once, including late subscriptions."""

        if not callable(callback):
            raise TypeError("completion observer must be callable")
        terminal: TaskCompletion | None
        with self._lock:
            terminal = self._terminal
            if terminal is None:
                self._callbacks.append(callback)
                return
        _notify_observer(callback, terminal)

    def bind_task_id(self, task_id: int) -> None:
        """Bind the controller task returned by ``BackgroundTaskController.start``."""

        resolved = _task_id(task_id)
        with self._lock:
            if self._task_id is not None and self._task_id != resolved:
                raise RuntimeError("completion is already bound to another task")
            if self._terminal is not None and self._terminal.task_id != resolved:
                raise ValueError("terminal task_id does not match the controller task")
            self._task_id = resolved

    def complete(
        self,
        completion: TaskCompletion,
        *,
        outcome: GuiCommandOutcome | None = None,
    ) -> bool:
        """Install one controller terminal value, returning false if already done."""

        if type(completion) is not TaskCompletion:
            raise TypeError("completion must be a TaskCompletion")
        callbacks: tuple[TaskCompletionObserver, ...]
        with self._lock:
            if self._terminal is not None:
                return False
            resolved_outcome = _completion_outcome(completion, outcome)
            terminal = replace(completion, value=None)
            if self._task_id is not None and completion.task_id != self._task_id:
                raise ValueError("terminal task_id does not match the controller task")
            self._task_id = completion.task_id
            self._terminal = terminal
            self._outcome = resolved_outcome
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            self._event.set()
        for callback in callbacks:
            _notify_observer(callback, terminal)
        return True

    def wait(self, timeout: float | None = None) -> TaskCompletion | None:
        """Wait for terminal completion and return ``None`` on timeout."""

        resolved_timeout = _wait_timeout(timeout)
        application = QCoreApplication.instance()
        if application is None or QThread.currentThread() != application.thread():
            self._event.wait(resolved_timeout)
            return self.terminal

        deadline = None if resolved_timeout is None else monotonic() + resolved_timeout
        while not self._event.is_set():
            application.processEvents()
            if self._event.is_set():
                break
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            if remaining == 0.0:
                break
            self._event.wait(0.001 if remaining is None else min(0.001, remaining))
        application.processEvents()
        return self.terminal

    def result(self, timeout: float | None = None) -> TaskCompletion:
        """Wait for and require the terminal completion."""

        terminal = self.wait(timeout)
        if terminal is None:
            raise TimeoutError("GUI command did not reach a terminal state")
        return terminal


@dataclass(frozen=True, slots=True)
class GuiCommandReceipt:
    """Uniform immediate return value for every public GUI command."""

    command_id: int
    status: GuiCommandStatus
    delta: SessionDelta | None = None
    diagnostic: GuiCommandDiagnostic | None = None
    completion: GuiCommandCompletion | None = None
    outcome: GuiCommandOutcome | None = None

    def __post_init__(self) -> None:
        command_id = _command_id(self.command_id)
        if type(self.status) is not GuiCommandStatus:
            raise TypeError("status must be a GuiCommandStatus")
        if self.delta is not None and type(self.delta) is not SessionDelta:
            raise TypeError("delta must be a SessionDelta or None")
        if self.diagnostic is not None and (
            type(self.diagnostic) is not GuiCommandDiagnostic
        ):
            raise TypeError("diagnostic must be a GuiCommandDiagnostic or None")
        if self.completion is not None and (
            type(self.completion) is not GuiCommandCompletion
        ):
            raise TypeError("completion must be a GuiCommandCompletion or None")
        if self.outcome is not None and type(self.outcome) is not GuiCommandOutcome:
            raise TypeError("outcome must be a GuiCommandOutcome or None")

        if self.status is GuiCommandStatus.ACCEPTED:
            if (
                (self.delta is None) == (self.outcome is None)
                or (self.delta is not None and not self.delta.accepted)
                or self.diagnostic is not None
                or self.completion is not None
            ):
                raise ValueError(
                    "an accepted receipt requires exactly one accepted delta or outcome"
                )
        elif self.status is GuiCommandStatus.REJECTED:
            if (
                self.delta is not None
                or self.diagnostic is None
                or self.completion is not None
                or self.outcome is not None
            ):
                raise ValueError("a rejected receipt requires one diagnostic only")
        elif (
            self.delta is not None
            or self.diagnostic is not None
            or self.completion is None
            or self.outcome is not None
            or self.completion.command_id != command_id
        ):
            raise ValueError("a pending receipt requires its matching completion only")
        object.__setattr__(self, "command_id", command_id)
        if self.outcome is not None:
            object.__setattr__(self, "outcome", deepcopy(self.outcome))

    @classmethod
    @overload
    def accepted(
        cls,
        command_id: int,
        delta: SessionDelta,
        *,
        outcome: None = None,
    ) -> GuiCommandReceipt: ...

    @classmethod
    @overload
    def accepted(
        cls,
        command_id: int,
        delta: None = None,
        *,
        outcome: GuiCommandOutcome,
    ) -> GuiCommandReceipt: ...

    @classmethod
    def accepted(
        cls,
        command_id: int,
        delta: SessionDelta | None = None,
        *,
        outcome: GuiCommandOutcome | None = None,
    ) -> GuiCommandReceipt:
        return cls(
            command_id=command_id,
            status=GuiCommandStatus.ACCEPTED,
            delta=delta,
            outcome=outcome,
        )

    @classmethod
    def rejected(
        cls,
        command_id: int,
        diagnostic: GuiCommandDiagnostic,
    ) -> GuiCommandReceipt:
        return cls(
            command_id=command_id,
            status=GuiCommandStatus.REJECTED,
            diagnostic=diagnostic,
        )

    @classmethod
    def pending(
        cls,
        command_id: int,
        completion: GuiCommandCompletion,
    ) -> GuiCommandReceipt:
        return cls(
            command_id=command_id,
            status=GuiCommandStatus.PENDING,
            completion=completion,
        )


@dataclass(frozen=True, slots=True)
class NewNativeProjectCommand:
    """Inputs for the public new-native-project command."""

    name: str = "模型-1"
    expected_session_revision: int | None = None
    part_name: str = "部件-1"
    body_name: str = "实体-1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, "project name"),
        )
        object.__setattr__(
            self,
            "part_name",
            _required_text(self.part_name, "part name"),
        )
        object.__setattr__(
            self,
            "body_name",
            _required_text(self.body_name, "body name"),
        )
        object.__setattr__(
            self,
            "expected_session_revision",
            _optional_revision(self.expected_session_revision),
        )


@dataclass(frozen=True, slots=True)
class CloseSessionCommand:
    """Inputs for the public close-session command."""

    expected_session_revision: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_session_revision",
            _optional_revision(self.expected_session_revision),
        )


@dataclass(frozen=True, slots=True)
class NativeGeometryEdit:
    """One detached, revision-bound replacement of native geometry inputs."""

    base_session_revision: int
    parts: tuple[NativePart, ...]
    recipe: NativeGeometry | None
    mesh_settings: MeshSettings | None | Unset = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_session_revision",
            _revision(self.base_session_revision, "base_session_revision"),
        )
        parts = deepcopy(tuple(self.parts))
        if any(type(part) is not NativePart for part in parts):
            raise TypeError("parts must contain only NativePart values")
        recipe = deepcopy(self.recipe)
        if recipe is not None and not isinstance(
            recipe,
            NATIVE_GEOMETRY_TYPES,
        ):
            raise TypeError("recipe must be a NativeGeometry or None")
        if recipe is None and parts:
            raise ValueError("a cleared native geometry must not retain parts")
        if recipe is not None and not parts:
            raise ValueError("native geometry requires at least one part")
        mesh_settings = deepcopy(self.mesh_settings)
        if (
            mesh_settings is not None
            and type(mesh_settings) is not MeshSettings
            and type(mesh_settings) is not Unset
        ):
            raise TypeError("mesh_settings must be MeshSettings, None, or UNSET")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "recipe", recipe)
        object.__setattr__(self, "mesh_settings", mesh_settings)


@dataclass(frozen=True, slots=True)
class MeshInputEdit:
    """One detached global/cell-shape/local-control mesh input replacement."""

    base_session_revision: int
    settings: MeshSettings | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_session_revision",
            _revision(self.base_session_revision, "base_session_revision"),
        )
        settings = deepcopy(self.settings)
        if settings is not None and type(settings) is not MeshSettings:
            raise TypeError("settings must be a MeshSettings or None")
        object.__setattr__(self, "settings", settings)


def _command_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("command_id must be a positive integer")
    return int(value)


def _task_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("task_id must be a positive integer")
    return int(value)


def _revision(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _optional_revision(value: Any) -> int | None:
    if value is None:
        return None
    return _revision(value, "expected_session_revision")


def _optional_output_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, Path):
        raise TypeError("output_path must be a Path or None")
    path = Path(value)
    if (
        not path.name.strip()
        or path.name in {".", ".."}
        or any(not character.isprintable() for character in str(path))
    ):
        raise ValueError("output_path must identify a file")
    return path


def _owned_result_source(value: Any) -> ResultSourceKey:
    if type(value) is not ResultSourceKey:
        raise TypeError("source must be a ResultSourceKey")
    return deepcopy(value)


def _owned_scalar_selection(value: Any) -> ScalarFieldSelection:
    if type(value) is not ScalarFieldSelection:
        raise TypeError("selection must be a ScalarFieldSelection")
    return deepcopy(value)


def _materialization_generation(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("materialization_generation must be an integer")
    if value < 0:
        raise ValueError("materialization_generation must be a non-negative integer")
    return value


def _optional_record_count(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("record_count must be an integer or None")
    if value < 0:
        raise ValueError("record_count must be non-negative")
    return value


def _stable_diagnostic_summary(value: Any) -> str:
    if type(value) is not str:
        raise TypeError("diagnostic_summary must be a string")
    if any(
        not character.isprintable() and not character.isspace() for character in value
    ):
        raise ValueError("diagnostic_summary must not contain control characters")
    return " ".join(value.split())


def _finite_deformation_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("deformation_scale must be a real number")
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("deformation_scale must be finite")
    return scale


def _completion_outcome(
    completion: TaskCompletion,
    explicit: GuiCommandOutcome | None,
) -> GuiCommandOutcome | None:
    if explicit is not None and type(explicit) is not GuiCommandOutcome:
        raise TypeError("outcome must be a GuiCommandOutcome or None")
    value_outcome = (
        completion.value if type(completion.value) is GuiCommandOutcome else None
    )
    if explicit is not None and value_outcome is not None:
        raise ValueError("completion outcome must come from one explicit source only")
    outcome = explicit if explicit is not None else value_outcome
    if outcome is not None and completion.state is not BackgroundTaskState.SUCCEEDED:
        raise ValueError("only a succeeded completion may carry a command outcome")
    return None if outcome is None else deepcopy(outcome)


def _required_text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value.strip()


def _wait_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("timeout must be a finite non-negative number or None")
    return float(value)


def _notify_observer(
    callback: TaskCompletionObserver,
    terminal: TaskCompletion,
) -> None:
    try:
        callback(terminal)
    except Exception:
        logging.exception("GUI command completion observer failed")


__all__ = [
    "CloseSessionCommand",
    "GuiCommandCompletion",
    "GuiCommandDiagnostic",
    "GuiCommandOutcome",
    "GuiCommandReceipt",
    "GuiCommandStatus",
    "MeshInputEdit",
    "NativeGeometryEdit",
    "NewNativeProjectCommand",
    "ResultCsvExportSpec",
    "ResultVtkExportSpec",
]
