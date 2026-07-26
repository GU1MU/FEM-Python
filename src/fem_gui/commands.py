"""Typed values shared by GUI actions and public workflow commands."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
from threading import Event, Lock
from time import monotonic
from typing import Any

from PySide6.QtCore import QCoreApplication, QThread

from fem.application import NativePart, SessionDelta, UNSET, Unset
from fem.geometry import NATIVE_GEOMETRY_TYPES, NativeGeometry
from fem.mesh.settings import MeshSettings

from .task_controller import TaskCompletion


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


TaskCompletionObserver = Callable[[TaskCompletion], None]


class GuiCommandCompletion:
    """Thread-safe, observable terminal handle for one pending command.

    The handle retains a metadata-only copy of :class:`TaskCompletion`.
    Worker payloads are deliberately removed so the public command layer never
    becomes an owner of a Session snapshot, model, or result.
    """

    __slots__ = (
        "_callbacks",
        "_command_id",
        "_event",
        "_lock",
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

    def complete(self, completion: TaskCompletion) -> bool:
        """Install one controller terminal value, returning false if already done."""

        if type(completion) is not TaskCompletion:
            raise TypeError("completion must be a TaskCompletion")
        terminal = replace(completion, value=None)
        callbacks: tuple[TaskCompletionObserver, ...]
        with self._lock:
            if self._terminal is not None:
                return False
            if self._task_id is not None and completion.task_id != self._task_id:
                raise ValueError("terminal task_id does not match the controller task")
            self._task_id = completion.task_id
            self._terminal = terminal
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
        if (
            application is None
            or QThread.currentThread() != application.thread()
        ):
            self._event.wait(resolved_timeout)
            return self.terminal

        deadline = (
            None
            if resolved_timeout is None
            else monotonic() + resolved_timeout
        )
        while not self._event.is_set():
            application.processEvents()
            if self._event.is_set():
                break
            remaining = (
                None if deadline is None else max(0.0, deadline - monotonic())
            )
            if remaining == 0.0:
                break
            self._event.wait(
                0.001 if remaining is None else min(0.001, remaining)
            )
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

        if self.status is GuiCommandStatus.ACCEPTED:
            if (
                self.delta is None
                or not self.delta.accepted
                or self.diagnostic is not None
                or self.completion is not None
            ):
                raise ValueError("an accepted receipt requires one accepted delta only")
        elif self.status is GuiCommandStatus.REJECTED:
            if (
                self.delta is not None
                or self.diagnostic is None
                or self.completion is not None
            ):
                raise ValueError("a rejected receipt requires one diagnostic only")
        elif (
            self.delta is not None
            or self.diagnostic is not None
            or self.completion is None
            or self.completion.command_id != command_id
        ):
            raise ValueError("a pending receipt requires its matching completion only")
        object.__setattr__(self, "command_id", command_id)

    @classmethod
    def accepted(
        cls,
        command_id: int,
        delta: SessionDelta,
    ) -> GuiCommandReceipt:
        return cls(
            command_id=command_id,
            status=GuiCommandStatus.ACCEPTED,
            delta=delta,
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

    name: str = "Model-1"
    expected_session_revision: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, "project name"),
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
    "GuiCommandReceipt",
    "GuiCommandStatus",
    "MeshInputEdit",
    "NativeGeometryEdit",
    "NewNativeProjectCommand",
]
