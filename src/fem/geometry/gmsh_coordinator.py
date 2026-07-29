"""Process-wide ownership gate for every in-process Gmsh lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock, get_ident
from typing import Callable


class GmshExecutionCancelled(RuntimeError):
    """Raised while a queued owner is cancelled before it acquires Gmsh."""


@dataclass(frozen=True, slots=True)
class GmshOwnershipSnapshot:
    owner_thread_id: int | None
    owner_operation: str | None
    depth: int


class _GmshExecutionLease:
    __slots__ = ("_coordinator", "_released", "operation")

    def __init__(
        self,
        coordinator: GmshExecutionCoordinator,
        operation: str,
    ) -> None:
        self._coordinator = coordinator
        self.operation = operation
        self._released = False

    def __enter__(self) -> _GmshExecutionLease:
        return self

    def __exit__(self, *_args: object) -> bool:
        self.release()
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._coordinator._release()


class GmshExecutionCoordinator:
    """A cancellable re-entrant mutex independent of GUI busy state."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state_lock = RLock()
        self._owner_thread_id: int | None = None
        self._owner_operation: str | None = None
        self._depth = 0

    def acquire(
        self,
        operation: str,
        *,
        cancelled: Event | Callable[[], bool] | None = None,
        poll_interval: float = 0.05,
    ) -> _GmshExecutionLease:
        normalized = str(operation).strip()
        if not normalized:
            raise ValueError("Gmsh operation must be non-empty")
        if poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive")
        current_thread = get_ident()
        while True:
            if _is_cancelled(cancelled):
                raise GmshExecutionCancelled(
                    f"cancelled while waiting for Gmsh ownership: {normalized}"
                )
            if self._lock.acquire(timeout=float(poll_interval)):
                break
        with self._state_lock:
            if self._depth == 0:
                self._owner_thread_id = current_thread
                self._owner_operation = normalized
            elif self._owner_thread_id != current_thread:
                self._lock.release()
                raise RuntimeError("Gmsh ownership state is inconsistent")
            self._depth += 1
        return _GmshExecutionLease(self, normalized)

    def snapshot(self) -> GmshOwnershipSnapshot:
        with self._state_lock:
            return GmshOwnershipSnapshot(
                self._owner_thread_id,
                self._owner_operation,
                self._depth,
            )

    def _release(self) -> None:
        current_thread = get_ident()
        with self._state_lock:
            if self._depth <= 0 or self._owner_thread_id != current_thread:
                raise RuntimeError("Gmsh lease released by a non-owner")
            self._depth -= 1
            if self._depth == 0:
                self._owner_thread_id = None
                self._owner_operation = None
        self._lock.release()


def _is_cancelled(value: Event | Callable[[], bool] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, Event):
        return value.is_set()
    if callable(value):
        return bool(value())
    raise TypeError("cancelled must be a threading.Event, callable, or None")


PROCESS_GMSH_COORDINATOR = GmshExecutionCoordinator()


__all__ = [
    "GmshExecutionCancelled",
    "GmshExecutionCoordinator",
    "GmshOwnershipSnapshot",
    "PROCESS_GMSH_COORDINATOR",
]
