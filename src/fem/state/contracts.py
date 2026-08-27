"""State contracts shared by materials, contact and analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True, order=True)
class StateKey:
    """Stable typed identity for one history-bearing local point."""

    namespace: str
    entity_id: int
    point_id: int

    def __post_init__(self) -> None:
        namespace = str(self.namespace).strip()
        if not namespace:
            raise ValueError("state namespace must be nonblank")
        for name in ("entity_id", "point_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"state {name} must be an integer")
            if value < 1:
                raise ValueError(f"state {name} must be >= 1")
        object.__setattr__(self, "namespace", namespace)

    @classmethod
    def material_point(
        cls,
        entity_id: int,
        point_id: int,
        *,
        namespace: str = "mechanics.material",
    ) -> "StateKey":
        return cls(namespace, entity_id, point_id)


@runtime_checkable
class StateManager(Protocol):
    """Transactional store for committed and trial local states."""

    def register(self, key: StateKey, state: Mapping[str, Any] | None = None) -> None:
        """Register one local state identity."""

    def committed(self, key: StateKey) -> Mapping[str, Any]:
        """Return a detached committed state view."""

    def stage(self, key: StateKey, state: Mapping[str, Any]) -> None:
        """Stage a trial state for the current increment."""

    def begin_increment(self) -> None:
        """Discard stale trials and start a new increment transaction."""

    def commit(self) -> None:
        """Commit all staged trial states."""

    def rollback(self) -> None:
        """Discard all staged trial states."""


__all__ = ["StateKey", "StateManager"]
