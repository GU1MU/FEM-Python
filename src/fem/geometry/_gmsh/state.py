"""Private state machine for one scripted Gmsh geometry model."""

from __future__ import annotations

from collections.abc import Collection
from enum import Enum, auto

from ..errors import GeometryStateError


class _State(Enum):
    NEW = auto()
    BUILDING_GEOMETRY = auto()
    CONFIGURING_MESH = auto()
    MESHED = auto()
    MESH_FAILED = auto()
    CLOSED = auto()


_QUERY_STATES = frozenset(
    {_State.BUILDING_GEOMETRY, _State.CONFIGURING_MESH, _State.MESH_FAILED}
)
_GEOMETRY_MUTATION_STATES = frozenset({_State.BUILDING_GEOMETRY})
_MESH_CONTROL_STATES = frozenset({_State.CONFIGURING_MESH})


class _ModelStateMachine:
    """Own and validate the public lifecycle state of one geometry model."""

    __slots__ = ("_model_name", "_state")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._state = _State.NEW

    @property
    def state(self) -> _State:
        """Return the current lifecycle state."""
        return self._state

    def check(self, operation: str, allowed: Collection[_State]) -> None:
        """Raise the established contextual error unless the state is allowed."""
        if self._state not in allowed:
            allowed_text = ", ".join(
                state.name for state in sorted(allowed, key=lambda item: item.name)
            )
            raise self.error(
                operation,
                f"state {self._state.name} does not permit this operation "
                f"(expected {allowed_text})",
            )

    def enter_geometry(self) -> None:
        """Move a newly entered model into its geometry-building phase."""
        if self._state is not _State.NEW:
            raise self.error("context entry", "model context is not new")
        self._state = _State.BUILDING_GEOMETRY

    def begin_mesh_configuration(self, operation: str = "Mesher binding") -> None:
        """Seal ordinary OCC mutation and begin mesh configuration."""
        self.check(operation, _GEOMETRY_MUTATION_STATES)
        self._state = _State.CONFIGURING_MESH

    def mark_meshed(self, operation: str = "mesh generation") -> None:
        """Record successful completion of the sole native mesh attempt."""
        self.check(operation, _MESH_CONTROL_STATES)
        self._state = _State.MESHED

    def mark_mesh_failed(self, operation: str = "mesh generation") -> None:
        """Record terminal failure after native meshing work began."""
        self.check(operation, _MESH_CONTROL_STATES)
        self._state = _State.MESH_FAILED

    def close(self) -> None:
        """Forbid public operations, including after an earlier close."""
        self._state = _State.CLOSED

    def error(self, operation: str, detail: str) -> GeometryStateError:
        """Create the established contextual geometry-state exception."""
        return GeometryStateError(
            f"geometry model {self._model_name!r}: {operation} failed because {detail}"
        )
