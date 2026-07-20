"""Public exceptions raised by scripted geometry."""

from __future__ import annotations


class GeometryError(RuntimeError):
    """Base error raised by the scripted Gmsh geometry facade."""


class GeometryStateError(GeometryError):
    """Raised when a geometry operation is invalid in the current state."""


class EntityOwnershipError(GeometryError):
    """Raised when an entity belongs to a different geometry model."""


class StaleEntityError(GeometryError):
    """Raised when an entity reference no longer denotes a live OCC entity."""


__all__ = [
    "EntityOwnershipError",
    "GeometryError",
    "GeometryStateError",
    "StaleEntityError",
]
