"""Public exceptions raised by the native Gmsh meshing backend."""

from __future__ import annotations

from fem.geometry.errors import GeometryError


class MeshCellShapeError(GeometryError):
    """Raised when an automatic mesh violates its top-cell contract."""


class MeshControlConflictError(GeometryError):
    """Raised when explicit topology controls conflict with automatic meshing."""


class MeshFieldOwnershipError(GeometryError):
    """Raised when a mesh field belongs to a different mesh runtime."""


class StaleMeshFieldError(GeometryError):
    """Raised when a mesh-field reference no longer denotes a live field."""


class StaleGmshMeshError(GeometryError):
    """Raised when a generated native mesh is no longer available to import."""


__all__ = [
    "MeshCellShapeError",
    "MeshControlConflictError",
    "MeshFieldOwnershipError",
    "StaleGmshMeshError",
    "StaleMeshFieldError",
]
