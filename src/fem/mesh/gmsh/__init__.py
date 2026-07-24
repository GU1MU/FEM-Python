"""Native Gmsh mesh controls and generation for scripted OCC geometry."""

from __future__ import annotations

from .errors import (
    MeshCellShapeError,
    MeshControlConflictError,
    MeshFieldOwnershipError,
    StaleGmshMeshError,
    StaleMeshFieldError,
)
from .mesher import Mesher
from .specs import AutoMeshSpec, MeshSpec
from .types import GmshMeshRef, MeshFieldRef


__all__ = [
    "AutoMeshSpec",
    "GmshMeshRef",
    "MeshCellShapeError",
    "MeshControlConflictError",
    "MeshFieldOwnershipError",
    "MeshFieldRef",
    "MeshSpec",
    "Mesher",
    "StaleGmshMeshError",
    "StaleMeshFieldError",
]
