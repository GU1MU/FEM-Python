"""Native Gmsh mesh controls and generation for scripted OCC geometry."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

from fem import geometry as _geometry
from fem.geometry._gmsh.model import (
    GmshMeshRef,
    MeshCellShapeError,
    MeshControlConflictError,
    MeshFieldOwnershipError,
    MeshFieldRef,
    StaleGmshMeshError,
    StaleMeshFieldError,
)

_CellShape = Literal["tri", "tri-quad", "quad", "tet", "hex"]
_CELL_SHAPES = frozenset({"tri", "tri-quad", "quad", "tet", "hex"})


@dataclass(frozen=True, slots=True)
class MeshSpec:
    """Immutable controls for explicit native Gmsh mesh generation."""

    size: float | None = None
    order: Literal[1, 2] = 1
    recombine: bool = False

    def __post_init__(self) -> None:
        normalized_size = (
            None if self.size is None else _positive_float(self.size, "size")
        )
        object.__setattr__(self, "size", normalized_size)
        _validate_order(self.order)
        if not isinstance(self.recombine, bool):
            raise TypeError(
                f"recombine must be a boolean, got {self.recombine!r}"
            )


@dataclass(frozen=True, slots=True)
class AutoMeshSpec:
    """Immutable controls for level-scaled strict-shape mesh generation."""

    level: Literal[1, 2, 3, 4, 5] = 3
    cell_shape: _CellShape | None = None
    order: Literal[1, 2] = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level not in range(1, 6)
        ):
            raise ValueError(
                "level must be a Python integer from 1 through 5, "
                f"got {self.level!r}"
            )
        if self.cell_shape is not None and (
            not isinstance(self.cell_shape, str)
            or self.cell_shape not in _CELL_SHAPES
        ):
            raise ValueError(
                "cell_shape must be exactly 'tri', 'tri-quad', 'quad', "
                f"'tet', 'hex', or None, got {self.cell_shape!r}"
            )
        _validate_order(self.order)


class Mesher:
    """Configure and generate one native mesh for one live geometry model."""

    __slots__ = ("_geometry", "_mesher_token")

    def __init__(self, geometry: _geometry.GeometryModel) -> None:
        if not isinstance(geometry, _geometry.GeometryModel):
            raise TypeError(
                "geometry must be a live fem.geometry.GeometryModel, "
                f"got {geometry!r}"
            )
        mesher_token = geometry._bind_mesher()
        self._geometry = geometry
        self._mesher_token = mesher_token

    def transfinite_curve(
        self,
        curve: _geometry.EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        """Set the primary-node count on one curve."""
        operation = "transfinite_curve"
        self._geometry._mesher_transfinite_curve(
            self._mesher_token,
            curve,
            num_nodes=num_nodes,
        )
        self._complete(operation)

    def transfinite_surface(
        self,
        surface: _geometry.EntityRef,
        *,
        corners: Sequence[_geometry.EntityRef] = (),
    ) -> None:
        """Mark one surface as transfinite with optional boundary corners."""
        operation = "transfinite_surface"
        self._geometry._mesher_transfinite_surface(
            self._mesher_token,
            surface,
            corners=corners,
        )
        self._complete(operation)

    def transfinite_volume(
        self,
        volume: _geometry.EntityRef,
        *,
        corners: Sequence[_geometry.EntityRef] = (),
    ) -> None:
        """Mark one volume as transfinite with optional boundary corners."""
        operation = "transfinite_volume"
        self._geometry._mesher_transfinite_volume(
            self._mesher_token,
            volume,
            corners=corners,
        )
        self._complete(operation)

    def recombine(self, surface: _geometry.EntityRef) -> None:
        """Request native Gmsh recombination on one surface."""
        operation = "recombine"
        self._geometry._mesher_recombine(self._mesher_token, surface)
        self._complete(operation)

    def mesh_size(
        self,
        points: Iterable[_geometry.EntityRef],
        *,
        size: float,
    ) -> None:
        """Assign one mesh size to selected live OCC points."""
        operation = "mesh_size"
        self._geometry._mesher_mesh_size(
            self._mesher_token,
            points,
            size=size,
        )
        self._complete(operation)

    def distance_field(
        self,
        *,
        points: Iterable[_geometry.EntityRef] = (),
        curves: Iterable[_geometry.EntityRef] = (),
        surfaces: Iterable[_geometry.EntityRef] = (),
        sampling: int = 20,
    ) -> MeshFieldRef:
        """Create a field measuring distance from selected OCC entities."""
        operation = "distance_field"
        mesh_field = self._geometry._mesher_distance_field(
            self._mesher_token,
            points=points,
            curves=curves,
            surfaces=surfaces,
            sampling=sampling,
        )
        self._complete(operation)
        return mesh_field

    def threshold_field(
        self,
        distance: MeshFieldRef,
        *,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
    ) -> MeshFieldRef:
        """Map one distance field to near- and far-field mesh sizes."""
        operation = "threshold_field"
        mesh_field = self._geometry._mesher_threshold_field(
            self._mesher_token,
            distance,
            size_min=size_min,
            size_max=size_max,
            dist_min=dist_min,
            dist_max=dist_max,
        )
        self._complete(operation)
        return mesh_field

    def min_field(self, fields: Sequence[MeshFieldRef]) -> MeshFieldRef:
        """Create the pointwise minimum of two or more size fields."""
        operation = "min_field"
        mesh_field = self._geometry._mesher_min_field(
            self._mesher_token,
            fields,
        )
        self._complete(operation)
        return mesh_field

    def background_field(self, field: MeshFieldRef) -> None:
        """Select one size-producing field as the background field."""
        operation = "background_field"
        self._geometry._mesher_background_field(self._mesher_token, field)
        self._complete(operation)

    def structured_extrude(
        self,
        entities: Iterable[_geometry.EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int] = (),
        heights: Sequence[float] = (),
        recombine: bool = False,
    ) -> _geometry.FeatureResult:
        """Create an OCC extrusion carrying native structured-layer controls."""
        return self._geometry._structured_extrude(
            self._mesher_token,
            entities,
            dx,
            dy,
            dz,
            num_elements=num_elements,
            heights=heights,
            recombine=recombine,
        )

    def generate(self, spec: MeshSpec | AutoMeshSpec) -> GmshMeshRef:
        """Generate the one native mesh permitted for the bound geometry."""
        if isinstance(spec, MeshSpec):
            return self._geometry._mesher_generate_mesh(
                self._mesher_token,
                size=spec.size,
                order=spec.order,
                recombine=spec.recombine,
            )
        if isinstance(spec, AutoMeshSpec):
            return self._geometry._mesher_generate_auto_mesh(
                self._mesher_token,
                level=spec.level,
                cell_shape=spec.cell_shape,
                order=spec.order,
            )
        raise TypeError(
            "spec must be a MeshSpec or AutoMeshSpec, "
            f"got {spec!r}"
        )

    def _complete(self, operation: str) -> None:
        self._geometry._complete_mesh_configuration_operation(
            self._mesher_token,
            operation,
        )


def _validate_order(value: Any) -> Literal[1, 2]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError(f"order must be integer 1 or 2, got {value!r}")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be finite and > 0, got {value!r}"
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return normalized


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
