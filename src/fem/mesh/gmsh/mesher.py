"""Thin public mesher bound to one mesh-owned runtime."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from fem import geometry as _geometry

from ._runtime import _GmshMeshRuntime
from .specs import AutoMeshSpec, MeshSpec
from .types import GmshMeshRef, MeshFieldRef


class Mesher:
    """Configure and generate one native mesh for one live geometry model."""

    __slots__ = ("_runtime",)

    def __init__(self, geometry: _geometry.GeometryModel) -> None:
        if not isinstance(geometry, _geometry.GeometryModel):
            raise TypeError(
                "geometry must be a live fem.geometry.GeometryModel, "
                f"got {geometry!r}"
            )
        port = geometry._acquire_meshing_port()
        self._runtime = _GmshMeshRuntime(port)

    def transfinite_curve(
        self,
        curve: _geometry.EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        """Set the primary-node count on one curve."""
        self._runtime.transfinite_curve(curve, num_nodes=num_nodes)

    def transfinite_surface(
        self,
        surface: _geometry.EntityRef,
        *,
        corners: Sequence[_geometry.EntityRef] = (),
    ) -> None:
        """Mark one surface as transfinite with optional boundary corners."""
        self._runtime.transfinite_surface(surface, corners=corners)

    def transfinite_volume(
        self,
        volume: _geometry.EntityRef,
        *,
        corners: Sequence[_geometry.EntityRef] = (),
    ) -> None:
        """Mark one volume as transfinite with optional boundary corners."""
        self._runtime.transfinite_volume(volume, corners=corners)

    def recombine(self, surface: _geometry.EntityRef) -> None:
        """Request native Gmsh recombination on one surface."""
        self._runtime.recombine(surface)

    def mesh_size(
        self,
        points: Iterable[_geometry.EntityRef],
        *,
        size: float,
    ) -> None:
        """Assign one mesh size to selected live OCC points."""
        self._runtime.mesh_size(points, size=size)

    def distance_field(
        self,
        *,
        points: Iterable[_geometry.EntityRef] = (),
        curves: Iterable[_geometry.EntityRef] = (),
        surfaces: Iterable[_geometry.EntityRef] = (),
        sampling: int = 20,
    ) -> MeshFieldRef:
        """Create a field measuring distance from selected OCC entities."""
        return self._runtime.distance_field(
            points=points,
            curves=curves,
            surfaces=surfaces,
            sampling=sampling,
        )

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
        return self._runtime.threshold_field(
            distance,
            size_min=size_min,
            size_max=size_max,
            dist_min=dist_min,
            dist_max=dist_max,
        )

    def min_field(self, fields: Sequence[MeshFieldRef]) -> MeshFieldRef:
        """Create the pointwise minimum of two or more size fields."""
        return self._runtime.min_field(fields)

    def background_field(self, field: MeshFieldRef) -> None:
        """Select one size-producing field as the background field."""
        self._runtime.background_field(field)

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
        return self._runtime.structured_extrude(
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
        return self._runtime.generate(spec)


__all__ = ["Mesher"]
