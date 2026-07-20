"""Bound capability for configuring and generating one Gmsh mesh."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from fem.geometry.types import EntityRef, FeatureResult


class _MeshingPortOwner(Protocol):
    """Structural owner contract used without importing the concrete model."""

    def _mesher_transfinite_curve(
        self,
        authority: object,
        curve: EntityRef,
        *,
        num_nodes: int,
    ) -> None: ...

    def _mesher_transfinite_surface(
        self,
        authority: object,
        surface: EntityRef,
        *,
        corners: Sequence[EntityRef],
    ) -> None: ...

    def _mesher_transfinite_volume(
        self,
        authority: object,
        volume: EntityRef,
        *,
        corners: Sequence[EntityRef],
    ) -> None: ...

    def _mesher_recombine(
        self,
        authority: object,
        surface: EntityRef,
    ) -> None: ...

    def _mesher_mesh_size(
        self,
        authority: object,
        points: Iterable[EntityRef],
        *,
        size: float,
    ) -> None: ...

    def _mesher_distance_field(
        self,
        authority: object,
        *,
        points: Iterable[EntityRef],
        curves: Iterable[EntityRef],
        surfaces: Iterable[EntityRef],
        sampling: int,
    ) -> Any: ...

    def _mesher_threshold_field(
        self,
        authority: object,
        distance: Any,
        *,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
    ) -> Any: ...

    def _mesher_min_field(
        self,
        authority: object,
        fields: Sequence[Any],
    ) -> Any: ...

    def _mesher_background_field(
        self,
        authority: object,
        field: Any,
    ) -> None: ...

    def _structured_extrude(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int],
        heights: Sequence[float],
        recombine: bool,
    ) -> FeatureResult: ...

    def _mesher_generate_mesh(
        self,
        authority: object,
        *,
        size: float | None,
        order: Literal[1, 2],
        recombine: bool,
    ) -> Any: ...

    def _mesher_generate_auto_mesh(
        self,
        authority: object,
        *,
        level: Literal[1, 2, 3, 4, 5],
        cell_shape: Literal["tri", "tri-quad", "quad", "tet", "hex"] | None,
        order: Literal[1, 2],
    ) -> Any: ...


class _BoundMeshingPort:
    """Restricted, identity-bearing access to one model's mesh transactions."""

    __slots__ = ("__owner",)

    def __init__(self, owner: _MeshingPortOwner) -> None:
        self.__owner = owner

    def transfinite_curve(
        self,
        curve: EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        self.__owner._mesher_transfinite_curve(
            self,
            curve,
            num_nodes=num_nodes,
        )

    def transfinite_surface(
        self,
        surface: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        self.__owner._mesher_transfinite_surface(
            self,
            surface,
            corners=corners,
        )

    def transfinite_volume(
        self,
        volume: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        self.__owner._mesher_transfinite_volume(
            self,
            volume,
            corners=corners,
        )

    def recombine(self, surface: EntityRef) -> None:
        self.__owner._mesher_recombine(self, surface)

    def mesh_size(
        self,
        points: Iterable[EntityRef],
        *,
        size: float,
    ) -> None:
        self.__owner._mesher_mesh_size(self, points, size=size)

    def distance_field(
        self,
        *,
        points: Iterable[EntityRef] = (),
        curves: Iterable[EntityRef] = (),
        surfaces: Iterable[EntityRef] = (),
        sampling: int = 20,
    ) -> Any:
        return self.__owner._mesher_distance_field(
            self,
            points=points,
            curves=curves,
            surfaces=surfaces,
            sampling=sampling,
        )

    def threshold_field(
        self,
        distance: Any,
        *,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
    ) -> Any:
        return self.__owner._mesher_threshold_field(
            self,
            distance,
            size_min=size_min,
            size_max=size_max,
            dist_min=dist_min,
            dist_max=dist_max,
        )

    def min_field(self, fields: Sequence[Any]) -> Any:
        return self.__owner._mesher_min_field(self, fields)

    def background_field(self, field: Any) -> None:
        self.__owner._mesher_background_field(self, field)

    def structured_extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int] = (),
        heights: Sequence[float] = (),
        recombine: bool = False,
    ) -> FeatureResult:
        return self.__owner._structured_extrude(
            self,
            entities,
            dx,
            dy,
            dz,
            num_elements=num_elements,
            heights=heights,
            recombine=recombine,
        )

    def generate_mesh(
        self,
        *,
        size: float | None,
        order: Literal[1, 2],
        recombine: bool,
    ) -> Any:
        return self.__owner._mesher_generate_mesh(
            self,
            size=size,
            order=order,
            recombine=recombine,
        )

    def generate_auto_mesh(
        self,
        *,
        level: Literal[1, 2, 3, 4, 5],
        cell_shape: Literal["tri", "tri-quad", "quad", "tet", "hex"] | None,
        order: Literal[1, 2],
    ) -> Any:
        return self.__owner._mesher_generate_auto_mesh(
            self,
            level=level,
            cell_shape=cell_shape,
            order=order,
        )
