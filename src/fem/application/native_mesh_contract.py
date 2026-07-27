"""Shared native recipe-to-element intent resolution.

The authoring layers must agree on the dimensional mesh shape and the FEM
formulation before they touch Gmsh or compile model definitions.  This
module is deliberately Qt-free and owns that small contract in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.elements import canonical_element_type, get_element_capabilities
from fem.geometry.recipes import (
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    geometry_dimension,
)
from fem.mesh.settings import MeshSettings


NativeDimension = Literal[1, 2, 3]
LineElementType = Literal["Truss2", "Beam2"]


class NativeMeshContractError(ValueError):
    """A native recipe and mesh settings have no compatible mesh contract."""


@dataclass(frozen=True, slots=True)
class NativeMeshContract:
    """Resolved dimensional and formulation intent for one native recipe."""

    dimension: NativeDimension
    cell_shape: str | None
    order: int | None
    canonical_element_type: str | None
    line_element_type: LineElementType | None
    complete: bool

    def __post_init__(self) -> None:
        if self.dimension not in (1, 2, 3):
            raise ValueError("native mesh contract dimension must be 1, 2, or 3")
        if self.cell_shape is not None and (
            type(self.cell_shape) is not str or not self.cell_shape.strip()
        ):
            raise ValueError("native mesh contract cell_shape must be text or None")
        if self.order is not None and (
            isinstance(self.order, bool) or self.order not in (1, 2)
        ):
            raise ValueError("native mesh contract order must be 1, 2, or None")
        if self.line_element_type not in (None, "Truss2", "Beam2"):
            raise ValueError("native mesh contract line element type is invalid")
        if type(self.complete) is not bool:
            raise TypeError("native mesh contract complete must be a boolean")
        if self.complete and self.canonical_element_type is None:
            raise ValueError("a complete native mesh contract needs an element type")
        if not self.complete and self.canonical_element_type is not None:
            raise ValueError(
                "an incomplete native mesh contract cannot have an element type"
            )


_CANONICAL_BY_SHAPE = {
    ("triangle", 1): "Tri3",
    ("triangle", 2): "Tri6",
    ("quadrilateral", 1): "Quad4",
    ("quadrilateral", 2): "Quad8",
    ("tetrahedron", 1): "Tet4",
    ("tetrahedron", 2): "Tet10",
    ("hexahedron", 1): "Hex8",
    ("hexahedron", 2): "Hex20",
}
_SHAPES_BY_DIMENSION = {
    1: frozenset({"line"}),
    2: frozenset({"triangle", "quadrilateral"}),
    3: frozenset({"tetrahedron", "hexahedron"}),
}
_DEFAULT_SHAPE_BY_DIMENSION = {2: "triangle", 3: "tetrahedron"}


def describe_native_mesh_contract(
    recipe: NativeGeometry,
    settings: MeshSettings | None,
) -> NativeMeshContract:
    """Resolve native dimensional and FEM element intent without side effects."""

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("recipe must be a native geometry recipe")
    if settings is not None and type(settings) is not MeshSettings:
        raise TypeError("settings must be MeshSettings or None")

    return _resolve_native_mesh_contract(geometry_dimension(recipe), settings)


def describe_native_mesh_settings_contract(
    settings: MeshSettings,
) -> NativeMeshContract:
    """Resolve mesh intent when a prospective recipe is not available yet."""

    if type(settings) is not MeshSettings:
        raise TypeError("settings must be MeshSettings")
    if settings.cell_shape == "line":
        dimension: NativeDimension = 1
    elif settings.cell_shape in _SHAPES_BY_DIMENSION[2]:
        dimension = 2
    else:
        dimension = 3
    return _resolve_native_mesh_contract(dimension, settings)


def _resolve_native_mesh_contract(
    dimension: NativeDimension,
    settings: MeshSettings | None,
) -> NativeMeshContract:
    if dimension == 1 and settings is None:
        return NativeMeshContract(
            dimension=1,
            cell_shape="line",
            order=None,
            canonical_element_type=None,
            line_element_type=None,
            complete=False,
        )

    if settings is None:
        cell_shape = _DEFAULT_SHAPE_BY_DIMENSION[dimension]
        order = 1
        line_element_type = None
    else:
        cell_shape = settings.cell_shape
        order = settings.order
        line_element_type = settings.line_element_type

    # ``MeshSettings`` historically used ``triangle`` as its cross-
    # dimensional default.  Preserve that continuum convenience for a
    # three-dimensional recipe while exposing the canonical Tet mapping to
    # downstream consumers.
    if dimension == 3 and cell_shape == "triangle":
        cell_shape = "tetrahedron"

    supported_shapes = _SHAPES_BY_DIMENSION[dimension]
    if cell_shape not in supported_shapes:
        expected = ", ".join(sorted(supported_shapes))
        raise NativeMeshContractError(
            f"native recipe dimension {dimension} cell_shape {cell_shape!r} "
            f"is not supported; expected one of {{{expected}}}"
        )

    if dimension == 1:
        if order != 1:
            raise NativeMeshContractError(
                "native line meshes require first-order two-node elements"
            )
        if line_element_type not in ("Truss2", "Beam2"):
            return NativeMeshContract(
                dimension=1,
                cell_shape="line",
                order=order,
                canonical_element_type=None,
                line_element_type=None,
                complete=False,
            )
        canonical = line_element_type
    else:
        if line_element_type is not None:
            raise NativeMeshContractError(
                "line_element_type is only valid for dimension-one recipes"
            )
        canonical = _CANONICAL_BY_SHAPE[(cell_shape, order)]

    # Keep the element registry authoritative for canonical names and verify
    # that the mapping never drifts away from the registered capability.
    canonical = canonical_element_type(canonical)
    capabilities = get_element_capabilities(canonical)
    if (
        capabilities.topological_dimension != dimension
        or capabilities.canonical_type != canonical
    ):
        raise NativeMeshContractError(
            f"element registry contract for {canonical!r} is incompatible "
            f"with recipe dimension {dimension}"
        )
    return NativeMeshContract(
        dimension=dimension,
        cell_shape=cell_shape,
        order=order,
        canonical_element_type=canonical,
        line_element_type=line_element_type,
        complete=True,
    )


def require_complete_native_mesh_contract(
    recipe: NativeGeometry,
    settings: MeshSettings | None,
) -> NativeMeshContract:
    """Resolve a contract and reject incomplete native line authoring intent."""

    contract = describe_native_mesh_contract(recipe, settings)
    if not contract.complete:
        raise NativeMeshContractError(
            "native 1D mesh settings require explicit line_element_type "
            "('Truss2' or 'Beam2')"
        )
    return contract


__all__ = [
    "LineElementType",
    "NativeDimension",
    "NativeMeshContract",
    "NativeMeshContractError",
    "describe_native_mesh_contract",
    "describe_native_mesh_settings_contract",
    "require_complete_native_mesh_contract",
]
