"""Native Gmsh generation policies and exact option composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fem.geometry import GeometryStateError

from ._validation import _validate_order, _validate_runtime_level


_AutoCellShape = Literal["tri", "tri-quad", "quad", "tet", "hex"]
_AutoMeshMode = Literal["line", "tri", "tri-quad", "quad", "tet", "hex"]
_GenerationOperation = Literal["MeshSpec generation", "AutoMeshSpec generation"]
_GenerationSizeMode = Literal["none", "uniform", "point", "background"]

_POINT_SIZE_OPTION_NAME = "Mesh.MeshSizeFromPoints"
_GMSH_TOP_CELL_TYPE_NAMES = {
    1: "Line 2",
    2: "Triangle 3",
    3: "Quadrilateral 4",
    4: "Tetrahedron 4",
    5: "Hexahedron 8",
    9: "Triangle 6",
    11: "Tetrahedron 10",
    16: "Quadrilateral 8",
    17: "Hexahedron 20",
}


@dataclass(frozen=True, slots=True)
class _AutoMeshPolicy:
    mode: _AutoMeshMode
    option_overrides: tuple[tuple[str, float], ...]
    order_one_types: frozenset[int]
    order_two_types: frozenset[int] | None

    def allowed_types(self, order: Literal[1, 2]) -> frozenset[int]:
        if order == 1:
            return self.order_one_types
        if self.order_two_types is None:
            raise ValueError(
                f"order must be 1 for automatic {self.mode!r} mesh generation"
            )
        return self.order_two_types


@dataclass(frozen=True, slots=True)
class _MeshGenerationPolicy:
    operation: _GenerationOperation
    order: Literal[1, 2]
    option_overrides: tuple[tuple[str, float], ...]
    mesh_size_factor: float | None = None
    requested_cell_shape: _AutoCellShape | None = None
    resolved_cell_shape: _AutoMeshMode | None = None
    allowed_top_cell_types: frozenset[int] | None = None
    strict_cell_shape: bool = False


_AUTO_MESH_POLICIES = {
    "line": _AutoMeshPolicy(
        "line",
        (("Mesh.RecombineAll", 0.0), ("Mesh.SubdivisionAlgorithm", 0.0)),
        frozenset({1}),
        None,
    ),
    "tri": _AutoMeshPolicy(
        "tri",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({2}),
        frozenset({9}),
    ),
    "tri-quad": _AutoMeshPolicy(
        "tri-quad",
        (
            ("Mesh.RecombineAll", 1.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.RecombinationAlgorithm", 1.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({2, 3}),
        frozenset({9, 16}),
    ),
    "quad": _AutoMeshPolicy(
        "quad",
        (
            ("Mesh.RecombineAll", 1.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.RecombinationAlgorithm", 3.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({3}),
        frozenset({16}),
    ),
    "tet": _AutoMeshPolicy(
        "tet",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.Algorithm3D", 1.0),
            ("Mesh.Recombine3DAll", 0.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({4}),
        frozenset({11}),
    ),
    "hex": _AutoMeshPolicy(
        "hex",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.Algorithm3D", 1.0),
            ("Mesh.Recombine3DAll", 0.0),
            ("Mesh.SubdivisionAlgorithm", 2.0),
        ),
        frozenset({5}),
        frozenset({17}),
    ),
}


def _explicit_policy(
    dimension: int,
    *,
    order: Any,
    recombine: Any,
) -> _MeshGenerationPolicy:
    normalized_order = _validate_order(order)
    if not isinstance(recombine, bool):
        raise TypeError(f"recombine must be a boolean, got {recombine!r}")
    if dimension == 1 and normalized_order != 1:
        raise ValueError("order must be 1 for a one-dimensional geometry model")
    if dimension == 1 and recombine:
        raise ValueError(
            "recombine must be False for a one-dimensional geometry model"
        )
    return _MeshGenerationPolicy(
        operation="MeshSpec generation",
        order=normalized_order,
        option_overrides=(
            ("Mesh.ElementOrder", float(normalized_order)),
            (
                "Mesh.SecondOrderIncomplete",
                1.0 if normalized_order == 2 else 0.0,
            ),
            ("Mesh.RecombineAll", 1.0 if recombine else 0.0),
        ),
    )


def _automatic_policy(
    dimension: Literal[1, 2, 3],
    *,
    level: Any,
    cell_shape: Any,
    order: Any,
) -> _MeshGenerationPolicy:
    normalized_level = _validate_runtime_level(level)
    resolved_cell_shape = _resolve_auto_mesh_mode(dimension, cell_shape)
    normalized_order = _validate_order(order)
    if dimension == 1 and normalized_order != 1:
        raise ValueError("order must be 1 for a one-dimensional geometry model")
    auto_policy = _AUTO_MESH_POLICIES[resolved_cell_shape]
    size_factor = 2.0 ** ((3 - normalized_level) / dimension)
    return _MeshGenerationPolicy(
        operation="AutoMeshSpec generation",
        order=normalized_order,
        option_overrides=(
            ("Mesh.ElementOrder", float(normalized_order)),
            (
                "Mesh.SecondOrderIncomplete",
                1.0 if normalized_order == 2 else 0.0,
            ),
            *auto_policy.option_overrides,
        ),
        mesh_size_factor=size_factor,
        requested_cell_shape=cell_shape,
        resolved_cell_shape=resolved_cell_shape,
        allowed_top_cell_types=auto_policy.allowed_types(normalized_order),
        strict_cell_shape=True,
    )


def _compose_numeric_options(
    policy: _MeshGenerationPolicy,
    *,
    size_mode: _GenerationSizeMode,
    model_name: str,
) -> tuple[tuple[str, float], ...]:
    requested: dict[str, float] = {}
    for option_name, option_value in policy.option_overrides:
        if option_name in requested:
            raise GeometryStateError(
                f"geometry model {model_name!r}: generation policy contains "
                f"duplicate Gmsh option {option_name!r}"
            )
        requested[option_name] = float(option_value)

    if size_mode == "uniform":
        requested[_POINT_SIZE_OPTION_NAME] = 1.0
    elif size_mode in {"point", "background"}:
        requested.update(
            {
                _POINT_SIZE_OPTION_NAME: 1.0 if size_mode == "point" else 0.0,
                "Mesh.MeshSizeFromCurvature": 0.0,
                "Mesh.MeshSizeExtendFromBoundary": (
                    1.0 if size_mode == "point" else 0.0
                ),
                "Mesh.MeshSizeMin": 0.0,
                "Mesh.MeshSizeMax": 1.0e22,
            }
        )
    if policy.mesh_size_factor is not None:
        requested["Mesh.MeshSizeFactor"] = policy.mesh_size_factor
    elif size_mode in {"point", "background"}:
        requested["Mesh.MeshSizeFactor"] = 1.0
    return tuple(requested.items())


def _resolve_auto_mesh_mode(
    dimension: Literal[1, 2, 3],
    cell_shape: Any,
) -> _AutoMeshMode:
    if dimension == 1:
        if cell_shape is not None:
            raise ValueError(
                "AutoMeshSpec cell_shape must be None for dimension 1, "
                f"got {cell_shape!r}"
            )
        return "line"
    if dimension == 2:
        if cell_shape is None:
            return "tri-quad"
        if not isinstance(cell_shape, str) or cell_shape not in {
            "tri",
            "tri-quad",
            "quad",
        }:
            raise ValueError(
                "AutoMeshSpec cell_shape for dimension 2 must be exactly "
                f"'tri', 'tri-quad', or 'quad', got {cell_shape!r}"
            )
        return cell_shape
    if cell_shape is None:
        return "tet"
    if not isinstance(cell_shape, str) or cell_shape not in {"tet", "hex"}:
        raise ValueError(
            "AutoMeshSpec cell_shape for dimension 3 must be exactly "
            f"'tet' or 'hex', got {cell_shape!r}"
        )
    return cell_shape
