from __future__ import annotations

import pytest

from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.post.vtk import cells
from fem.post.vtk.cells import (
    UnsupportedVTKCellTypeError,
    vtk_cell_type,
)


@pytest.mark.parametrize(
    ("canonical_type", "expected"),
    (
        ("Truss2", 3),
        ("Beam2", 3),
        ("Tri3", 5),
        ("Tri6", 22),
        ("Quad4", 9),
        ("Quad8", 23),
        ("Tet4", 10),
        ("Tet10", 24),
        ("Hex8", 12),
        ("Hex20", 25),
    ),
)
def test_vtk_cell_type_maps_every_supported_canonical_element(
    canonical_type: str,
    expected: int,
) -> None:
    assert vtk_cell_type(canonical_type) == expected


@pytest.mark.parametrize(
    "canonical_type",
    ("", "Tri4", "tri3", "CPS3", "C3D8R"),
)
def test_vtk_cell_type_rejects_unknown_alias_and_unsupported_types(
    canonical_type: str,
) -> None:
    with pytest.raises(
        UnsupportedVTKCellTypeError,
        match="Unsupported canonical element type",
    ):
        vtk_cell_type(canonical_type)


@pytest.mark.parametrize("canonical_type", (None, True, 5, object()))
def test_vtk_cell_type_requires_an_exact_string(canonical_type: object) -> None:
    with pytest.raises(TypeError, match="canonical_type"):
        vtk_cell_type(canonical_type)  # type: ignore[arg-type]


def test_legacy_mesh_builder_keeps_alias_compatibility_through_canonicalization() -> (
    None
):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[Element2D(10, [1, 2, 3], "CPS3")],
    )

    connectivity, cell_types, elements = cells.build(mesh)

    assert connectivity == [[3, 0, 1, 2]]
    assert cell_types == [5]
    assert elements == mesh.elements


def test_legacy_mesh_builder_reports_the_typed_mapping_error() -> None:
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[Element2D(10, [1, 2, 3], "UnsupportedCell")],
    )

    with pytest.raises(
        UnsupportedVTKCellTypeError,
        match="Unsupported element type for VTK export: UnsupportedCell",
    ):
        cells.build(mesh)


def test_cells_module_declares_the_strict_mapping_api() -> None:
    assert {
        "UnsupportedVTKCellTypeError",
        "vtk_cell_type",
    }.issubset(cells.__all__)
