from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.mesh.quality import analyze_mesh


_PLANE_NODES = {
    "Tri3": (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 3.0**0.5 / 2.0),
    ),
    "Tri6": (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 3.0**0.5 / 2.0),
        (0.5, 0.0),
        (0.75, 3.0**0.5 / 4.0),
        (0.25, 3.0**0.5 / 4.0),
    ),
    "Quad4": (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ),
    "Quad8": (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (1.0, 0.5),
        (0.5, 1.0),
        (0.0, 0.5),
    ),
}

_SPATIAL_NODES = {
    "Tet4": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "Tet10": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ),
    "Hex8": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ),
    "Hex20": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
        (1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.5),
        (1.0, 1.0, 0.5),
        (0.0, 1.0, 0.5),
    ),
    "Truss2": ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    "Beam2": ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
}


@pytest.mark.parametrize("element_type", tuple(_PLANE_NODES))
def test_plane_element_families_share_the_mesh_protocol_entry(
    element_type: str,
) -> None:
    coordinates = _PLANE_NODES[element_type]
    mesh = Mesh2D(
        [Node2D(index, *point) for index, point in enumerate(coordinates, 1)],
        [Element2D(7, list(range(1, len(coordinates) + 1)), element_type)],
    )

    report = analyze_mesh(mesh)

    assert report.checked_count == 1
    assert report.unchecked_count == 0
    assert 0.0 < report.minimum <= report.maximum <= 1.0


@pytest.mark.parametrize("element_type", tuple(_SPATIAL_NODES))
def test_spatial_and_line_element_families_share_the_mesh_protocol_entry(
    element_type: str,
) -> None:
    coordinates = _SPATIAL_NODES[element_type]
    mesh = Mesh3D(
        [Node3D(index, *point) for index, point in enumerate(coordinates, 1)],
        [Element3D(9, list(range(1, len(coordinates) + 1)), element_type)],
        dofs_per_node=6 if element_type == "Beam2" else 3,
    )

    report = analyze_mesh(mesh)

    assert report.checked_count == 1
    assert report.unchecked_count == 0
    assert 0.0 < report.minimum <= report.maximum <= 1.0


def test_degenerate_element_is_checked_with_zero_quality() -> None:
    mesh = Mesh2D(
        [Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 2.0, 0.0)],
        [Element2D(3, [1, 2, 3], "Tri3")],
    )

    report = analyze_mesh(mesh)

    assert report.checked_count == 1
    assert report.unchecked_count == 0
    assert report.minimum == report.mean == report.maximum == 0.0
    assert report.worst_elements == ((3, 0.0),)


def test_unknown_and_malformed_types_are_explicitly_unchecked() -> None:
    mesh = Mesh2D(
        [
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        [
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [1, 2, 3], "Future9"),
            Element2D(3, [1, 2], "Tri3"),
        ],
    )

    report = analyze_mesh(mesh)

    assert report.element_count == 3
    assert report.checked_count == 1
    assert report.unchecked_count == 2
    assert report.unchecked_element_types == (("Future9", 1), ("Tri3", 1))
    assert report.checked_count + report.unchecked_count == report.element_count


def test_worst_element_order_is_deterministic_for_equal_scores() -> None:
    mesh = Mesh2D(
        [
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.5, 3.0**0.5 / 2.0),
        ],
        [
            Element2D(20, [1, 2, 3], "Tri3"),
            Element2D(10, [1, 2, 3], "Tri3"),
        ],
    )

    worst = analyze_mesh(mesh).worst_elements

    assert tuple(element_id for element_id, _score in worst) == (10, 20)
    assert tuple(score for _element_id, score in worst) == pytest.approx(
        (1.0, 1.0)
    )


def test_quality_module_has_no_gui_import() -> None:
    path = Path(__file__).parents[2] / "src" / "fem" / "mesh" / "quality.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        name == "fem_gui"
        or name.startswith("fem_gui.")
        or name == "PySide6"
        or name.startswith("PySide6.")
        for name in imports
    )


def test_quality_entry_rejects_non_mesh_values() -> None:
    with pytest.raises(TypeError, match="MeshProtocol"):
        analyze_mesh(object())
