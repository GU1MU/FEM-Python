from __future__ import annotations

import builtins
import inspect
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from fem import materials, post, steps
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    Mesh2D,
    Mesh3D,
    Node2D,
    NodeSet,
    Surface,
    validate_model,
)
from fem.elements import get_element_kernel
from fem.elements.hexahedron import HEX20_NATURAL_NODE_COORDS
from fem.elements.tetrahedron import TET10_NATURAL_NODE_COORDS
from fem.io import gmsh as gmsh_io
from fem.selection import edges as edge_selection
from fem.selection import faces as face_selection
from fem.solvers import static_linear


ROOT = Path(__file__).resolve().parents[1]


class _UnreadableModel:
    @property
    def mesh(self):
        raise AssertionError("the backend must not be read before option validation")


_ELEMENT_PROPERTIES = {
    1: ("Line 2", 1, 1, 2, [], 2),
    2: ("Triangle 3", 2, 1, 3, [], 3),
    3: ("Quadrilateral 4", 2, 1, 4, [], 4),
    4: ("Tetrahedron 4", 3, 1, 4, [], 4),
    5: ("Hexahedron 8", 3, 1, 8, [], 8),
    6: ("Prism 6", 3, 1, 6, [], 6),
    8: ("Line 3", 1, 2, 3, [], 2),
    9: ("Triangle 6", 2, 2, 6, [], 3),
    10: ("Quadrilateral 9", 2, 2, 9, [], 4),
    11: ("Tetrahedron 10", 3, 2, 10, [], 4),
    12: ("Hexahedron 27", 3, 2, 27, [], 8),
    16: ("Quadrilateral 8", 2, 2, 8, [], 4),
    17: ("Hexahedron 20", 3, 2, 20, [], 8),
}


class _FakeMesh:
    def __init__(
        self,
        *,
        node_tags,
        coordinates,
        elements,
        properties=None,
        entity_elements=None,
        physical_nodes=None,
    ):
        self.node_tags = node_tags
        self.coordinates = coordinates
        self.elements = elements
        self.properties = dict(_ELEMENT_PROPERTIES)
        if properties:
            self.properties.update(properties)
        self.entity_elements = entity_elements or {}
        self.physical_nodes = physical_nodes or {}
        self.element_calls = []
        self.node_calls = 0
        self.property_calls = []

    def getElements(self, dimension, entity_tag):
        self.element_calls.append((dimension, entity_tag))
        if entity_tag == -1:
            return self.elements
        return self.entity_elements.get((dimension, entity_tag), ([], [], []))

    def getElementProperties(self, element_type):
        self.property_calls.append(element_type)
        return self.properties[element_type]

    def getNodes(self):
        self.node_calls += 1
        return self.node_tags, self.coordinates, []

    def getNodesForPhysicalGroup(self, dimension, group_tag):
        node_tags = self.physical_nodes.get((dimension, group_tag), [])
        return node_tags, []


class _FakeModel:
    def __init__(self, mesh, physical_groups=(), physical_names=None, entities=None):
        self.mesh = mesh
        self.physical_groups = list(physical_groups)
        self.physical_names = physical_names or {}
        self.entities = entities or {}
        self.physical_group_calls = 0

    def getPhysicalGroups(self):
        self.physical_group_calls += 1
        return self.physical_groups

    def getPhysicalName(self, dimension, group_tag):
        return self.physical_names.get((dimension, group_tag), "")

    def getEntitiesForPhysicalGroup(self, dimension, group_tag):
        return self.entities.get((dimension, group_tag), [])


def _fake_model(*, node_coordinates, elements, **mesh_kwargs):
    node_tags = list(node_coordinates)
    coordinates = [
        coordinate
        for node_tag in node_tags
        for coordinate in node_coordinates[node_tag]
    ]
    return _FakeModel(
        _FakeMesh(
            node_tags=node_tags,
            coordinates=coordinates,
            elements=elements,
            **mesh_kwargs,
        )
    )


def _physical_curve_model(
    *,
    node_coordinates,
    top_elements,
    boundary_entities,
    physical_nodes,
    group_entities=None,
    group_tag=7,
    name="BOUNDARY",
):
    mesh = _FakeMesh(
        node_tags=list(node_coordinates),
        coordinates=[
            coordinate
            for node_tag in node_coordinates
            for coordinate in node_coordinates[node_tag]
        ],
        elements=top_elements,
        entity_elements={
            (1, entity_tag): blocks
            for entity_tag, blocks in boundary_entities.items()
        },
        physical_nodes={(1, group_tag): physical_nodes},
    )
    return _FakeModel(
        mesh,
        physical_groups=[(1, group_tag)],
        physical_names={(1, group_tag): name},
        entities={
            (1, group_tag): (
                list(boundary_entities) if group_entities is None else group_entities
            )
        },
    )


def _physical_surface_model(
    *,
    node_coordinates,
    top_elements,
    boundary_entities,
    physical_nodes,
    group_entities=None,
    group_tag=7,
    name="BOUNDARY",
):
    mesh = _FakeMesh(
        node_tags=list(node_coordinates),
        coordinates=[
            coordinate
            for node_tag in node_coordinates
            for coordinate in node_coordinates[node_tag]
        ],
        elements=top_elements,
        entity_elements={
            (2, entity_tag): blocks
            for entity_tag, blocks in boundary_entities.items()
        },
        physical_nodes={(2, group_tag): physical_nodes},
    )
    return _FakeModel(
        mesh,
        physical_groups=[(2, group_tag)],
        physical_names={(2, group_tag): name},
        entities={
            (2, group_tag): (
                list(boundary_entities) if group_entities is None else group_entities
            )
        },
    )


def test_cad_extra_declares_supported_gmsh_range():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["cad"] == [
        "gmsh>=4.15.2,<5"
    ]


def test_importing_fem_io_gmsh_does_not_import_external_gmsh():
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fem.io import gmsh as gmsh_io
assert gmsh_io.__name__ == "fem.io.gmsh"
"""

    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("dimension", [0, 1, 4, "2", None])
def test_from_model_rejects_invalid_dimension_before_reading_backend(dimension):
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        gmsh_io.from_model(dimension=dimension, gmsh_model=_UnreadableModel())


@pytest.mark.parametrize("plane_type", ["", "plane", None, 3])
def test_from_model_rejects_invalid_plane_type_before_reading_backend(plane_type):
    with pytest.raises(ValueError, match="plane_type.*stress.*strain"):
        gmsh_io.from_model(
            dimension=2,
            plane_type=plane_type,
            gmsh_model=_UnreadableModel(),
        )


@pytest.mark.parametrize("thickness", [0.0, -1.0, float("inf"), float("nan")])
def test_from_model_rejects_invalid_thickness_before_reading_backend(thickness):
    with pytest.raises(ValueError, match="thickness must be finite and > 0"):
        gmsh_io.from_model(
            dimension=2,
            thickness=thickness,
            gmsh_model=_UnreadableModel(),
        )


@pytest.mark.parametrize(
    "z_tolerance", [-1.0, float("-inf"), float("inf"), float("nan")]
)
def test_from_model_rejects_invalid_z_tolerance_before_reading_backend(z_tolerance):
    with pytest.raises(ValueError, match="z_tolerance must be finite and >= 0"):
        gmsh_io.from_model(
            dimension=2,
            z_tolerance=z_tolerance,
            gmsh_model=_UnreadableModel(),
        )


def test_default_backend_reports_missing_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def missing_gmsh(name, *args, **kwargs):
        if name == "gmsh":
            raise ModuleNotFoundError("No module named 'gmsh'", name="gmsh")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_gmsh)

    with pytest.raises(ModuleNotFoundError, match=r"pip install -e .\[cad\]"):
        gmsh_io.from_model(dimension=2)


def test_default_backend_requires_initialized_session(monkeypatch):
    class _UninitializedGmsh:
        __version__ = "4.15.2"
        model = object()

        @staticmethod
        def isInitialized():
            return False

    monkeypatch.setitem(sys.modules, "gmsh", _UninitializedGmsh())

    with pytest.raises(RuntimeError, match="not initialized"):
        gmsh_io.from_model(dimension=2)


def test_import_result_to_fem_model_copies_sets_and_metadata():
    mesh = Mesh2D(nodes=[Node2D(10, 0.0, 0.0)], elements=[])
    node_sets = {"LEFT": NodeSet("LEFT", [10])}
    element_sets = {"DOMAIN": ElementSet("DOMAIN", [20])}
    edges = {"BOTTOM": Edge("BOTTOM", [ElementEdge(20, 0, [10, 30])])}
    surfaces = {"FRONT": Surface("FRONT", [ElementFace(20, 0, [10, 30, 40])])}
    metadata = {"source": "gmsh", "dimension": 2}
    imported = gmsh_io.GmshImportResult(
        mesh=mesh,
        node_sets=node_sets,
        element_sets=element_sets,
        metadata=metadata,
        edges=edges,
        surfaces=surfaces,
    )

    model = imported.to_fem_model("plate")

    assert model.name == "plate"
    assert model.mesh is mesh
    assert model.node_sets == node_sets
    assert model.node_sets is not node_sets
    assert model.element_sets == element_sets
    assert model.element_sets is not element_sets
    assert model.metadata == metadata
    assert model.metadata is not metadata
    assert model.materials == {}
    assert model.sections == []
    assert model.steps == []
    assert model.edges == edges
    assert model.edges is not edges
    assert model.surfaces == surfaces
    assert model.surfaces is not surfaces

    model.node_sets.clear()
    model.element_sets.clear()
    model.metadata.clear()
    model.edges.clear()
    model.surfaces.clear()
    assert imported.node_sets == node_sets
    assert imported.element_sets == element_sets
    assert imported.metadata == metadata
    assert imported.edges == edges
    assert imported.surfaces == surfaces


def test_import_result_four_argument_boundary_defaults_are_independent():
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])

    first = gmsh_io.GmshImportResult(mesh, {}, {}, {})
    second = gmsh_io.GmshImportResult(mesh, {}, {}, {})

    assert first.edges == {}
    assert first.surfaces == {}
    assert first.edges is not second.edges
    assert first.surfaces is not second.surfaces


def test_from_model_public_signature_is_unchanged():
    signature = inspect.signature(gmsh_io.from_model)

    assert tuple(signature.parameters) == (
        "dimension",
        "gmsh_model",
        "plane_type",
        "thickness",
        "z_tolerance",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    ("gmsh_type", "expected"),
    [
        (1, ("Line 2", 1, 1, 2, 2, "line")),
        (8, ("Line 3", 1, 2, 3, 2, "line")),
        (2, ("Triangle 3", 2, 1, 3, 3, "triangle")),
        (9, ("Triangle 6", 2, 2, 6, 3, "triangle")),
        (3, ("Quadrilateral 4", 2, 1, 4, 4, "quadrilateral")),
        (16, ("Quadrilateral 8", 2, 2, 8, 4, "quadrilateral")),
    ],
)
def test_boundary_specs_define_supported_matching_contract(gmsh_type, expected):
    spec = gmsh_io._BOUNDARY_ELEMENT_SPECS[gmsh_type]

    assert (
        spec.gmsh_name,
        spec.dimension,
        spec.order,
        spec.node_count,
        spec.primary_node_count,
        spec.shape,
    ) == expected


@pytest.mark.parametrize(
    (
        "element_type",
        "node_coordinates",
        "top_node_ids",
        "boundary_type",
        "boundary_node_ids",
        "expected_local_index",
        "expected_node_ids",
    ),
    [
        (
            2,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (0.0, 1.0, 0.0),
            },
            [1, 2, 3],
            1,
            [2, 1],
            0,
            (1, 2),
        ),
        (
            3,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
                4: (0.0, 1.0, 0.0),
            },
            [1, 2, 3, 4],
            1,
            [3, 2],
            1,
            (2, 3),
        ),
        (
            9,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (0.0, 1.0, 0.0),
                4: (0.5, 0.0, 0.0),
                5: (0.5, 0.5, 0.0),
                6: (0.0, 0.5, 0.0),
            },
            [1, 2, 3, 4, 5, 6],
            8,
            [2, 1, 4],
            0,
            (1, 4, 2),
        ),
        (
            16,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
                4: (0.0, 1.0, 0.0),
                5: (0.5, 0.0, 0.0),
                6: (1.0, 0.5, 0.0),
                7: (0.5, 1.0, 0.0),
                8: (0.0, 0.5, 0.0),
            },
            [1, 2, 3, 4, 5, 6, 7, 8],
            8,
            [4, 3, 7],
            2,
            (3, 7, 4),
        ),
    ],
    ids=["tri3-line2", "quad4-line2", "tri6-line3", "quad8-line3"],
)
def test_physical_curve_maps_to_edge_in_fem_owner_order(
    element_type,
    node_coordinates,
    top_node_ids,
    boundary_type,
    boundary_node_ids,
    expected_local_index,
    expected_node_ids,
):
    model = _physical_curve_model(
        node_coordinates=node_coordinates,
        top_elements=([element_type], [[101]], [top_node_ids]),
        boundary_entities={
            71: ([boundary_type], [[901]], [boundary_node_ids]),
        },
        physical_nodes=boundary_node_ids,
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.node_sets["BOUNDARY"] == NodeSet(
        "BOUNDARY",
        sorted(set(boundary_node_ids)),
    )
    assert result.edges["BOUNDARY"] == Edge(
        "BOUNDARY",
        [ElementEdge(101, expected_local_index, expected_node_ids)],
    )
    assert result.surfaces == {}
    assert result.metadata["physical_groups"]["BOUNDARY"] == {
        "dimension": 1,
        "tag": 7,
        "kind": "node_set",
        "boundary_kind": "edge",
        "boundary_entry_count": 1,
    }
    validate_model(result.to_fem_model("edge_owner"))


def test_physical_curve_combines_entities_and_sorts_noncontiguous_owners():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (2.0, 0.0, 0.0),
            5: (3.0, 0.0, 0.0),
            6: (2.0, 1.0, 0.0),
        },
        top_elements=([2], [[205, 101]], [[4, 5, 6, 1, 2, 3]]),
        boundary_entities={
            72: ([1], [[990]], [[5, 4]]),
            71: ([1], [[870]], [[2, 1]]),
        },
        group_entities=[72, 71],
        physical_nodes=[5, 4, 2, 1],
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.edges["BOUNDARY"].edges == (
        ElementEdge(101, 0, [1, 2]),
        ElementEdge(205, 0, [4, 5]),
    )


def test_multiple_physical_curves_scan_fem_edge_topology_once(monkeypatch):
    mesh = _FakeMesh(
        node_tags=[1, 2, 3],
        coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        elements=([2], [[101]], [[1, 2, 3]]),
        entity_elements={
            (1, 71): ([1], [[901]], [[1, 2]]),
            (1, 72): ([1], [[902]], [[2, 3]]),
        },
        physical_nodes={(1, 7): [1, 2], (1, 8): [2, 3]},
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(1, 7), (1, 8)],
        physical_names={(1, 7): "BOTTOM", (1, 8): "DIAGONAL"},
        entities={(1, 7): [71], (1, 8): [72]},
    )
    real_all = edge_selection.all
    calls = 0

    def counted_all(imported_mesh):
        nonlocal calls
        calls += 1
        return real_all(imported_mesh)

    monkeypatch.setattr(edge_selection, "all", counted_all)

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert set(result.edges) == {"BOTTOM", "DIAGONAL"}
    assert calls == 1


def test_physical_point_remains_node_set_only():
    mesh = _FakeMesh(
        node_tags=[1, 2, 3],
        coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        elements=([2], [[101]], [[1, 2, 3]]),
        physical_nodes={(0, 7): [1]},
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(0, 7)],
        physical_names={(0, 7): "POINT"},
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.node_sets == {"POINT": NodeSet("POINT", [1])}
    assert result.edges == {}


def test_physical_curve_rejects_boundary_without_fem_owner():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (1.0, 1.0, 0.0),
            4: (0.0, 1.0, 0.0),
        },
        top_elements=([3], [[101]], [[1, 2, 3, 4]]),
        boundary_entities={71: ([1], [[901]], [[1, 3]])},
        physical_nodes=[1, 3],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*no FEM owner"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_physical_curve_rejects_internal_edge_with_all_candidate_owners():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
        },
        top_elements=([2], [[101, 205]], [[1, 2, 3, 2, 4, 3]]),
        boundary_entities={71: ([1], [[901]], [[3, 2]])},
        physical_nodes=[2, 3],
    )

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=2, gmsh_model=model)

    message = str(exc_info.value)
    assert "BOUNDARY" in message
    assert "901" in message
    assert "multiple FEM owners" in message
    assert "(101, 1)" in message
    assert "(205, 2)" in message


def test_physical_curve_rejects_partial_retained_node_membership():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            99: (2.0, 0.0, 0.0),
        },
        top_elements=([2], [[101]], [[1, 2, 3]]),
        boundary_entities={71: ([1], [[901]], [[1, 99]])},
        physical_nodes=[1, 99],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*partial retained-node"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_physical_curve_rejects_linear_boundary_for_high_order_owner():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.5, 0.0, 0.0),
            5: (0.5, 0.5, 0.0),
            6: (0.0, 0.5, 0.0),
        },
        top_elements=([9], [[101]], [[1, 2, 3, 4, 5, 6]]),
        boundary_entities={71: ([1], [[901]], [[1, 2]])},
        physical_nodes=[1, 2],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*full node set"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


@pytest.mark.parametrize(
    ("boundary_tags", "boundary_node_ids", "message"),
    [
        ([901], [1, 1], r"BOUNDARY.*901.*repeated node tags"),
        ([901, 901], [1, 2, 2, 3], r"BOUNDARY.*duplicate boundary element tag 901"),
        ([901, 902], [1, 2, 2, 1], r"BOUNDARY.*duplicate FEM owner \(101, 0\)"),
    ],
)
def test_physical_curve_rejects_duplicate_source_or_owner_data(
    boundary_tags,
    boundary_node_ids,
    message,
):
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        top_elements=([2], [[101]], [[1, 2, 3]]),
        boundary_entities={71: ([1], [boundary_tags], [boundary_node_ids])},
        physical_nodes=[1, 2, 3],
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_physical_curve_rejects_retained_nodes_without_retained_elements():
    model = _physical_curve_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            99: (2.0, 0.0, 0.0),
            100: (3.0, 0.0, 0.0),
        },
        top_elements=([2], [[101]], [[1, 2, 3]]),
        boundary_entities={71: ([1], [[901]], [[99, 100]])},
        physical_nodes=[1, 99, 100],
    )

    with pytest.raises(
        ValueError,
        match=r"BOUNDARY.*retained nodes.*no retained boundary elements",
    ):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def _three_dimensional_coordinates(node_ids):
    return {
        node_id: (
            float(node_id),
            float(node_id % 3),
            float(node_id % 5),
        )
        for node_id in node_ids
    }


@pytest.mark.parametrize(
    (
        "element_type",
        "raw_top_node_ids",
        "boundary_type",
        "boundary_node_ids",
        "expected_local_index",
        "expected_node_ids",
    ),
    [
        (4, [1, 2, 3, 4], 2, [3, 1, 2], 3, (1, 2, 3)),
        (
            11,
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 9],
            9,
            [3, 1, 2, 7, 5, 6],
            3,
            (1, 2, 3, 5, 6, 7),
        ),
        (5, list(range(1, 9)), 3, [2, 3, 4, 1], 0, (1, 4, 3, 2)),
        (
            17,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 10, 18, 11, 19, 20, 13, 16, 14, 15],
            16,
            [2, 3, 4, 1, 10, 11, 12, 9],
            0,
            (1, 4, 3, 2, 12, 11, 10, 9),
        ),
    ],
    ids=["tet4-triangle3", "tet10-triangle6", "hex8-quad4", "hex20-quad8"],
)
def test_physical_surface_maps_to_face_in_fem_owner_order(
    element_type,
    raw_top_node_ids,
    boundary_type,
    boundary_node_ids,
    expected_local_index,
    expected_node_ids,
):
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates(raw_top_node_ids),
        top_elements=([element_type], [[101]], [raw_top_node_ids]),
        boundary_entities={
            71: ([boundary_type], [[901]], [boundary_node_ids]),
        },
        physical_nodes=boundary_node_ids,
    )

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert result.node_sets["BOUNDARY"] == NodeSet(
        "BOUNDARY",
        sorted(set(boundary_node_ids)),
    )
    assert result.surfaces["BOUNDARY"] == Surface(
        "BOUNDARY",
        [ElementFace(101, expected_local_index, expected_node_ids)],
    )
    assert result.edges == {}
    assert result.metadata["physical_groups"]["BOUNDARY"] == {
        "dimension": 2,
        "tag": 7,
        "kind": "node_set",
        "boundary_kind": "surface",
        "boundary_entry_count": 1,
    }
    validate_model(result.to_fem_model("surface_owner"))


def test_physical_surface_supports_mixed_triangle_and_quadrilateral_blocks():
    node_ids = [1, 2, 3, 4, *range(11, 19)]
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates(node_ids),
        top_elements=(
            [4, 5],
            [[205], [101]],
            [[1, 2, 3, 4], list(range(11, 19))],
        ),
        boundary_entities={
            71: (
                [2, 3],
                [[901], [902]],
                [[3, 1, 2], [12, 13, 14, 11]],
            ),
        },
        physical_nodes=[1, 2, 3, 11, 12, 13, 14],
    )

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert result.surfaces["BOUNDARY"].faces == (
        ElementFace(101, 0, [11, 14, 13, 12]),
        ElementFace(205, 3, [1, 2, 3]),
    )


def test_multiple_physical_surfaces_scan_fem_face_topology_once(monkeypatch):
    mesh = _FakeMesh(
        node_tags=[1, 2, 3, 4],
        coordinates=[
            coordinate
            for node_id in [1, 2, 3, 4]
            for coordinate in _three_dimensional_coordinates([node_id])[node_id]
        ],
        elements=([4], [[101]], [[1, 2, 3, 4]]),
        entity_elements={
            (2, 71): ([2], [[901]], [[1, 2, 3]]),
            (2, 72): ([2], [[902]], [[2, 3, 4]]),
        },
        physical_nodes={(2, 7): [1, 2, 3], (2, 8): [2, 3, 4]},
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(2, 7), (2, 8)],
        physical_names={(2, 7): "BASE", (2, 8): "SIDE"},
        entities={(2, 7): [71], (2, 8): [72]},
    )
    real_all = face_selection.all
    calls = 0

    def counted_all(imported_mesh):
        nonlocal calls
        calls += 1
        return real_all(imported_mesh)

    monkeypatch.setattr(face_selection, "all", counted_all)

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert set(result.surfaces) == {"BASE", "SIDE"}
    assert calls == 1


def test_physical_curve_and_point_in_3d_remain_node_sets_only():
    coordinates = _three_dimensional_coordinates([1, 2, 3, 4])
    mesh = _FakeMesh(
        node_tags=list(coordinates),
        coordinates=[
            coordinate
            for node_id in coordinates
            for coordinate in coordinates[node_id]
        ],
        elements=([4], [[101]], [[1, 2, 3, 4]]),
        physical_nodes={(1, 7): [1, 2], (0, 8): [1]},
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(1, 7), (0, 8)],
        physical_names={(1, 7): "CURVE", (0, 8): "POINT"},
    )

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert result.node_sets == {
        "CURVE": NodeSet("CURVE", [1, 2]),
        "POINT": NodeSet("POINT", [1]),
    }
    assert result.edges == {}
    assert result.surfaces == {}


def test_physical_surface_rejects_triangle_shape_for_hex_owner():
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates(range(1, 9)),
        top_elements=([5], [[101]], [list(range(1, 9))]),
        boundary_entities={71: ([2], [[901]], [[1, 2, 3]])},
        physical_nodes=[1, 2, 3],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*no FEM owner"):
        gmsh_io.from_model(dimension=3, gmsh_model=model)


def test_physical_surface_rejects_internal_face_with_all_candidate_owners():
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates([1, 2, 3, 4, 5]),
        top_elements=([4], [[101, 205]], [[1, 2, 3, 4, 1, 3, 2, 5]]),
        boundary_entities={71: ([2], [[901]], [[3, 1, 2]])},
        physical_nodes=[1, 2, 3],
    )

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=3, gmsh_model=model)

    message = str(exc_info.value)
    assert "BOUNDARY" in message
    assert "901" in message
    assert "multiple FEM owners" in message
    assert "(101, 3)" in message
    assert "(205, 3)" in message


def test_physical_surface_rejects_partial_high_order_node_membership():
    raw_top_node_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates([*raw_top_node_ids, 99]),
        top_elements=([11], [[101]], [raw_top_node_ids]),
        boundary_entities={71: ([9], [[901]], [[1, 2, 3, 5, 6, 99]])},
        physical_nodes=[1, 2, 3, 5, 6, 99],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*partial retained-node"):
        gmsh_io.from_model(dimension=3, gmsh_model=model)


def test_physical_surface_rejects_linear_boundary_for_high_order_owner():
    raw_top_node_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates(raw_top_node_ids),
        top_elements=([11], [[101]], [raw_top_node_ids]),
        boundary_entities={71: ([2], [[901]], [[1, 2, 3]])},
        physical_nodes=[1, 2, 3],
    )

    with pytest.raises(ValueError, match=r"BOUNDARY.*901.*full node set"):
        gmsh_io.from_model(dimension=3, gmsh_model=model)


def test_physical_surface_rejects_unsupported_face_type():
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates(range(1, 10)),
        top_elements=([5], [[101]], [list(range(1, 9))]),
        boundary_entities={
            71: ([10], [[901]], [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        },
        physical_nodes=list(range(1, 10)),
    )

    with pytest.raises(
        ValueError,
        match=r"BOUNDARY.*unsupported codimension-one Gmsh element type 10",
    ) as exc_info:
        gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert "901" in str(exc_info.value)


@pytest.mark.parametrize(
    ("boundary_blocks", "message"),
    [
        (([2], [[901]], []), r"BOUNDARY.*inconsistent boundary element block counts"),
        (([2], [[901]], [[1, 2]]), r"BOUNDARY.*flattened connectivity.*expected 3"),
        (([2], [[0]], [[1, 2, 3]]), r"BOUNDARY.*boundary element tags.*positive"),
        (([2], [[901]], [[1, 2, 0]]), r"BOUNDARY.*boundary node tags.*positive"),
        (([2], [None], [[1, 2, 3]]), r"BOUNDARY.*malformed boundary element-tag block"),
        (([2], [[901]], [None]), r"BOUNDARY.*malformed boundary connectivity block"),
    ],
)
def test_physical_surface_rejects_malformed_boundary_blocks(
    boundary_blocks,
    message,
):
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates([1, 2, 3, 4]),
        top_elements=([4], [[101]], [[1, 2, 3, 4]]),
        boundary_entities={71: boundary_blocks},
        physical_nodes=[1, 2, 3],
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=3, gmsh_model=model)


def test_physical_surface_rejects_boundary_property_mismatch():
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates([1, 2, 3, 4]),
        top_elements=([4], [[101]], [[1, 2, 3, 4]]),
        boundary_entities={71: ([2], [[901]], [[1, 2, 3]])},
        physical_nodes=[1, 2, 3],
    )
    model.mesh.properties[2] = ("Triangle 3", 2, 2, 3, [], 3)

    with pytest.raises(
        ValueError,
        match=r"BOUNDARY.*properties do not match the adapter contract",
    ) as exc_info:
        gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert "901" in str(exc_info.value)


def test_physical_surface_rejects_malformed_entity_tags():
    model = _physical_surface_model(
        node_coordinates=_three_dimensional_coordinates([1, 2, 3, 4]),
        top_elements=([4], [[101]], [[1, 2, 3, 4]]),
        boundary_entities={71: ([2], [[901]], [[1, 2, 3]])},
        physical_nodes=[1, 2, 3],
    )
    model.entities[(2, 7)] = None

    with pytest.raises(ValueError, match=r"BOUNDARY.*entity tags.*malformed"):
        gmsh_io.from_model(dimension=3, gmsh_model=model)


@pytest.mark.parametrize(
    ("gmsh_type", "expected"),
    [
        (2, ("Triangle 3", 2, 1, 3, 3, "Tri3", (0, 1, 2))),
        (3, ("Quadrilateral 4", 2, 1, 4, 4, "Quad4", (0, 1, 2, 3))),
        (4, ("Tetrahedron 4", 3, 1, 4, 4, "Tet4", (0, 1, 2, 3))),
        (5, ("Hexahedron 8", 3, 1, 8, 8, "Hex8", tuple(range(8)))),
        (9, ("Triangle 6", 2, 2, 6, 3, "Tri6", tuple(range(6)))),
        (16, ("Quadrilateral 8", 2, 2, 8, 4, "Quad8", tuple(range(8)))),
        (11, ("Tetrahedron 10", 3, 2, 10, 4, "Tet10", (0, 1, 2, 3, 4, 5, 6, 7, 9, 8))),
        (
            17,
            (
                "Hexahedron 20",
                3,
                2,
                20,
                8,
                "Hex20",
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15),
            ),
        ),
    ],
)
def test_element_specs_define_complete_supported_ordering_contract(gmsh_type, expected):
    spec = gmsh_io._ELEMENT_SPECS[gmsh_type]

    assert (
        spec.gmsh_name,
        spec.dimension,
        spec.order,
        spec.node_count,
        spec.primary_node_count,
        spec.fem_type,
        spec.connectivity_permutation,
    ) == expected
    assert sorted(spec.connectivity_permutation) == list(range(spec.node_count))


@pytest.mark.parametrize(
    ("dimension", "element_type", "canonical_type", "node_coordinates", "node_ids"),
    [
        (
            2,
            2,
            "Tri3",
            {30: (0.0, 1.0, 0.0), 10: (0.0, 0.0, 0.0), 20: (1.0, 0.0, 0.0)},
            [10, 20, 30],
        ),
        (
            2,
            3,
            "Quad4",
            {
                40: (0.0, 1.0, 0.0),
                10: (0.0, 0.0, 0.0),
                30: (1.0, 1.0, 0.0),
                20: (1.0, 0.0, 0.0),
            },
            [10, 20, 30, 40],
        ),
        (
            3,
            4,
            "Tet4",
            {
                10: (0.0, 0.0, 0.0),
                20: (1.0, 0.0, 0.0),
                30: (0.0, 1.0, 0.0),
                40: (0.0, 0.0, 1.0),
            },
            [10, 20, 30, 40],
        ),
        (
            3,
            5,
            "Hex8",
            {
                10: (0.0, 0.0, 0.0),
                20: (1.0, 0.0, 0.0),
                30: (1.0, 1.0, 0.0),
                40: (0.0, 1.0, 0.0),
                50: (0.0, 0.0, 1.0),
                60: (1.0, 0.0, 1.0),
                70: (1.0, 1.0, 1.0),
                80: (0.0, 1.0, 1.0),
            },
            [10, 20, 30, 40, 50, 60, 70, 80],
        ),
    ],
    ids=["tri3", "quad4", "tet4", "hex8"],
)
def test_from_model_converts_each_supported_linear_type(
    dimension, element_type, canonical_type, node_coordinates, node_ids
):
    model = _fake_model(
        node_coordinates=node_coordinates,
        elements=([element_type], [[901]], [node_ids]),
    )

    result = gmsh_io.from_model(
        dimension=dimension,
        gmsh_model=model,
        plane_type="STRAIN",
        thickness=2,
    )

    expected_mesh_type = Mesh2D if dimension == 2 else Mesh3D
    assert isinstance(result.mesh, expected_mesh_type)
    assert result.mesh.dofs_per_node == dimension
    assert result.mesh.node_ids == sorted(node_ids)
    assert [node.id for node in result.mesh.nodes] == list(node_coordinates)
    assert result.mesh.elements[0].id == 901
    assert result.mesh.elements[0].node_ids == node_ids
    assert result.mesh.elements[0].type == canonical_type
    expected_props = {"plane_type": "strain", "thickness": 2.0} if dimension == 2 else {}
    assert result.mesh.elements[0].props == expected_props
    assert model.mesh.element_calls == [(dimension, -1)]
    assert model.mesh.property_calls == [element_type]
    assert model.mesh.node_calls == 1


@pytest.mark.parametrize(
    (
        "dimension",
        "element_type",
        "canonical_type",
        "node_coordinates",
        "gmsh_node_ids",
        "expected_node_ids",
    ),
    [
        (
            2,
            9,
            "Tri6",
            {
                10: (0.0, 0.0, 0.0),
                30: (1.0, 0.0, 0.0),
                50: (0.0, 1.0, 0.0),
                70: (0.5, 0.0, 0.0),
                90: (0.5, 0.5, 0.0),
                110: (0.0, 0.5, 0.0),
            },
            [10, 30, 50, 70, 90, 110],
            [10, 30, 50, 70, 90, 110],
        ),
        (
            2,
            16,
            "Quad8",
            {
                11: (0.0, 0.0, 0.0),
                22: (1.0, 0.0, 0.0),
                33: (1.0, 1.0, 0.0),
                44: (0.0, 1.0, 0.0),
                55: (0.5, 0.0, 0.0),
                66: (1.0, 0.5, 0.0),
                77: (0.5, 1.0, 0.0),
                88: (0.0, 0.5, 0.0),
            },
            [11, 22, 33, 44, 55, 66, 77, 88],
            [11, 22, 33, 44, 55, 66, 77, 88],
        ),
        (
            3,
            11,
            "Tet10",
            {
                node_id: tuple(coordinates)
                for node_id, coordinates in zip(
                    range(51, 61),
                    TET10_NATURAL_NODE_COORDS,
                    strict=True,
                )
            },
            [51, 52, 53, 54, 55, 56, 57, 58, 60, 59],
            list(range(51, 61)),
        ),
        (
            3,
            17,
            "Hex20",
            {
                node_id: tuple(coordinates)
                for node_id, coordinates in zip(
                    range(101, 121),
                    HEX20_NATURAL_NODE_COORDS,
                    strict=True,
                )
            },
            [
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                112,
                117,
                110,
                118,
                111,
                119,
                120,
                113,
                116,
                114,
                115,
            ],
            list(range(101, 121)),
        ),
    ],
    ids=["tri6", "quad8", "tet10", "hex20"],
)
def test_from_model_converts_each_supported_quadratic_type(
    dimension,
    element_type,
    canonical_type,
    node_coordinates,
    gmsh_node_ids,
    expected_node_ids,
):
    model = _fake_model(
        node_coordinates=node_coordinates,
        elements=([element_type], [[901]], [gmsh_node_ids]),
    )

    result = gmsh_io.from_model(
        dimension=dimension,
        gmsh_model=model,
        plane_type="STRAIN",
        thickness=2,
    )

    element = result.mesh.elements[0]
    assert element.id == 901
    assert element.node_ids == expected_node_ids
    assert element.type == canonical_type
    expected_props = (
        {"plane_type": "strain", "thickness": 2.0}
        if dimension == 2
        else {}
    )
    assert element.props == expected_props


def test_from_model_preserves_mixed_linear_and_quadratic_2d_block_order():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
            5: (0.5, 0.0, 0.0),
            6: (0.5, 0.5, 0.0),
            7: (0.0, 0.5, 0.0),
            8: (1.0, 0.5, 0.0),
            9: (0.5, 1.0, 0.0),
        },
        elements=(
            [2, 9, 3, 16],
            [[101], [102], [103], [104]],
            [
                [1, 2, 3],
                [1, 2, 3, 5, 6, 7],
                [1, 2, 4, 3],
                [1, 2, 4, 3, 5, 8, 9, 7],
            ],
        ),
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert [element.id for element in result.mesh.elements] == [101, 102, 103, 104]
    assert [element.type for element in result.mesh.elements] == [
        "Tri3",
        "Tri6",
        "Quad4",
        "Quad8",
    ]


def test_from_model_preserves_mixed_linear_and_quadratic_3d_block_order():
    model = _fake_model(
        node_coordinates={
            node_id: (float(node_id), float(node_id % 3), float(node_id % 5))
            for node_id in range(1, 31)
        },
        elements=(
            [4, 11, 5, 17],
            [[201], [202], [203], [204]],
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5, 6, 7, 8, 10, 9],
                list(range(11, 19)),
                [
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    22,
                    27,
                    20,
                    28,
                    21,
                    29,
                    30,
                    23,
                    26,
                    24,
                    25,
                ],
            ],
        ),
    )

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert [element.id for element in result.mesh.elements] == [201, 202, 203, 204]
    assert [element.type for element in result.mesh.elements] == [
        "Tet4",
        "Tet10",
        "Hex8",
        "Hex20",
    ]
    assert result.mesh.elements[1].node_ids == list(range(1, 11))
    assert result.mesh.elements[3].node_ids == list(range(11, 31))


def test_from_model_normalizes_clockwise_quadratic_elements_with_midsides():
    node_coordinates = {
        10: (0.0, 0.0, 0.0),
        20: (1.0, 0.0, 0.0),
        30: (0.0, 1.0, 0.0),
        40: (0.0, 0.5, 0.0),
        50: (0.5, 0.5, 0.0),
        60: (0.5, 0.0, 0.0),
        70: (1.0, 1.0, 0.0),
        80: (0.5, 1.0, 0.0),
        90: (1.0, 0.5, 0.0),
    }
    model = _fake_model(
        node_coordinates=node_coordinates,
        elements=(
            [9, 16],
            [[301], [302]],
            [
                [10, 30, 20, 40, 50, 60],
                [10, 30, 70, 20, 40, 80, 90, 60],
            ],
        ),
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.mesh.elements[0].node_ids == [10, 20, 30, 60, 50, 40]
    assert result.mesh.elements[1].node_ids == [10, 20, 70, 30, 60, 90, 80, 40]
    for element, edge_nodes in zip(
        result.mesh.elements,
        (
            ((0, 1, 3), (1, 2, 4), (2, 0, 5)),
            ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)),
        ),
        strict=True,
    ):
        for start, end, midside in edge_nodes:
            start_xy = np.asarray(node_coordinates[element.node_ids[start]][:2])
            end_xy = np.asarray(node_coordinates[element.node_ids[end]][:2])
            midside_xy = np.asarray(node_coordinates[element.node_ids[midside]][:2])
            np.testing.assert_allclose(midside_xy, 0.5 * (start_xy + end_xy))


def test_from_model_regroups_flattened_mixed_2d_blocks_and_normalizes_orientation():
    model = _fake_model(
        node_coordinates={
            999: (20.0, 20.0, 5.0),
            10: (0.0, 0.0, 0.0),
            20: (1.0, 0.0, 0.0),
            30: (1.0, 1.0, 0.0),
            40: (0.0, 1.0, 0.0),
        },
        elements=(
            [2, 3],
            [[501, 502], [700]],
            [[10, 30, 20, 10, 20, 40], [10, 40, 30, 20]],
        ),
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert [node.id for node in result.mesh.nodes] == [10, 20, 30, 40]
    assert [element.id for element in result.mesh.elements] == [501, 502, 700]
    assert [element.type for element in result.mesh.elements] == [
        "Tri3",
        "Tri3",
        "Quad4",
    ]
    assert [element.node_ids for element in result.mesh.elements] == [
        [10, 20, 30],
        [10, 20, 40],
        [10, 20, 30, 40],
    ]


def test_from_model_preserves_mixed_3d_block_order_and_coordinates():
    coordinates = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0),
        5: (1.0, 1.0, 0.0),
        6: (1.0, 0.0, 1.0),
        7: (1.0, 1.0, 1.0),
        8: (0.0, 1.0, 1.0),
    }
    model = _fake_model(
        node_coordinates=coordinates,
        elements=(
            [4, 5],
            [[91], [305]],
            [[1, 2, 3, 4], [1, 2, 5, 3, 4, 6, 7, 8]],
        ),
    )

    result = gmsh_io.from_model(dimension=3, gmsh_model=model)

    assert [element.id for element in result.mesh.elements] == [91, 305]
    assert [element.type for element in result.mesh.elements] == ["Tet4", "Hex8"]
    assert result.mesh.elements[0].node_ids == [1, 2, 3, 4]
    assert result.mesh.elements[1].node_ids == [1, 2, 5, 3, 4, 6, 7, 8]
    assert [(node.x, node.y, node.z) for node in result.mesh.nodes] == list(
        coordinates.values()
    )


@pytest.mark.parametrize(
    "elements",
    [
        ([], [], []),
        ([2], [[]], [[]]),
    ],
)
def test_from_model_rejects_mesh_without_top_dimensional_elements(elements):
    model = _fake_model(
        node_coordinates={1: (0.0, 0.0, 0.0)},
        elements=elements,
    )

    with pytest.raises(ValueError, match="no top-dimensional elements.*dimension 2"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


@pytest.mark.parametrize(
    (
        "dimension",
        "element_type",
        "node_count",
        "reported_name",
        "available_fem_type",
        "primary_node_count",
    ),
    [
        (2, 10, 9, "Quadrilateral 9", "Quad8", 4),
        (3, 12, 27, "Hexahedron 27", "Hex20", 8),
    ],
    ids=["quad9", "hex27"],
)
def test_from_model_rejects_complete_second_order_types_with_guidance_atomically(
    dimension,
    element_type,
    node_count,
    reported_name,
    available_fem_type,
    primary_node_count,
):
    model = _fake_model(
        node_coordinates={
            index: (float(index), 0.0, 0.0)
            for index in range(1, node_count + 1)
        },
        elements=([element_type], [[2]], [list(range(1, node_count + 1))]),
    )

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=dimension, gmsh_model=model)

    message = str(exc_info.value)
    assert f"unsupported Gmsh element type {element_type}" in message
    assert reported_name in message
    assert f"dimension={dimension}" in message
    assert "order=2" in message
    assert f"nodes={node_count}" in message
    assert f"primary_nodes={primary_node_count}" in message
    assert (
        f"{reported_name} is unsupported because FEM-Python provides "
        f"{available_fem_type}."
    ) in message
    assert "Mesh.SecondOrderIncomplete = 1" in message
    assert model.mesh.node_calls == 0
    assert model.physical_group_calls == 0


@pytest.mark.parametrize(
    ("element_types", "connectivity_blocks", "message"),
    [
        (
            [2, 10],
            [[1, 2, 3], list(range(1, 10))],
            "unsupported Gmsh element type 10",
        ),
        (
            [2, 9],
            [[1, 2, 3], [1, 2, 3, 4, 5]],
            "flattened connectivity.*expected 6",
        ),
    ],
    ids=["late-unsupported-quad9", "late-malformed-tri6"],
)
def test_from_model_rejects_late_invalid_mixed_block_before_reading_nodes(
    element_types,
    connectivity_blocks,
    message,
):
    model = _fake_model(
        node_coordinates={
            node_id: (float(node_id), 0.0, 0.0)
            for node_id in range(1, 10)
        },
        elements=(element_types, [[1], [2]], connectivity_blocks),
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert model.mesh.property_calls == element_types
    assert model.mesh.node_calls == 0
    assert model.physical_group_calls == 0


def test_from_model_keeps_unrelated_unsupported_type_diagnostic_generic():
    model = _fake_model(
        node_coordinates={
            index: (float(index), 0.0, 0.0)
            for index in range(1, 7)
        },
        elements=([6], [[2]], [list(range(1, 7))]),
    )

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=3, gmsh_model=model)

    message = str(exc_info.value)
    assert "unsupported Gmsh element type 6" in message
    assert "Prism 6" in message
    assert "dimension=3, order=1, nodes=6, primary_nodes=6" in message
    assert "SecondOrderIncomplete" not in message
    assert model.mesh.node_calls == 0
    assert model.physical_group_calls == 0


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        (([2, 3], [[1]], [[1, 2, 3]]), "inconsistent element block counts"),
        (([2], [[1]], [[1, 2]]), "flattened connectivity.*expected 3"),
        (([2], [[1, 2]], [[1, 2, 3]]), "flattened connectivity.*expected 6"),
    ],
)
def test_from_model_rejects_malformed_element_block_arrays(elements, message):
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=elements,
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


@pytest.mark.parametrize(
    ("dimension", "element_type", "node_ids", "message"),
    [
        (2, 9, [1, 2, 3, 4, 5], "flattened connectivity.*expected 6"),
        (3, 17, list(range(1, 20)), "flattened connectivity.*expected 20"),
    ],
    ids=["tri6", "hex20"],
)
def test_from_model_rejects_malformed_quadratic_connectivity(
    dimension, element_type, node_ids, message
):
    model = _fake_model(
        node_coordinates={
            node_id: (float(node_id), 0.0, 0.0)
            for node_id in range(1, 21)
        },
        elements=([element_type], [[1]], [node_ids]),
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=dimension, gmsh_model=model)


@pytest.mark.parametrize(
    "reported_properties",
    [
        ("Wrong Triangle", 2, 1, 3, [], 3),
        ("Triangle 3", 3, 1, 3, [], 3),
        ("Triangle 3", 2, 2, 3, [], 3),
        ("Triangle 3", 2, 1, 6, [], 3),
        ("Triangle 3", 2, 1, 3, [], 4),
    ],
)
def test_from_model_validates_supported_type_properties(reported_properties):
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[1]], [[1, 2, 3]]),
        properties={2: reported_properties},
    )

    with pytest.raises(ValueError, match="Gmsh element type 2 properties do not match"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert model.mesh.node_calls == 0


def test_from_model_rejects_duplicate_element_tags_across_blocks():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (1.0, 1.0, 0.0),
            4: (0.0, 1.0, 0.0),
        },
        elements=([2, 3], [[7], [7]], [[1, 2, 3], [1, 2, 3, 4]]),
    )

    with pytest.raises(ValueError, match="duplicate element tag 7"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


@pytest.mark.parametrize(
    ("node_tags", "coordinates", "message"),
    [
        ([1, 1, 3], [0.0, 0.0, 0.0] * 3, "duplicate node tag 1"),
        ([0, 2, 3], [0.0, 0.0, 0.0] * 3, "node tags must be positive integers"),
        ([1, 2, 3], [0.0] * 8, "coordinate vector length.*expected 9"),
        ([1, 2, 3], [0.0, 0.0, 0.0, 1.0, float("nan"), 0.0, 0.0, 1.0, 0.0], "coordinates must be finite"),
    ],
)
def test_from_model_rejects_malformed_node_arrays(node_tags, coordinates, message):
    mesh = _FakeMesh(
        node_tags=node_tags,
        coordinates=coordinates,
        elements=([2], [[1]], [[1, 2, 3]]),
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=_FakeModel(mesh))


def test_from_model_rejects_missing_referenced_node():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[1]], [[1, 2, 99]]),
    )

    with pytest.raises(ValueError, match="element references missing node tag 99"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_from_model_rejects_missing_referenced_midside_node():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.5, 0.0, 0.0),
            5: (0.5, 0.5, 0.0),
        },
        elements=([9], [[1]], [[1, 2, 3, 4, 5, 99]]),
    )

    with pytest.raises(ValueError, match="element references missing node tag 99"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_from_model_rejects_nonplanar_retained_node_beyond_tolerance():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 1e-4),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[1]], [[1, 2, 3]]),
    )

    with pytest.raises(ValueError, match=r"node 2.*z=.*outside.*tolerance"):
        gmsh_io.from_model(
            dimension=2,
            gmsh_model=model,
            z_tolerance=1e-6,
        )


def test_from_model_rejects_nonplanar_retained_midside_node():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (0.5, 0.0, 1e-4),
            5: (0.5, 0.5, 0.0),
            6: (0.0, 0.5, 0.0),
        },
        elements=([9], [[1]], [[1, 2, 3, 4, 5, 6]]),
    )

    with pytest.raises(ValueError, match=r"node 4.*z=.*outside.*tolerance"):
        gmsh_io.from_model(
            dimension=2,
            gmsh_model=model,
            z_tolerance=1e-6,
        )


def test_from_model_keeps_quadratic_midsides_in_physical_boundary_node_sets():
    mesh = _FakeMesh(
        node_tags=[10, 20, 30, 40, 50, 60, 999],
        coordinates=[
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            2.0,
            2.0,
            0.0,
        ],
        elements=([9], [[501]], [[10, 20, 30, 40, 50, 60]]),
        entity_elements={(1, 71): ([8], [[701]], [[20, 10, 40]])},
        physical_nodes={(1, 7): [10, 20, 40, 999]},
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(1, 7)],
        physical_names={(1, 7): "BOTTOM"},
        entities={(1, 7): [71]},
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.node_sets["BOTTOM"] == NodeSet("BOTTOM", [10, 20, 40])
    assert result.edges["BOTTOM"] == Edge(
        "BOTTOM",
        [ElementEdge(501, 0, [10, 40, 20])],
    )


def _physical_group_model():
    mesh = _FakeMesh(
        node_tags=[40, 10, 30, 20, 999],
        coordinates=[
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            5.0,
            5.0,
            0.0,
        ],
        elements=([2], [[502, 501]], [[10, 30, 40, 10, 20, 30]]),
        entity_elements={
            (1, 201): ([1], [[701]], [[40, 10]]),
            (2, 101): ([2], [[502, 900]], [[10, 30, 40, 10, 20, 30]]),
            (2, 102): ([2], [[501]], [[10, 20, 30]]),
            (2, 103): ([2], [[999]], [[10, 20, 30]]),
        },
        physical_nodes={
            (1, 2): [40, 999, 10],
            (0, 3): [20],
            (1, 4): [999],
        },
    )
    return _FakeModel(
        mesh,
        physical_groups=[(2, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 6)],
        physical_names={
            (2, 1): "DOMAIN",
            (1, 2): "LEFT",
            (0, 3): "",
            (1, 4): "EMPTY_BOUNDARY",
            (2, 5): "EMPTY_DOMAIN",
            (3, 6): "OUTSIDE_SUBMODEL",
        },
        entities={(1, 2): [201], (2, 1): [101, 102], (2, 5): [103]},
    )


def test_from_model_converts_named_unnamed_and_skipped_physical_groups():
    result = gmsh_io.from_model(dimension=2, gmsh_model=_physical_group_model())

    assert result.node_sets == {
        "LEFT": NodeSet("LEFT", [10, 40]),
        "physical_0_3": NodeSet("physical_0_3", [20]),
    }
    assert result.element_sets == {
        "DOMAIN": ElementSet("DOMAIN", [501, 502]),
    }
    assert result.metadata == {
        "source": "gmsh",
        "dimension": 2,
        "physical_groups": {
            "DOMAIN": {"dimension": 2, "tag": 1, "kind": "element_set"},
            "LEFT": {
                "dimension": 1,
                "tag": 2,
                "kind": "node_set",
                "boundary_kind": "edge",
                "boundary_entry_count": 1,
            },
            "physical_0_3": {"dimension": 0, "tag": 3, "kind": "node_set"},
        },
        "skipped_physical_groups": (
            {
                "name": "EMPTY_BOUNDARY",
                "dimension": 1,
                "tag": 4,
                "kind": "node_set",
            },
            {
                "name": "EMPTY_DOMAIN",
                "dimension": 2,
                "tag": 5,
                "kind": "element_set",
            },
        ),
    }


def test_from_model_records_live_gmsh_version_in_metadata(monkeypatch):
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[1]], [[1, 2, 3]]),
    )
    monkeypatch.setattr(
        gmsh_io,
        "_resolve_live_backend",
        lambda: (model, "4.15.2"),
    )

    result = gmsh_io.from_model(dimension=2)

    assert result.metadata["gmsh_version"] == "4.15.2"


@pytest.mark.parametrize(
    ("groups", "names", "physical_nodes", "entities", "entity_elements", "message"),
    [
        (
            [(1, 1), (0, 2)],
            {(1, 1): "DUPLICATE", (0, 2): "DUPLICATE"},
            {(1, 1): [1, 2], (0, 2): [2]},
            {(1, 1): [201]},
            {(1, 201): ([1], [[301]], [[1, 2]])},
            "duplicate physical-group name 'DUPLICATE' in node-set namespace",
        ),
        (
            [(2, 1), (2, 2)],
            {(2, 1): "DUPLICATE", (2, 2): "DUPLICATE"},
            {},
            {(2, 1): [101], (2, 2): [102]},
            {
                (2, 101): ([2], [[1]], [[1, 2, 3]]),
                (2, 102): ([2], [[1]], [[1, 2, 3]]),
            },
            "duplicate physical-group name 'DUPLICATE' in element-set namespace",
        ),
    ],
)
def test_from_model_rejects_physical_name_collisions_within_namespace(
    groups, names, physical_nodes, entities, entity_elements, message
):
    mesh = _FakeMesh(
        node_tags=[1, 2, 3],
        coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        elements=([2], [[1]], [[1, 2, 3]]),
        physical_nodes=physical_nodes,
        entity_elements=entity_elements,
    )
    model = _FakeModel(
        mesh,
        physical_groups=groups,
        physical_names=names,
        entities=entities,
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


def test_same_physical_name_is_allowed_once_in_each_set_namespace():
    mesh = _FakeMesh(
        node_tags=[1, 2, 3],
        coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        elements=([2], [[11]], [[1, 2, 3]]),
        physical_nodes={(1, 1): [1, 2]},
        entity_elements={
            (1, 201): ([1], [[301]], [[1, 2]]),
            (2, 101): ([2], [[11]], [[1, 2, 3]]),
        },
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(1, 1), (2, 2)],
        physical_names={(1, 1): "SHARED", (2, 2): "SHARED"},
        entities={(1, 1): [201], (2, 2): [101]},
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.node_sets == {"SHARED": NodeSet("SHARED", [1, 2])}
    assert result.element_sets == {"SHARED": ElementSet("SHARED", [11])}
    assert result.metadata["physical_groups"] == {
        "SHARED": (
            {
                "dimension": 1,
                "tag": 1,
                "kind": "node_set",
                "boundary_kind": "edge",
                "boundary_entry_count": 1,
            },
            {"dimension": 2, "tag": 2, "kind": "element_set"},
        )
    }


def test_cross_namespace_metadata_does_not_overwrite_literal_prefixed_name():
    mesh = _FakeMesh(
        node_tags=[1, 2, 3],
        coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        elements=([2], [[11]], [[1, 2, 3]]),
        physical_nodes={(1, 1): [1, 2], (0, 2): [2]},
        entity_elements={
            (1, 201): ([1], [[301]], [[1, 2]]),
            (2, 101): ([2], [[11]], [[1, 2, 3]]),
        },
    )
    model = _FakeModel(
        mesh,
        physical_groups=[(1, 1), (0, 2), (2, 3)],
        physical_names={
            (1, 1): "node_set:SHARED",
            (0, 2): "SHARED",
            (2, 3): "SHARED",
        },
        entities={(1, 1): [201], (2, 3): [101]},
    )

    result = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert result.metadata["physical_groups"] == {
        "node_set:SHARED": {
            "dimension": 1,
            "tag": 1,
            "kind": "node_set",
            "boundary_kind": "edge",
            "boundary_entry_count": 1,
        },
        "SHARED": (
            {"dimension": 0, "tag": 2, "kind": "node_set"},
            {"dimension": 2, "tag": 3, "kind": "element_set"},
        ),
    }


def test_to_fem_model_deep_copies_import_metadata():
    imported = gmsh_io.from_model(dimension=2, gmsh_model=_physical_group_model())

    model = imported.to_fem_model("copied")
    model.metadata["physical_groups"]["LEFT"]["tag"] = 999
    model.metadata["skipped_physical_groups"][0]["tag"] = 999

    assert imported.metadata["physical_groups"]["LEFT"]["tag"] == 2
    assert imported.metadata["skipped_physical_groups"][0]["tag"] == 4


@pytest.fixture
def live_gmsh():
    gmsh = pytest.importorskip("gmsh")
    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()

    option_names = (
        "General.Terminal",
        "Mesh.ElementOrder",
        "Mesh.SecondOrderIncomplete",
        "Mesh.RecombineAll",
    )
    previous_options = {
        name: gmsh.option.getNumber(name)
        for name in option_names
    }
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.clear()
        gmsh.model.add("fem_gmsh_adapter_test")
        yield gmsh
    finally:
        gmsh.clear()
        for name, value in previous_options.items():
            gmsh.option.setNumber(name, value)
        if owns_session:
            gmsh.finalize()


@pytest.mark.parametrize(
    "element_type",
    sorted(gmsh_io._BOUNDARY_ELEMENT_SPECS),
)
def test_real_gmsh_boundary_element_properties_match_adapter_contract(
    live_gmsh,
    element_type,
):
    spec = gmsh_io._BOUNDARY_ELEMENT_SPECS[element_type]

    assert gmsh_io._reported_element_properties(
        live_gmsh.model.mesh,
        element_type,
    ) == (
        spec.gmsh_name,
        spec.dimension,
        spec.order,
        spec.node_count,
        spec.primary_node_count,
    )


@pytest.mark.parametrize(
    ("element_type", "expected_coordinates"),
    [
        (
            9,
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.0],
                    [0.5, 0.5],
                    [0.0, 0.5],
                ]
            ),
        ),
        (
            16,
            np.array(
                [
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
        ),
        (11, np.asarray(TET10_NATURAL_NODE_COORDS)),
        (17, HEX20_NATURAL_NODE_COORDS),
    ],
    ids=["tri6", "quad8", "tet10", "hex20"],
)
def test_real_gmsh_high_order_local_coordinates_match_fem_order(
    live_gmsh, element_type, expected_coordinates
):
    properties = live_gmsh.model.mesh.getElementProperties(element_type)
    spec = gmsh_io._ELEMENT_SPECS[element_type]
    local_coordinates = np.asarray(properties[4], dtype=float).reshape(
        spec.node_count,
        spec.dimension,
    )

    assert properties[5] == spec.primary_node_count
    np.testing.assert_allclose(
        local_coordinates[list(spec.connectivity_permutation)],
        expected_coordinates,
    )


def test_real_gmsh_complete_quad9_reports_incomplete_second_order_guidance(live_gmsh):
    gmsh = live_gmsh
    point_1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    point_2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)
    point_3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
    point_4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
    lines = [
        gmsh.model.geo.addLine(point_1, point_2),
        gmsh.model.geo.addLine(point_2, point_3),
        gmsh.model.geo.addLine(point_3, point_4),
        gmsh.model.geo.addLine(point_4, point_1),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    for line in lines:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 3)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.model.mesh.generate(2)
    assert 10 in gmsh.model.mesh.getElementTypes(2)

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=2)

    message = str(exc_info.value)
    assert "Quadrilateral 9 is unsupported because FEM-Python provides Quad8." in message
    assert "Mesh.SecondOrderIncomplete = 1" in message


def _vertical_boundary_tags(gmsh, owner, x):
    tags = []
    for boundary_dimension, boundary_tag in gmsh.model.getBoundary(
        [owner],
        oriented=False,
        recursive=False,
    ):
        if boundary_dimension != owner[0] - 1:
            continue
        x_min, _, _, x_max, _, _ = gmsh.model.getBoundingBox(
            boundary_dimension,
            boundary_tag,
        )
        if abs(x_min - x) <= 1e-6 and abs(x_max - x) <= 1e-6:
            tags.append(boundary_tag)
    return tags


def _entity_tags_at_x(gmsh, dimension, x):
    tags = []
    for _, tag in gmsh.model.getEntities(dimension):
        x_min, _, _, x_max, _, _ = gmsh.model.getBoundingBox(dimension, tag)
        if abs(x_min - x) <= 1e-6 and abs(x_max - x) <= 1e-6:
            tags.append(tag)
    return tags


def _assert_kernel_accepts_imported_order(mesh):
    for element in mesh.elements:
        element.props.update({"E": 1000.0, "nu": 0.3})
        stiffness = get_element_kernel(element.type).stiffness(mesh, element)
        assert np.all(np.isfinite(stiffness))


def _quadratic_boundary_midside_ids(gmsh, entity_tags):
    midside_ids = set()
    for entity_tag in entity_tags:
        element_types, _, connectivity_blocks = gmsh.model.mesh.getElements(
            1,
            entity_tag,
        )
        for element_type, connectivity in zip(
            element_types,
            connectivity_blocks,
            strict=True,
        ):
            properties = gmsh.model.mesh.getElementProperties(element_type)
            if int(properties[2]) != 2:
                continue
            node_count = int(properties[3])
            primary_node_count = int(properties[5])
            node_ids = [int(value) for value in connectivity]
            for start in range(0, len(node_ids), node_count):
                midside_ids.update(
                    node_ids[
                        start + primary_node_count : start + node_count
                    ]
                )
    return midside_ids


def _assert_edge_midsides_are_geometric_midpoints(mesh, edge_node_indices):
    coordinates = {}
    for node in mesh.nodes:
        values = [node.x, node.y]
        if hasattr(node, "z"):
            values.append(node.z)
        coordinates[node.id] = np.asarray(values, dtype=float)

    for element in mesh.elements:
        for start, end, midside in edge_node_indices:
            start_coordinates = coordinates[element.node_ids[start]]
            end_coordinates = coordinates[element.node_ids[end]]
            midside_coordinates = coordinates[element.node_ids[midside]]
            np.testing.assert_allclose(
                midside_coordinates,
                0.5 * (start_coordinates + end_coordinates),
                atol=1e-12,
            )


def _assembled_resultant(model, step):
    load_vector = build_load_vector(
        model.mesh,
        boundary_for_step(model, step),
    )
    return np.asarray(
        [
            sum(
                load_vector[model.mesh.global_dof(node.id, component)]
                for node in model.mesh.nodes
            )
            for component in range(model.mesh.dofs_per_node)
        ]
    )


def _solve_imported_plate(imported, name):
    model = imported.to_fem_model(name)
    material_name = f"{name}_elastic"
    elastic = materials.linear_elastic.material(material_name, E=1000.0, nu=0.3)
    materials.add(model, elastic)
    materials.assign(model, material_name, "DOMAIN")
    load_step = steps.static("pull")
    steps.displacement(load_step, "LEFT", components=(1, 2))
    steps.edge_traction(load_step, "RIGHT", vector=(2.0, 0.0))
    steps.add(model, load_step)

    validate_model(model)
    np.testing.assert_allclose(_assembled_resultant(model, load_step), (2.0, 0.0))

    result = static_linear.solve(model, "pull")
    right_displacements = [
        result.U[model.mesh.global_dof(node_id, 0)]
        for node_id in model.node_sets["RIGHT"].node_ids
    ]
    assert np.all(np.isfinite(result.U))
    assert np.mean(right_displacements) > 0.0
    return result


def test_real_gmsh_rectangle_reaches_solver_and_vtk(live_gmsh, tmp_path):
    gmsh = live_gmsh
    surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
    gmsh.model.occ.synchronize()
    left = _vertical_boundary_tags(gmsh, (2, surface), 0.0)
    right = _vertical_boundary_tags(gmsh, (2, surface), 2.0)
    assert left and right

    domain_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, domain_group, "DOMAIN")
    left_group = gmsh.model.addPhysicalGroup(1, left)
    gmsh.model.setPhysicalName(1, left_group, "LEFT")
    right_group = gmsh.model.addPhysicalGroup(1, right)
    gmsh.model.setPhysicalName(1, right_group, "RIGHT")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.35)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(2)

    imported = gmsh_io.from_model(dimension=2)

    assert isinstance(imported.mesh, Mesh2D)
    assert {element.type for element in imported.mesh.elements} == {"Tri3"}
    assert all(node.id > 0 for node in imported.mesh.nodes)
    assert all(element.id > 0 for element in imported.mesh.elements)
    assert set(imported.element_sets["DOMAIN"].element_ids) == {
        element.id for element in imported.mesh.elements
    }
    assert imported.node_sets["LEFT"].node_ids
    assert imported.node_sets["RIGHT"].node_ids
    assert imported.edges["LEFT"].edges
    assert imported.edges["RIGHT"].edges
    assert imported.metadata["gmsh_version"] == gmsh.__version__

    coordinates = {node.id: (node.x, node.y) for node in imported.mesh.nodes}
    for element in imported.mesh.elements:
        signed_twice_area = 0.0
        for index, node_id in enumerate(element.node_ids):
            next_node_id = element.node_ids[(index + 1) % len(element.node_ids)]
            x, y = coordinates[node_id]
            next_x, next_y = coordinates[next_node_id]
            signed_twice_area += x * next_y - next_x * y
        assert signed_twice_area > 0.0

    model = imported.to_fem_model("gmsh_rectangle")
    steel = materials.linear_elastic.material("steel", E=1000.0, nu=0.3)
    materials.add(model, steel)
    materials.assign(model, "steel", "DOMAIN")
    load_step = steps.static("pull")
    steps.displacement(load_step, "LEFT", components=(1, 2))
    steps.edge_traction(load_step, "RIGHT", vector=(2.0, 0.0))
    steps.add(model, load_step)

    validate_model(model)
    np.testing.assert_allclose(_assembled_resultant(model, load_step), (2.0, 0.0))

    result = static_linear.solve(model, "pull")
    right_displacements = [
        result.U[model.mesh.global_dof(node_id, 0)]
        for node_id in model.node_sets["RIGHT"].node_ids
    ]
    assert np.all(np.isfinite(result.U))
    assert np.mean(right_displacements) > 0.0

    post.vtk.export.from_result(
        result,
        output_dir=tmp_path,
        name="gmsh_rectangle",
    )
    assert (tmp_path / "gmsh_rectangle.vtk").is_file()


def test_real_gmsh_tetrahedral_box_reaches_surface_load_solver_and_vtk(
    live_gmsh,
    tmp_path,
):
    gmsh = live_gmsh
    volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    left_faces = _vertical_boundary_tags(gmsh, (3, volume), 0.0)
    right_faces = _vertical_boundary_tags(gmsh, (3, volume), 1.0)
    assert left_faces and right_faces

    volume_group = gmsh.model.addPhysicalGroup(3, [volume])
    gmsh.model.setPhysicalName(3, volume_group, "VOLUME")
    left_group = gmsh.model.addPhysicalGroup(2, left_faces)
    gmsh.model.setPhysicalName(2, left_group, "LEFT")
    right_group = gmsh.model.addPhysicalGroup(2, right_faces)
    gmsh.model.setPhysicalName(2, right_group, "RIGHT")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(3)

    imported = gmsh_io.from_model(dimension=3)

    assert isinstance(imported.mesh, Mesh3D)
    assert {element.type for element in imported.mesh.elements} == {"Tet4"}
    assert imported.element_sets["VOLUME"].element_ids
    assert imported.node_sets["LEFT"].node_ids
    assert imported.node_sets["RIGHT"].node_ids
    assert imported.surfaces["LEFT"].faces
    assert imported.surfaces["RIGHT"].faces
    _assert_kernel_accepts_imported_order(imported.mesh)

    model = imported.to_fem_model("gmsh_box")
    steel = materials.linear_elastic.material("steel", E=1000.0, nu=0.3)
    materials.add(model, steel)
    materials.assign(model, "steel", "VOLUME")
    load_step = steps.static("pull")
    steps.displacement(load_step, "LEFT", components=(1, 2, 3))
    steps.surface_traction(load_step, "RIGHT", vector=(1.5, 0.0, 0.0))
    steps.add(model, load_step)

    validate_model(model)
    np.testing.assert_allclose(
        _assembled_resultant(model, load_step),
        (1.5, 0.0, 0.0),
        atol=1e-12,
    )
    result = static_linear.solve(model, "pull")
    right_displacements = [
        result.U[model.mesh.global_dof(node_id, 0)]
        for node_id in model.node_sets["RIGHT"].node_ids
    ]
    assert np.all(np.isfinite(result.U))
    assert np.mean(right_displacements) > 0.0

    post.vtk.export.from_result(
        result,
        output_dir=tmp_path,
        name="gmsh_box",
    )
    assert (tmp_path / "gmsh_box.vtk").is_file()

    cells, cell_types, elements = post.vtk.cells.build(imported.mesh)
    assert len(cells) == len(imported.mesh.elements)
    assert cell_types == [10] * len(imported.mesh.elements)
    assert elements == imported.mesh.elements


def test_real_gmsh_extrusion_confirms_hex8_kernel_order(live_gmsh):
    gmsh = live_gmsh
    point_1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    point_2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)
    point_3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
    point_4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
    lines = [
        gmsh.model.geo.addLine(point_1, point_2),
        gmsh.model.geo.addLine(point_2, point_3),
        gmsh.model.geo.addLine(point_3, point_4),
        gmsh.model.geo.addLine(point_4, point_1),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    for line in lines:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 3)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    extruded = gmsh.model.geo.extrude(
        [(2, surface)],
        0.0,
        0.0,
        1.0,
        numElements=[1],
        recombine=True,
    )
    gmsh.model.geo.synchronize()
    assert any(entity_dimension == 3 for entity_dimension, _ in extruded)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)

    imported = gmsh_io.from_model(dimension=3)

    assert {element.type for element in imported.mesh.elements} == {"Hex8"}
    _assert_kernel_accepts_imported_order(imported.mesh)


def test_real_gmsh_tri6_rectangle_reaches_solver_and_vtk(live_gmsh):
    gmsh = live_gmsh
    surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
    gmsh.model.occ.synchronize()
    left = _vertical_boundary_tags(gmsh, (2, surface), 0.0)
    right = _vertical_boundary_tags(gmsh, (2, surface), 2.0)
    assert left and right

    domain_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, domain_group, "DOMAIN")
    left_group = gmsh.model.addPhysicalGroup(1, left)
    gmsh.model.setPhysicalName(1, left_group, "LEFT")
    right_group = gmsh.model.addPhysicalGroup(1, right)
    gmsh.model.setPhysicalName(1, right_group, "RIGHT")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.4)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(2)

    imported = gmsh_io.from_model(dimension=2)

    assert isinstance(imported.mesh, Mesh2D)
    assert {element.type for element in imported.mesh.elements} == {"Tri6"}
    left_midsides = _quadratic_boundary_midside_ids(gmsh, left)
    right_midsides = _quadratic_boundary_midside_ids(gmsh, right)
    assert left_midsides
    assert right_midsides
    assert left_midsides <= set(imported.node_sets["LEFT"].node_ids)
    assert right_midsides <= set(imported.node_sets["RIGHT"].node_ids)
    assert left_midsides <= {
        node_id
        for edge in imported.edges["LEFT"].edges
        for node_id in edge.node_ids
    }
    assert right_midsides <= {
        node_id
        for edge in imported.edges["RIGHT"].edges
        for node_id in edge.node_ids
    }

    _solve_imported_plate(imported, "gmsh_tri6_rectangle")
    _, cell_types, _ = post.vtk.cells.build(imported.mesh)
    assert cell_types == [22] * len(imported.mesh.elements)


def test_real_gmsh_quad8_rectangle_reaches_solver_and_vtk(live_gmsh):
    gmsh = live_gmsh
    point_1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    point_2 = gmsh.model.geo.addPoint(2.0, 0.0, 0.0)
    point_3 = gmsh.model.geo.addPoint(2.0, 1.0, 0.0)
    point_4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
    lines = [
        gmsh.model.geo.addLine(point_1, point_2),
        gmsh.model.geo.addLine(point_2, point_3),
        gmsh.model.geo.addLine(point_3, point_4),
        gmsh.model.geo.addLine(point_4, point_1),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    for line in lines:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 3)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    gmsh.model.geo.synchronize()

    domain_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, domain_group, "DOMAIN")
    left_group = gmsh.model.addPhysicalGroup(1, [lines[3]])
    gmsh.model.setPhysicalName(1, left_group, "LEFT")
    right_group = gmsh.model.addPhysicalGroup(1, [lines[1]])
    gmsh.model.setPhysicalName(1, right_group, "RIGHT")
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)
    assert 16 in gmsh.model.mesh.getElementTypes(2)

    imported = gmsh_io.from_model(dimension=2)

    assert {element.type for element in imported.mesh.elements} == {"Quad8"}
    _assert_edge_midsides_are_geometric_midpoints(
        imported.mesh,
        ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)),
    )
    left_midsides = _quadratic_boundary_midside_ids(gmsh, [lines[3]])
    right_midsides = _quadratic_boundary_midside_ids(gmsh, [lines[1]])
    assert left_midsides <= set(imported.node_sets["LEFT"].node_ids)
    assert right_midsides <= set(imported.node_sets["RIGHT"].node_ids)
    assert left_midsides <= {
        node_id
        for edge in imported.edges["LEFT"].edges
        for node_id in edge.node_ids
    }
    assert right_midsides <= {
        node_id
        for edge in imported.edges["RIGHT"].edges
        for node_id in edge.node_ids
    }

    _solve_imported_plate(imported, "gmsh_quad8_rectangle")
    _, cell_types, _ = post.vtk.cells.build(imported.mesh)
    assert cell_types == [23] * len(imported.mesh.elements)


def test_real_gmsh_tet10_box_reaches_kernel_sets_and_vtk(live_gmsh):
    gmsh = live_gmsh
    volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    left_faces = _vertical_boundary_tags(gmsh, (3, volume), 0.0)
    assert left_faces

    volume_group = gmsh.model.addPhysicalGroup(3, [volume])
    gmsh.model.setPhysicalName(3, volume_group, "VOLUME")
    left_group = gmsh.model.addPhysicalGroup(2, left_faces)
    gmsh.model.setPhysicalName(2, left_group, "LEFT")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(3)

    element_types, element_tag_blocks, connectivity_blocks = (
        gmsh.model.mesh.getElements(3, -1)
    )
    block_index = list(element_types).index(11)
    first_element_id = int(element_tag_blocks[block_index][0])
    raw_node_ids = [int(value) for value in connectivity_blocks[block_index][:10]]

    imported = gmsh_io.from_model(dimension=3)

    assert isinstance(imported.mesh, Mesh3D)
    assert {element.type for element in imported.mesh.elements} == {"Tet10"}
    imported_by_id = {element.id: element for element in imported.mesh.elements}
    expected_node_ids = [
        raw_node_ids[index]
        for index in (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
    ]
    assert imported_by_id[first_element_id].node_ids == expected_node_ids
    assert imported.element_sets["VOLUME"].element_ids
    assert imported.node_sets["LEFT"].node_ids
    model = imported.to_fem_model("gmsh_tet10_box")
    assert model.mesh is imported.mesh
    assert model.element_sets["VOLUME"].element_ids
    assert model.node_sets["LEFT"].node_ids
    _assert_kernel_accepts_imported_order(imported.mesh)
    _, cell_types, _ = post.vtk.cells.build(imported.mesh)
    assert cell_types == [24] * len(imported.mesh.elements)


def test_real_gmsh_hex20_extrusion_reaches_kernel_sets_and_vtk(live_gmsh):
    gmsh = live_gmsh
    point_1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    point_2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)
    point_3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
    point_4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
    lines = [
        gmsh.model.geo.addLine(point_1, point_2),
        gmsh.model.geo.addLine(point_2, point_3),
        gmsh.model.geo.addLine(point_3, point_4),
        gmsh.model.geo.addLine(point_4, point_1),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    for line in lines:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 3)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    extruded = gmsh.model.geo.extrude(
        [(2, surface)],
        0.0,
        0.0,
        1.0,
        numElements=[1],
        recombine=True,
    )
    gmsh.model.geo.synchronize()
    volume = next(tag for dimension, tag in extruded if dimension == 3)
    left_faces = _vertical_boundary_tags(gmsh, (3, volume), 0.0)
    right_faces = _vertical_boundary_tags(gmsh, (3, volume), 1.0)
    assert left_faces and right_faces

    volume_group = gmsh.model.addPhysicalGroup(3, [volume])
    gmsh.model.setPhysicalName(3, volume_group, "VOLUME")
    left_group = gmsh.model.addPhysicalGroup(2, left_faces)
    gmsh.model.setPhysicalName(2, left_group, "LEFT")
    right_group = gmsh.model.addPhysicalGroup(2, right_faces)
    gmsh.model.setPhysicalName(2, right_group, "RIGHT")
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(3)
    assert 17 in gmsh.model.mesh.getElementTypes(3)

    imported = gmsh_io.from_model(dimension=3)

    assert {element.type for element in imported.mesh.elements} == {"Hex20"}
    _assert_edge_midsides_are_geometric_midpoints(
        imported.mesh,
        (
            (0, 1, 8),
            (1, 2, 9),
            (2, 3, 10),
            (3, 0, 11),
            (4, 5, 12),
            (5, 6, 13),
            (6, 7, 14),
            (7, 4, 15),
            (0, 4, 16),
            (1, 5, 17),
            (2, 6, 18),
            (3, 7, 19),
        ),
    )
    assert imported.element_sets["VOLUME"].element_ids
    assert imported.node_sets["LEFT"].node_ids
    assert imported.node_sets["RIGHT"].node_ids
    assert imported.surfaces["LEFT"].faces
    assert imported.surfaces["RIGHT"].faces
    assert all(len(face.node_ids) == 8 for face in imported.surfaces["RIGHT"].faces)
    model = imported.to_fem_model("gmsh_hex20_extrusion")
    assert model.mesh is imported.mesh
    assert model.element_sets["VOLUME"].element_ids
    assert model.node_sets["LEFT"].node_ids
    load_step = steps.static("traction")
    steps.surface_traction(load_step, "RIGHT", vector=(2.0, 0.0, 0.0))
    steps.add(model, load_step)
    validate_model(model)
    np.testing.assert_allclose(
        _assembled_resultant(model, load_step),
        (2.0, 0.0, 0.0),
        atol=1e-12,
    )
    _assert_kernel_accepts_imported_order(imported.mesh)
    _, cell_types, _ = post.vtk.cells.build(imported.mesh)
    assert cell_types == [25] * len(imported.mesh.elements)


def test_real_gmsh_internal_curve_reports_all_owners(live_gmsh):
    gmsh = live_gmsh
    left = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    right = gmsh.model.occ.addRectangle(1.0, 0.0, 0.0, 1.0, 1.0)
    gmsh.model.occ.fragment([(2, left)], [(2, right)])
    gmsh.model.occ.synchronize()

    surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
    interfaces = _entity_tags_at_x(gmsh, 1, 1.0)
    assert len(surfaces) == 2
    assert interfaces
    domain_group = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(2, domain_group, "DOMAIN")
    interface_group = gmsh.model.addPhysicalGroup(1, interfaces)
    gmsh.model.setPhysicalName(1, interface_group, "INTERFACE")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.4)
    gmsh.model.mesh.generate(2)

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=2)

    message = str(exc_info.value)
    assert "INTERFACE" in message
    assert "multiple FEM owners" in message


def test_real_gmsh_internal_surface_reports_all_owners(live_gmsh):
    gmsh = live_gmsh
    left = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    right = gmsh.model.occ.addBox(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    gmsh.model.occ.fragment([(3, left)], [(3, right)])
    gmsh.model.occ.synchronize()

    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    interfaces = _entity_tags_at_x(gmsh, 2, 1.0)
    assert len(volumes) == 2
    assert interfaces
    volume_group = gmsh.model.addPhysicalGroup(3, volumes)
    gmsh.model.setPhysicalName(3, volume_group, "VOLUME")
    interface_group = gmsh.model.addPhysicalGroup(2, interfaces)
    gmsh.model.setPhysicalName(2, interface_group, "INTERFACE")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.model.mesh.generate(3)

    with pytest.raises(ValueError) as exc_info:
        gmsh_io.from_model(dimension=3)

    message = str(exc_info.value)
    assert "INTERFACE" in message
    assert "multiple FEM owners" in message
