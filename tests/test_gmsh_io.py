from __future__ import annotations

import builtins
import inspect
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import get_type_hints

import numpy as np
import pytest

from fem.core import Mesh2D, Mesh3D
from fem.elements import get_element_kernel
from fem.elements.hexahedron import HEX20_NATURAL_NODE_COORDS
from fem.elements.tetrahedron import TET10_NATURAL_NODE_COORDS
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


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
    ):
        self.node_tags = node_tags
        self.coordinates = coordinates
        self.elements = elements
        self.properties = dict(_ELEMENT_PROPERTIES)
        if properties:
            self.properties.update(properties)
        self.element_calls = []
        self.node_calls = 0
        self.property_calls = []

    def getElements(self, dimension, entity_tag):
        self.element_calls.append((dimension, entity_tag))
        return self.elements

    def getElementProperties(self, element_type):
        self.property_calls.append(element_type)
        return self.properties[element_type]

    def getNodes(self):
        self.node_calls += 1
        return self.node_tags, self.coordinates, []


class _FakeModel:
    """Minimal backend exposing only the mesh conversion contract."""

    def __init__(self, mesh):
        self.mesh = mesh


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


class _FakeGmshMeshRef:
    def __init__(self, dimension, gmsh_model, *, borrow_error=None):
        self.dimension = dimension
        self._gmsh_model = gmsh_model
        self._borrow_error = borrow_error
        self.borrow_calls = 0

    def _borrow_model(self):
        self.borrow_calls += 1
        if self._borrow_error is not None:
            raise self._borrow_error
        return self._gmsh_model


def _install_fake_mesh_ref(monkeypatch):
    monkeypatch.setattr(gmsh_io, "GmshMeshRef", _FakeGmshMeshRef)

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


@pytest.mark.parametrize("dimension", [0, 4, "2", None])
def test_from_model_rejects_invalid_dimension_before_reading_backend(dimension):
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        gmsh_io.from_model(dimension=dimension, gmsh_model=_UnreadableModel())


@pytest.mark.parametrize("line_element_type", [None, "", "truss2", "Beam3", 1])
def test_from_model_requires_canonical_line_element_type_before_reading_backend(
    line_element_type,
):
    with pytest.raises(ValueError, match="line_element_type.*Truss2.*Beam2"):
        gmsh_io.from_model(
            dimension=1,
            line_element_type=line_element_type,
            gmsh_model=_UnreadableModel(),
        )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("line_element_type", ["Truss2", "Beam2", "Line2"])
def test_from_model_rejects_line_element_type_for_other_dimensions_before_backend(
    dimension,
    line_element_type,
):
    with pytest.raises(ValueError, match="line_element_type.*dimension 1"):
        gmsh_io.from_model(
            dimension=dimension,
            line_element_type=line_element_type,
            gmsh_model=_UnreadableModel(),
        )


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

def test_public_api_exports_only_mesh_conversion_entry_points():
    assert gmsh_io.__all__ == ["read", "from_model"]


def test_public_signatures_keep_formulation_arguments_in_io_layer():
    read_signature = inspect.signature(gmsh_io.read)
    from_model_signature = inspect.signature(gmsh_io.from_model)

    assert list(read_signature.parameters) == [
        "source",
        "line_element_type",
        "plane_type",
        "thickness",
        "z_tolerance",
    ]
    assert list(from_model_signature.parameters) == [
        "dimension",
        "gmsh_model",
        "line_element_type",
        "plane_type",
        "thickness",
        "z_tolerance",
    ]
    assert read_signature.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in read_signature.parameters.items()
        if name != "source"
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in from_model_signature.parameters.values()
    )
    assert get_type_hints(gmsh_io.read)["source"] is gmsh_meshing.GmshMeshRef
    assert gmsh_io.GmshMeshRef is gmsh_meshing.GmshMeshRef


def test_from_model_returns_mesh_with_minimal_backend_contract():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[11]], [[1, 2, 3]]),
    )

    mesh = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert isinstance(mesh, Mesh2D)
    assert [element.type for element in mesh.elements] == ["Tri3"]


def test_read_rejects_foreign_source_before_borrow(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)

    with pytest.raises(
        TypeError,
        match=r"source must be a GmshMeshRef.*fem\.mesh\.gmsh\.Mesher\.generate",
    ):
        gmsh_io.read(object())


def test_read_infers_dimension_and_forwards_plane_formulation(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[11]], [[1, 2, 3]]),
    )
    source = _FakeGmshMeshRef(2, model)

    mesh = gmsh_io.read(
        source,
        plane_type="STRAIN",
        thickness=0.25,
        z_tolerance=0.0,
    )

    assert isinstance(mesh, Mesh2D)
    assert source.borrow_calls == 1
    assert mesh.elements[0].props == {
        "plane_type": "strain",
        "thickness": 0.25,
    }


@pytest.mark.parametrize(
    ("line_element_type", "expected_dofs"),
    [("Truss2", 3), ("Beam2", 6)],
)
def test_read_forwards_line_formulation(monkeypatch, line_element_type, expected_dofs):
    _install_fake_mesh_ref(monkeypatch)
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 2.0, 3.0),
        },
        elements=([1], [[11]], [[1, 2]]),
    )
    source = _FakeGmshMeshRef(1, model)

    mesh = gmsh_io.read(source, line_element_type=line_element_type)

    assert isinstance(mesh, Mesh3D)
    assert mesh.dofs_per_node == expected_dofs
    assert mesh.elements[0].type == line_element_type


def test_read_allows_repeated_conversion_of_same_live_source(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[11]], [[1, 2, 3]]),
    )
    source = _FakeGmshMeshRef(2, model)

    first = gmsh_io.read(source)
    second = gmsh_io.read(source)

    assert source.borrow_calls == 2
    assert first is not second
    assert first.nodes == second.nodes
    assert first.elements == second.elements
    assert model.mesh.element_calls == [(2, -1), (2, -1)]
    assert model.mesh.node_calls == 2


def test_read_can_retry_after_io_conversion_failure(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 1e-4),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[11]], [[1, 2, 3]]),
    )
    source = _FakeGmshMeshRef(2, model)

    with pytest.raises(ValueError, match=r"node 2.*outside.*tolerance"):
        gmsh_io.read(source, z_tolerance=1e-6)

    mesh = gmsh_io.read(source, z_tolerance=1e-3)

    assert isinstance(mesh, Mesh2D)
    assert source.borrow_calls == 2
    assert model.mesh.element_calls == [(2, -1), (2, -1)]
    assert model.mesh.node_calls == 2


def test_read_propagates_stale_handle_error_before_backend_access(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)
    model = _UnreadableModel()
    stale_error = gmsh_meshing.StaleGmshMeshError(
        "mesh 'plate' must be imported inside its geometry context"
    )
    source = _FakeGmshMeshRef(2, model, borrow_error=stale_error)

    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="inside"):
        gmsh_io.read(source)

    assert source.borrow_calls == 1


def test_read_validates_source_dimension_before_borrow(monkeypatch):
    _install_fake_mesh_ref(monkeypatch)
    source = _FakeGmshMeshRef(4, _UnreadableModel())

    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        gmsh_io.read(source)

    assert source.borrow_calls == 0



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


def test_real_gmsh_line_import_supports_truss2_and_beam2(live_gmsh):
    gmsh = live_gmsh
    point_1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    point_2 = gmsh.model.geo.addPoint(2.0, 1.0, 0.5)
    line = gmsh.model.geo.addLine(point_1, point_2)
    gmsh.model.geo.mesh.setTransfiniteCurve(line, 4)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)

    truss_mesh = gmsh_io.from_model(dimension=1, line_element_type="Truss2")
    beam_mesh = gmsh_io.from_model(dimension=1, line_element_type="Beam2")

    assert {element.type for element in truss_mesh.elements} == {"Truss2"}
    assert {element.type for element in beam_mesh.elements} == {"Beam2"}
    assert truss_mesh.dofs_per_node == 3
    assert beam_mesh.dofs_per_node == 6
    assert [element.node_ids for element in truss_mesh.elements] == [
        element.node_ids for element in beam_mesh.elements
    ]


@pytest.mark.parametrize(
    ("order", "expected_type"),
    [(1, "Tri3"), (2, "Tri6")],
)
def test_real_gmsh_triangle_imports(live_gmsh, order, expected_type):
    gmsh = live_gmsh
    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.4)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(2)

    mesh = gmsh_io.from_model(dimension=2)

    assert isinstance(mesh, Mesh2D)
    assert {element.type for element in mesh.elements} == {expected_type}
    assert all(node.id > 0 for node in mesh.nodes)
    assert all(element.id > 0 for element in mesh.elements)
    coordinates = {node.id: (node.x, node.y) for node in mesh.nodes}
    for element in mesh.elements:
        corner_count = 3
        signed_twice_area = 0.0
        for index in range(corner_count):
            node_id = element.node_ids[index]
            next_node_id = element.node_ids[(index + 1) % corner_count]
            x, y = coordinates[node_id]
            next_x, next_y = coordinates[next_node_id]
            signed_twice_area += x * next_y - next_x * y
        assert signed_twice_area > 0.0


@pytest.mark.parametrize(
    ("order", "incomplete", "expected_type"),
    [(1, 0, "Quad4"), (2, 1, "Quad8")],
)
def test_real_gmsh_quadrilateral_imports(
    live_gmsh,
    order,
    incomplete,
    expected_type,
):
    gmsh = live_gmsh
    points = [
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(2.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(2.0, 1.0, 0.0),
        gmsh.model.geo.addPoint(0.0, 1.0, 0.0),
    ]
    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0]),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    for line in lines:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 3)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", incomplete)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    mesh = gmsh_io.from_model(dimension=2)

    assert {element.type for element in mesh.elements} == {expected_type}
    _assert_kernel_accepts_imported_order(mesh)


@pytest.mark.parametrize(
    ("order", "expected_type"),
    [(1, "Tet4"), (2, "Tet10")],
)
def test_real_gmsh_tetrahedron_imports(live_gmsh, order, expected_type):
    gmsh = live_gmsh
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.generate(3)

    mesh = gmsh_io.from_model(dimension=3)

    assert isinstance(mesh, Mesh3D)
    assert {element.type for element in mesh.elements} == {expected_type}
    _assert_kernel_accepts_imported_order(mesh)


@pytest.mark.parametrize(
    ("order", "incomplete", "expected_type"),
    [(1, 0, "Hex8"), (2, 1, "Hex20")],
)
def test_real_gmsh_hexahedron_imports(
    live_gmsh,
    order,
    incomplete,
    expected_type,
):
    gmsh = live_gmsh
    points = [
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(1.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(1.0, 1.0, 0.0),
        gmsh.model.geo.addPoint(0.0, 1.0, 0.0),
    ]
    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0]),
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
    assert any(dimension == 3 for dimension, _ in extruded)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", incomplete)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(3)

    mesh = gmsh_io.from_model(dimension=3)

    assert {element.type for element in mesh.elements} == {expected_type}
    _assert_kernel_accepts_imported_order(mesh)


def test_real_gmsh_complete_quad9_reports_incomplete_second_order_guidance(live_gmsh):
    gmsh = live_gmsh
    points = [
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(1.0, 0.0, 0.0),
        gmsh.model.geo.addPoint(1.0, 1.0, 0.0),
        gmsh.model.geo.addPoint(0.0, 1.0, 0.0),
    ]
    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0]),
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


def _assert_kernel_accepts_imported_order(mesh):
    for element in mesh.elements:
        element.props.update({"E": 1000.0, "nu": 0.3})
        stiffness = get_element_kernel(element.type).stiffness(mesh, element)
        assert np.all(np.isfinite(stiffness))


@pytest.mark.parametrize(
    ("gmsh_type", "expected"),
    [
        (1, ("Line 2", 1, 1, 2, 2, None, (0, 1))),
        (2, ("Triangle 3", 2, 1, 3, 3, "Tri3", (0, 1, 2))),
        (3, ("Quadrilateral 4", 2, 1, 4, 4, "Quad4", (0, 1, 2, 3))),
        (4, ("Tetrahedron 4", 3, 1, 4, 4, "Tet4", (0, 1, 2, 3))),
        (5, ("Hexahedron 8", 3, 1, 8, 8, "Hex8", tuple(range(8)))),
        (9, ("Triangle 6", 2, 2, 6, 3, "Tri6", tuple(range(6)))),
        (16, ("Quadrilateral 8", 2, 2, 8, 4, "Quad8", tuple(range(8)))),
        (
            11,
            (
                "Tetrahedron 10",
                3,
                2,
                10,
                4,
                "Tet10",
                (0, 1, 2, 3, 4, 5, 6, 7, 9, 8),
            ),
        ),
        (
            17,
            (
                "Hexahedron 20",
                3,
                2,
                20,
                8,
                "Hex20",
                (
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 11,
                    13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
                ),
            ),
        ),
    ],
)
def test_element_specs_preserve_supported_connectivity_contract(gmsh_type, expected):
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
    ("line_element_type", "expected_dofs"),
    [("Truss2", 3), ("Beam2", 6)],
)
def test_from_model_imports_spatial_line2(line_element_type, expected_dofs):
    model = _fake_model(
        node_coordinates={
            31: (3.5, -2.0, 8.25),
            17: (-1.0, 4.0, -0.5),
            999: (100.0, 100.0, 100.0),
        },
        elements=([1], [[901]], [[31, 17]]),
    )

    mesh = gmsh_io.from_model(
        dimension=1,
        line_element_type=line_element_type,
        gmsh_model=model,
    )

    assert isinstance(mesh, Mesh3D)
    assert mesh.dofs_per_node == expected_dofs
    assert [(node.id, node.x, node.y, node.z) for node in mesh.nodes] == [
        (31, 3.5, -2.0, 8.25),
        (17, -1.0, 4.0, -0.5),
    ]
    assert mesh.elements[0].id == 901
    assert mesh.elements[0].node_ids == [31, 17]
    assert mesh.elements[0].type == line_element_type
    assert mesh.elements[0].props == {}
    assert model.mesh.element_calls == [(1, -1)]
    assert model.mesh.property_calls == [1]
    assert model.mesh.node_calls == 1


_SUPPORTED_CELL_CASES = [
    (2, 2, "Tri3", [(0, 0), (1, 0), (0, 1)]),
    (
        2,
        9,
        "Tri6",
        [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)],
    ),
    (2, 3, "Quad4", [(0, 0), (1, 0), (1, 1), (0, 1)]),
    (
        2,
        16,
        "Quad8",
        [
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
        ],
    ),
    (3, 4, "Tet4", [tuple(value) for value in TET10_NATURAL_NODE_COORDS[:4]]),
    (3, 11, "Tet10", [tuple(value) for value in TET10_NATURAL_NODE_COORDS]),
    (3, 5, "Hex8", [tuple(value) for value in HEX20_NATURAL_NODE_COORDS[:8]]),
    (3, 17, "Hex20", [tuple(value) for value in HEX20_NATURAL_NODE_COORDS]),
]


@pytest.mark.parametrize(
    ("dimension", "gmsh_type", "fem_type", "coordinates"),
    _SUPPORTED_CELL_CASES,
    ids=["tri3", "tri6", "quad4", "quad8", "tet4", "tet10", "hex8", "hex20"],
)
def test_from_model_converts_every_supported_area_and_volume_cell(
    dimension,
    gmsh_type,
    fem_type,
    coordinates,
):
    spec = gmsh_io._ELEMENT_SPECS[gmsh_type]
    expected_node_ids = list(range(101, 101 + spec.node_count))
    raw_node_ids = [0] * spec.node_count
    for fem_index, gmsh_index in enumerate(spec.connectivity_permutation):
        raw_node_ids[gmsh_index] = expected_node_ids[fem_index]
    node_coordinates = {
        node_id: (*coordinate, 0.0) if dimension == 2 else tuple(coordinate)
        for node_id, coordinate in zip(expected_node_ids, coordinates, strict=True)
    }
    model = _fake_model(
        node_coordinates=node_coordinates,
        elements=([gmsh_type], [[901]], [raw_node_ids]),
    )

    mesh = gmsh_io.from_model(
        dimension=dimension,
        gmsh_model=model,
        plane_type="STRAIN",
        thickness=2,
    )

    assert isinstance(mesh, Mesh2D if dimension == 2 else Mesh3D)
    assert mesh.dofs_per_node == dimension
    assert mesh.elements[0].id == 901
    assert mesh.elements[0].node_ids == expected_node_ids
    assert mesh.elements[0].type == fem_type
    assert mesh.elements[0].props == (
        {"plane_type": "strain", "thickness": 2.0}
        if dimension == 2
        else {}
    )


def test_from_model_preserves_mixed_supported_block_order():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
        },
        elements=(
            [2, 3],
            [[101, 102], [205]],
            [[1, 2, 3, 2, 4, 3], [1, 2, 4, 3]],
        ),
    )

    mesh = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert [element.id for element in mesh.elements] == [101, 102, 205]
    assert [element.type for element in mesh.elements] == [
        "Tri3",
        "Tri3",
        "Quad4",
    ]


def test_from_model_normalizes_clockwise_quadratic_connectivity():
    model = _fake_model(
        node_coordinates={
            10: (0.0, 0.0, 0.0),
            20: (1.0, 0.0, 0.0),
            30: (0.0, 1.0, 0.0),
            40: (0.0, 0.5, 0.0),
            50: (0.5, 0.5, 0.0),
            60: (0.5, 0.0, 0.0),
            70: (1.0, 1.0, 0.0),
            80: (0.5, 1.0, 0.0),
            90: (1.0, 0.5, 0.0),
        },
        elements=(
            [9, 16],
            [[301], [302]],
            [
                [10, 30, 20, 40, 50, 60],
                [10, 30, 70, 20, 40, 80, 90, 60],
            ],
        ),
    )

    mesh = gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert mesh.elements[0].node_ids == [10, 20, 30, 60, 50, 40]
    assert mesh.elements[1].node_ids == [10, 20, 70, 30, 60, 90, 80, 40]


@pytest.mark.parametrize(
    ("dimension", "element_type", "node_ids", "message"),
    [
        (
            1,
            8,
            [1, 2, 3],
            "unsupported Gmsh element type 8.*first-order.*two-node",
        ),
        (
            2,
            10,
            list(range(1, 10)),
            "Quadrilateral 9.*Quad8.*SecondOrderIncomplete",
        ),
        (
            3,
            12,
            list(range(1, 28)),
            "Hexahedron 27.*Hex20.*SecondOrderIncomplete",
        ),
        (3, 6, list(range(1, 7)), "unsupported Gmsh element type 6.*Prism 6"),
    ],
    ids=["line3", "quad9", "hex27", "prism6"],
)
def test_from_model_rejects_unsupported_top_cells(
    dimension,
    element_type,
    node_ids,
    message,
):
    model = _fake_model(
        node_coordinates={
            node_id: (float(node_id), 0.0, 0.0)
            for node_id in node_ids
        },
        elements=([element_type], [[7]], [node_ids]),
    )
    kwargs = {"line_element_type": "Beam2"} if dimension == 1 else {}

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=dimension, gmsh_model=model, **kwargs)

    assert model.mesh.node_calls == 0


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        (([], [], []), "no top-dimensional elements"),
        (([2, 3], [[1]], [[1, 2, 3]]), "inconsistent element block counts"),
        (([2], [[1]], [[1, 2]]), "flattened connectivity.*expected 3"),
        (([2], [[1, 2]], [[1, 2, 3]]), "flattened connectivity.*expected 6"),
        (([2], [[1]], [[1, 1, 3]]), "repeated node tags"),
        (
            ([2, 3], [[7], [7]], [[1, 2, 3], [1, 2, 3, 4]]),
            "duplicate element tag 7",
        ),
        (([2], [[0]], [[1, 2, 3]]), "element tags must be positive integers"),
        (
            ([2], [[1]], [[0, 2, 3]]),
            "connectivity node tags must be positive integers",
        ),
    ],
)
def test_from_model_rejects_invalid_element_blocks(elements, message):
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
            4: (1.0, 1.0, 0.0),
        },
        elements=elements,
    )

    with pytest.raises(ValueError, match=message):
        gmsh_io.from_model(dimension=2, gmsh_model=model)


@pytest.mark.parametrize(
    ("node_tags", "coordinates", "message"),
    [
        ([1, 1, 3], [0.0, 0.0, 0.0] * 3, "duplicate node tag 1"),
        ([0, 2, 3], [0.0, 0.0, 0.0] * 3, "node tags must be positive integers"),
        ([1, 2, 3], [0.0] * 8, "coordinate vector length.*expected 9"),
        (
            [1, 2, 3],
            [
                0.0, 0.0, 0.0,
                1.0, float("nan"), 0.0,
                0.0, 1.0, 0.0,
            ],
            "coordinates must be finite",
        ),
    ],
)
def test_from_model_rejects_invalid_node_arrays(node_tags, coordinates, message):
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


@pytest.mark.parametrize(
    ("element_type", "node_coordinates", "connectivity", "bad_node"),
    [
        (
            2,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 1e-4),
                3: (0.0, 1.0, 0.0),
            },
            [1, 2, 3],
            2,
        ),
        (
            9,
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (0.0, 1.0, 0.0),
                4: (0.5, 0.0, 1e-4),
                5: (0.5, 0.5, 0.0),
                6: (0.0, 0.5, 0.0),
            },
            [1, 2, 3, 4, 5, 6],
            4,
        ),
    ],
    ids=["corner", "midside"],
)
def test_from_model_enforces_planar_tolerance(
    element_type,
    node_coordinates,
    connectivity,
    bad_node,
):
    model = _fake_model(
        node_coordinates=node_coordinates,
        elements=([element_type], [[1]], [connectivity]),
    )

    with pytest.raises(ValueError, match=rf"node {bad_node}.*outside.*tolerance"):
        gmsh_io.from_model(dimension=2, gmsh_model=model, z_tolerance=1e-6)


def test_from_model_validates_reported_element_properties_before_nodes():
    model = _fake_model(
        node_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0),
        },
        elements=([2], [[1]], [[1, 2, 3]]),
        properties={2: ("Wrong Triangle", 2, 1, 3, [], 3)},
    )

    with pytest.raises(ValueError, match="properties do not match the adapter contract"):
        gmsh_io.from_model(dimension=2, gmsh_model=model)

    assert model.mesh.node_calls == 0
