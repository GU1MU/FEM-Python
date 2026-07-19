from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from fem.geometry import gmsh as geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as meshing


ROOT = Path(__file__).resolve().parents[1]


def _model_name(label: str) -> str:
    return f"fem_meshing_{label}_{uuid.uuid4().hex}"


@pytest.fixture
def real_gmsh():
    gmsh = pytest.importorskip("gmsh")
    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()

    original_models = {str(name) for name in gmsh.model.list()}
    original_current = str(gmsh.model.getCurrent())
    original_terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", 0.0)
    try:
        yield gmsh
    finally:
        if gmsh.isInitialized():
            for model_name in tuple(str(name) for name in gmsh.model.list()):
                if model_name not in original_models:
                    gmsh.model.setCurrent(model_name)
                    gmsh.model.remove()
            remaining_models = {str(name) for name in gmsh.model.list()}
            if original_current in remaining_models:
                gmsh.model.setCurrent(original_current)
            gmsh.option.setNumber("General.Terminal", original_terminal)
        if owns_session and gmsh.isInitialized():
            gmsh.finalize()


def _create_top_entity(cad: geometry.GeometryModel):
    if cad.dimension == 1:
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(2.0, 0.5, 0.25)
        return cad.line(start, end)
    if cad.dimension == 2:
        return cad.rectangle(0.0, 0.0, 2.0, 1.0)
    return cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def _read_native(native_mesh: meshing.GmshMeshRef):
    if native_mesh.dimension == 1:
        return gmsh_io.read(native_mesh, line_element_type="Truss2")
    return gmsh_io.read(native_mesh)


def test_mesh_specs_are_frozen_slotted_and_normalize_values():
    explicit = meshing.MeshSpec(size=2, order=2, recombine=True)
    automatic = meshing.AutoMeshSpec(level=4, cell_shape="quad", order=2)

    assert explicit.size == 2.0
    assert not hasattr(explicit, "__dict__")
    assert not hasattr(automatic, "__dict__")
    with pytest.raises(FrozenInstanceError):
        explicit.order = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        automatic.level = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"size": 0.0}, ValueError),
        ({"size": -1.0}, ValueError),
        ({"size": float("inf")}, ValueError),
        ({"size": float("nan")}, ValueError),
        ({"size": True}, ValueError),
        ({"order": 0}, ValueError),
        ({"order": 3}, ValueError),
        ({"order": True}, ValueError),
        ({"recombine": 1}, TypeError),
    ],
)
def test_mesh_spec_rejects_invalid_values(kwargs, error_type):
    with pytest.raises(error_type):
        meshing.MeshSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level": 0},
        {"level": 6},
        {"level": True},
        {"level": 2.0},
        {"cell_shape": "line"},
        {"cell_shape": "HEX"},
        {"order": 0},
        {"order": 3},
        {"order": True},
    ],
)
def test_auto_mesh_spec_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        meshing.AutoMeshSpec(**kwargs)


def test_importing_meshing_facade_does_not_import_external_gmsh():
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fem.mesh import gmsh as gmsh_meshing
assert gmsh_meshing.__name__ == "fem.mesh.gmsh"
"""

    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_canonical_exports_share_owner_coupled_class_identity():
    assert meshing.__all__ == [
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
    for name in (
        "GmshMeshRef",
        "MeshCellShapeError",
        "MeshControlConflictError",
        "MeshFieldOwnershipError",
        "MeshFieldRef",
        "StaleGmshMeshError",
        "StaleMeshFieldError",
    ):
        assert getattr(meshing, name) is getattr(geometry, name)
        assert name not in geometry.__all__


def test_geometry_model_no_longer_exposes_public_meshing_methods():
    old_names = (
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
        "recombine",
        "mesh_size",
        "distance_field",
        "threshold_field",
        "min_field",
        "background_field",
        "generate_mesh",
        "generate_auto_mesh",
    )

    assert all(not hasattr(geometry.GeometryModel, name) for name in old_names)


def test_mesher_requires_live_geometry_and_failed_new_binding_is_retryable(
    real_gmsh,
):
    with pytest.raises(TypeError, match="GeometryModel"):
        meshing.Mesher(object())  # type: ignore[arg-type]

    source = geometry.model(_model_name("binding_lifecycle"), dimension=2)
    with pytest.raises(geometry.GeometryStateError, match="NEW"):
        meshing.Mesher(source)

    with source as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        first = meshing.Mesher(cad)
        with pytest.raises(geometry.GeometryStateError, match="already bound"):
            meshing.Mesher(cad)

        native_mesh = first.generate(meshing.MeshSpec(size=0.4))
        assert _read_native(native_mesh).num_elements > 0

    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        meshing.Mesher(source)


def test_binding_seals_representative_geometry_mutations_but_preserves_queries(
    real_gmsh,
):
    with geometry.model(_model_name("sealed_geometry"), dimension=3) as cad:
        start = cad.point(0.0, 0.0, 0.0)
        center = cad.point(0.5, 0.0, 0.0)
        end = cad.point(1.0, 0.0, 0.0)
        curve = cad.line(start, end)
        surface_a = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        surface_b = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        box_a = cad.box(0.0, 2.0, 0.0, 1.0, 1.0, 1.0)
        box_b = cad.box(2.0, 2.0, 0.0, 1.0, 1.0, 1.0)
        meshing.Mesher(cad)

        mutations = (
            lambda: cad.point(3.0, 0.0, 0.0),
            lambda: cad.line(start, center),
            lambda: cad.circular_arc(start, center, end),
            lambda: cad.spline((start, center, end)),
            lambda: cad.curve_loop(()),
            lambda: cad.rectangle(4.0, 0.0, 1.0, 1.0),
            lambda: cad.box(4.0, 2.0, 0.0, 1.0, 1.0, 1.0),
            lambda: cad.cylinder(4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5),
            lambda: cad.fuse((box_a,), (box_b,)),
            lambda: cad.cut((surface_a,), (surface_b,)),
            lambda: cad.translate((curve,), 0.1, 0.0, 0.0),
            lambda: cad.rotate((surface_a,), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1),
            lambda: cad.extrude((surface_a,), 0.0, 0.0, 1.0),
        )
        for mutation in mutations:
            with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
                mutation()

        surfaces = cad.entities(2)
        boundary = cad.boundary((surface_a,))
        assert surface_a in surfaces
        assert boundary
        assert cad.select(boundary, x=0.0)
        assert cad.bounding_box(surface_a)[0] == pytest.approx(0.0, abs=1.0e-6)
        assert cad.area(surface_a) == pytest.approx(1.0)
        assert cad.center_of_mass(surface_a) == pytest.approx((0.5, 0.5, 0.0))
        assert cad.orient(curve).curve == curve


@pytest.mark.parametrize("attribute", ["raw_model", "raw_occ"])
def test_raw_access_is_rejected_after_mesher_binding(real_gmsh, attribute):
    with geometry.model(_model_name(f"sealed_{attribute}"), dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        meshing.Mesher(cad)

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            getattr(cad, attribute)


def test_prebinding_raw_access_preserves_documented_meshing_matrix(real_gmsh):
    with geometry.model(_model_name("raw_matrix"), dimension=2) as cad:
        original = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        original_tag = original.tag
        cad.raw_occ
        mesh_builder = meshing.Mesher(cad)

        with pytest.raises(geometry.StaleEntityError):
            cad.boundary((original,))

        reacquired = cad.entity(2, original_tag)
        assert cad.area(reacquired) == pytest.approx(1.0)
        with pytest.raises(meshing.MeshControlConflictError, match="scope unknown"):
            mesh_builder.generate(
                meshing.AutoMeshSpec(level=2, cell_shape="tri")
            )

        mesh_builder.recombine(reacquired)
        native_mesh = mesh_builder.generate(meshing.MeshSpec(size=0.3))
        assert _read_native(native_mesh).num_elements > 0


@pytest.mark.parametrize(
    ("dimension", "geometry_kind", "spec", "allowed_types"),
    [
        (1, "line", meshing.AutoMeshSpec(level=2), {"Truss2"}),
        (
            2,
            "rectangle",
            meshing.AutoMeshSpec(level=2, cell_shape="tri", order=2),
            {"Tri6"},
        ),
        (
            2,
            "rectangle",
            meshing.AutoMeshSpec(level=2, cell_shape="tri-quad"),
            {"Tri3", "Quad4"},
        ),
        (
            2,
            "disk",
            meshing.AutoMeshSpec(level=2, cell_shape="quad", order=2),
            {"Quad8"},
        ),
        (
            3,
            "box",
            meshing.AutoMeshSpec(level=2, cell_shape="tet", order=2),
            {"Tet10"},
        ),
        (
            3,
            "box",
            meshing.AutoMeshSpec(level=1, cell_shape="hex", order=2),
            {"Hex20"},
        ),
    ],
)
def test_auto_mesh_spec_dimension_and_cell_shape_matrix(
    real_gmsh,
    dimension,
    geometry_kind,
    spec,
    allowed_types,
):
    with geometry.model(
        _model_name(f"auto_{dimension}_{geometry_kind}"),
        dimension=dimension,
    ) as cad:
        if geometry_kind == "line":
            _create_top_entity(cad)
        elif geometry_kind == "rectangle":
            cad.rectangle(0.0, 0.0, 2.0, 1.0)
        elif geometry_kind == "disk":
            cad.disk(0.0, 0.0, 1.0)
        else:
            cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        native_mesh = meshing.Mesher(cad).generate(spec)
        mesh = _read_native(native_mesh)

    actual_types = {element.type for element in mesh.elements}
    assert actual_types
    assert actual_types.issubset(allowed_types)


@pytest.mark.parametrize(
    ("dimension", "invalid_specs"),
    [
        (
            1,
            (
                meshing.MeshSpec(order=2),
                meshing.MeshSpec(recombine=True),
                meshing.AutoMeshSpec(cell_shape="tri"),
                meshing.AutoMeshSpec(order=2),
            ),
        ),
        (
            2,
            (
                meshing.AutoMeshSpec(cell_shape="tet"),
                meshing.AutoMeshSpec(cell_shape="hex"),
            ),
        ),
        (
            3,
            (
                meshing.AutoMeshSpec(cell_shape="tri"),
                meshing.AutoMeshSpec(cell_shape="tri-quad"),
                meshing.AutoMeshSpec(cell_shape="quad"),
            ),
        ),
    ],
)
def test_invalid_dimension_spec_combinations_are_retryable(
    real_gmsh,
    dimension,
    invalid_specs,
):
    with geometry.model(
        _model_name(f"invalid_matrix_{dimension}"),
        dimension=dimension,
    ) as cad:
        _create_top_entity(cad)
        mesh_builder = meshing.Mesher(cad)

        for invalid_spec in invalid_specs:
            with pytest.raises(ValueError):
                mesh_builder.generate(invalid_spec)

        native_mesh = mesh_builder.generate(meshing.MeshSpec(size=0.5))
        assert _read_native(native_mesh).num_elements > 0


def test_consecutive_structured_extrusions_generate_hex_mesh(real_gmsh):
    with geometry.model(_model_name("structured_consecutive"), dimension=3) as cad:
        first_surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        second_surface = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        mesh_builder = meshing.Mesher(cad)

        first_outputs = mesh_builder.structured_extrude(
            (first_surface,),
            0.0,
            0.0,
            1.0,
            num_elements=(2,),
            recombine=True,
        )
        second_outputs = mesh_builder.structured_extrude(
            (second_surface,),
            0.0,
            0.0,
            1.0,
            num_elements=(3,),
            heights=(1.0,),
            recombine=True,
        )

        for outputs in (first_outputs, second_outputs):
            volumes = {
                (entity.dimension, entity.tag)
                for entity in outputs
                if entity.dimension == 3
            }
            assert len(volumes) == 1

        native_mesh = mesh_builder.generate(
            meshing.MeshSpec(size=0.5, recombine=True)
        )
        mesh = gmsh_io.read(native_mesh)

    assert {element.type for element in mesh.elements} == {"Hex8"}


def test_structured_extrusion_preflight_failure_is_retryable(real_gmsh):
    with geometry.model(_model_name("structured_retry"), dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        mesh_builder = meshing.Mesher(cad)

        with pytest.raises(ValueError, match="positive integers"):
            mesh_builder.structured_extrude(
                (surface,),
                0.0,
                0.0,
                1.0,
                num_elements=(0,),
                recombine=True,
            )

        outputs = mesh_builder.structured_extrude(
            (surface,),
            0.0,
            0.0,
            1.0,
            num_elements=(1,),
            recombine=True,
        )
        assert any(entity.dimension == 3 for entity in outputs)


def test_successful_ordinary_control_closes_structured_extrusion_subphase(
    real_gmsh,
):
    with geometry.model(_model_name("structured_closed"), dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        curves = cad.boundary((surface,))
        mesh_builder = meshing.Mesher(cad)

        mesh_builder.transfinite_curve(curves[0], num_nodes=3)
        with pytest.raises(geometry.GeometryStateError, match="subphase was closed"):
            mesh_builder.structured_extrude(
                (surface,),
                0.0,
                0.0,
                1.0,
                num_elements=(1,),
                recombine=True,
            )

        assert cad.entities(3) == ()


@pytest.mark.parametrize("operation", ("control", "field"))
def test_native_ordinary_configuration_failure_closes_structured_subphase(
    real_gmsh,
    monkeypatch,
    operation,
):
    with geometry.model(
        _model_name(f"structured_failed_{operation}"),
        dimension=3,
    ) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        curve = cad.boundary((surface,))[0]
        mesh_builder = meshing.Mesher(cad)

        def fail_configuration(*args, **kwargs):
            raise RuntimeError(f"injected {operation} failure")

        if operation == "control":
            monkeypatch.setattr(
                real_gmsh.model.mesh,
                "setTransfiniteCurve",
                fail_configuration,
            )
            def configure():
                mesh_builder.transfinite_curve(
                    curve,
                    num_nodes=3,
                )
        else:
            monkeypatch.setattr(
                real_gmsh.model.mesh.field,
                "add",
                fail_configuration,
            )
            def configure():
                mesh_builder.distance_field(curves=(curve,))

        with pytest.raises(RuntimeError, match=f"injected {operation} failure"):
            configure()
        with pytest.raises(geometry.GeometryStateError, match="subphase was closed"):
            mesh_builder.structured_extrude(
                (surface,),
                0.0,
                0.0,
                1.0,
                num_elements=(1,),
                recombine=True,
            )

        assert cad.entities(3) == ()


def test_native_structured_extrusion_failure_is_terminal(
    real_gmsh,
    monkeypatch,
):
    with geometry.model(_model_name("structured_failure"), dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        mesh_builder = meshing.Mesher(cad)

        def fail_extrusion(*args, **kwargs):
            raise RuntimeError("injected structured extrusion failure")

        monkeypatch.setattr(real_gmsh.model.occ, "extrude", fail_extrusion)
        with pytest.raises(RuntimeError, match="injected structured"):
            mesh_builder.structured_extrude(
                (surface,),
                0.0,
                0.0,
                1.0,
                num_elements=(1,),
                recombine=True,
            )

        assert surface in cad.entities(2)
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            mesh_builder.generate(meshing.MeshSpec(size=0.4))
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.rectangle(2.0, 0.0, 1.0, 1.0)


def test_local_fields_route_through_canonical_mesher(real_gmsh):
    with geometry.model(_model_name("local_fields"), dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curves = cad.boundary((surface,))
        mesh_builder = meshing.Mesher(cad)

        first_distance = mesh_builder.distance_field(curves=(curves[0],))
        first_threshold = mesh_builder.threshold_field(
            first_distance,
            size_min=0.08,
            size_max=0.35,
            dist_min=0.05,
            dist_max=0.5,
        )
        second_distance = mesh_builder.distance_field(curves=(curves[-1],))
        second_threshold = mesh_builder.threshold_field(
            second_distance,
            size_min=0.1,
            size_max=0.35,
            dist_min=0.05,
            dist_max=0.5,
        )
        minimum = mesh_builder.min_field((first_threshold, second_threshold))
        mesh_builder.background_field(minimum)

        assert isinstance(first_distance, meshing.MeshFieldRef)
        assert isinstance(minimum, meshing.MeshFieldRef)
        assert (first_distance.field_type, minimum.field_type) == ("Distance", "Min")
        native_mesh = mesh_builder.generate(meshing.MeshSpec())
        assert gmsh_io.read(native_mesh).num_elements > 0


def test_generated_mesh_supports_repeated_reads_and_becomes_stale_on_close(
    real_gmsh,
):
    with geometry.model(_model_name("handle_lifecycle"), dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        native_mesh = meshing.Mesher(cad).generate(
            meshing.MeshSpec(size=0.3, order=2)
        )
        first = gmsh_io.read(native_mesh)
        second = gmsh_io.read(native_mesh)

        assert first is not second
        assert first.nodes == second.nodes
        assert first.elements == second.elements

    with pytest.raises(meshing.StaleGmshMeshError, match="inside"):
        gmsh_io.read(native_mesh)
