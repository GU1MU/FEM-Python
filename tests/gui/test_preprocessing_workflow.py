from __future__ import annotations

from dataclasses import replace
import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
import pytest

from fem.application import (
    DeleteIntent,
    MeshEntityRef,
    NamedRegion,
    NamedRegionEditBatch,
    RenameIntent,
)
from fem.application.preprocessing import generate_fem_model
from fem.application.native_scope_materialization import (
    materialize_native_scopes,
)
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    supports_structured_hexahedron,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.mesh.quality import analyze_mesh
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces
from fem_gui.geometry_preview import build_geometry_preview
from fem_gui.main_window import FEMMainWindow
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    _geometry_edge_polydata,
    _geometry_point_polydata,
    _point_to_segment_distance,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    deadline = monotonic() + 2.0
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def _set_native_mesh_inputs(
    window: FEMMainWindow,
    recipe: object,
    settings: MeshSettings,
) -> None:
    window._set_native_geometry(recipe, "测试几何")
    assert window._apply_session_delta(
        window.session.replace_mesh_settings(settings)
    )


@pytest.mark.gmsh
def test_native_rectangle_mesh_joins_the_existing_model_workflow() -> None:
    _application()
    window = FEMMainWindow()
    recipe = RectangleGeometry("gui-native-rectangle", 2.0, 1.0)
    settings = MeshSettings(0.25, order=1, cell_shape="triangle")
    _set_native_mesh_inputs(window, recipe, settings)

    window.generate_native_mesh()
    _wait_for_task(window)

    assert window.document.source_kind == "native"
    assert window.document.geometry_recipe == recipe
    assert window.document.mesh_settings == settings
    assert window.document.path is None
    assert not window.actions["reload"].isEnabled()
    assert window.document.model is not None
    assert window.geometry is not None
    assert window.document.model.name == window.document.model_name
    assert {element.type for element in window.document.model.mesh.elements} == {"Tri3"}
    assert not window.document.model.node_sets
    assert not window.document.model.element_sets
    assert not window.document.model.edges
    assert not window.document.model.surfaces
    assert window.ribbon.tab_bar.tabText(window.ribbon.tab_bar.currentIndex()) == "模型"
    assert window.actions["mesh_generate"].isEnabled()
    assert window.actions["mesh_clear"].isEnabled()
    assert window.actions["mesh_quality"].isEnabled()
    assert window.actions["geometry_region"].isEnabled()
    assert not window.actions["geometry_regions"].isEnabled()
    assert window._analysis_region_names() == ([], [], [])
    assert window._analysis_element_regions() == []

    report = analyze_mesh(window.document.model.mesh)
    assert report.checked_count == report.element_count
    assert 0.0 < report.minimum <= report.mean <= report.maximum <= 1.0

    window.clear_native_mesh()
    assert window.document.model is None
    assert window.document.geometry_recipe == recipe
    assert window.document.mesh_settings == settings
    assert window.viewport._geometry_preview is not None
    assert window.actions["mesh_generate"].isEnabled()
    assert not window.actions["geometry_region"].isEnabled()
    window.close()


@pytest.mark.gmsh
def test_scope_creation_starts_from_the_meshed_model() -> None:
    _application()
    window = FEMMainWindow()
    recipe = RectangleGeometry("scope-plate", 2.0, 1.0)
    _set_native_mesh_inputs(window, recipe, MeshSettings(0.25))
    window.generate_native_mesh()
    _wait_for_task(window)
    before_nodes = tuple(window.document.model.mesh.nodes)
    before_elements = tuple(window.document.model.mesh.elements)

    window._request_analysis_geometry_selection("scope", "edge")
    assert window._pending_analysis_selection == "scope"
    assert window._scope_selection_overlay_active
    topology = window._scope_selection_topology()
    selected_geometry_edge = next(
        reference
        for reference in topology.mesh_references
        if reference.kind == "edge"
    )
    expected_mesh_edges = topology.mesh_references[
        selected_geometry_edge
    ]
    assert len(expected_mesh_edges) > 1
    window._on_geometry_entity_pick(selected_geometry_edge)
    window.viewport_panel.scope_creation_bar.name_edit.setText("Support")
    window._confirm_guided_selection()
    _application().processEvents()

    assert tuple(window.document.named_regions) == ("Support",)
    assert window.document.model is not None
    assert tuple(window.document.model.mesh.nodes) == before_nodes
    assert tuple(window.document.model.mesh.elements) == before_elements
    assert "Support" in window.document.model.edges
    node_regions, edge_regions, face_regions = (
        window._analysis_region_names()
    )
    assert node_regions == []
    assert [region.name for region in edge_regions] == ["Support"]
    assert face_regions == []
    assert not window._scope_selection_overlay_active

    support = window.document.named_regions["Support"]
    assert support.references == tuple(
        replace(reference, part_id=window.document.active_part_id)
        for reference in expected_mesh_edges
    )
    rename_receipt = window.apply_named_region_edit(
        NamedRegionEditBatch(
            window.document.session_revision,
            (NamedRegion("Fixed", support.references),),
            renames=(RenameIntent("Support", "Fixed"),),
        )
    )
    assert rename_receipt.diagnostic is None
    assert window.document.model is not None
    assert tuple(window.document.model.mesh.nodes) == before_nodes
    assert tuple(window.document.model.mesh.elements) == before_elements
    assert "Support" not in window.document.model.edges
    assert "Fixed" in window.document.model.edges

    delete_receipt = window.apply_named_region_edit(
        NamedRegionEditBatch(
            window.document.session_revision,
            (),
            deletes=(DeleteIntent("Fixed"),),
        )
    )
    assert delete_receipt.diagnostic is None
    assert window.document.model is not None
    assert tuple(window.document.model.mesh.nodes) == before_nodes
    assert tuple(window.document.model.mesh.elements) == before_elements
    assert "Fixed" not in window.document.model.edges
    window.close()


@pytest.mark.gmsh
def test_generic_sketch_replaces_special_plate_with_hole_entry() -> None:
    recipe = SketchGeometry(
        "generic-plate-with-hole",
        (
            SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
            SketchCircle("cut", 1.0, 0.5, 0.2),
        ),
    )

    preview = build_geometry_preview(recipe, segments=16)
    model = generate_fem_model(recipe, MeshSettings(0.2))

    assert preview.faces
    assert len(preview.edges) == 2
    assert all(len(face) == 4 for face in preview.faces)
    assert len(preview.faces) == 20
    outer_points = preview.points[len(preview.points) // 2:]
    assert any(
        point[0] == pytest.approx(0.0, abs=1.0e-8)
        and point[1] == pytest.approx(0.0, abs=1.0e-8)
        for point in outer_points
    )
    assert any(
        point[0] == pytest.approx(2.0, abs=1.0e-8)
        and point[1] == pytest.approx(1.0, abs=1.0e-8)
        for point in outer_points
    )
    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


def test_rectangle_cut_preview_shows_an_open_frame() -> None:
    recipe = BooleanGeometry(
        "rectangular-cut",
        "cut",
        RectangleGeometry("plate", 2.0, 1.0),
        MovedGeometry(RectangleGeometry("tool", 0.5, 0.25), 0.75, 0.375),
    )

    preview = build_geometry_preview(recipe)

    assert len(preview.faces) == 4
    assert len(preview.edges) == 2
    assert all(len(face) == 4 for face in preview.faces)


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("cell_shape", "expected_type"),
    (("tetrahedron", "Tet4"), ("hexahedron", "Hex8")),
)
def test_rectangle_sketch_extrusion_uses_existing_solid_mesh_workflow(
    cell_shape,
    expected_type,
) -> None:
    recipe = ExtrudedGeometry(
        SketchGeometry(
            "sketch-extrusion",
            (SketchRectangle("material", 0.0, 0.0, 1.0, 0.8),),
        ),
        0.6,
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(0.3, cell_shape=cell_shape),
    )

    assert {element.type for element in model.mesh.elements} == {expected_type}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


@pytest.mark.gmsh
def test_quadrilateral_setting_reaches_the_same_gui_adapter() -> None:
    _application()
    window = FEMMainWindow()
    _set_native_mesh_inputs(
        window,
        RectangleGeometry(
            "gui-native-quad",
            2.0,
            1.0,
        ),
        MeshSettings(
            0.25,
            order=2,
            cell_shape="quadrilateral",
        ),
    )

    window.generate_native_mesh()
    _wait_for_task(window)

    assert window.document.model is not None
    assert {element.type for element in window.document.model.mesh.elements} == {"Quad8"}
    assert len(window.geometry.cells) == len(window.document.model.mesh.elements)
    window.close()


@pytest.mark.gmsh
def test_plate_with_hole_applies_local_refinement_without_implicit_scopes() -> None:
    _application()
    window = FEMMainWindow()
    recipe = PlateWithHoleGeometry(
        "gui-plate-with-hole",
        2.0,
        1.0,
        1.0,
        0.5,
        0.2,
    )
    settings = MeshSettings(
        0.20,
        order=1,
        cell_shape="triangle",
        local_controls=(
            LocalMeshControl(
                LogicalEntityRef("edge:hole-loop"),
                0.04,
                MeshSizeFalloff("target_radius", 0.25, 2.0),
            ),
        ),
    )
    _set_native_mesh_inputs(window, recipe, settings)

    window.generate_native_mesh()
    _wait_for_task(window)

    model = window.document.model
    assert model is not None
    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert len(mesh_edges.boundary(model.mesh)) >= 12
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces
    window.close()


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("recipe", "expected_type"),
    (
        (DiskGeometry("gui-disk", 1.0), "Tri3"),
        (
            BoxGeometry("gui-box", 1.0, 0.8, 0.6),
            "Tet4",
        ),
        (
            CylinderGeometry("gui-cylinder", 0.5, 1.0),
            "Tet4",
        ),
    ),
)
def test_added_basic_geometries_generate_canonical_models(
    recipe,
    expected_type,
) -> None:
    settings = MeshSettings(
        0.25,
        cell_shape="tetrahedron" if expected_type == "Tet4" else "triangle",
    )

    model = generate_fem_model(recipe, settings)

    assert {element.type for element in model.mesh.elements} == {expected_type}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


@pytest.mark.gmsh
def test_box_supports_structured_hexahedral_mesh() -> None:
    recipe = BoxGeometry("gui-structured-box", 1.0, 0.8, 0.6)

    model = generate_fem_model(
        recipe,
        MeshSettings(0.3, cell_shape="hexahedron"),
    )

    assert {element.type for element in model.mesh.elements} == {"Hex8"}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


@pytest.mark.parametrize(
    "recipe",
    (
        RectangleGeometry("preview-rectangle", 2.0, 1.0),
        DiskGeometry("preview-disk", 1.0),
        PlateWithHoleGeometry("preview-hole", 2.0, 1.0, 1.0, 0.5, 0.2),
        BoxGeometry("preview-box", 2.0, 1.0, 0.5),
        CylinderGeometry("preview-cylinder", 0.5, 1.0),
    ),
)
def test_each_native_geometry_has_a_surface_preview(recipe) -> None:
    preview = build_geometry_preview(recipe, segments=16)

    assert preview.points
    assert preview.faces
    assert preview.edges
    assert max(index for face in preview.faces for index in face) < len(preview.points)


def test_cylinder_preview_edge_mesh_contains_only_logical_line_cells() -> None:
    import pyvista

    preview = build_geometry_preview(
        CylinderGeometry("preview-edge-cells", 0.5, 1.0),
        segments=16,
    )

    edge_mesh = _geometry_edge_polydata(
        pyvista,
        np.asarray(preview.points, dtype=float),
        preview,
        tuple(range(1, len(preview.edges) + 1)),
    )

    assert edge_mesh.n_cells == len(preview.edges)
    assert len(edge_mesh.cell_data["geometry_pick_id"]) == len(preview.edges)
    assert edge_mesh.active_scalars_name is None


def test_box_preview_exposes_all_twelve_logical_edges() -> None:
    recipe = BoxGeometry("preview-box-edges", 2.0, 1.0, 0.5)
    preview = build_geometry_preview(recipe)

    assert len(preview.edges) == 12
    assert all(logical_id is not None for logical_id in preview.edge_logical_ids)
    assert len(set(preview.edge_logical_ids)) == 12
    assert all(len(edge) == 2 for edge in preview.edges)


def test_rigidly_transformed_box_keeps_hexahedron_authoring_support() -> None:
    box = BoxGeometry("hex-box", 2.0, 1.0, 0.5)

    assert supports_structured_hexahedron(
        MovedGeometry(box, 4.0, -2.0, 1.0)
    )
    assert supports_structured_hexahedron(RotatedGeometry(box, "z", 35.0))


def test_rectangle_preview_cells_follow_catalog_semantics() -> None:
    recipe = RectangleGeometry("semantic-rectangle", 2.0, 1.0)
    preview = build_geometry_preview(recipe)
    point_by_index = dict(enumerate(preview.point_logical_ids))

    expected_points = {
        "point:bottom-left": (0.0, 0.0, 0.0),
        "point:bottom-right": (2.0, 0.0, 0.0),
        "point:top-right": (2.0, 1.0, 0.0),
        "point:top-left": (0.0, 1.0, 0.0),
    }
    assert {
        logical_id: preview.points[index]
        for index, logical_id in point_by_index.items()
    } == expected_points

    expected_edges = {
        "edge:bottom": {"point:bottom-left", "point:bottom-right"},
        "edge:right": {"point:bottom-right", "point:top-right"},
        "edge:top": {"point:top-right", "point:top-left"},
        "edge:left": {"point:top-left", "point:bottom-left"},
    }
    assert {
        logical_id: {point_by_index[index] for index in edge}
        for edge, logical_id in zip(
            preview.edges,
            preview.edge_logical_ids,
            strict=True,
        )
    } == expected_edges
    assert all(logical_id is not None for logical_id in preview.edge_logical_ids)


def test_box_preview_cells_follow_catalog_semantics() -> None:
    recipe = BoxGeometry("semantic-box", 2.0, 1.0, 0.5)
    preview = build_geometry_preview(recipe)
    point_by_index = dict(enumerate(preview.point_logical_ids))

    expected_points = {
        "point:bottom-front-left": (0.0, 0.0, 0.0),
        "point:bottom-front-right": (2.0, 0.0, 0.0),
        "point:bottom-back-right": (2.0, 1.0, 0.0),
        "point:bottom-back-left": (0.0, 1.0, 0.0),
        "point:top-front-left": (0.0, 0.0, 0.5),
        "point:top-front-right": (2.0, 0.0, 0.5),
        "point:top-back-right": (2.0, 1.0, 0.5),
        "point:top-back-left": (0.0, 1.0, 0.5),
    }
    assert {
        logical_id: preview.points[index]
        for index, logical_id in point_by_index.items()
    } == expected_points

    expected_edges = {
        "edge:bottom-front": {
            "point:bottom-front-left",
            "point:bottom-front-right",
        },
        "edge:bottom-right": {
            "point:bottom-front-right",
            "point:bottom-back-right",
        },
        "edge:bottom-back": {
            "point:bottom-back-right",
            "point:bottom-back-left",
        },
        "edge:bottom-left": {
            "point:bottom-back-left",
            "point:bottom-front-left",
        },
        "edge:top-front": {"point:top-front-left", "point:top-front-right"},
        "edge:top-right": {"point:top-front-right", "point:top-back-right"},
        "edge:top-back": {"point:top-back-right", "point:top-back-left"},
        "edge:top-left": {"point:top-back-left", "point:top-front-left"},
        "edge:vertical-front-left": {
            "point:bottom-front-left",
            "point:top-front-left",
        },
        "edge:vertical-front-right": {
            "point:bottom-front-right",
            "point:top-front-right",
        },
        "edge:vertical-back-right": {
            "point:bottom-back-right",
            "point:top-back-right",
        },
        "edge:vertical-back-left": {
            "point:bottom-back-left",
            "point:top-back-left",
        },
    }
    assert {
        logical_id: {point_by_index[index] for index in edge}
        for edge, logical_id in zip(
            preview.edges,
            preview.edge_logical_ids,
            strict=True,
        )
    } == expected_edges

    expected_faces = {
        "face:bottom": {
            "point:bottom-front-left",
            "point:bottom-front-right",
            "point:bottom-back-right",
            "point:bottom-back-left",
        },
        "face:top": {
            "point:top-front-left",
            "point:top-front-right",
            "point:top-back-right",
            "point:top-back-left",
        },
        "face:front": {
            "point:bottom-front-left",
            "point:bottom-front-right",
            "point:top-front-right",
            "point:top-front-left",
        },
        "face:right": {
            "point:bottom-front-right",
            "point:bottom-back-right",
            "point:top-back-right",
            "point:top-front-right",
        },
        "face:back": {
            "point:bottom-back-right",
            "point:bottom-back-left",
            "point:top-back-left",
            "point:top-back-right",
        },
        "face:left": {
            "point:bottom-back-left",
            "point:bottom-front-left",
            "point:top-front-left",
            "point:top-back-left",
        },
    }
    assert {
        logical_id: {point_by_index[index] for index in face}
        for face, logical_id in zip(
            preview.faces,
            preview.face_logical_ids,
            strict=True,
        )
    } == expected_faces
    for kind, logical_ids in (
        ("point", preview.point_logical_ids),
        ("edge", preview.edge_logical_ids),
        ("face", preview.face_logical_ids),
    ):
        assert all(logical_id is not None for logical_id in logical_ids), kind


def test_preview_rejects_cell_map_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="face_logical_ids"):
        viewport_module.GeometryPreview(
            points=((0.0, 0.0, 0.0),) * 3,
            faces=((0, 1, 2),),
            edges=(),
            face_logical_ids=("face:front", "face:back"),
        )


def test_large_plate_with_small_hole_keeps_corner_identity() -> None:
    recipe = PlateWithHoleGeometry(
        "large-plate-small-hole",
        1000.0,
        800.0,
        500.0,
        400.0,
        1.0,
    )
    preview = build_geometry_preview(recipe, segments=16)
    expected = {
        "point:bottom-left": (0.0, 0.0, 0.0),
        "point:bottom-right": (1000.0, 0.0, 0.0),
        "point:top-right": (1000.0, 800.0, 0.0),
        "point:top-left": (0.0, 800.0, 0.0),
    }

    for logical_id, coordinates in expected.items():
        index = preview.point_logical_ids.index(logical_id)
        assert preview.points[index] == pytest.approx(coordinates, abs=1.0e-9)


@pytest.mark.parametrize(
    "recipe",
    (
        RectangleGeometry("catalog-rectangle", 2.0, 1.0),
        DiskGeometry("catalog-disk", 1.0),
        PlateWithHoleGeometry("catalog-plate", 2.0, 1.0, 1.0, 0.5, 0.2),
        BoxGeometry("catalog-box", 2.0, 1.0, 0.5),
        CylinderGeometry("catalog-cylinder", 0.5, 1.0),
        BooleanGeometry(
            "catalog-rectangular-cut",
            "cut",
            RectangleGeometry("outer", 2.0, 1.0),
            MovedGeometry(RectangleGeometry("inner", 0.5, 0.25), 0.75, 0.375),
        ),
        BooleanGeometry(
            "catalog-circular-cut",
            "cut",
            RectangleGeometry("outer", 2.0, 1.0),
            MovedGeometry(DiskGeometry("inner", 0.2), 1.0, 0.5),
        ),
        ExtrudedGeometry(RectangleGeometry("catalog-extrude", 2.0, 1.0), 0.5),
    ),
)
def test_selectable_preview_logical_ids_match_recipe_catalog(recipe) -> None:
    preview = build_geometry_preview(recipe, segments=16)
    topology = describe_recipe_topology(recipe)

    for logical_ids in (
        preview.point_logical_ids,
        preview.edge_logical_ids,
        preview.face_logical_ids,
    ):
        for logical_id in logical_ids:
            if logical_id is None:
                continue
            assert topology.entity(logical_id).selectable
    assert preview.body_logical_id == "body:domain"


@pytest.mark.parametrize(
    "recipe",
    (
        DiskGeometry("preview-disk-points", 1.0),
        CylinderGeometry("preview-cylinder-points", 0.5, 1.0),
    ),
)
def test_curved_display_samples_are_not_selectable_cad_points(recipe) -> None:
    import pyvista

    preview = build_geometry_preview(recipe, segments=16)
    point_mesh = _geometry_point_polydata(
        pyvista,
        np.asarray(preview.points, dtype=float),
        preview,
        (0,) * len(preview.points),
    )

    assert point_mesh.n_points == 0
    assert describe_recipe_topology(recipe).entities_of("point") == ()


def test_unproven_boolean_preview_has_no_selectable_subentities() -> None:
    recipe = BooleanGeometry(
        "overlapping-fuse",
        "fuse",
        RectangleGeometry("left", 2.0, 1.0),
        MovedGeometry(RectangleGeometry("right", 2.0, 1.0), 1.0, 0.0),
    )

    preview = build_geometry_preview(recipe)

    assert not any(preview.point_logical_ids)
    assert not any(preview.edge_logical_ids)
    assert not any(preview.face_logical_ids)
    assert preview.body_logical_id is None


def test_geometry_preview_never_exposes_internal_entity_ids_as_a_legend(
    monkeypatch,
) -> None:
    import pyvista
    from pyvista.plotting.scalar_bars import ScalarBars

    class Actor:
        def SetVisibility(self, _visible):
            pass

    class Plotter:
        def __init__(self):
            self.calls = []
            self.scalar_bars = ScalarBars(self)
            self.scalar_bars._scalar_bar_actors["stale"] = object()

        def add_mesh(self, _data, **kwargs):
            self.calls.append(kwargs)
            return Actor()

        def remove_actor(self, *_args, **_kwargs):
            pass

        def remove_scalar_bar(self, title, **_kwargs):
            self.scalar_bars._scalar_bar_actors.pop(title, None)

        def reset_camera(self, **_kwargs):
            pass

        def render(self):
            pass

    _application()
    viewport = FEMViewport()
    plotter = Plotter()
    viewport._plotter = plotter
    monkeypatch.setattr(viewport_module, "_pyvista", pyvista)
    monkeypatch.setattr(
        viewport_module,
        "is_offscreen_environment",
        lambda: False,
    )
    monkeypatch.setattr(viewport, "_ensure_plotter", lambda: True)

    viewport.show_geometry_preview(
        build_geometry_preview(
            RectangleGeometry("preview", 2.0, 1.0)
        )
    )

    assert tuple(plotter.scalar_bars.keys()) == ()
    assert plotter.calls
    assert all(
        call.get("show_scalar_bar") is False
        for call in plotter.calls
    )
    assert viewport._geometry_preview_surface.active_scalars_name is None


def test_geometry_pick_overlay_keeps_pick_data_without_rendering_over_mesh(
    monkeypatch,
) -> None:
    import pyvista

    class Actor:
        def SetVisibility(self, _visible):
            pass

    class Plotter:
        def __init__(self):
            self.calls = []
            self.render_count = 0
            self.scalar_bars = {}

        def add_mesh(self, _data, **kwargs):
            self.calls.append(kwargs)
            return Actor()

        def remove_actor(self, *_args, **_kwargs):
            pass

        def render(self):
            self.render_count += 1

    _application()
    viewport = FEMViewport()
    plotter = Plotter()
    viewport._plotter = plotter
    monkeypatch.setattr(viewport_module, "_pyvista", pyvista)
    monkeypatch.setattr(
        viewport_module,
        "is_offscreen_environment",
        lambda: False,
    )
    monkeypatch.setattr(viewport, "_ensure_plotter", lambda: True)

    viewport.show_geometry_preview(
        build_geometry_preview(
            RectangleGeometry("preview", 2.0, 1.0)
        ),
        preserve_model=True,
        render=False,
    )

    assert plotter.calls == []
    assert viewport._geometry_preview_surface is not None
    assert viewport._geometry_preview_edges is not None
    viewport.set_selection_mode("geometry_face")
    viewport.hide_geometry_selection_overlay()
    assert plotter.render_count == 0
    viewport.close()


def test_selecting_a_solid_geometry_prepares_tetrahedral_settings_and_preview() -> None:
    _application()
    window = FEMMainWindow()
    recipe = BoxGeometry("preview-box", 2.0, 1.0, 0.5)

    window._set_native_geometry(recipe, "长方体")

    geometry = window.document.geometry_recipe
    assert geometry == recipe
    assert len(window.document.parts) == 1
    assert window.document.active_part.geometry_recipe == recipe
    assert window.document.mesh_settings.cell_shape == "tetrahedron"
    assert window.viewport._geometry_preview is not None
    assert window.model_tree.topLevelItemCount() == 1
    assert (
        window.model_tree.topLevelItem(0).text(0)
        == window.document.model_name
    )
    assert "未打开模型" not in window.model_tree.topLevelItem(0).text(0)
    assert window.actions["mesh_generate"].isEnabled()
    window.close()


def test_renderer_failure_cannot_leave_valid_geometry_actions_disabled(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    render_preview = window.viewport.show_geometry_preview
    calls = 0
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    def fail_preview(preview, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("preview backend failed")
        render_preview(preview, **kwargs)

    monkeypatch.setattr(window.viewport, "show_geometry_preview", fail_preview)
    window.ribbon.set_current("几何")
    window._set_native_geometry(
        CylinderGeometry("renderer-failure", 0.5, 1.0),
        "圆柱",
    )

    assert calls == 1
    assert errors == [("编辑几何", "preview backend failed")]
    window._on_geometry_entity_pick(LogicalEntityRef("body:P1/domain"))
    assert window.actions["geometry_move"].isEnabled()
    assert window.actions["select_face"].isEnabled()
    assert window.actions["mesh_settings"].isEnabled()
    window.close()


@pytest.mark.gmsh
def test_geometry_feature_chain_is_shared_by_preview_and_gmsh() -> None:
    recipe = ExtrudedGeometry(
        RotatedGeometry(
            MovedGeometry(RectangleGeometry("feature-chain", 1.0, 0.5), 0.2, 0.1),
            "z",
            30.0,
        ),
        0.4,
    )

    preview = build_geometry_preview(recipe, segments=16)
    model = generate_fem_model(
        recipe,
        MeshSettings(0.2, cell_shape="tetrahedron"),
    )

    assert any(point[2] == pytest.approx(0.4) for point in preview.points)
    assert {element.type for element in model.mesh.elements} == {"Tet4"}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


@pytest.mark.gmsh
def test_extruded_face_scope_is_created_from_mesh_faces() -> None:
    recipe = ExtrudedGeometry(
        SketchGeometry(
            "face-region",
            (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
        ),
        0.5,
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.3,
            cell_shape="tetrahedron",
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("face:side/right"),
                    0.15,
                ),
            ),
        ),
    )
    nodes_by_id = {
        int(node.id): node
        for node in model.mesh.nodes
    }
    right_faces = tuple(
        row
        for row in mesh_faces.boundary(model.mesh)
        if all(
            float(nodes_by_id[int(node_id)].x)
            == pytest.approx(2.0, abs=1.0e-8)
            for node_id in row[2]
        )
    )

    scoped = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(
            NamedRegion(
                "LoadedFace",
                tuple(MeshEntityRef.face(*row) for row in right_faces),
            ),
        ),
    )

    assert right_faces
    assert scoped.surfaces["LoadedFace"].faces
    assert "LoadedFace" not in scoped.node_sets
    assert not model.surfaces


@pytest.mark.gmsh
def test_extruded_hole_mesh_face_scopes_keep_inner_and_outer_sides_distinct() -> None:
    recipe = ExtrudedGeometry(
        SketchGeometry(
            "extruded-hole-faces",
            (
                SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
                SketchCircle("cut", 1.0, 0.5, 0.2),
            ),
        ),
        0.5,
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(0.3, cell_shape="tetrahedron"),
    )
    nodes_by_id = {
        int(node.id): node
        for node in model.mesh.nodes
    }
    side_faces = tuple(
        row
        for row in mesh_faces.boundary(model.mesh)
        if (
            max(float(nodes_by_id[int(node_id)].z) for node_id in row[2])
            - min(float(nodes_by_id[int(node_id)].z) for node_id in row[2])
            > 1.0e-8
        )
    )
    hole_faces = tuple(
        row
        for row in side_faces
        if all(
            np.hypot(
                float(nodes_by_id[int(node_id)].x) - 1.0,
                float(nodes_by_id[int(node_id)].y) - 0.5,
            )
            == pytest.approx(0.2, abs=1.0e-7)
            for node_id in row[2]
        )
    )
    outer_faces = tuple(row for row in side_faces if row not in hole_faces)
    scoped = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(
            NamedRegion(
                "HoleSide",
                tuple(MeshEntityRef.face(*row) for row in hole_faces),
            ),
            NamedRegion(
                "OuterSide",
                tuple(MeshEntityRef.face(*row) for row in outer_faces),
            ),
        ),
    )

    hole_nodes = {
        node_id
        for face in scoped.surfaces["HoleSide"].faces
        for node_id in face.node_ids
    }
    outer_nodes = {
        node_id
        for face in scoped.surfaces["OuterSide"].faces
        for node_id in face.node_ids
    }
    assert hole_nodes
    assert outer_nodes
    assert hole_nodes.isdisjoint(outer_nodes)


@pytest.mark.gmsh
def test_selected_geometry_edge_can_drive_local_mesh_refinement() -> None:
    recipe = DiskGeometry("locally-refined-disk", 1.0)
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.25,
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:outer"),
                    0.05,
                ),
            ),
        ),
    )

    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert len(mesh_edges.boundary(model.mesh)) >= 20
    assert not model.node_sets
    assert not model.edges


@pytest.mark.gmsh
def test_generic_hole_inner_edge_local_control_increases_mesh_density() -> None:
    recipe = SketchGeometry(
        "locally-refined-generic-hole",
        (
            SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
            SketchCircle("cut", 1.0, 0.5, 0.2),
        ),
    )

    baseline = generate_fem_model(recipe, MeshSettings(0.25))
    refined = generate_fem_model(
        recipe,
        MeshSettings(
            0.25,
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:hole-loop"),
                    0.05,
                ),
            ),
        ),
    )

    assert len(refined.mesh.nodes) > len(baseline.mesh.nodes)
    assert len(refined.mesh.elements) > len(baseline.mesh.elements)


def test_local_mesh_command_enters_viewport_edge_selection_first() -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        SketchGeometry(
            "select-edge-first",
            (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
        ),
        "草图",
    )

    window.set_local_mesh_control()

    assert window._pending_local_mesh_selection
    assert window.viewport._selection_mode == "geometry_edge"
    window.close()


def test_geometry_edge_distance_uses_display_pixels() -> None:
    distance, fraction = _point_to_segment_distance(
        np.asarray((25.0, 16.0)),
        np.asarray((10.0, 10.0)),
        np.asarray((40.0, 10.0)),
    )

    assert distance == pytest.approx(6.0)
    assert fraction == pytest.approx(0.5)


@pytest.mark.gmsh
@pytest.mark.parametrize("operation", ("fuse", "cut", "fragment"))
def test_boolean_features_are_meshed_by_the_same_native_workflow(operation) -> None:
    object_geometry = RectangleGeometry(f"boolean-object-{operation}", 2.0, 1.0)
    tool_geometry = MovedGeometry(
        DiskGeometry(f"boolean-tool-{operation}", 0.2),
        1.0,
        0.5,
    )
    recipe = BooleanGeometry(
        f"boolean-result-{operation}",
        operation,
        object_geometry,
        tool_geometry,
    )

    preview = build_geometry_preview(recipe, segments=16)
    model = generate_fem_model(recipe, MeshSettings(0.2))

    assert preview.faces
    assert preview.edges
    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces
