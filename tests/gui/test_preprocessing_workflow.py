from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
import pytest

from fem_gui.document import NamedRegion
from fem_gui.main_window import FEMMainWindow
from fem_gui.mesh_quality import analyze_mesh
from fem_gui.preprocessing import (
    BoxGeometry,
    BooleanGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LocalMeshControl,
    MeshSettings,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    build_geometry_preview,
    generate_fem_model,
)
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    _geometry_edge_polydata,
    _point_to_segment_distance,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    deadline = monotonic() + 10.0
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def test_native_rectangle_mesh_joins_the_existing_model_workflow() -> None:
    _application()
    window = FEMMainWindow()
    recipe = RectangleGeometry("gui-native-rectangle", 2.0, 1.0)
    settings = MeshSettings(0.25, order=1, cell_shape="triangle")
    window.document.geometry_recipe = recipe
    window.document.mesh_settings = settings
    window.document.native_mesh_current = False
    window._update_action_states()

    window.generate_native_mesh()
    _wait_for_task(window)

    assert window.document.source_kind == "native"
    assert window.document.geometry_recipe is recipe
    assert window.document.mesh_settings is settings
    assert window.document.path is None
    assert not window.actions["reload"].isEnabled()
    assert window.document.model is not None
    assert window.geometry is not None
    assert window.document.model.name == recipe.name
    assert {element.type for element in window.document.model.mesh.elements} == {"Tri3"}
    assert "DOMAIN" in window.document.model.element_sets
    assert {"LEFT", "RIGHT", "BOTTOM", "TOP"}.issubset(
        window.document.model.node_sets
    )
    assert window.ribbon.tab_bar.tabText(window.ribbon.tab_bar.currentIndex()) == "模型"
    assert window.actions["mesh_generate"].isEnabled()
    assert window.actions["mesh_clear"].isEnabled()
    assert window.actions["mesh_quality"].isEnabled()

    report = analyze_mesh(window.document.model)
    assert report.checked_count == report.element_count
    assert 0.0 < report.minimum <= report.mean <= report.maximum <= 1.0

    window.clear_native_mesh()
    assert window.document.model is None
    assert window.document.geometry_recipe is recipe
    assert window.document.mesh_settings is settings
    assert window.viewport._geometry_preview is not None
    assert window.actions["mesh_generate"].isEnabled()
    window.close()


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
    assert "DOMAIN" in model.element_sets
    assert "BOUNDARY" not in model.node_sets


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
    assert {"BOTTOM", "TOP", "OUTER"}.issubset(model.node_sets)


def test_quadrilateral_setting_reaches_the_same_gui_adapter() -> None:
    _application()
    window = FEMMainWindow()
    window.document.geometry_recipe = RectangleGeometry(
        "gui-native-quad",
        2.0,
        1.0,
    )
    window.document.mesh_settings = MeshSettings(
        0.25,
        order=2,
        cell_shape="quadrilateral",
    )

    window.generate_native_mesh()
    _wait_for_task(window)

    assert window.document.model is not None
    assert {element.type for element in window.document.model.mesh.elements} == {"Quad8"}
    assert len(window.geometry.cells) == len(window.document.model.mesh.elements)
    window.close()


def test_plate_with_hole_imports_named_boundary_and_local_refinement() -> None:
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
        local_size=0.04,
    )
    window.document.geometry_recipe = recipe
    window.document.mesh_settings = settings

    window.generate_native_mesh()
    _wait_for_task(window)

    model = window.document.model
    assert model is not None
    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert "HOLE" in model.node_sets
    assert "HOLE" in model.edges
    assert len(model.node_sets["HOLE"].node_ids) >= 12
    window.close()


@pytest.mark.parametrize(
    ("recipe", "expected_type", "boundary_names"),
    (
        (DiskGeometry("gui-disk", 1.0), "Tri3", {"OUTER"}),
        (
            BoxGeometry("gui-box", 1.0, 0.8, 0.6),
            "Tet4",
            {"LEFT", "RIGHT", "FRONT", "BACK", "BOTTOM", "TOP"},
        ),
        (
            CylinderGeometry("gui-cylinder", 0.5, 1.0),
            "Tet4",
            {"BOTTOM", "TOP", "OUTER"},
        ),
    ),
)
def test_added_basic_geometries_generate_canonical_models(
    recipe,
    expected_type,
    boundary_names,
) -> None:
    settings = MeshSettings(
        0.25,
        cell_shape="tetrahedron" if expected_type == "Tet4" else "triangle",
    )

    model = generate_fem_model(recipe, settings)

    assert {element.type for element in model.mesh.elements} == {expected_type}
    assert "DOMAIN" in model.element_sets
    assert boundary_names.issubset(model.node_sets)


def test_box_supports_structured_hexahedral_mesh() -> None:
    recipe = BoxGeometry("gui-structured-box", 1.0, 0.8, 0.6)

    model = generate_fem_model(
        recipe,
        MeshSettings(0.3, cell_shape="hexahedron"),
    )

    assert {element.type for element in model.mesh.elements} == {"Hex8"}
    assert {"LEFT", "RIGHT", "FRONT", "BACK", "BOTTOM", "TOP"}.issubset(
        model.node_sets
    )


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
    )

    assert edge_mesh.n_cells == len(preview.edges)
    assert len(edge_mesh.cell_data["geometry_entity_id"]) == len(preview.edges)
    assert edge_mesh.active_scalars_name is None


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

        def reset_camera(self):
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


def test_selecting_a_solid_geometry_prepares_tetrahedral_settings_and_preview() -> None:
    _application()
    window = FEMMainWindow()
    recipe = BoxGeometry("preview-box", 2.0, 1.0, 0.5)

    window._set_native_geometry(recipe, "长方体")

    assert window.document.geometry_recipe is recipe
    assert window.document.mesh_settings.cell_shape == "tetrahedron"
    assert window.viewport._geometry_preview is not None
    assert window.model_tree.topLevelItemCount() == 1
    assert window.model_tree.topLevelItem(0).text(0) == recipe.name
    assert "未打开模型" not in window.model_tree.topLevelItem(0).text(0)
    assert window.actions["mesh_generate"].isEnabled()
    window.close()


def test_renderer_failure_cannot_leave_valid_geometry_actions_disabled(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()

    def fail_preview(_preview) -> None:
        raise RuntimeError("preview backend failed")

    monkeypatch.setattr(window.viewport, "show_geometry_preview", fail_preview)
    with pytest.raises(RuntimeError, match="preview backend failed"):
        window._set_native_geometry(
            CylinderGeometry("renderer-failure", 0.5, 1.0),
            "圆柱",
        )

    assert window.actions["geometry_move"].isEnabled()
    assert window.actions["geometry_select_face"].isEnabled()
    assert window.actions["mesh_settings"].isEnabled()
    window.close()


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
    assert {"BOTTOM", "TOP", "OUTER"}.issubset(model.node_sets)


def test_extruded_face_named_region_reuses_the_resolved_cad_face_groups() -> None:
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
            local_controls=(LocalMeshControl("face", 4, 0.15),),
        ),
        named_regions=(NamedRegion("LoadedFace", "face", (4,)),),
    )

    assert "LoadedFace" in model.node_sets
    assert model.node_sets["LoadedFace"].node_ids
    assert set(model.node_sets["LoadedFace"].node_ids) < set(
        model.node_sets["OUTER"].node_ids
    )


def test_extruded_hole_keeps_inner_and_outer_side_face_ids_distinct() -> None:
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
        named_regions=(
            NamedRegion("HoleSide", "face", (3,)),
            NamedRegion("OuterSide", "face", (4,)),
        ),
    )

    hole_nodes = set(model.node_sets["HoleSide"].node_ids)
    outer_nodes = set(model.node_sets["OuterSide"].node_ids)
    assert hole_nodes
    assert outer_nodes
    assert hole_nodes.isdisjoint(outer_nodes)


def test_selected_geometry_edge_can_drive_local_mesh_refinement() -> None:
    recipe = DiskGeometry("locally-refined-disk", 1.0)
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.25,
            local_controls=(LocalMeshControl("edge", 1, 0.05),),
        ),
    )

    assert {element.type for element in model.mesh.elements} == {"Tri3"}
    assert len(model.node_sets["OUTER"].node_ids) >= 20


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
            local_controls=(LocalMeshControl("edge", 1, 0.05),),
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
    assert "DOMAIN" in model.element_sets
    assert "BOUNDARY" not in model.node_sets
