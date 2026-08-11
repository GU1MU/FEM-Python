from __future__ import annotations

from dataclasses import replace
import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import RegionRef, SectionDefinition
from fem.application.preprocessing import generate_fem_model
from fem.core.model import MaterialDefinition
from fem.geometry import (
    BoxGeometry,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.mesh.settings import MeshSettings
import fem_gui.main_window as main_window_module
from fem_gui.analysis_definition_dialogs import LoadDialogState
from fem_gui.scope_selection import build_scope_selection_topology
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


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
def test_native_rectangle_uses_exact_whole_geometry_edges() -> None:
    recipe = RectangleGeometry("NativeScopeSelection", 2.0, 1.0)
    model = generate_fem_model(recipe, MeshSettings(0.2))

    topology = build_scope_selection_topology(model, recipe)
    edge_groups = {
        reference.logical_id: mesh_references
        for reference, mesh_references in topology.mesh_references.items()
        if reference.kind == "edge"
    }

    assert set(edge_groups) == {
        "edge:bottom",
        "edge:left",
        "edge:right",
        "edge:top",
    }
    assert all(len(references) > 1 for references in edge_groups.values())
    assert all(
        mesh_reference.kind == "edge"
        for references in edge_groups.values()
        for mesh_reference in references
    )
    surface_groups = [
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "face"
    ]
    assert len(surface_groups) == 1
    assert all(
        mesh_reference.kind == "element"
        for mesh_reference in surface_groups[0]
    )


@pytest.mark.gmsh
def test_imported_planar_mesh_infers_continuous_geometry_edges() -> None:
    model = generate_fem_model(
        RectangleGeometry("ImportedPlanarScope", 2.0, 1.0),
        MeshSettings(0.2),
    )
    model.metadata.clear()

    topology = build_scope_selection_topology(model)
    edge_groups = [
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "edge"
    ]

    assert len(edge_groups) == 4
    assert all(len(references) > 1 for references in edge_groups)
    surface_groups = [
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "face"
    ]
    assert len(surface_groups) == 1
    assert all(
        mesh_reference.kind == "element"
        for mesh_reference in surface_groups[0]
    )


@pytest.mark.gmsh
def test_imported_solid_mesh_infers_surfaces_and_feature_edges() -> None:
    model = generate_fem_model(
        BoxGeometry("ImportedSolidScope", 1.0, 1.0, 1.0),
        MeshSettings(0.35, cell_shape="tetrahedron"),
    )
    model.metadata.clear()

    topology = build_scope_selection_topology(model)
    kinds = {
        reference.kind
        for reference in topology.mesh_references
    }
    surface_groups = [
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "face"
    ]

    assert kinds == {"body", "edge", "face"}
    assert len(surface_groups) == 6
    assert all(references for references in surface_groups)
    volume = next(
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "body"
    )
    assert all(reference.kind == "element" for reference in volume)


@pytest.mark.gmsh
def test_native_wire_edges_expand_to_line_elements() -> None:
    recipe = WireGeometry(
        "WireScope",
        (
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0, 0.0),
        ),
        (WireMember("M1", "P1", "P2"),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.2,
            cell_shape="line",
            line_element_type="Truss2",
        ),
    )

    topology = build_scope_selection_topology(model, recipe)
    edge_scopes = [
        references
        for reference, references in topology.mesh_references.items()
        if reference.kind == "edge"
    ]

    assert len(edge_scopes) == 1
    assert edge_scopes[0]
    assert all(
        reference.kind == "element"
        for reference in edge_scopes[0]
    )
    assert {
        reference.kind
        for reference in topology.mesh_references
    } == {"edge"}


@pytest.mark.gmsh
def test_geometry_edge_box_selection_expands_and_ctrl_toggles(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    recipe = RectangleGeometry("BoxScopeSelection", 2.0, 1.0)
    _set_native_mesh_inputs(window, recipe, MeshSettings(0.2))
    window.generate_native_mesh()
    _wait_for_task(window)
    window._request_analysis_geometry_selection("scope", "edge")
    topology = window._scope_selection_topology()
    geometry_edges = tuple(
        reference
        for reference in topology.mesh_references
        if reference.kind == "edge"
    )
    selected = geometry_edges[:2]
    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: False,
    )

    window._on_geometry_entities_box_selected(selected)

    assert window._selected_geometry_refs == set(selected)
    assert window._selected_mesh_scope_refs == {
        mesh_reference
        for reference in selected
        for mesh_reference in topology.mesh_references[reference]
    }

    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: True,
    )
    window._on_geometry_entities_box_selected((selected[1],))

    assert window._selected_geometry_refs == {selected[0]}
    assert window._selected_mesh_scope_refs == set(
        topology.mesh_references[selected[0]]
    )
    window._on_geometry_entity_pick(selected[0])

    assert not window._selected_geometry_refs
    assert not window._selected_mesh_scope_refs
    assert not (
        window.viewport_panel.scope_creation_bar.create_button.isEnabled()
    )
    window.close()


def test_box_selection_direction_uses_containment_then_crossing(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    viewport = window.viewport
    monkeypatch.setattr(
        viewport,
        "_qt_to_vtk_position",
        lambda x, y: (int(x), int(y)),
    )

    _bounds, left_to_right = viewport._vtk_rectangle(
        (10.0, 10.0),
        (50.0, 50.0),
    )
    _bounds, right_to_left = viewport._vtk_rectangle(
        (50.0, 10.0),
        (10.0, 50.0),
    )

    assert left_to_right
    assert not right_to_left
    window.close()


def test_completed_scope_creation_reopens_load_editor_with_new_scope(
    monkeypatch,
) -> None:
    application = _application()
    window = FEMMainWindow()
    definition_key = ("edge_load", 0, 0)
    window._pending_analysis_selection = "load"
    window._pending_scope_kind = "edge"
    window._pending_analysis_edit = (definition_key, "edge", ())
    monkeypatch.setattr(
        window,
        "_canonical_mesh_scope_selection",
        lambda: (object(),),
    )
    monkeypatch.setattr(
        window,
        "_create_region_from_current_mesh_selection",
        lambda **_kwargs: "EdgeSet-2",
    )
    resumed = []
    monkeypatch.setattr(
        window,
        "_edit_analysis_definition_key",
        lambda key, **kwargs: resumed.append((key, kwargs)),
    )

    window._complete_scope_creation_from_bar()
    application.processEvents()

    assert resumed == [
        (
            definition_key,
            {
                "selected_region": RegionRef("edge", "EdgeSet-2"),
                "steps": (),
            },
        ),
    ]
    assert window._pending_analysis_edit is None
    window.close()


def test_completed_scope_creation_restores_new_load_form_and_scope(
    monkeypatch,
) -> None:
    application = _application()
    window = FEMMainWindow()
    state = LoadDialogState(
        "surface",
        "Step-2",
        "pressure",
        "global",
        1,
        12.5,
        (),
    )
    window._pending_analysis_selection = "load"
    window._pending_analysis_requested_scope_kind = "surface"
    window._pending_analysis_dialog_state = state
    window._pending_scope_kind = "face"
    window._pending_analysis_edit = None
    resumed = []
    monkeypatch.setattr(
        window,
        "create_load",
        lambda selected_region, **kwargs: resumed.append(
            (selected_region, kwargs)
        ),
    )

    window._finish_scope_creation_from_bar("LoadedFace")
    application.processEvents()

    assert resumed == [
        (
            RegionRef("surface", "LoadedFace"),
            {"dialog_state": state},
        )
    ]
    window.close()


@pytest.mark.gmsh
def test_scope_dispatch_offers_dimension_appropriate_semantic_types(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    recipe = BoxGeometry("ScopeDispatch", 1.0, 1.0, 1.0)
    _set_native_mesh_inputs(
        window,
        recipe,
        MeshSettings(0.4, cell_shape="tetrahedron"),
    )
    window.generate_native_mesh()
    _wait_for_task(window)
    captured: dict[str, tuple[str, ...]] = {}

    def choose(
        _parent,
        _title,
        _label,
        items,
        _current,
        _editable,
    ):
        captured["items"] = tuple(items)
        return "Set", True

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getItem",
        choose,
    )
    fit_calls = []
    monkeypatch.setattr(
        window.viewport,
        "fit",
        lambda: fit_calls.append(True),
    )

    assert window._choose_mesh_scope_kind() == "node"
    _application().processEvents()
    assert captured["items"] == ("Set", "Edge", "Surface", "Volume")
    assert fit_calls == []
    assert window.actions["select_edge"].text() == "选择边"
    window._start_edge_scope_selection()
    assert window._pending_scope_kind == "edge"
    assert window.viewport._selection_mode == "geometry_edge"
    window.close()


@pytest.mark.gmsh
def test_imported_mesh_can_create_an_inferred_whole_edge_scope(
    tmp_path,
) -> None:
    _application()
    model = generate_fem_model(
        RectangleGeometry("ImportedGuiScope", 2.0, 1.0),
        MeshSettings(0.2),
    )
    model.metadata.clear()
    window = FEMMainWindow()
    task = window.session.prepare_import(tmp_path / "imported.inp")
    delta = window.session.accept_imported_model(task.token, model)
    assert window._apply_session_delta(
        delta,
        model_geometry=build_model_geometry(model),
        source_label="ImportedGuiScope",
    )

    window._request_analysis_geometry_selection("scope", "edge")
    topology = window._scope_selection_topology()
    geometry_edge = next(
        reference
        for reference in topology.mesh_references
        if reference.kind == "edge"
    )
    expected = topology.mesh_references[geometry_edge]

    window._on_geometry_entity_pick(geometry_edge)
    window.viewport_panel.scope_creation_bar.name_edit.setText(
        "ImportedBoundary"
    )
    window._confirm_guided_selection()
    _application().processEvents()

    assert window.document.source_kind == "imported"
    assert (
        window.document.named_regions["ImportedBoundary"].references
        == expected
    )
    assert "ImportedBoundary" in window.document.model.edges
    window.close()


@pytest.mark.gmsh
def test_2d_section_scope_creation_uses_surface_bar_and_element_set(
    monkeypatch,
) -> None:
    application = _application()
    window = FEMMainWindow()
    window.show()
    application.processEvents()
    recipe = RectangleGeometry("SectionSurfaceScope", 2.0, 1.0)
    _set_native_mesh_inputs(window, recipe, MeshSettings(0.2))
    window.generate_native_mesh()
    _wait_for_task(window)
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            (
                MaterialDefinition(
                    "Steel",
                    {"E": 210000.0, "nu": 0.3},
                ),
            ),
            (SectionDefinition("PlateSection", "Steel"),),
            (),
            (),
        )
    )
    resumed: list[object] = []
    monkeypatch.setattr(
        window,
        "_exec_dialog",
        lambda dialog: resumed.append(dialog) or False,
    )

    window._request_analysis_geometry_selection("section", "element_set")

    bar = window.viewport_panel.scope_creation_bar
    assert not bar.isHidden()
    assert bar.type_value.text() == "Surface"
    assert window._pending_scope_kind == "face"
    assert window.viewport._selection_mode == "geometry_face"
    topology = window._scope_selection_topology()
    geometry_face = next(
        reference
        for reference in topology.mesh_references
        if reference.kind == "face"
    )

    window._on_geometry_entities_box_selected((geometry_face,))

    assert bar.create_button.isEnabled()
    assert {
        reference.kind
        for reference in window._selected_mesh_scope_refs
    } == {"element"}
    bar.name_edit.setText("PlateSurface")
    bar.create_button.click()
    application.processEvents()

    assert (
        window.document.named_regions["PlateSurface"].references
        == tuple(
            replace(reference, part_id=window.document.active_part_id)
            for reference in topology.mesh_references[geometry_face]
        )
    )
    assert "PlateSurface" in window.document.model.element_sets
    assert len(resumed) == 1
    assert (
        resumed[0].region_combo.currentData().name
        == "PlateSurface"
    )
    assert bar.isHidden()
    window._confirm_discard_changes = lambda: True
    window.close()
