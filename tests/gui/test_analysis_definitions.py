from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RegionRef,
    SectionDefinition,
)
from fem.application.results import (
    ElementResultProfile,
    ResultCapabilityCatalog,
    ResultModelFamily,
)
from fem.application.preprocessing import generate_fem_model
from fem.core.model import (
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    OutputSourceEvidence,
)
from fem.geometry import ExtrudedGeometry, LogicalEntityRef, RectangleGeometry
from fem.mesh.settings import MeshSettings
from fem.solvers.static_linear import solve, validate_problem
from fem.steps.factory import static
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    LoadDialog,
    OutputRequestDialog,
    StaticStepDialog,
)
from fem_gui.dialogs import AdaptivePrecisionDoubleSpinBox
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_new_static_step_uses_a_chinese_default_name(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(RectangleGeometry("plate", 2.0, 1.0), "矩形")
    names: list[str] = []

    class _Dialog:
        def __init__(self, name, parent):
            del parent
            names.append(name)

    monkeypatch.setattr("fem_gui.main_window.StaticStepDialog", _Dialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: False)

    window.create_static_step()

    assert names == ["分析步-1"]
    window.close()


def test_analysis_manager_reuses_one_authoring_projection(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("cached-authoring", 2.0, 1.0),
        "矩形",
    )
    window._analysis_definitions_changed(
        "测试分析步",
        [static("Step-1")],
    )
    window._session_authoring_cache = None
    calls = []
    original = main_window_module.describe_session_authoring

    def record_projection(snapshot):
        calls.append(snapshot)
        return original(snapshot)

    monkeypatch.setattr(
        main_window_module,
        "describe_session_authoring",
        record_projection,
    )

    dialog = window._analysis_manager_dialog()

    assert dialog is not None
    assert calls == [window.document]
    dialog.close()
    window.close()


def test_boundary_scope_highlight_reuses_entity_highlighter(monkeypatch):
    _application()
    window = FEMMainWindow()
    highlighted = []
    cleared = []
    monkeypatch.setattr(
        window,
        "highlight_entity",
        lambda kind, key: highlighted.append((kind, key)),
    )
    monkeypatch.setattr(
        window.viewport,
        "clear_selection",
        lambda: cleared.append(True),
    )

    window._highlight_analysis_scope(RegionRef("surface", "FixedFace"))
    window._highlight_analysis_scope(None)

    assert highlighted == [("surface", "FixedFace")]
    assert cleared == [True]
    window.close()


def _regions(kind: str, *names: str) -> list[RegionRef]:
    return [RegionRef(kind, name) for name in names]


def _output_candidates(
    family: ResultModelFamily = ResultModelFamily.PLANE_CONTINUUM,
):
    values = {
        ResultModelFamily.PLANE_CONTINUUM: (
            ("Quad4",),
            ("plane_continuum",),
            ("U1", "U2"),
            ("Fx", "Fy"),
        ),
        ResultModelFamily.BEAM: (
            ("Beam2",),
            ("beam",),
            ("U1", "U2", "U3", "UR1", "UR2", "UR3"),
            ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        ),
        ResultModelFamily.TRUSS: (
            ("Truss2",),
            ("truss",),
            ("U1", "U2", "U3"),
            ("Fx", "Fy", "Fz"),
        ),
    }
    element_types, element_families, dofs, forces = values[family]
    profile = ElementResultProfile(
        family=family,
        canonical_element_types=element_types,
        element_families=element_families,
        dofs_per_node=len(dofs),
        dof_labels=dofs,
        force_labels=forces,
        primary_compatible=True,
        stress_compatible=True,
    )
    return ResultCapabilityCatalog.from_profile(profile).candidates


def _output_capability(
    operation: str,
    status: AuthoringStatus,
) -> AuthoringCapability:
    return AuthoringCapability(operation, status)


@pytest.mark.gmsh
def test_native_linear_static_definition_reuses_the_existing_solver():
    recipe = RectangleGeometry("plate", 2.0, 1.0)
    settings = MeshSettings(0.5)
    model = generate_fem_model(recipe, settings)
    x_coordinates = tuple(node.x for node in model.mesh.nodes)
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    tolerance = max(1.0, abs(x_min), abs(x_max)) * 1.0e-9
    regions = (
        NamedRegion(
            "DOMAIN",
            tuple(
                MeshEntityRef.element(element.id)
                for element in model.mesh.elements
            ),
        ),
        NamedRegion(
            "LEFT",
            tuple(
                MeshEntityRef.node(node.id)
                for node in model.mesh.nodes
                if abs(node.x - x_min) <= tolerance
            ),
        ),
        NamedRegion(
            "RIGHT",
            tuple(
                MeshEntityRef.node(node.id)
                for node in model.mesh.nodes
                if abs(node.x - x_max) <= tolerance
            ),
        ),
    )
    step = static("Step-1")
    step.boundaries = (DisplacementConstraint("LEFT", 1, 2, 0.0),)
    step.cloads = (NodalLoad("RIGHT", 1, 10.0),)
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(session.snapshot().parts, recipe)
    session.replace_mesh_settings(settings)
    mesh_task = session.prepare_mesh_generation()
    session.accept_generated_model(mesh_task.token, model)
    session.replace_named_regions(regions)
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (
            SectionDefinition(
                "Section-1",
                "Steel",
                properties={"thickness": 1.0},
            ),
        ),
        (RegionAssignment("Section-1", "DOMAIN"),),
        (step,),
    )
    compiled_model = session.snapshot().model

    selected = validate_problem(compiled_model, "Step-1")
    result = solve(compiled_model, selected)

    assert selected is not None
    assert result.U.size == compiled_model.mesh.num_dofs


def test_analysis_dialogs_define_only_supported_kernel_objects():
    _application()
    step_dialog = StaticStepDialog("Load")
    assert step_dialog.step().procedure == "static"
    boundary_dialog = DisplacementDialog(
        ["Load"],
        _regions("node_set", "LEFT", "Fixed"),
        2,
        selected_region=RegionRef("node_set", "Fixed"),
    )
    step_name, boundaries = boundary_dialog.definitions()
    assert step_name == "Load"
    assert boundaries[0].target == "Fixed"
    assert boundaries[0].target_kind == "node_set"
    load_dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "RIGHT"),
        _regions("edge", "TOP"),
        [],
        2,
        selected_region=RegionRef("edge", "TOP"),
        preferred_kind="edge",
    )
    load_dialog.x_spin.setValue(1.0)
    step_name, load = load_dialog.definition()
    assert step_name == "Load"
    assert load.edge == "TOP"
    assert load_dialog.component_combo.itemText(0) == "Fx"
    candidates = _output_candidates()
    output_dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
    )
    output_dialog.candidate_list.item(0).setCheckState(
        Qt.CheckState.Checked
    )
    step_name, output = output_dialog.definition()
    assert step_name == "Load"
    assert output == candidates[0].authoring_request


def test_analysis_dialog_region_catalogs_reject_untyped_strings():
    _application()

    with pytest.raises(TypeError, match="RegionRef"):
        DisplacementDialog(["Load"], ["Fixed"], 2)

    with pytest.raises(TypeError, match="RegionRef"):
        LoadDialog(["Load"], ["Loaded"], [], [], 2)


def test_displacement_dialog_creates_independent_checked_dofs():
    _application()
    dialog = DisplacementDialog(
        ["Load"],
        _regions("node_set", "Fixed"),
        3,
    )
    dialog.component_checks[1].setChecked(False)
    dialog.name_edit.setText("支座位移")
    dialog.component_checks[2].setChecked(True)
    dialog.component_values[2].setValue(0.25)
    dialog.component_checks[3].setChecked(True)
    dialog.component_values[3].setValue(-0.5)

    step_name, boundaries = dialog.definitions()

    assert step_name == "Load"
    assert [
        (item.name, item.first_component, item.last_component, item.value)
        for item in boundaries
    ] == [
        ("支座位移", 2, 2, 0.25),
        ("支座位移-2", 3, 3, -0.5),
    ]


def test_displacement_dialog_reports_selected_scope_changes():
    _application()
    dialog = DisplacementDialog(
        ["Load"],
        [
            *_regions("node_set", "Fixed", "Roller"),
            *_regions("edge", "SupportedEdge"),
        ],
        2,
    )
    selected = []
    dialog.scopeChanged.connect(selected.append)

    dialog.region_combo.setCurrentIndex(1)
    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("edge")
    )

    assert selected == [
        RegionRef("node_set", "Roller"),
        RegionRef("edge", "SupportedEdge"),
    ]


def test_boundary_and_load_inputs_share_adaptive_precision():
    _application()
    boundary = DisplacementDialog(
        ["Load"],
        _regions("node_set", "Fixed"),
        2,
        selected_region=RegionRef("node_set", "Fixed"),
        current=DisplacementConstraint("Fixed", 1, 1, 0.123456),
    )
    load = LoadDialog(
        ["Load"],
        _regions("node_set", "Loaded"),
        [],
        [],
        2,
        selected_region=RegionRef("node_set", "Loaded"),
        current=NodalLoad("Loaded", 1, 3.456789),
    )

    boundary_value = boundary.component_values[1]
    assert isinstance(boundary_value, AdaptivePrecisionDoubleSpinBox)
    assert isinstance(load.value_spin, AdaptivePrecisionDoubleSpinBox)
    assert all(editor.decimals() == 12 for editor in boundary.component_values.values())
    assert all(
        editor.decimals() == 12
        for editor in (load.value_spin, load.x_spin, load.y_spin, load.z_spin)
    )
    assert boundary_value.text() == "0.12"
    assert load.value_spin.text() == "3.46"

    boundary_value.selectAll()
    QTest.keyClicks(boundary_value, "0.123456")
    QTest.keyClick(boundary_value, Qt.Key.Key_Return)
    load.value_spin.selectAll()
    QTest.keyClicks(load.value_spin, "3.456789")
    QTest.keyClick(load.value_spin, Qt.Key.Key_Return)

    assert boundary_value.text() == "0.123456"
    assert boundary.definitions()[1][0].value == 0.123456
    assert load.value_spin.text() == "3.456789"
    assert load.definition()[1].value == 3.456789


def test_displacement_dialog_merges_adjacent_equal_dofs():
    _application()
    dialog = DisplacementDialog(
        ["Load"],
        _regions("surface", "FixedFace"),
        3,
        selected_region=RegionRef("surface", "FixedFace"),
    )
    for component in (1, 2, 3):
        dialog.component_checks[component].setChecked(True)
        dialog.component_values[component].setValue(0.0)

    _step_name, boundaries = dialog.definitions()

    assert boundaries == (
        DisplacementConstraint(
            "FixedFace",
            1,
            3,
            0.0,
            target_kind="surface",
        ),
    )


def test_displacement_dialog_restores_form_after_scope_creation():
    _application()
    original = DisplacementDialog(
        ["Step-1", "Step-2"],
        _regions("surface", "OldFace"),
        3,
        selected_region=RegionRef("surface", "OldFace"),
        scope_selection_kinds=("surface",),
    )
    original.step_combo.setCurrentText("Step-2")
    original.name_edit.setText("固定端")
    for component in (1, 2, 3):
        original.component_checks[component].setChecked(True)
        original.component_values[component].setValue(0.125)
    original.scope_pick_button.click()

    restored = DisplacementDialog(
        ["Step-1", "Step-2"],
        _regions("surface", "OldFace", "NewFace"),
        3,
        selected_region=RegionRef("surface", "NewFace"),
        scope_selection_kinds=("surface",),
        form_state=original.form_state(),
    )
    step_name, boundaries = restored.definitions()

    assert original.requested_scope_kind() == "surface"
    assert restored.kind_combo.currentData() == "surface"
    assert restored.region_combo.currentData() == RegionRef(
        "surface",
        "NewFace",
    )
    assert step_name == "Step-2"
    assert boundaries == (
        DisplacementConstraint(
            "NewFace",
            1,
            3,
            0.125,
            target_kind="surface",
            name="固定端",
        ),
    )


def test_load_dialog_restores_surface_load_after_scope_creation():
    _application()
    original = LoadDialog(
        ["Step-1", "Step-2"],
        _regions("node_set", "Nodes"),
        [],
        _regions("surface", "OldFace"),
        3,
        selected_region=RegionRef("surface", "OldFace"),
        preferred_kind="surface",
        scope_selection_kinds=("surface",),
    )
    original.step_combo.setCurrentText("Step-2")
    original.name_edit.setText("端面压力")
    original.load_type_combo.setCurrentIndex(
        original.load_type_combo.findData("pressure")
    )
    original.value_spin.setValue(12.5)
    original.scope_pick_button.click()

    restored = LoadDialog(
        ["Step-1", "Step-2"],
        _regions("node_set", "Nodes"),
        [],
        _regions("surface", "OldFace", "NewFace"),
        3,
        selected_region=RegionRef("surface", "NewFace"),
        scope_selection_kinds=("surface",),
        form_state=original.form_state(),
    )
    step_name, load = restored.definition()

    assert original.requested_scope_kind() == "surface"
    assert restored.kind_combo.currentData() == "surface"
    assert restored.region_combo.currentData() == RegionRef(
        "surface",
        "NewFace",
    )
    assert restored.load_type_combo.currentData() == "pressure"
    assert step_name == "Step-2"
    assert load.magnitude == 12.5
    assert load.surface == "NewFace"
    assert load.name == "端面压力"


@pytest.mark.parametrize(
    ("kind", "name"),
    (("edge", "FixedEdge"), ("surface", "FixedSurface")),
)
def test_displacement_dialog_accepts_edge_and_surface_regions(kind, name):
    _application()
    dialog = DisplacementDialog(
        ["Load"],
        [
            RegionRef("node_set", "FixedNodes"),
            RegionRef("edge", "FixedEdge"),
            RegionRef("surface", "FixedSurface"),
        ],
        3,
        selected_region=RegionRef(kind, name),
    )

    step_name, boundaries = dialog.definitions()

    assert step_name == "Load"
    assert boundaries[0].target == name
    assert boundaries[0].target_kind == kind
    assert not hasattr(boundaries[0], "node_ids")


def test_analysis_manager_uses_a_copy_and_deletes_selected_definition():
    _application()
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 2, 0.0),)
    step.cloads = (NodalLoad("Loaded", 1, 10.0),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        _regions("node_set", "Fixed", "Loaded"),
        _regions("edge", "Loaded"),
        [],
        2,
    )

    assert manager.table.rowCount() == 3
    manager.table.selectRow(1)
    manager._delete()

    assert len(step.boundaries) == 1
    assert manager.values()[0].boundaries == ()
    assert manager.values()[0].cloads[0].target == "Loaded"


def test_load_dialog_can_edit_an_existing_distributed_load():
    _application()
    dialog = LoadDialog(
        ["Load"],
        [],
        _regions("edge", "Loaded"),
        [],
        2,
        current=EdgeLoad(
            "Loaded",
            (2.0, -3.0),
            load_type="traction",
            name="剪切载荷",
        ),
    )

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load.edge == "Loaded"
    assert load.vector == (2.0, -3.0)
    assert dialog.name_edit.text() == "剪切载荷"
    assert load.name == "剪切载荷"


@pytest.mark.parametrize(
    "kind",
    ("node", "edge", "surface", "line", "body", "gravity"),
)
def test_load_dialog_saves_names_for_every_load_kind(kind):
    _application()
    dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "Nodes"),
        _regions("edge", "Edges"),
        _regions("surface", "Faces"),
        3,
        spatial_dimensions=3,
        line_regions=_regions("element_set", "Lines"),
        body_regions=_regions("element_set", "Domain"),
    )
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData(kind))
    dialog.name_edit.setText("工况载荷")
    if kind in {"edge", "surface"}:
        dialog.x_spin.setValue(1.0)

    _step_name, load = dialog.definition()

    assert load.name == "工况载荷"


def test_analysis_manager_can_rename_boundary_and_load(monkeypatch):
    _application()
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint(
            "Fixed",
            1,
            2,
            name="旧位移约束",
        ),
    )
    step.edge_loads = (
        EdgeLoad(
            "Loaded",
            (2.0, -3.0),
            name="旧边载荷",
        ),
    )
    manager = AnalysisDefinitionManagerDialog(
        [step],
        _regions("node_set", "Fixed"),
        _regions("edge", "Loaded"),
        [],
        2,
    )

    def rename_boundary(dialog):
        dialog.name_edit.setText("固定端位移")
        return True

    monkeypatch.setattr(DisplacementDialog, "exec", rename_boundary)
    assert manager.edit_definition(("boundary", 0, 0))

    def rename_load(dialog):
        dialog.name_edit.setText("加载边牵引")
        return True

    monkeypatch.setattr(LoadDialog, "exec", rename_load)
    assert manager.edit_definition(("edge_load", 0, 0))

    updated = manager.values()[0]
    assert updated.boundaries[0].name == "固定端位移"
    assert updated.edge_loads[0].name == "加载边牵引"


def test_analysis_manager_forwards_boundary_scope_changes(monkeypatch):
    _application()
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint("Fixed", 1, 2),
    )
    manager = AnalysisDefinitionManagerDialog(
        [step],
        _regions("node_set", "Fixed", "Roller"),
        [],
        [],
        2,
    )
    selected = []
    manager.scopeChanged.connect(selected.append)

    def select_scope(dialog):
        assert selected == [RegionRef("node_set", "Fixed")]
        dialog.region_combo.setCurrentIndex(1)
        return False

    monkeypatch.setattr(DisplacementDialog, "exec", select_scope)

    assert not manager.edit_definition(("boundary", 0, 0))
    assert selected == [
        RegionRef("node_set", "Fixed"),
        RegionRef("node_set", "Roller"),
    ]


def test_edge_load_editor_refreshes_only_after_dialog_construction(
    monkeypatch,
):
    _application()
    refresh_states = []
    original_refresh = LoadDialog._refresh

    def tracked_refresh(dialog):
        refresh_states.append(
            (
                hasattr(dialog, "buttons"),
                dialog.kind_combo.currentData(),
            )
        )
        original_refresh(dialog)

    monkeypatch.setattr(LoadDialog, "_refresh", tracked_refresh)

    dialog = LoadDialog(
        ["Load"],
        [],
        _regions("edge", "Loaded"),
        [],
        2,
        current=EdgeLoad(
            "Loaded",
            (0.0, 0.0),
            magnitude=-1.0,
            load_type="pressure",
        ),
    )

    assert refresh_states == [(True, "edge")]
    assert dialog.windowTitle() == "编辑载荷"
    dialog.close()


def test_load_dialog_creates_global_gravity_without_a_named_region():
    _application()
    dialog = LoadDialog(
        ["Load"],
        [],
        [],
        [],
        3,
        spatial_dimensions=3,
    )

    assert dialog.kind_combo.currentData() == "gravity"
    assert not dialog.form.isRowVisible(dialog.region_combo)
    assert dialog.form.labelForField(dialog.x_spin).text() == "ax"
    assert dialog.form.labelForField(dialog.z_spin).text() == "az"

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load == GravityLoad((0.0, 0.0, -9.81))


def test_load_dialog_keeps_gravity_and_distributed_vectors_separate():
    _application()
    dialog = LoadDialog(
        ["Load"],
        [],
        _regions("edge", "EdgeSet"),
        [],
        2,
        spatial_dimensions=2,
    )

    assert dialog.kind_combo.currentData() == "edge"
    assert dialog.y_spin.value() == 0.0
    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("gravity")
    )
    assert dialog.y_spin.value() == -9.81
    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("edge")
    )
    assert dialog.y_spin.value() == 0.0


def test_load_dialog_exposes_five_physical_categories_and_builds_body_force():
    _application()
    dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "Nodes"),
        _regions("edge", "Edges"),
        _regions("surface", "Faces"),
        3,
        spatial_dimensions=3,
        body_regions=_regions("element_set", "Domain"),
    )

    assert [
        dialog.kind_combo.itemText(index)
        for index in range(dialog.kind_combo.count())
    ] == ["节点力", "边力", "面力", "体力", "重力"]
    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("body")
    )
    dialog.x_spin.setValue(1.5)
    dialog.y_spin.setValue(-2.0)
    dialog.z_spin.setValue(3.25)

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load == BodyForce("Domain", (1.5, -2.0, 3.25))
    assert dialog.form.labelForField(dialog.x_spin).text() == "bx"
    assert not dialog.form.isRowVisible(dialog.load_type_combo)


def test_analysis_manager_lists_and_deletes_gravity_loads():
    _application()
    step = static("Load")
    step.gravity_loads = (GravityLoad((0.0, -9.81)),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        2,
        spatial_dimensions=2,
    )

    assert manager.table.rowCount() == 2
    assert manager.table.item(1, 0).text() == "重力"
    manager.table.selectRow(1)
    manager._delete()

    assert step.gravity_loads == (GravityLoad((0.0, -9.81)),)
    assert manager.values()[0].gravity_loads == ()


def test_load_dialog_only_shows_parameters_for_the_selected_load_kind():
    _application()
    dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "NodeSet"),
        _regions("edge", "EdgeSet"),
        [],
        2,
    )

    assert dialog.form.isRowVisible(dialog.component_combo)
    assert dialog.form.isRowVisible(dialog.value_spin)
    assert not dialog.form.isRowVisible(dialog.load_type_combo)
    assert not dialog.form.isRowVisible(dialog.x_spin)

    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("edge")
    )
    assert dialog.region_combo.currentText() == "EdgeSet"
    assert dialog.form.isRowVisible(dialog.load_type_combo)
    assert not dialog.form.isRowVisible(dialog.component_combo)
    assert dialog.form.isRowVisible(dialog.x_spin)
    assert dialog.form.isRowVisible(dialog.y_spin)
    assert not dialog.form.isRowVisible(dialog.z_spin)
    assert not dialog.form.isRowVisible(dialog.value_spin)

    dialog.load_type_combo.setCurrentIndex(
        dialog.load_type_combo.findData("pressure")
    )
    assert dialog.form.isRowVisible(dialog.value_spin)
    assert dialog.form.labelForField(dialog.value_spin).text() == "压力值"
    assert not dialog.form.isRowVisible(dialog.x_spin)


def test_load_dialog_separates_nodal_dofs_from_spatial_vector_dimension():
    _application()
    dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "NodeSet"),
        [],
        [],
        6,
        spatial_dimensions=1,
    )

    assert [
        dialog.component_combo.itemText(index)
        for index in range(dialog.component_combo.count())
    ] == ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    surface_dialog = LoadDialog(
        ["Load"],
        [],
        [],
        _regions("surface", "Surface"),
        6,
        spatial_dimensions=3,
    )
    assert surface_dialog.form.isRowVisible(surface_dialog.z_spin)
    surface_dialog.z_spin.setValue(1.0)
    _step, load = surface_dialog.definition()
    assert len(load.vector) == 3


def test_main_window_filters_distributed_load_regions_by_model_dimension():
    _application()
    window = FEMMainWindow()
    rectangle = RectangleGeometry("plate", 2.0, 1.0)
    planar_regions = (
        NamedRegion(
            "NodeSet",
            (LogicalEntityRef("point:P1/bottom-left"),),
        ),
        NamedRegion(
            "EdgeSet",
            (LogicalEntityRef("edge:P1/bottom"),),
        ),
        NamedRegion(
            "Surface",
            (LogicalEntityRef("face:P1/domain"),),
        ),
    )
    window._set_native_geometry(rectangle, "矩形")
    assert window._apply_session_delta(
        window.session.replace_named_regions(planar_regions)
    )

    (
        node_regions,
        edge_regions,
        face_regions,
        line_regions,
        body_regions,
    ) = (
        window._supported_load_regions()
    )
    assert node_regions == [
        RegionRef("node_set", "EdgeSet"),
        RegionRef("node_set", "NodeSet"),
    ]
    assert edge_regions == [
        RegionRef("edge", "EdgeSet"),
    ]
    assert face_regions == []
    assert line_regions == []
    assert body_regions == [
        RegionRef("element_set", "Surface"),
    ]
    assert set(window._supported_boundary_regions()) == {
        RegionRef("node_set", "EdgeSet"),
        RegionRef("edge", "EdgeSet"),
        RegionRef("node_set", "NodeSet"),
    }

    window._set_native_geometry(ExtrudedGeometry(rectangle, 1.0), "拉伸体")
    solid_regions = (
        NamedRegion(
            "NodeSet",
            (LogicalEntityRef("point:P1/bottom/bottom-left"),),
        ),
        NamedRegion(
            "EdgeSet",
            (LogicalEntityRef("edge:P1/bottom/bottom"),),
        ),
        NamedRegion(
            "Surface",
            (LogicalEntityRef("face:P1/bottom"),),
        ),
    )
    assert window._apply_session_delta(
        window.session.replace_named_regions(solid_regions)
    )
    (
        node_regions,
        edge_regions,
        face_regions,
        line_regions,
        body_regions,
    ) = (
        window._supported_load_regions()
    )
    assert node_regions == [
        RegionRef("node_set", "EdgeSet"),
        RegionRef("node_set", "NodeSet"),
        RegionRef("node_set", "Surface"),
    ]
    assert edge_regions == []
    assert face_regions == [
        RegionRef("surface", "Surface"),
    ]
    assert line_regions == []
    assert body_regions == []
    assert set(window._supported_boundary_regions()) == {
        RegionRef("node_set", "EdgeSet"),
        RegionRef("node_set", "NodeSet"),
        RegionRef("node_set", "Surface"),
        RegionRef("surface", "Surface"),
    }
    window.close()


def test_unmeshed_rectangle_publishes_exact_catalog_region_choices():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("catalog-plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.model is None
    assert window._analysis_region_names() == ([], [], [])
    assert window._analysis_element_regions() == []
    window.close()


def test_load_dialog_validates_region_and_builds_pressure():
    _application()
    edge_regions = _regions("edge", "Loaded")
    missing_region = LoadDialog(["Load"], [], edge_regions, [], 2)
    missing_region.region_combo.clear()
    with pytest.raises(ValueError, match="载荷作用域"):
        missing_region.definition()

    dialog = LoadDialog(["Load"], [], edge_regions, [], 2)
    dialog.load_type_combo.setCurrentIndex(
        dialog.load_type_combo.findData("pressure")
    )
    dialog.value_spin.setValue(12.5)

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load.edge == "Loaded"
    assert load.load_type == "pressure"
    assert load.magnitude == 12.5


def test_load_dialog_rejects_zero_distributed_loads():
    _application()
    dialog = LoadDialog(
        ["Load"],
        [],
        _regions("edge", "Loaded"),
        [],
        2,
    )

    with pytest.raises(ValueError, match="非零分量"):
        dialog.definition()

    dialog.load_type_combo.setCurrentIndex(
        dialog.load_type_combo.findData("pressure")
    )
    with pytest.raises(ValueError, match="压力值不能为 0"):
        dialog.definition()


def test_scope_pick_buttons_request_node_edge_and_surface_selection():
    _application()
    displacement = DisplacementDialog(
        ["Load"],
        [],
        2,
        allow_scope_selection=True,
    )
    assert not displacement.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert displacement.scope_pick_button.text() == "创建"
    assert displacement.scope_pick_button.toolTip() == ""
    displacement.scope_pick_button.click()
    assert displacement.requested_scope_kind() == "node"

    displacement = DisplacementDialog(
        ["Load"],
        [],
        3,
        scope_selection_kinds=("node", "edge", "surface"),
    )
    for kind, target_kind in (
        ("node", "node_set"),
        ("edge", "edge"),
        ("surface", "surface"),
    ):
        displacement.kind_combo.setCurrentIndex(
            displacement.kind_combo.findData(target_kind)
        )
        assert displacement.scope_pick_button.isEnabled()
        displacement._scope_selection_request = None
        displacement.scope_pick_button.click()
        assert displacement.requested_scope_kind() == kind

    load = LoadDialog(
        ["Load"],
        [],
        [],
        [],
        3,
        scope_selection_kinds=("node", "edge", "surface"),
    )
    assert [
        load.kind_combo.itemData(index)
        for index in range(load.kind_combo.count())
    ] == ["node", "edge", "surface", "gravity"]
    for kind in ("node", "edge", "surface"):
        load.kind_combo.setCurrentIndex(load.kind_combo.findData(kind))
        assert load.scope_pick_button.isEnabled()
        assert load.scope_pick_button.text() == "创建"
        assert load.scope_pick_button.toolTip() == ""
        load._scope_selection_request = None
        load.scope_pick_button.click()
        assert load.requested_scope_kind() == kind


def test_analysis_manager_edit_requests_a_new_load_scope(monkeypatch):
    _application()
    step = static("Load")
    step.edge_loads = (EdgeLoad("EdgeSet-1", (-10.0, 0.0)),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        _regions("edge", "EdgeSet-1"),
        [],
        2,
        load_scope_selection_kinds=("edge",),
    )

    def request_scope(dialog):
        assert dialog.scope_pick_button.isEnabled()
        dialog.scope_pick_button.click()
        return False

    monkeypatch.setattr(LoadDialog, "exec", request_scope)

    assert not manager.edit_definition(("edge_load", 0, 0))
    assert manager.requested_scope_selection() == (
        "edge",
        ("edge_load", 0, 0),
    )


def test_analysis_manager_restores_unsaved_load_editor_state(monkeypatch):
    _application()
    first_step = static("Step-1")
    first_step.edge_loads = (EdgeLoad("EdgeSet-1", (-10.0, 0.0)),)
    second_step = static("Step-2")
    manager = AnalysisDefinitionManagerDialog(
        [first_step, second_step],
        [],
        _regions("edge", "EdgeSet-1"),
        _regions("surface", "OldFace"),
        2,
        load_scope_selection_kinds=("surface",),
    )

    def request_scope(dialog):
        dialog.kind_combo.setCurrentIndex(
            dialog.kind_combo.findData("surface")
        )
        dialog.step_combo.setCurrentText("Step-2")
        dialog.load_type_combo.setCurrentIndex(
            dialog.load_type_combo.findData("pressure")
        )
        dialog.value_spin.setValue(8.5)
        dialog.scope_pick_button.click()
        return False

    monkeypatch.setattr(LoadDialog, "exec", request_scope)
    assert not manager.edit_definition(("edge_load", 0, 0))
    state = manager.requested_scope_dialog_state()

    restored_manager = AnalysisDefinitionManagerDialog(
        [first_step, second_step],
        [],
        _regions("edge", "EdgeSet-1"),
        _regions("surface", "OldFace", "NewFace"),
        2,
        load_scope_selection_kinds=("surface",),
    )

    def accept_restored(dialog):
        assert dialog.kind_combo.currentData() == "surface"
        assert dialog.step_combo.currentText() == "Step-2"
        assert dialog.region_combo.currentData() == RegionRef(
            "surface",
            "NewFace",
        )
        assert dialog.load_type_combo.currentData() == "pressure"
        assert dialog.value_spin.value() == 8.5
        return True

    monkeypatch.setattr(LoadDialog, "exec", accept_restored)
    assert restored_manager.edit_definition(
        ("edge_load", 0, 0),
        selected_region=RegionRef("surface", "NewFace"),
        dialog_state=state,
    )
    restored_steps = restored_manager.values()

    assert restored_steps[0].edge_loads == ()
    assert restored_steps[1].surface_loads[0].surface == "NewFace"
    assert restored_steps[1].surface_loads[0].magnitude == 8.5


def test_analysis_manager_edit_requests_a_new_boundary_scope(monkeypatch):
    _application()
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint(
            "NodeSet-1",
            1,
            1,
            0.0,
            target_kind="node_set",
        ),
    )
    manager = AnalysisDefinitionManagerDialog(
        [step],
        _regions("node_set", "NodeSet-1"),
        [],
        [],
        2,
        boundary_scope_selection_kinds=("node",),
    )

    def request_scope(dialog):
        assert dialog.scope_pick_button.isEnabled()
        dialog.scope_pick_button.click()
        return False

    monkeypatch.setattr(DisplacementDialog, "exec", request_scope)

    assert not manager.edit_definition(("boundary", 0, 0))
    assert manager.requested_scope_selection() == (
        "node",
        ("boundary", 0, 0),
    )


def test_edit_load_dialog_prefers_a_new_explicit_scope():
    _application()
    dialog = LoadDialog(
        ["Load"],
        [],
        _regions("edge", "EdgeSet-1", "EdgeSet-2"),
        [],
        2,
        selected_region=RegionRef("edge", "EdgeSet-2"),
        current=EdgeLoad("EdgeSet-1", (-10.0, 0.0)),
    )

    assert dialog.region_combo.currentData() == RegionRef(
        "edge",
        "EdgeSet-2",
    )
    assert dialog.x_spin.value() == -10.0


@pytest.mark.parametrize(
    "family",
    (
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.TRUSS,
        ResultModelFamily.BEAM,
    ),
)
def test_output_request_uses_only_published_candidate_order_and_dto(
    family,
):
    _application()
    candidates = _output_candidates(family)
    dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
    )

    expected = tuple(
        next(
            candidate
            for candidate in candidates
            if candidate.authoring_request.variables == (variable,)
        )
        for variable in ("U", "RF", "S")
    )
    assert dialog.candidate_list.count() == len(expected)
    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == ("U", "RF", "S")
    assert all(
        "position" not in dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    )
    displacement_item = dialog.candidate_list.item(0)
    assert displacement_item.checkState() == Qt.CheckState.Unchecked
    displacement_item.setCheckState(Qt.CheckState.Unchecked)
    assert displacement_item.checkState() == Qt.CheckState.Unchecked
    for index in range(dialog.candidate_list.count()):
        dialog.candidate_list.item(index).setCheckState(
            Qt.CheckState.Checked
        )
    step_name, outputs = dialog.definitions()
    assert step_name == "Load"
    assert outputs == tuple(
        candidate.authoring_request
        for candidate in expected
    )
    assert all(
        output is not candidate.authoring_request
        for output, candidate in zip(outputs, expected, strict=True)
    )


def test_output_request_discards_parsed_inp_details():
    _application()
    current = OutputRequest(
        "history",
        "preselect",
        ("PRESELECT", "PRESELECT", "Future"),
        {"variable": "PRESELECT"},
        OutputSourceEvidence(
            "abaqus",
            parent_parameters=(("frequency", "2"),),
            child_flags=("preselect",),
        ),
    )
    dialog = OutputRequestDialog(["Load"], current=current)

    step_name, output = dialog.definition()

    assert step_name == "Load"
    assert output != current
    assert output is not current
    assert output == OutputRequest(
        "history",
        "preselect",
        ("PRESELECT", "PRESELECT", "Future"),
    )
    assert not output.metadata
    assert output.source_evidence is None
    assert not dialog.step_combo.isEnabled()
    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == (
        "PRESELECT",
        "PRESELECT",
        "Future",
    )
    assert all(
        dialog.candidate_list.item(index).checkState()
        == Qt.CheckState.Checked
        for index in range(dialog.candidate_list.count())
    )
    assert not hasattr(dialog, "target_value")
    assert not hasattr(dialog, "metadata_value")
    assert not hasattr(dialog, "source_evidence_value")


def test_output_request_dialog_shows_existing_imported_requests_by_step():
    _application()
    candidates = _output_candidates(ResultModelFamily.PLANE_CONTINUUM)
    dialog = OutputRequestDialog(
        ["Load", "Empty"],
        candidates=candidates,
        existing_requests_by_step={
            "Load": (
                OutputRequest("field", "node", ("RF", "U")),
                OutputRequest("field", "element", ("S",)),
            ),
            "Empty": (),
        },
    )

    assert tuple(
        dialog.candidate_list.item(index).text()
        for index in range(dialog.candidate_list.count())
    ) == ("U", "RF", "S")
    assert all(
        dialog.candidate_list.item(index).checkState()
        == Qt.CheckState.Checked
        for index in range(dialog.candidate_list.count())
    )
    assert not hasattr(dialog, "existing_value")
    assert not hasattr(dialog, "kind_value")
    assert not hasattr(dialog, "variables_value")
    dialog.step_combo.setCurrentText("Empty")
    assert tuple(
        dialog.candidate_list.item(index).checkState()
        for index in range(dialog.candidate_list.count())
    ) == (
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
    )


def test_output_request_dialog_does_not_select_history_variables_as_fields():
    _application()
    candidates = _output_candidates(ResultModelFamily.PLANE_CONTINUUM)
    dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
        existing_requests_by_step={
            "Load": (
                OutputRequest("history", "node", ("U", "RF")),
                OutputRequest("history", "element", ("S", "MISES")),
            ),
        },
    )

    assert tuple(
        dialog.candidate_list.item(index).checkState()
        for index in range(dialog.candidate_list.count())
    ) == (
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
        Qt.CheckState.Unchecked,
    )


def test_analysis_manager_uses_readable_definition_summaries():
    _application()
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 1, 0.0),)
    step.outputs = (OutputRequest("field", "node", ("U", "RF")),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        _regions("node_set", "Fixed"),
        [],
        [],
        2,
    )

    assert manager.table.item(0, 3).text() == "线性静力"
    assert manager.table.item(1, 3).text() == "U1 = 0"
    assert manager.table.item(2, 0).text() == "输出"
    assert manager.table.item(2, 2).text() == "节点"


def test_model_tree_boundary_and_load_delete_preserve_other_definitions():
    first = DisplacementConstraint("Fixed", 1, 1, 0.0)
    second = DisplacementConstraint("Roller", 2, 2, 0.0)
    first_load = NodalLoad("Loaded", 1, 10.0)
    second_load = NodalLoad("Loaded", 2, 20.0)
    step = static("Load")
    step.boundaries = (first, second)
    step.cloads = (first_load, second_load)
    changes = []

    class WindowStub:
        def __init__(self):
            self.document = type(
                "Document",
                (),
                {"steps": (step,)},
            )()

        def _analysis_definitions_changed(self, reason, definitions):
            changes.append((reason, definitions))

    window = WindowStub()

    FEMMainWindow.delete_analysis_definition(
        window,
        "boundary",
        (0, 0),
    )

    assert step.boundaries == (first, second)
    assert changes[0][0] == "边界条件已删除，模型需要重新检查"
    assert changes[0][1][0].boundaries == (second,)

    FEMMainWindow.delete_analysis_definition(
        window,
        "cload",
        (0, 0),
    )
    assert changes[1][0] == "载荷已删除，模型需要重新检查"
    assert changes[1][1][0].cloads == (second_load,)

    FEMMainWindow.delete_analysis_definition(
        window,
        "output",
        (0, 0),
    )
    FEMMainWindow.delete_analysis_definition(
        window,
        "boundary",
        (9, 0),
    )
    assert len(changes) == 2


def test_output_view_is_read_only_and_preserves_unsupported_request(
    monkeypatch,
) -> None:
    _application()
    output = OutputRequest(
        "history",
        "preselect",
        ("Future", "Future", "PRESELECT"),
        {"future": {"mode": "opaque"}},
        OutputSourceEvidence(
            "abaqus",
            parent_flags=("history",),
            child_parameters=(("variable", "PRESELECT"),),
        ),
    )
    step = static("Load")
    step.outputs = (output,)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.READ_ONLY,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.UNAVAILABLE,
        ),
    )
    monkeypatch.setattr(
        OutputRequestDialog,
        "exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )
    before = manager.values()

    assert manager.select_definition(("output", 0, 0))
    assert manager.edit_button.text() == "查看"
    assert manager.edit_button.isEnabled()
    assert not manager.delete_button.isEnabled()
    assert not manager.edit_definition(("output", 0, 0))
    assert manager.values() == before
    assert manager.values()[0].outputs[0] == output


def test_output_delete_uses_independent_capability_and_protects_initial() -> None:
    _application()
    output = OutputRequest("history", "preselect", ("Future",))
    load = static("Load")
    load.outputs = (output,)
    denied = AnalysisDefinitionManagerDialog(
        [load],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.UNAVAILABLE,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.UNAVAILABLE,
        ),
    )
    assert denied.select_definition(("output", 0, 0))
    assert not denied.edit_button.isEnabled()
    assert not denied.delete_button.isEnabled()
    denied._delete()
    assert denied.values()[0].outputs == (output,)

    allowed = AnalysisDefinitionManagerDialog(
        [load],
        [],
        [],
        [],
        2,
        output_view_capability=_output_capability(
            "output_request.view",
            AuthoringStatus.UNAVAILABLE,
        ),
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert allowed.select_definition(("output", 0, 0))
    assert not allowed.edit_button.isEnabled()
    assert allowed.delete_button.isEnabled()
    allowed._delete()
    assert allowed.values()[0].outputs == ()

    required = OutputRequest("field", "node", ("U",))
    required_step = static("Load")
    required_step.outputs = (required,)
    required_manager = AnalysisDefinitionManagerDialog(
        [required_step],
        [],
        [],
        [],
        2,
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert required_manager.select_definition(("output", 0, 0))
    assert not required_manager.delete_button.isEnabled()
    required_manager._delete()
    assert required_manager.values()[0].outputs == (required,)

    initial = static("Initial")
    initial.outputs = (output,)
    protected = AnalysisDefinitionManagerDialog(
        [initial],
        [],
        [],
        [],
        2,
        output_delete_capability=_output_capability(
            "output_request.delete",
            AuthoringStatus.ENABLED,
        ),
    )
    assert protected.select_definition(("output", 0, 0))
    assert not protected.delete_button.isEnabled()
    protected._delete()
    assert protected.values()[0].outputs == (output,)
    assert protected.select_definition(("step", 0, None))
    assert not protected.delete_button.isEnabled()
    protected._delete()
    assert len(protected.values()) == 1
    assert protected.values()[0].outputs == (output,)


def test_output_dialog_has_no_gui_support_matrix_or_dto_rebuild() -> None:
    source_path = Path(inspect.getsourcefile(OutputRequestDialog) or "")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OutputRequestDialog"
    )
    string_values = {
        node.value
        for node in ast.walk(class_node)
        if isinstance(node, ast.Constant)
        and type(node.value) is str
    }
    output_request_calls = [
        node
        for node in ast.walk(class_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OutputRequest"
    ]
    attributes = {
        node.attr
        for node in ast.walk(class_node)
        if isinstance(node, ast.Attribute)
    }

    assert string_values.isdisjoint({"U", "UR", "RF", "RM", "S"})
    assert output_request_calls == []
    assert attributes.isdisjoint(
        {"upper", "split", "startswith", "endswith"}
    )
