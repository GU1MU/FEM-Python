from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import (
    ModelSession,
    NamedRegion,
    RegionAssignment,
    SectionDefinition,
)
from fem.core.model import (
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
)
from fem.solvers.static_linear import solve, validate_problem
from fem.steps.factory import static
from fem_gui.analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    LoadDialog,
    OutputRequestDialog,
    StaticStepDialog,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing import (
    ExtrudedGeometry,
    MeshSettings,
    RectangleGeometry,
    generate_fem_model,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_native_linear_static_definition_reuses_the_existing_solver():
    recipe = RectangleGeometry("plate", 2.0, 1.0)
    settings = MeshSettings(0.5)
    model = generate_fem_model(recipe, settings)
    step = static("Step-1")
    step.boundaries = (DisplacementConstraint("LEFT", 1, 2, 0.0),)
    step.cloads = (NodalLoad("RIGHT", 1, 10.0),)
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(session.snapshot().parts, recipe)
    session.replace_mesh_settings(settings)
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
    mesh_task = session.prepare_mesh_generation()
    session.accept_generated_model(mesh_task.token, model)
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
        ["LEFT", "Fixed"],
        2,
        selected_region="Fixed",
    )
    step_name, boundary = boundary_dialog.definition()
    assert step_name == "Load"
    assert boundary.target == "Fixed"
    load_dialog = LoadDialog(
        ["Load"],
        ["RIGHT"],
        ["TOP"],
        [],
        2,
        selected_region="TOP",
        preferred_kind="edge",
    )
    step_name, load = load_dialog.definition()
    assert step_name == "Load"
    assert load.edge == "TOP"
    assert load_dialog.component_combo.itemText(0) == "Fx"
    output_dialog = OutputRequestDialog(["Load"])
    step_name, output = output_dialog.definition()
    assert step_name == "Load"
    assert output.variables == ("U", "RF")
    assert output_dialog.kind_combo.findData("history") == -1


def test_displacement_dialog_creates_independent_checked_dofs():
    _application()
    dialog = DisplacementDialog(["Load"], ["Fixed"], 3)
    dialog.component_checks[1].setChecked(False)
    dialog.component_checks[2].setChecked(True)
    dialog.component_values[2].setValue(0.25)
    dialog.component_checks[3].setChecked(True)
    dialog.component_values[3].setValue(-0.5)

    step_name, boundaries = dialog.definitions()

    assert step_name == "Load"
    assert [
        (item.first_component, item.last_component, item.value)
        for item in boundaries
    ] == [(2, 2, 0.25), (3, 3, -0.5)]


def test_analysis_manager_uses_a_copy_and_deletes_selected_definition():
    _application()
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 2, 0.0),)
    step.cloads = (NodalLoad("Loaded", 1, 10.0),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        ["Fixed", "Loaded"],
        ["Loaded"],
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
        ["Loaded"],
        [],
        2,
        current=EdgeLoad("Loaded", (2.0, -3.0), load_type="traction"),
    )

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load.edge == "Loaded"
    assert load.vector == (2.0, -3.0)


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
        ["EdgeSet"],
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
        ["NodeSet"],
        ["EdgeSet"],
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
        ["NodeSet"],
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
        ["Surface"],
        6,
        spatial_dimensions=3,
    )
    assert surface_dialog.form.isRowVisible(surface_dialog.z_spin)
    _step, load = surface_dialog.definition()
    assert len(load.vector) == 3


def test_main_window_filters_distributed_load_regions_by_model_dimension():
    _application()
    window = FEMMainWindow()
    rectangle = RectangleGeometry("plate", 2.0, 1.0)
    regions = (
        NamedRegion("NodeSet", "point", (1,)),
        NamedRegion("EdgeSet", "edge", (1,)),
        NamedRegion("Surface", "face", (1,)),
    )
    window._set_native_geometry(rectangle, "矩形")
    assert window._apply_session_delta(
        window.session.replace_named_regions(regions)
    )

    node_regions, edge_regions, face_regions = (
        window._supported_load_region_names()
    )
    assert "NodeSet" in node_regions
    assert edge_regions == ["EdgeSet"]
    assert face_regions == []

    window._set_native_geometry(ExtrudedGeometry(rectangle, 1.0), "拉伸体")
    assert window._apply_session_delta(
        window.session.replace_named_regions(regions)
    )
    node_regions, edge_regions, face_regions = (
        window._supported_load_region_names()
    )
    assert "NodeSet" in node_regions
    assert edge_regions == []
    assert face_regions == ["Surface"]
    window.close()


def test_load_dialog_validates_region_and_builds_pressure():
    _application()
    missing_region = LoadDialog(["Load"], [], ["Loaded"], [], 2)
    missing_region.region_combo.clear()
    with pytest.raises(ValueError, match="载荷区域"):
        missing_region.definition()

    dialog = LoadDialog(["Load"], [], ["Loaded"], [], 2)
    dialog.load_type_combo.setCurrentIndex(
        dialog.load_type_combo.findData("pressure")
    )
    dialog.value_spin.setValue(12.5)

    step_name, load = dialog.definition()

    assert step_name == "Load"
    assert load.edge == "Loaded"
    assert load.load_type == "pressure"
    assert load.magnitude == 12.5


def test_output_request_uses_supported_target_specific_variables():
    _application()
    dialog = OutputRequestDialog(["Load"])

    assert not dialog.variable_checks["U"].isHidden()
    assert not dialog.variable_checks["RF"].isHidden()
    assert dialog.variable_checks["S"].isHidden()

    dialog.target_combo.setCurrentIndex(
        dialog.target_combo.findData("element")
    )
    step_name, output = dialog.definition()

    assert step_name == "Load"
    assert output.kind == "field"
    assert output.target == "element"
    assert output.variables == ("S",)
    assert dialog.variable_checks["U"].isHidden()
    assert not dialog.variable_checks["S"].isHidden()

    dialog.variable_checks["S"].setChecked(False)
    with pytest.raises(ValueError, match="至少选择"):
        dialog.definition()


def test_output_request_preserves_parsed_inp_history_metadata():
    _application()
    current = OutputRequest(
        "history",
        "preselect",
        ("PRESELECT",),
        {"variable": "PRESELECT"},
    )
    dialog = OutputRequestDialog(["Load"], current=current)

    step_name, output = dialog.definition()

    assert step_name == "Load"
    assert output.kind == "history"
    assert output.target == "preselect"
    assert output.variables == ("PRESELECT",)
    assert output.metadata == {"variable": "PRESELECT"}
    assert not dialog.kind_combo.isEnabled()
    assert "PRESELECT" in dialog.preserved_label.text()


def test_analysis_manager_uses_readable_definition_summaries():
    _application()
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 1, 0.0),)
    step.outputs = (OutputRequest("field", "node", ("U", "RF")),)
    manager = AnalysisDefinitionManagerDialog(
        [step],
        ["Fixed"],
        [],
        [],
        2,
    )

    assert manager.table.item(0, 3).text() == "线性静力"
    assert manager.table.item(1, 3).text() == "U1 = 0"
    assert manager.table.item(2, 0).text() == "字段输出"
    assert manager.table.item(2, 2).text() == "节点"
