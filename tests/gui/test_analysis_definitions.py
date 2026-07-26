from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
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
from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


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
        _regions("node_set", "LEFT", "Fixed"),
        2,
        selected_region=RegionRef("node_set", "Fixed"),
    )
    step_name, boundaries = boundary_dialog.definitions()
    assert step_name == "Load"
    assert boundaries[0].target == "Fixed"
    load_dialog = LoadDialog(
        ["Load"],
        _regions("node_set", "RIGHT"),
        _regions("edge", "TOP"),
        [],
        2,
        selected_region=RegionRef("edge", "TOP"),
        preferred_kind="edge",
    )
    step_name, load = load_dialog.definition()
    assert step_name == "Load"
    assert load.edge == "TOP"
    assert load_dialog.component_combo.itemText(0) == "Fx"
    candidates = _output_candidates()
    output_dialog = OutputRequestDialog(
        ["Load"],
        candidates=candidates,
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
    _step, load = surface_dialog.definition()
    assert len(load.vector) == 3


def test_main_window_filters_distributed_load_regions_by_model_dimension():
    _application()
    window = FEMMainWindow()
    rectangle = RectangleGeometry("plate", 2.0, 1.0)
    planar_regions = (
        NamedRegion(
            "NodeSet",
            (LogicalEntityRef("point:bottom-left"),),
        ),
        NamedRegion(
            "EdgeSet",
            (LogicalEntityRef("edge:bottom"),),
        ),
        NamedRegion(
            "Surface",
            (LogicalEntityRef("face:domain"),),
        ),
    )
    window._set_native_geometry(rectangle, "矩形")
    assert window._apply_session_delta(
        window.session.replace_named_regions(planar_regions)
    )

    node_regions, edge_regions, face_regions, line_regions = (
        window._supported_load_regions()
    )
    assert node_regions == [
        RegionRef("node_set", "BOTTOM"),
        RegionRef("node_set", "EdgeSet"),
        RegionRef("node_set", "LEFT"),
        RegionRef("node_set", "NodeSet"),
        RegionRef("node_set", "RIGHT"),
        RegionRef("node_set", "TOP"),
    ]
    assert edge_regions == [
        RegionRef("edge", "BOTTOM"),
        RegionRef("edge", "EdgeSet"),
        RegionRef("edge", "LEFT"),
        RegionRef("edge", "RIGHT"),
        RegionRef("edge", "TOP"),
    ]
    assert face_regions == []
    assert line_regions == []

    window._set_native_geometry(ExtrudedGeometry(rectangle, 1.0), "拉伸体")
    solid_regions = (
        NamedRegion(
            "NodeSet",
            (LogicalEntityRef("point:bottom/bottom-left"),),
        ),
        NamedRegion(
            "EdgeSet",
            (LogicalEntityRef("edge:bottom/bottom"),),
        ),
        NamedRegion(
            "Surface",
            (LogicalEntityRef("face:bottom"),),
        ),
    )
    assert window._apply_session_delta(
        window.session.replace_named_regions(solid_regions)
    )
    node_regions, edge_regions, face_regions, line_regions = (
        window._supported_load_regions()
    )
    assert node_regions == [
        RegionRef("node_set", "BOTTOM"),
        RegionRef("node_set", "EdgeSet"),
        RegionRef("node_set", "NodeSet"),
        RegionRef("node_set", "OUTER"),
        RegionRef("node_set", "Surface"),
        RegionRef("node_set", "TOP"),
    ]
    assert edge_regions == []
    assert face_regions == [
        RegionRef("surface", "BOTTOM"),
        RegionRef("surface", "OUTER"),
        RegionRef("surface", "Surface"),
        RegionRef("surface", "TOP"),
    ]
    assert line_regions == []
    window.close()


def test_unmeshed_rectangle_publishes_exact_catalog_region_choices():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("catalog-plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.model is None
    assert window._analysis_region_names() == (
        _regions("node_set", "BOTTOM", "LEFT", "RIGHT", "TOP"),
        _regions("edge", "BOTTOM", "LEFT", "RIGHT", "TOP"),
        [],
    )
    assert window._analysis_element_regions() == [
        RegionRef("element_set", "DOMAIN")
    ]
    window.close()


def test_load_dialog_validates_region_and_builds_pressure():
    _application()
    edge_regions = _regions("edge", "Loaded")
    missing_region = LoadDialog(["Load"], [], edge_regions, [], 2)
    missing_region.region_combo.clear()
    with pytest.raises(ValueError, match="载荷区域"):
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

    assert dialog.candidate_combo.count() == len(candidates)
    for index, candidate in enumerate(candidates):
        dialog.candidate_combo.setCurrentIndex(index)
        step_name, output = dialog.definition()
        assert step_name == "Load"
        assert output == candidate.authoring_request
        assert output is not candidate.authoring_request

    assert tuple(
        dialog.candidate_combo.itemData(index)
        for index in range(dialog.candidate_combo.count())
    ) == tuple(range(len(candidates)))


def test_output_request_preserves_parsed_inp_history_metadata():
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
    assert output == current
    assert output is not current
    assert not dialog.step_combo.isEnabled()
    assert dialog.kind_value.text() == "history"
    assert dialog.target_value.text() == "preselect"
    assert dialog.variables_value.text() == (
        "PRESELECT、PRESELECT、Future"
    )
    assert "variable=PRESELECT" in dialog.metadata_value.text()
    assert "frequency" in dialog.source_evidence_value.text()


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
    assert manager.table.item(2, 0).text() == "字段输出"
    assert manager.table.item(2, 2).text() == "节点"


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
