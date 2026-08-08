from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMenu, QToolButton

from fem.application import (
    ModelSession,
    NativePart,
    TokenStatus,
    describe_session_authoring,
)
from fem.application.results import (
    ElementResultInspectionRequest,
    FieldAssociation,
    FieldPosition,
    FieldRequest,
    FieldState,
    NodeResultInspectionRequest,
    ResultFieldId,
    ResultModelFamily,
    ResultQuery,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    prepare_result_export_snapshot,
    restore_result_provider,
)
from fem.core.model import AnalysisStep, FEMModel
from fem.geometry.recipes import BoxGeometry
from fem.io import project as project_module
from fem.io.project import (
    CURRENT_PROJECT_SCHEMA,
    decode_project,
    dumps_project,
    encode_project,
)
from fem.io.result_csv import read_result_csv, write_result_csv
from fem.io.result_vtk import read_result_vtk, write_result_vtk
from fem_gui.action_state import (
    ACTION_DESCRIPTORS,
    GuiActionContext,
    GuiActionKey,
    derive_action_availability,
)
from fem_gui.main_window import FEMMainWindow
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import make_solve_result_bundle


_RESULT_CASES = (
    (
        "continuum",
        make_continuum_nodal_semantics_result,
        ResultModelFamily.PLANE_CONTINUUM,
        ("Tri3",),
        2,
        (ResultVariable.U, ResultVariable.RF),
        (
            (ResultVariable.S, FieldPosition.INTEGRATION_POINT),
            (ResultVariable.S, FieldPosition.CENTROID),
            (ResultVariable.S, FieldPosition.ELEMENT_NODAL),
            (ResultVariable.S, FieldPosition.NODE_REGION),
            (ResultVariable.S, FieldPosition.RESOLVED_NODAL),
        ),
    ),
    (
        "truss2",
        make_truss_field_characterization_result,
        ResultModelFamily.TRUSS,
        ("Truss2",),
        3,
        (ResultVariable.U, ResultVariable.RF),
        (
            (ResultVariable.LE, FieldPosition.CENTROID),
            (ResultVariable.S, FieldPosition.CENTROID),
        ),
    ),
    (
        "beam2",
        make_beam_field_characterization_result,
        ResultModelFamily.BEAM,
        ("Beam2",),
        6,
        (ResultVariable.U, ResultVariable.UR, ResultVariable.RF,
         ResultVariable.RM),
        (
            (ResultVariable.S, FieldPosition.SECTION_END),
            (ResultVariable.S, FieldPosition.SECTION_NODE_ENVELOPE),
        ),
    ),
)


def _source(name: str) -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"phase0-result-{name}",
        session_id="phase0-session",
        artifact_id="phase0-artifact",
        model_revision=7,
        step_name="Phase-0",
        run_id=f"phase0-run-{name}",
    )


def _location_identity(location):
    return (
        location.association,
        location.coordinates,
        location.node_id,
        location.element_id,
        location.integration_point,
        location.local_node,
        location.region_key,
        location.averaged,
    )


@pytest.mark.parametrize(
    "name,builder,family,canonical_types,dofs,primary_variables,lazy_fields",
    _RESULT_CASES,
)
def test_phase0_provider_profile_catalog_and_topology_contract(
    name,
    builder,
    family,
    canonical_types,
    dofs,
    primary_variables,
    lazy_fields,
) -> None:
    result = builder()
    source = _source(name)
    provider = build_result_provider(source, result)

    assert provider.source == source
    assert provider.profile.family is family
    assert provider.profile.canonical_element_types == canonical_types
    assert provider.profile.dofs_per_node == dofs
    assert provider.profile.primary_compatible is True
    assert provider.profile.stress_compatible is True

    topology = provider.snapshot.topology
    mesh = result.model.mesh
    assert topology.source == source
    assert topology.node_ids == tuple(node.id for node in mesh.nodes)
    assert topology.element_ids == tuple(element.id for element in mesh.elements)
    assert topology.element_types == tuple(element.type for element in mesh.elements)
    assert topology.connectivity == tuple(
        tuple(element.node_ids) for element in mesh.elements
    )
    assert len(topology.element_region_keys) == len(mesh.elements)
    assert topology.node_coordinates.shape == (len(mesh.nodes), 3)
    assert topology.nodal_displacements.shape == (len(mesh.nodes), 3)
    assert np.isfinite(topology.node_coordinates).all()
    assert np.isfinite(topology.nodal_displacements).all()

    catalog = provider.catalog()
    assert catalog.source == source
    assert catalog.default_selection is not None
    assert catalog.default_selection.field_key.request.field_id == ResultFieldId(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    assert catalog.default_selection.component == "Magnitude"
    fields = provider.snapshot.fields
    assert tuple(item.key for item in fields) == tuple(
        item.key for item in catalog.fields if item.state is FieldState.READY
    )
    assert tuple(
        item.key.request.field_id.variable for item in catalog.fields
        if item.state is FieldState.READY
    ) == primary_variables
    assert tuple(
        (
            item.key.request.field_id.variable,
            item.key.request.field_id.position,
        )
        for item in catalog.fields
        if item.state is FieldState.LAZY
    ) == lazy_fields
    for field_data in fields:
        assert field_data.source == source
        assert field_data.values.shape == (
            len(field_data.locations),
            len(field_data.descriptor.columns),
        )
        assert np.isfinite(field_data.values).all()
        assert all(
            location.association is field_data.descriptor.association
            for location in field_data.locations
        )


@pytest.mark.parametrize(
    "name,builder",
    tuple((case[0], case[1]) for case in _RESULT_CASES),
)
def test_phase0_materialization_query_inspection_csv_and_vtk_parity(
    tmp_path: Path,
    name,
    builder,
) -> None:
    provider = build_result_provider(_source(name), builder())
    source = provider.source
    initial = provider.snapshot
    selection = provider.catalog().default_selection
    assert selection is not None
    field = provider.field(selection.field_key)

    query = provider.query(
        ResultQuery(selection.field_key, selection.component)
    )
    component_index = field.descriptor.columns.index(selection.component)
    assert query.source == source
    assert query.materialization_generation == 0
    assert tuple(record.location for record in query.records) == field.locations
    np.testing.assert_allclose(
        tuple(record.value for record in query.records),
        field.values[:, component_index],
    )

    node_request = NodeResultInspectionRequest(initial.topology.node_ids[0])
    node_inspection = provider.inspect_result(node_request)
    assert node_inspection.source == source
    assert node_inspection.materialization_generation == 0
    assert node_inspection.request == node_request
    for item in node_inspection.fields:
        if item.availability.state is not FieldState.READY:
            assert item.component_results == ()
            continue
        assert tuple(
            result.query.component for result in item.component_results
        ) == item.availability.descriptor.columns
        assert all(
            result.query.node_ids == (node_request.node_id,)
            for result in item.component_results
        )

    element_request = ElementResultInspectionRequest(
        initial.topology.element_ids[0]
    )
    element_inspection = provider.inspect_result(element_request)
    assert element_inspection.request == element_request
    assert all(
        result.query.element_ids == (element_request.element_id,)
        for item in element_inspection.fields
        for result in item.component_results
    )

    export = prepare_result_export_snapshot(initial, selection)
    csv_path = tmp_path / f"{name}.csv"
    vtk_path = tmp_path / f"{name}.vtk"
    write_result_csv(csv_path, export)
    write_result_vtk(vtk_path, export)
    csv_readback = read_result_csv(csv_path)
    vtk_readback = read_result_vtk(vtk_path)
    assert csv_readback.source == source
    assert csv_readback.materialization_generation == 0
    assert csv_readback.selection == selection
    assert csv_readback.association is FieldAssociation.NODE
    assert tuple(
        _location_identity(record.location)
        for record in csv_readback.records
    ) == tuple(_location_identity(location) for location in field.locations)
    np.testing.assert_allclose(
        tuple(record.value for record in csv_readback.records),
        field.values[:, component_index],
    )
    assert vtk_readback.source == source
    assert vtk_readback.materialization_generation == 0
    assert vtk_readback.selection == selection
    assert vtk_readback.association is FieldAssociation.NODE
    assert tuple(identity.node_id for identity in vtk_readback.point_locations) == (
        initial.topology.node_ids
    )
    np.testing.assert_allclose(
        vtk_readback.points,
        initial.topology.node_coordinates,
    )
    np.testing.assert_allclose(
        vtk_readback.values,
        field.values[:, component_index],
    )

    lazy_keys = tuple(
        item.key for item in provider.catalog().fields
        if item.state is FieldState.LAZY
    )
    patch = provider.materialize(lazy_keys)
    advanced = provider.advance(patch)
    assert patch.source == source
    assert advanced.source == source
    assert advanced.snapshot.generation == 1
    assert len(advanced.snapshot.fields) == len(provider.snapshot.fields) + len(
        patch.fields
    )
    assert all(
        advanced.field_status(key).state is FieldState.READY
        for key in lazy_keys
    )


def _session_with_success():
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0))
    session.replace_model_definitions(
        (),
        (),
        (),
        (AnalysisStep("Step-A"),),
    )
    mesh_task = session.prepare_mesh_generation()
    session.accept_generated_model(
        mesh_task.token,
        FEMModel(
            mesh=make_simple_truss_mesh(),
            steps=(AnalysisStep("Step-A"),),
        ),
    )
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    solve = session.prepare_solve("Step-A", "Phase-0-Job")
    session.begin_run(solve.token)
    session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=1.0),
    )
    return session, solve


def test_phase0_session_generation_stale_gate_and_close_lifecycle() -> None:
    session, solve = _session_with_success()
    initial = session.current_result()
    assert initial is not None
    assert session.snapshot().displayed_result_run_id == solve.run_id
    assert initial.materialization.generation == 0

    provider = restore_result_provider(
        initial.result,
        initial.materialization,
    )
    key = provider.resolve_request(
        FieldRequest(ResultFieldId(ResultVariable.S, FieldPosition.CENTROID))
    )
    task = session.prepare_result_materialization(solve.run_id, (key,))
    patch = provider.materialize(task.field_keys)
    accepted = session.accept_result_materialization(task.token, patch)
    assert accepted.accepted
    assert session.current_result().materialization.generation == 1
    assert session.validate_task_token(task.token) is TokenStatus.ALREADY_COMPLETED

    stale_task = session.prepare_result_projection(solve.run_id)
    before_close_id = session.session_id
    session.close()
    closed = session.snapshot()
    assert session.session_id != before_close_id
    assert closed.is_open is False
    assert closed.runs == ()
    assert closed.displayed_result is None
    assert session.validate_task_token(stale_task.token) is not TokenStatus.CURRENT


def _project_snapshot(source_path: Path | None = None):
    from fem.application.feature_history import derive_feature_history
    from fem.application.session import ProjectSnapshot
    from fem.geometry.recipes import RectangleGeometry
    from fem.mesh.settings import MeshSettings

    recipe = RectangleGeometry("Phase-0", 4.0, 2.0)
    return ProjectSnapshot(
        source_kind="native",
        source_path=source_path,
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(0.5),
        feature_history=derive_feature_history(recipe),
    )


def test_phase0_schema13_canonical_project_payload_excludes_results() -> None:
    snapshot = _project_snapshot()
    payload = encode_project(snapshot)
    assert CURRENT_PROJECT_SCHEMA == 13
    assert payload["format"] == "fem-python-project"
    assert payload["schema"] == 13
    assert set(payload) == {"format", "schema", "project"}
    assert "results" not in payload["project"]
    reopened = decode_project(payload).snapshot
    assert dumps_project(snapshot) == dumps_project(reopened)


@pytest.mark.parametrize("schema", tuple(range(1, 14)))
def test_phase0_project_router_dispatches_every_supported_schema(
    monkeypatch: pytest.MonkeyPatch,
    schema: int,
) -> None:
    source_path = Path(f"schema-{schema}.femproj")
    snapshot = _project_snapshot(source_path)
    if schema == 1:
        monkeypatch.setattr(
            project_module,
            "_decode_project_v1_loaded",
            lambda _payload, *, source_path: (snapshot, ()),
        )
    else:
        monkeypatch.setattr(
            project_module,
            f"decode_project_v{schema}",
            lambda _payload, *, source_path: snapshot,
        )
    monkeypatch.setattr(
        project_module,
        "migrate_project_snapshot_to_v5",
        lambda value: (value, ()),
    )
    monkeypatch.setattr(
        project_module,
        "migrate_project_snapshot_to_v7",
        lambda value: (value, ()),
    )
    monkeypatch.setattr(
        project_module,
        "with_compatibility_analysis_names",
        lambda value: value,
    )

    loaded = decode_project({"schema": schema}, source_path=source_path)
    assert loaded.source_schema == schema
    assert loaded.path == source_path
    assert loaded.snapshot.source_path == source_path


def test_phase0_reload_close_action_position_icon_and_state_projection():
    descriptors = {item.key: item for item in ACTION_DESCRIPTORS}
    assert tuple(item.key for item in ACTION_DESCRIPTORS[:6]) == (
        GuiActionKey.OPEN,
        GuiActionKey.NEW_NATIVE,
        GuiActionKey.OPEN_PROJECT,
        GuiActionKey.SAVE_PROJECT,
        GuiActionKey.RELOAD,
        GuiActionKey.CLOSE,
    )
    assert descriptors[GuiActionKey.RELOAD].text == "重新加载"
    assert descriptors[GuiActionKey.RELOAD].handler == "reload_model"
    assert descriptors[GuiActionKey.RELOAD].icon_name == "reload"
    assert descriptors[GuiActionKey.CLOSE].text == "关闭模型"
    assert descriptors[GuiActionKey.CLOSE].handler == "close_model"
    assert descriptors[GuiActionKey.CLOSE].icon_name == "close"

    snapshot = ModelSession().snapshot()
    states = {
        item.key: item
        for item in derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(),
        )
    }
    assert not states[GuiActionKey.RELOAD].enabled
    assert "已打开的 INP" in states[GuiActionKey.RELOAD].reason
    assert not states[GuiActionKey.CLOSE].enabled
    assert "没有打开" in states[GuiActionKey.CLOSE].reason

    application = QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    file_menu = window.findChild(QMenu, "menuFile")
    assert file_menu is not None
    assert [
        action.objectName() for action in file_menu.actions()[:6]
    ] == [
        "action_new_native",
        "action_open_project",
        "action_save_project",
        "action_open",
        "action_reload",
        "action_close",
    ]
    assert file_menu.actions()[6].isSeparator()
    assert file_menu.actions()[7] is window.actions["exit"]
    assert not window.actions["reload"].icon().isNull()
    assert not window.actions["close"].icon().isNull()

    project_page = window.ribbon.stack.widget(
        [
            window.ribbon.tab_bar.tabText(index)
            for index in range(window.ribbon.tab_bar.count())
        ].index("项目")
    )
    file_label = next(
        label
        for label in project_page.findChildren(QLabel)
        if label.objectName() == "ribbonGroupTitle" and label.text() == "文件"
    )
    file_group = file_label.parent()
    assert [
        button.defaultAction().objectName()
        for button in file_group.findChildren(QToolButton)
        if button.defaultAction() is not None
    ] == [
        "action_new_native",
        "action_open_project",
        "action_save_project",
        "action_reload",
        "action_close",
        "action_open",
    ]
    window.close()
    application.processEvents()
