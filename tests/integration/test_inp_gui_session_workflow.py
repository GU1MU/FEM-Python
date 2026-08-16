"""Public-command characterization of standard imported GUI workflows."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import (
    AuthoringStatus,
    BeamOrientation,
    DefinitionEditBatch,
    ModelDefinitions,
    RegionRef,
    describe_session_authoring,
    evaluate_authoring_candidate,
)
from fem.application.results import OutputExecutionStatus, ResultVariable
from fem.core.model import LineLoad
from fem_gui.commands import CloseSessionCommand
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import BackgroundTaskState
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
    require_rejected,
)
from tests.helpers.file_builders import write_inp


FIXTURES = (
    Path(__file__).parents[1]
    / "helpers" / "fixtures"
    / "inp"
    / "abaqus_standard"
)
MIXED_PLATE = (
    Path(__file__).parents[2]
    / "data"
    / "MixedPlateCps3Cps4_PerforatedJob.inp"
)
B31_NOTICE = "abaqus.b31.linear_timoshenko_support_boundary"

PUBLIC_GUI_WORKFLOW_ENTRYPOINTS = (
    "open_inp_path",
    "apply_definition_edit",
    "check_step",
    "submit_run",
    "save_project_path",
    "reload_imported_source",
    "close_session",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _open_fixture(window: FEMMainWindow, name: str) -> Path:
    path = FIXTURES / name
    await_succeeded(window.open_inp_path(path))
    assert window.document.source_kind == "imported"
    assert window.document.source_path == path
    assert window.document.artifact is not None
    assert window.geometry is not None
    assert (
        window.geometry.artifact_id
        == window.document.artifact.artifact_id
    )
    assert (
        window.viewport.artifact_id
        == window.document.artifact.artifact_id
    )
    assert window.inspection_service is not None
    assert not window.document.can_save
    assert window.document.can_reload
    assert not window.actions["save_project"].isEnabled()
    assert window.actions["reload"].isEnabled()
    return path


def _write_inline_beam(tmp_path: Path, name: str, *, malformed: bool) -> Path:
    lines = [
        "*Heading",
        "Inline public B31 GUI workflow",
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        "*Material, name=STEEL",
        "*Elastic",
        "210000., 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 0., 1.",
        "*Step, name=LOAD",
        "*Static",
        "*Output, field, variable=PRESELECT",
        "*Node Output",
        "U, RF",
        "*Boundary",
        "1, ENCASTRE",
        "*Cload",
        "2, 2, -1.",
        "*End Step",
    ]
    if malformed:
        lines.extend(
            (
                "*Normal, type=ELEMENT",
                "1, 1, 0., 0.",
            )
        )
    return write_inp(tmp_path, name, lines)


def _check_and_solve(
    window: FEMMainWindow,
    *,
    step_name: str,
    run_name: str,
) -> str:
    await_succeeded(window.check_step(step_name))
    assert window.document.validation_current(step_name)
    assert window.actions["submit_job"].isEnabled()

    await_succeeded(window.submit_run(run_name, step_name))
    run = window.session.find_run(run_name)
    assert run is not None and run.has_result
    result = window.session.current_result()
    assert result is not None
    assert result.provenance.run_id == run.run_id
    assert result.provenance.step_name == step_name
    assert result.provenance.artifact_id == (
        window.document.artifact.artifact_id
    )
    provider = window.result_provider
    selection = window.result_selection
    payload = window.viewport._result_render_payload
    assert provider is not None
    assert selection is not None
    assert payload is not None
    assert provider.source.run_id == run.run_id
    assert selection.field_key.request.field_id.variable is ResultVariable.U
    assert provider.field(selection.field_key).key == selection.field_key
    assert payload.topology.source == provider.source
    assert window.actions["query"].isEnabled()
    return run.run_id


def _definition_batch(
    window: FEMMainWindow,
    *,
    sections=None,
    assignments=None,
    steps=None,
) -> DefinitionEditBatch:
    snapshot = window.document
    return DefinitionEditBatch(
        base_session_revision=snapshot.session_revision,
        materials=snapshot.materials,
        sections=(
            snapshot.sections
            if sections is None
            else tuple(sections)
        ),
        assignments=(
            snapshot.assignments
            if assignments is None
            else tuple(assignments)
        ),
        steps=snapshot.steps if steps is None else tuple(steps),
    )


def test_imported_t3d2_public_edit_check_solve_and_reload(
    tmp_path,
) -> None:
    _application()
    window = FEMMainWindow()
    source_path = _open_fixture(window, "truss2_tension.inp")

    assert window.import_notices == ()
    projection = describe_session_authoring(window.document)
    assert projection.report.canonical_element_types == ("Truss2",)
    assert RegionRef("element_set", "TRUSS") in {
        target.region for target in projection.targets
    }

    original_section = window.document.sections[0]
    original_area = float(original_section.properties["area"])
    properties = dict(original_section.properties)
    properties["area"] = original_area * 1.25
    require_accepted(
        window.apply_definition_edit(
            _definition_batch(
                window,
                sections=(
                    replace(
                        original_section,
                        properties=properties,
                    ),
                ),
            )
        )
    )
    assert window.document.sections[0].properties["area"] == (
        original_area * 1.25
    )

    require_rejected(
        window.save_project_path(tmp_path / "imported.femproj"),
        code="project.save.unavailable",
    )
    first_run_id = _check_and_solve(
        window,
        step_name="Tension",
        run_name="Truss-Edited",
    )

    await_succeeded(window.reload_imported_source())
    assert window.document.source_path == source_path
    assert window.document.sections[0].properties["area"] == original_area
    assert window.document.runs == ()
    assert window.session.find_run(first_run_id) is None
    assert window.session.current_result() is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None
    assert window.import_notices == ()

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_imported_output_overlay_executes_then_reload_restores_source() -> None:
    _application()
    window = FEMMainWindow()
    source_path = _open_fixture(window, "truss2_tension.inp")
    source_step = next(
        step for step in window.document.steps if step.name == "Tension"
    )
    source_outputs = tuple(source_step.outputs)
    assert tuple(output.variables for output in source_outputs) == (
        ("U", "RF"),
        ("S",),
    )

    snapshot = window.document
    authoring = describe_session_authoring(snapshot)
    catalog = authoring.output_request_catalog
    assert catalog is not None
    created = next(
        candidate.authoring_request
        for candidate in catalog.candidates
        if candidate.authoring_request.target == "node"
        and candidate.authoring_request.variables == ("U",)
    )
    overlay_outputs = (source_outputs[1], created)
    edited_steps = tuple(
        replace(step, outputs=overlay_outputs)
        if step.name == "Tension"
        else step
        for step in snapshot.steps
    )
    require_accepted(
        window.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision=snapshot.session_revision,
                materials=snapshot.materials,
                sections=snapshot.sections,
                assignments=snapshot.assignments,
                steps=edited_steps,
            )
        )
    )
    assert next(
        step for step in window.document.steps if step.name == "Tension"
    ).outputs == overlay_outputs

    run_id = _check_and_solve(
        window,
        step_name="Tension",
        run_name="Truss-Output-Overlay",
    )
    record = window.session.current_result()
    assert record is not None
    assert tuple(
        execution.status for execution in record.output_report.requests
    ) == (
        OutputExecutionStatus.EXECUTED,
        OutputExecutionStatus.EXECUTED,
    )
    assert tuple(
        tuple(
            variable.canonical_variable
            for variable in execution.variables
        )
        for execution in record.output_report.requests
    ) == (
        (ResultVariable.S,),
        (ResultVariable.U,),
    )
    field_keys = tuple(
        key
        for execution in record.output_report.requests
        for variable in execution.variables
        for key in variable.field_keys
    )
    assert field_keys
    for execution in record.output_report.requests:
        assert execution.executable_request is not None
        assert tuple(
            key.request
            for variable in execution.variables
            for key in variable.field_keys
        ) == execution.executable_request.field_requests
    materialized_keys = {
        field.key for field in record.materialization.fields
    }
    assert set(field_keys).issubset(materialized_keys)

    await_succeeded(window.reload_imported_source())
    restored_step = next(
        step for step in window.document.steps if step.name == "Tension"
    )
    assert window.document.source_path == source_path
    assert restored_step.outputs == source_outputs
    assert window.document.runs == ()
    assert window.session.find_run(run_id) is None
    assert window.session.current_result() is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_mixed_plate_import_and_gui_solve_publish_u_rf_and_s() -> None:
    _application()
    window = FEMMainWindow()

    await_succeeded(window.open_inp_path(MIXED_PLATE))
    step = next(
        item for item in window.document.steps if item.name == "LOAD"
    )
    assert tuple(
        (request.target, request.variables)
        for request in step.outputs
    ) == (
        ("node", ("RF", "U")),
        ("element", ("S",)),
    )

    _check_and_solve(
        window,
        step_name="LOAD",
        run_name="Mixed-Plate-Output",
    )

    provider = window.result_provider
    assert provider is not None
    assert {
        availability.descriptor.field_id.variable
        for availability in provider.catalog().fields
    } == {
        ResultVariable.U,
        ResultVariable.RF,
        ResultVariable.S,
    }
    result_root = window.result_tree.topLevelItem(0)
    result_step = result_root.child(0)
    assert {
        result_step.child(index).text(0)
        for index in range(result_step.childCount())
    } == {
        "位移 U",
        "反力 RF",
        "应力 S",
    }
    window.close()


def test_imported_b31_public_edit_candidates_check_and_solve() -> None:
    _application()
    window = FEMMainWindow()
    _open_fixture(window, "beam2_rectangle_uniform_load.inp")

    assert tuple(notice.code for notice in window.import_notices) == (
        B31_NOTICE,
    )
    projection = describe_session_authoring(window.document)
    assert projection.report.canonical_element_types == ("Beam2",)
    assert RegionRef("element_set", "BEAM") in {
        target.region for target in projection.targets
    }

    original_section = window.document.sections[0]
    properties = dict(original_section.properties)
    properties.update({"height": 0.12, "width": 0.025})
    section = replace(original_section, properties=properties)
    assignment = replace(
        window.document.assignments[0],
        beam_orientation=BeamOrientation((0.0, 1.0, 0.0)),
    )
    local_load = LineLoad(
        "BEAM",
        (0.0, -375.0, 0.0),
        "local",
    )
    global_load = replace(local_load, coordinate_system="global")
    steps = tuple(
        replace(step, line_loads=(local_load,))
        if step.name == "UniformLoad"
        else step
        for step in window.document.steps
    )
    definitions = ModelDefinitions(
        materials=window.document.materials,
        sections=(section,),
        assignments=(assignment,),
        steps=steps,
    )
    for operation, candidate in (
        ("load.line.local", local_load),
        ("load.line.global", global_load),
    ):
        decision = evaluate_authoring_candidate(
            window.document.model,
            definitions,
            operation=operation,
            candidate=candidate,
            step_name="UniformLoad",
        )
        assert decision.status is AuthoringStatus.ENABLED
        assert decision.can_submit

    require_accepted(
        window.apply_definition_edit(
            _definition_batch(
                window,
                sections=(section,),
                assignments=(assignment,),
                steps=steps,
            )
        )
    )
    assert window.document.assignments[0].beam_orientation == (
        BeamOrientation((0.0, 1.0, 0.0))
    )
    installed_step = next(
        step
        for step in window.document.steps
        if step.name == "UniformLoad"
    )
    assert installed_step.line_loads == (local_load,)

    _check_and_solve(
        window,
        step_name="UniformLoad",
        run_name="Beam-Local",
    )
    assert tuple(notice.code for notice in window.import_notices) == (
        B31_NOTICE,
    )

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_inline_inp_open_check_solve_projection_and_failed_reload_are_atomic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    valid_path = _write_inline_beam(
        tmp_path,
        "inline_valid.inp",
        malformed=False,
    )
    malformed_path = _write_inline_beam(
        tmp_path,
        "inline_malformed.inp",
        malformed=True,
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    try:
        receipt = window.open_inp_path(valid_path)
        await_succeeded(receipt)
        assert window.document.source_kind == "imported"
        assert window.document.source_path == valid_path
        assert window.import_notices

        run_id = _check_and_solve(
            window,
            step_name="LOAD",
            run_name="Inline-B31",
        )
        old_snapshot = window.document
        old_artifact_id = old_snapshot.artifact.artifact_id
        old_model = old_snapshot.model
        old_result = window.session.current_result()
        old_payload = window.viewport._result_render_payload
        old_provider = window.result_provider
        old_selection = window.result_selection
        old_notices = window.import_notices
        old_viewport_identity = (
            window.viewport.artifact_id,
            window.viewport.run_id,
        )
        assert old_result is not None
        assert old_payload is not None
        assert old_provider is not None
        assert old_selection is not None

        failed = window.open_inp_path(malformed_path)
        assert failed.completion is not None
        deadline = monotonic() + 30.0
        application = QApplication.instance() or QApplication([])
        while not failed.completion.done and monotonic() < deadline:
            application.processEvents()
            QThread.msleep(1)
        application.processEvents()
        terminal = failed.completion.result(0.0)
        assert terminal.state is BackgroundTaskState.FAILED
        assert not window.busy
        assert errors

        assert window.document.artifact.artifact_id == old_artifact_id
        assert window.document.model is old_model
        current_result = window.session.current_result()
        assert current_result is not None
        assert current_result.result_id == old_result.result_id
        assert current_result.provenance == old_result.provenance
        assert window.session.find_run(run_id) is not None
        assert window.viewport._result_render_payload is old_payload
        assert window.result_provider is old_provider
        assert window.result_selection is old_selection
        assert (
            window.viewport.artifact_id,
            window.viewport.run_id,
        ) == old_viewport_identity
        assert window.import_notices == old_notices
    finally:
        window.close()
