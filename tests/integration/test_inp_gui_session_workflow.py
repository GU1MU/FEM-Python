"""Public-command characterization of standard imported GUI workflows."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
    require_rejected,
)


FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)
B31_NOTICE = "abaqus.b31.euler_bernoulli_approximation"

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
    assert window.result_data is not None
    assert window.result_data.run_id == run.run_id
    assert window.result_data.field_ready("U")
    assert window.viewport.run_id == run.run_id
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
    assert window.result_data is None
    assert window.viewport.run_id is None
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


def test_imported_b31_public_edit_candidates_check_and_solve() -> None:
    _application()
    window = FEMMainWindow()
    _open_fixture(window, "beam2_rectangle_uniform_load.inp")

    assert len(window.import_notices) == 1
    assert window.import_notices[0].code == B31_NOTICE
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
    assert window.import_notices[0].code == B31_NOTICE

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()
