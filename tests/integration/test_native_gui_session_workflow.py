"""Public-command characterization of the native GUI workflow."""

from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import (
    DefinitionEditBatch,
    MeshEntityRef,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    SectionDefinition,
    describe_session_authoring,
)
from fem.application.results import OutputExecutionStatus, ResultVariable
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import SketchGeometry, SketchRectangle
from fem.mesh.settings import MeshSettings
from fem_gui.commands import (
    CloseSessionCommand,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
)
from fem_gui.main_window import FEMMainWindow
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
)


PUBLIC_GUI_WORKFLOW_ENTRYPOINTS = (
    "new_native_project",
    "apply_native_geometry_edit",
    "apply_named_region_edit",
    "apply_mesh_input_edit",
    "apply_definition_edit",
    "generate_mesh",
    "check_step",
    "submit_run",
    "save_project_path",
    "close_session",
    "open_project_path",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _install_native_authoring(window: FEMMainWindow) -> None:
    recipe = SketchGeometry(
        "Plate",
        (
            SketchRectangle(
                "material",
                0.0,
                0.0,
                2.0,
                1.0,
            ),
        ),
    )
    require_accepted(
        window.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=window.document.session_revision,
                parts=(
                    NativePart(
                        id="P1",
                        name="Plate",
                        geometry_recipe=recipe,
                    ),
                ),
                recipe=recipe,
            )
        )
    )
    require_accepted(
        window.apply_mesh_input_edit(
            MeshInputEdit(
                base_session_revision=window.document.session_revision,
                settings=MeshSettings(0.5),
            )
        )
    )
    await_succeeded(window.generate_mesh())
    generated = window.document.model
    assert generated is not None
    require_accepted(
        window.apply_named_region_edit(
            NamedRegionEditBatch(
                base_session_revision=window.document.session_revision,
                regions=(
                    NamedRegion(
                        "Fixed",
                        tuple(
                            MeshEntityRef.node(node.id)
                            for node in generated.mesh.nodes
                            if abs(float(node.x)) <= 1.0e-9
                        ),
                    ),
                    NamedRegion(
                        "Loaded",
                        tuple(
                            MeshEntityRef.node(node.id)
                            for node in generated.mesh.nodes
                            if abs(float(node.x) - 2.0) <= 1.0e-9
                        ),
                    ),
                    NamedRegion(
                        "DOMAIN",
                        tuple(
                            MeshEntityRef.element(element.id)
                            for element in generated.mesh.elements
                        ),
                    ),
                ),
            )
        )
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
        cloads=(NodalLoad("Loaded", 1, 10.0),),
    )
    require_accepted(
        window.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision=window.document.session_revision,
                materials=(
                    MaterialDefinition(
                        "Steel",
                        {"E": 210000.0, "nu": 0.3},
                    ),
                ),
                sections=(
                    SectionDefinition(
                        "Section-1",
                        "Steel",
                        properties={
                            "plane_type": "stress",
                            "thickness": 1.0,
                        },
                    ),
                ),
                assignments=(
                    RegionAssignment("Section-1", "DOMAIN"),
                ),
                steps=(step,),
            )
        )
    )


def _mesh_check_and_solve(
    window: FEMMainWindow,
    *,
    run_name: str,
) -> str:
    if window.document.model is None:
        await_succeeded(window.generate_mesh())
    artifact = window.document.artifact
    assert artifact is not None and artifact.source_kind == "native"
    assert window.geometry is not None
    assert window.geometry.artifact_id == artifact.artifact_id
    assert window.viewport.artifact_id == artifact.artifact_id
    assert window.actions["check_model"].isEnabled()

    await_succeeded(window.check_step("Load"))
    assert window.document.validation_current("Load")
    assert window.actions["submit_job"].isEnabled()

    await_succeeded(window.submit_run(run_name, "Load"))
    run = window.session.find_run(run_name)
    assert run is not None and run.has_result
    result = window.session.current_result()
    assert result is not None
    assert result.provenance.run_id == run.run_id
    assert result.provenance.artifact_id == artifact.artifact_id
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


def test_native_public_workflow_saves_reopens_remeshes_and_resolves(
    tmp_path,
) -> None:
    _application()
    window = FEMMainWindow()

    require_accepted(
        window.new_native_project(NewNativeProjectCommand("Plate Project"))
    )
    assert window.document.source_kind == "native"
    assert window.actions["geometry_sketch"].isEnabled()

    _install_native_authoring(window)
    assert window.document.dirty
    assert window.actions["mesh_generate"].isEnabled()
    first_run_id = _mesh_check_and_solve(window, run_name="Job-1")

    project_path = tmp_path / "plate-public.femproj"
    await_succeeded(window.save_project_path(project_path))
    assert project_path.is_file()
    assert window.document.project_path == project_path
    assert not window.document.dirty

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    assert window.document.source_kind is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None

    await_succeeded(window.open_project_path(project_path))
    assert window.document.source_kind == "native"
    assert window.document.project_path == project_path
    assert window.document.geometry_recipe is not None
    assert window.document.mesh_settings == MeshSettings(0.5)
    assert window.document.runs == ()
    assert window.session.current_result() is None

    reopened_run_id = _mesh_check_and_solve(
        window,
        run_name="Job-Reopened",
    )
    assert reopened_run_id != first_run_id

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_native_output_request_survives_save_reopen_and_executes(
    tmp_path,
) -> None:
    _application()
    window = FEMMainWindow()
    require_accepted(
        window.new_native_project(NewNativeProjectCommand("Output Project"))
    )
    _install_native_authoring(window)

    snapshot = window.document
    authoring = describe_session_authoring(snapshot)
    catalog = authoring.output_request_catalog
    assert catalog is not None
    request = next(
        candidate.authoring_request
        for candidate in catalog.candidates
        if candidate.authoring_request.variables == ("S",)
    )
    edited_steps = tuple(
        replace(step, outputs=(*step.outputs, request))
        if step.name == "Load"
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
    assert window.document.steps[0].outputs == (request,)

    project_path = tmp_path / "output-request-public.femproj"
    await_succeeded(window.save_project_path(project_path))
    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    await_succeeded(window.open_project_path(project_path))
    assert window.document.steps[0].outputs == (request,)

    _mesh_check_and_solve(window, run_name="Output-Reopened")
    record = window.session.current_result()
    assert record is not None
    assert len(record.output_report.requests) == 1
    execution = record.output_report.requests[0]
    assert execution.status is OutputExecutionStatus.EXECUTED
    assert tuple(
        variable.canonical_variable for variable in execution.variables
    ) == (ResultVariable.S,)
    assert execution.executable_request is not None
    field_keys = tuple(
        key
        for variable in execution.variables
        for key in variable.field_keys
    )
    assert field_keys
    assert tuple(key.request for key in field_keys) == (
        execution.executable_request.field_requests
    )
    materialized_keys = {
        field.key for field in record.materialization.fields
    }
    assert set(field_keys).issubset(materialized_keys)

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()
