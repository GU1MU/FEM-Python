import fem_agent.worker as worker_module
from fem_agent.artifacts import ArtifactStore, atomic_write_json, read_json_file
from fem_agent.confirmation import ConfirmationStore
from fem_agent.diagnostics import make_diagnostic
from fem_agent.schemas import (
    ExportFormat,
    ImportAnalysisSpec,
    ResourceLimits,
    ResultQuery,
    ResultQueryKind,
    RunStatus,
    UnitContext,
)
from fem_agent.state import RevisionStore

from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def _units():
    return UnitContext(
        length="mm",
        force="N",
        stress="MPa",
        density="tonne/mm^3",
        acceleration="mm/s^2",
    )


def _prepared_revision(
    tmp_path,
    *,
    confirmed=False,
    unit_context=None,
    resource_limits=None,
    requested_queries=None,
    export_formats=None,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "worker_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    artifacts = ArtifactStore(workspace)
    session_id = artifacts.create_session("ses_worker")
    artifact = artifacts.copy_input(session_id, source)
    revisions = RevisionStore(workspace)
    revisions.create_session(session_id)
    record = revisions.initialize(
        ImportAnalysisSpec(
            session_id=session_id,
            revision=1,
            source_artifact_id=artifact.artifact_id,
            source_sha256=artifact.sha256,
            unit_context=unit_context or _units(),
            analysis_step="Step-1",
            requested_queries=(
                requested_queries
                if requested_queries is not None
                else (
                    ResultQuery(
                        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE
                    ),
                    ResultQuery(
                        ResultQueryKind.REACTION_SUM,
                        component=1,
                        node_set="Set-left",
                    ),
                )
            ),
            export_formats=(
                export_formats
                if export_formats is not None
                else (ExportFormat.CSV,)
            ),
            resource_limits=resource_limits or ResourceLimits(),
        ),
        idempotency_key="initialize_worker",
    )
    if confirmed:
        ConfirmationStore(workspace, revisions).confirm(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
        )
    return workspace, artifacts, revisions, record


def _mark_persisted_worker_inactive(run):
    state_path = run.path / "logs" / "worker-process.json"
    state = read_json_file(state_path)
    state["supervisor_pid"] = 2_147_483_647
    atomic_write_json(state_path, state, overwrite=True)


def _persist_failure_response(
    artifacts,
    record,
    run,
    diagnostic_code,
    *,
    elapsed_seconds=0.01,
):
    response_path = run.path / "logs" / "worker-response.json"
    diagnostic = make_diagnostic(
        diagnostic_code,
        "The persisted worker stage failed.",
        source="test.worker",
    )
    return worker_module._write_terminal_failure(
        artifacts,
        record,
        run,
        response_path,
        RunStatus.FAILED,
        diagnostic,
        elapsed_seconds=elapsed_seconds,
    )
