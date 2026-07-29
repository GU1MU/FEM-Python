from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fem.application import (
    ModelSession,
    PreflightDiagnostic,
    PreflightFacts,
    PreflightReport,
    PreflightSeverity,
    PreflightStage,
    UnitContext,
    ValidationRecord,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    NodalLoad,
    OutputRequest,
)
from fem.geometry import PlateWithHoleGeometry
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import (
    AgentProposal,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalKind,
)
from fem_agent.solve_authoring import (
    SolveAuthoringError,
    SolveValidationStamp,
    build_solve_summary,
    create_solve_proposal,
    solve_operation_identity,
)
from fem_agent.tools.registry import AgentToolRegistry
from tests.helpers.model_builders import make_static_pull_truss_model


STEP_NAME = "分析步-静力"


def _recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def _model():
    model = make_static_pull_truss_model()
    model.name = "模型-偏心孔板"
    model.steps = [
        AnalysisStep(
            STEP_NAME,
            boundaries=(
                DisplacementConstraint(
                    "FIXED",
                    1,
                    3,
                    0.0,
                    name="位移-固定端",
                ),
                DisplacementConstraint(
                    2,
                    2,
                    3,
                    0.0,
                    name="位移-横向稳定",
                ),
            ),
            cloads=(
                NodalLoad("TIP", 1, 100.0, name="载荷-拉伸"),
            ),
            outputs=(
                OutputRequest(
                    "field",
                    "node",
                    ("U", "RF"),
                    name="结果请求-位移反力",
                ),
            ),
            metadata={"nlgeom": False},
        )
    ]
    return model


def _native_session(*, blocked: bool = False) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(1.0, cell_shape="triangle"),
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model())
    validation = session.prepare_validation(STEP_NAME)
    diagnostics = (
        (
            PreflightDiagnostic(
                code="static.boundary.missing",
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.BOUNDARY,
                message="missing displacement",
                path=("steps", STEP_NAME, "boundary"),
            ),
        )
        if blocked
        else (
            PreflightDiagnostic(
                code="test.warning",
                severity=PreflightSeverity.WARNING,
                stage=PreflightStage.OUTPUT,
                message="bounded warning",
                path=("steps", STEP_NAME, "outputs"),
                details={"local_path": r"D:\private\model.inp"},
            ),
        )
    )
    report = PreflightReport(
        step_name=STEP_NAME,
        diagnostics=diagnostics,
        facts=PreflightFacts(
            model_name="模型-偏心孔板",
            step_name=STEP_NAME,
            procedure="static",
            node_count=2,
            element_count=1,
            dof_count=6,
            displacement_count=2,
            nodal_load_count=1,
        ),
        numerical_stability_checked=True,
        session_id=validation.token.session_id,
        artifact_id=validation.token.artifact_id,
        model_revision=dict(validation.token.dependency_revisions)[
            "model_revision"
        ],
    )
    session.accept_validation(validation.token, report)
    return session


def _proposal(session: ModelSession, proposal_id: str = "proposal-a6"):
    return create_solve_proposal(
        proposal_id=proposal_id,
        agent_session_id="agent-session-a6",
        turn_id=f"turn-{proposal_id}",
        source_tool_call_ids=(f"call-{proposal_id}",),
        snapshot=session.snapshot(),
        draft_revision=6,
        step_name=STEP_NAME,
        job_name="作业-静力1",
    )


def test_a6_validation_stamp_is_deterministic_and_tracks_report_content() -> None:
    session = _native_session()
    record = session.validation_for(STEP_NAME)
    assert record is not None

    first = SolveValidationStamp.from_record(record)
    repeated = SolveValidationStamp.from_record(
        ValidationRecord(record.stamp, record.report)
    )
    changed_report = replace(
        record.report,
        diagnostics=(
            replace(record.report.diagnostics[0], message="changed warning"),
        ),
    )
    changed = SolveValidationStamp.from_record(
        ValidationRecord(record.stamp, changed_report)
    )

    assert first == repeated
    assert first.report_hash != changed.report_hash
    assert first.stamp_hash != changed.stamp_hash


def test_a6_blocking_diagnostic_cannot_create_executable_proposal() -> None:
    session = _native_session(blocked=True)

    with pytest.raises(SolveAuthoringError, match="blocking"):
        _proposal(session)

    assert session.snapshot().runs == ()


def test_a6_summary_and_proposal_are_bounded_and_provider_safe() -> None:
    session = _native_session()
    snapshot = session.snapshot()
    summary = build_solve_summary(
        snapshot,
        STEP_NAME,
        "作业-静力1",
    )
    proposal = _proposal(session)
    operation = proposal.operations[0]
    step, job, artifact, revision, stamp = solve_operation_identity(operation)
    encoded = json.dumps(
        proposal.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
    )

    assert proposal.proposal_kind is ProposalKind.SOLVE
    assert proposal.preconditions["authoring_phase"] == "A6"
    assert operation.kind is OperationKind.REQUEST_SOLVE
    assert step == STEP_NAME
    assert job == "作业-静力1"
    assert artifact == snapshot.artifact.artifact_id
    assert revision == snapshot.model_revision
    assert stamp.stamp_hash in encoded
    assert summary.node_count == 2
    assert summary.element_count == 1
    assert summary.constraint_names == ("位移-固定端", "位移-横向稳定")
    assert summary.load_names == ("载荷-拉伸",)
    assert summary.output_names == ("结果请求-位移反力",)
    assert r"D:\private" not in encoded
    assert "node_ids" not in encoded
    assert "elements" not in encoded
    assert AgentProposal.from_dict(proposal.to_dict()) == proposal


def test_a6_preserves_legacy_solve_operation_but_rejects_mixed_shape() -> None:
    legacy = ModelOperation(
        OperationKind.REQUEST_SOLVE,
        {"step_name": "Static-1", "validation_stamp": "legacy-stamp"},
    )

    assert ModelOperation.from_dict(legacy.to_dict()) == legacy
    with pytest.raises(AuthoringContractError, match="legacy or exact A6"):
        ModelOperation(
            OperationKind.REQUEST_SOLVE,
            {
                "step_name": "Static-1",
                "job_name": "作业-1",
                "validation_stamp": "legacy-stamp",
            },
        )
    with pytest.raises(SolveAuthoringError, match="exact schema"):
        solve_operation_identity(legacy)


def test_a6_provider_tool_catalog_exposes_no_confirmation_authority(
    tmp_path,
) -> None:
    names = {
        item.name
        for item in AgentToolRegistry(tmp_path / "workspace").definitions
    }

    assert "validate_analysis" in names
    assert "confirm_solve" not in names
    assert "accept_proposal" not in names
    assert "confirm_mesh" not in names
