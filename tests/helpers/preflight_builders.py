from __future__ import annotations

from fem.application import (
    PreflightDiagnostic,
    PreflightFacts,
    PreflightReport,
    PreflightSeverity,
    PreflightStage,
    TaskToken,
)


def _validation_identity(token: TaskToken) -> tuple[str, int]:
    if token.step_name is None:
        raise ValueError("validation token must identify a step")
    if token.artifact_id is None:
        raise ValueError("validation token must identify an artifact")
    try:
        model_revision = dict(token.dependency_revisions)["model_revision"]
    except KeyError as exc:
        raise ValueError(
            "validation token must identify a model revision"
        ) from exc
    return str(token.step_name), int(model_revision)


def passing_preflight_report(token: TaskToken) -> PreflightReport:
    step_name, model_revision = _validation_identity(token)
    return PreflightReport(
        step_name=step_name,
        facts=PreflightFacts(step_name=step_name),
        session_id=token.session_id,
        artifact_id=token.artifact_id,
        model_revision=model_revision,
    )


def failing_preflight_report(
    token: TaskToken,
    code: str = "test.preflight.failed",
    message: str = "preflight validation failed",
) -> PreflightReport:
    step_name, model_revision = _validation_identity(token)
    diagnostic = PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.STRUCTURE,
        message=message,
        subject=step_name,
        path=("steps", step_name),
    )
    return PreflightReport(
        step_name=step_name,
        diagnostics=(diagnostic,),
        facts=PreflightFacts(step_name=step_name),
        session_id=token.session_id,
        artifact_id=token.artifact_id,
        model_revision=model_revision,
    )
