"""Application-owned assembly of one solved result and its output artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from fem.application.revisions import SolveTaskSnapshot, TaskToken
from fem.core.model import OutputRequest
from fem.core.result import ModelResult

from ._materializers import check_cancellation
from .data import ResultMaterializationSnapshot
from .execution import (
    OutputExecutionStatus,
    ResultExecutionReport,
    execute_output_requests,
)
from .fields import ResultSourceKey
from .provider import build_result_provider


@dataclass(frozen=True, slots=True)
class SolveResultBundle:
    """Detached worker result ready for one atomic Session acceptance."""

    source: ResultSourceKey
    result: ModelResult
    execution_report: ResultExecutionReport
    initial_materialization: ResultMaterializationSnapshot

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.result) is not ModelResult:
            raise TypeError("result must be exactly ModelResult")
        if type(self.execution_report) is not ResultExecutionReport:
            raise TypeError(
                "execution_report must be ResultExecutionReport"
            )
        if (
            type(self.initial_materialization)
            is not ResultMaterializationSnapshot
        ):
            raise TypeError(
                "initial_materialization must be "
                "ResultMaterializationSnapshot"
            )
        if self.execution_report.source != self.source:
            raise ValueError("execution report source must match bundle source")
        if self.initial_materialization.source != self.source:
            raise ValueError(
                "initial materialization source must match bundle source"
            )
        if self.initial_materialization.generation != 0:
            raise ValueError(
                "initial materialization generation must be zero"
            )

        materialized_keys = {
            field_data.key
            for field_data in self.initial_materialization.fields
        }
        executed_keys = {
            key
            for request in self.execution_report.requests
            if request.status is OutputExecutionStatus.EXECUTED
            for variable in request.variables
            for key in variable.field_keys
        }
        if not executed_keys.issubset(materialized_keys):
            raise ValueError(
                "every executed output field must be present in the "
                "initial materialization"
            )


def build_solve_result_bundle(
    task: SolveTaskSnapshot,
    result: ModelResult,
    *,
    cancellation: object | None = None,
) -> SolveResultBundle:
    """Execute authored outputs and return a source-bound solve result bundle."""

    source, outputs = _validate_solve_result(task, result)
    check_cancellation(cancellation)
    provider = build_result_provider(source, result)
    outcome = execute_output_requests(
        provider,
        outputs,
        cancellation=cancellation,
    )
    check_cancellation(cancellation)
    return SolveResultBundle(
        source=source,
        result=result,
        execution_report=outcome.report,
        initial_materialization=outcome.provider_draft.snapshot,
    )


def _validate_solve_result(
    task: SolveTaskSnapshot,
    result: ModelResult,
) -> tuple[ResultSourceKey, tuple[OutputRequest, ...]]:
    if type(task) is not SolveTaskSnapshot:
        raise TypeError("task must be exactly SolveTaskSnapshot")
    if type(result) is not ModelResult:
        raise TypeError("result must be exactly ModelResult")

    token = task.token
    if type(token) is not TaskToken:
        raise TypeError("task token must be exactly TaskToken")
    if token.task_kind != "solve":
        raise ValueError("solve result bundle requires a solve task token")
    if token.artifact_id is None:
        raise ValueError("solve task token requires artifact_id")
    if token.step_name != task.step_name:
        raise ValueError("solve task token step must match task step")
    if token.run_id != task.run_id:
        raise ValueError("solve task token run must match task run")
    if token.result_id != task.result_id:
        raise ValueError("solve task token result must match task result")

    dependencies = dict(token.dependency_revisions)
    if set(dependencies) != {"model_revision"}:
        raise ValueError(
            "solve task token must depend exactly on model_revision"
        )
    model_revision = dependencies["model_revision"]

    if result.model is not task.model:
        raise ValueError("solved result model must be the task model")
    try:
        steps = tuple(task.model.steps)
    except (AttributeError, TypeError) as error:
        raise TypeError("solve task model must expose iterable steps") from error
    matching_steps = tuple(
        step
        for step in steps
        if getattr(step, "name", None) == task.step_name
    )
    if len(matching_steps) != 1:
        raise ValueError(
            "solve task model must contain exactly one matching step"
        )
    step = matching_steps[0]
    if result.step is not step:
        raise ValueError("solved result step must be the task model step")
    try:
        outputs = tuple(step.outputs)
    except (AttributeError, TypeError) as error:
        raise TypeError("solve task step must expose iterable outputs") from error
    if any(type(request) is not OutputRequest for request in outputs):
        raise TypeError(
            "solve task outputs must contain exact OutputRequest values"
        )

    return (
        ResultSourceKey(
            result_id=task.result_id,
            session_id=token.session_id,
            artifact_id=token.artifact_id,
            model_revision=model_revision,
            step_name=task.step_name,
            run_id=task.run_id,
        ),
        outputs,
    )


__all__ = ["SolveResultBundle", "build_solve_result_bundle"]
