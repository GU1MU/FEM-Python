"""Application-owned assembly of one solved result and its output artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from .provider import ResultProvider, build_result_provider


@dataclass(frozen=True, slots=True)
class SolveResultBundle:
    """Detached worker result ready for one atomic Session acceptance."""

    source: ResultSourceKey
    result: ModelResult
    execution_report: ResultExecutionReport
    initial_materialization: ResultMaterializationSnapshot
    _provider: ResultProvider | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

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

    @classmethod
    def _from_provider(
        cls,
        *,
        source: ResultSourceKey,
        result: ModelResult,
        execution_report: ResultExecutionReport,
        provider: ResultProvider,
    ) -> SolveResultBundle:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be exactly ResultProvider")
        if provider.source != source:
            raise ValueError("provider source must match bundle source")
        bundle = cls(
            source=source,
            result=result,
            execution_report=execution_report,
            initial_materialization=provider.snapshot,
        )
        object.__setattr__(bundle, "_provider", provider)
        return bundle


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
    published_keys = {
        key
        for request in outcome.report.requests
        if request.status is OutputExecutionStatus.EXECUTED
        for variable in request.variables
        for key in variable.field_keys
    }
    published_provider = outcome.provider_draft.publish_fields(
        published_keys
    )
    return SolveResultBundle._from_provider(
        source=source,
        result=result,
        execution_report=outcome.report,
        provider=published_provider,
    )


def validate_solve_result_model_identity(
    result: ModelResult,
    expected_model: object,
    step_name: str,
) -> None:
    """Require a solved model to match the accepted artifact and step inputs."""

    if type(result) is not ModelResult:
        raise TypeError("result must be exactly ModelResult")
    if type(step_name) is not str or not step_name.strip():
        raise ValueError("step_name must be a nonblank string")
    result_step = _unique_model_step(result.model, step_name)
    expected_step = _unique_model_step(expected_model, step_name)
    if result.step is not result_step:
        raise ValueError("solved result step must belong to its result model")
    if _mesh_identity(result.model) != _mesh_identity(expected_model):
        raise ValueError(
            "solved result model topology must match the accepted artifact"
        )
    try:
        step_matches = result_step == expected_step
    except (TypeError, ValueError) as error:
        raise ValueError(
            "solved result step cannot be matched to the accepted artifact"
        ) from error
    if type(step_matches) is not bool or not step_matches:
        raise ValueError(
            "solved result step must match the accepted artifact step"
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


def _unique_model_step(model: object, step_name: str) -> object:
    try:
        steps = tuple(model.steps)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError("result model must expose iterable steps") from error
    matching = tuple(
        step
        for step in steps
        if getattr(step, "name", None) == step_name
    )
    if len(matching) != 1:
        raise ValueError(
            "result model must contain exactly one matching step"
        )
    return matching[0]


def _mesh_identity(model: object) -> tuple[object, ...]:
    try:
        mesh = model.mesh  # type: ignore[attr-defined]
        nodes = tuple(mesh.nodes)
        elements = tuple(mesh.elements)
        node_ids = tuple(int(value) for value in mesh.node_ids)
        dofs_per_node = int(mesh.dofs_per_node)
        num_dofs = int(mesh.num_dofs)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "result model mesh must expose canonical topology and DOFs"
        ) from error

    node_identity = tuple(
        (
            int(node.id),
            float(node.x),
            float(node.y),
            float(getattr(node, "z", 0.0)),
        )
        for node in nodes
    )
    element_identity = tuple(
        (
            int(element.id),
            str(element.type).strip().casefold(),
            tuple(int(node_id) for node_id in element.node_ids),
        )
        for element in elements
    )
    return (
        node_ids,
        node_identity,
        element_identity,
        dofs_per_node,
        num_dofs,
    )


__all__ = [
    "SolveResultBundle",
    "build_solve_result_bundle",
    "validate_solve_result_model_identity",
]
