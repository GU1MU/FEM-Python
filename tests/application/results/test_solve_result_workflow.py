from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

import fem.application.results.workflow as workflow_module
from fem.application.results import (
    OutputExecutionStatus,
    ResultProvider,
    SolveResultBundle,
    build_solve_result_bundle,
    validate_solve_result_model_identity,
)
from fem.application.revisions import SolveTaskSnapshot, TaskToken
from fem.core.model import AnalysisStep, OutputRequest
from fem.core.result import ModelResult
from tests.helpers.phase8_result_characterization import (
    make_truss_field_characterization_result,
)


class _Cancelled(RuntimeError):
    pass


class _Cancellation:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def checkpoint(self) -> None:
        if self.cancelled:
            raise _Cancelled("cancelled")


def _task_and_result(
    outputs: tuple[OutputRequest, ...] = (),
) -> tuple[SolveTaskSnapshot, ModelResult]:
    result = make_truss_field_characterization_result()
    step = AnalysisStep("Step-1", outputs=outputs)
    result.model.steps = [step]
    result.step = step
    result.name = "Run-1"
    token = TaskToken(
        session_id="session-1",
        task_id="task-1",
        task_kind="solve",
        dependency_revisions=(("model_revision", 7),),
        artifact_id="artifact-1",
        step_name=step.name,
        run_id="run-1",
        result_id="result-1",
    )
    task = SolveTaskSnapshot(
        token=token,
        model=result.model,
        step_name=step.name,
        run_name="Run-1",
        run_id="run-1",
        result_id="result-1",
    )
    return task, result


def test_build_bundle_owns_source_and_initial_generation() -> None:
    task, result = _task_and_result(
        (OutputRequest("field", "element", ("S",)),)
    )

    bundle = build_solve_result_bundle(task, result)

    assert type(bundle) is SolveResultBundle
    assert bundle.result is result
    assert bundle.source.result_id == task.result_id
    assert bundle.source.session_id == task.token.session_id
    assert bundle.source.artifact_id == task.token.artifact_id
    assert bundle.source.model_revision == 7
    assert bundle.source.step_name == task.step_name
    assert bundle.source.run_id == task.run_id
    assert bundle.execution_report.source == bundle.source
    assert bundle.initial_materialization.source == bundle.source
    assert bundle.initial_materialization.generation == 0
    assert tuple(
        request.status for request in bundle.execution_report.requests
    ) == (OutputExecutionStatus.EXECUTED,)
    executed_keys = {
        key
        for request in bundle.execution_report.requests
        for variable in request.variables
        for key in variable.field_keys
    }
    assert executed_keys.issubset(
        {
            field_data.key
            for field_data in bundle.initial_materialization.fields
        }
    )


def test_empty_and_unsupported_outputs_still_build_a_success_bundle() -> None:
    empty_task, empty_result = _task_and_result()
    unsupported_task, unsupported_result = _task_and_result(
        (OutputRequest("history", "node", ("U",)),)
    )

    empty_bundle = build_solve_result_bundle(empty_task, empty_result)
    unsupported_bundle = build_solve_result_bundle(
        unsupported_task,
        unsupported_result,
    )

    assert empty_bundle.execution_report.requests == ()
    assert tuple(
        request.status
        for request in unsupported_bundle.execution_report.requests
    ) == (OutputExecutionStatus.UNSUPPORTED,)


def test_base_provider_failure_propagates_without_output_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, result = _task_and_result()
    executed = False

    def fail_provider(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("base provider failed")

    def unexpected_execution(*_args: object, **_kwargs: object) -> object:
        nonlocal executed
        executed = True
        raise AssertionError("output execution must not run")

    monkeypatch.setattr(
        workflow_module,
        "build_result_provider",
        fail_provider,
    )
    monkeypatch.setattr(
        workflow_module,
        "execute_output_requests",
        unexpected_execution,
    )

    with pytest.raises(RuntimeError, match="base provider failed"):
        build_solve_result_bundle(task, result)

    assert not executed


def test_derived_recovery_failure_is_reported_without_failing_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, result = _task_and_result(
        (OutputRequest("field", "element", ("S",)),)
    )

    def fail_materialization(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> object:
        del self, keys, cancellation
        raise RuntimeError("derived recovery failed")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        fail_materialization,
    )

    bundle = build_solve_result_bundle(task, result)

    assert tuple(
        request.status for request in bundle.execution_report.requests
    ) == (OutputExecutionStatus.FAILED,)
    assert bundle.execution_report.diagnostics


def test_cancellation_before_provider_construction_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, result = _task_and_result()
    provider_called = False

    def unexpected_provider(*_args: object, **_kwargs: object) -> object:
        nonlocal provider_called
        provider_called = True
        raise AssertionError("cancelled workflow must not build provider")

    monkeypatch.setattr(
        workflow_module,
        "build_result_provider",
        unexpected_provider,
    )

    with pytest.raises(_Cancelled, match="cancelled"):
        build_solve_result_bundle(
            task,
            result,
            cancellation=_Cancellation(cancelled=True),
        )

    assert not provider_called


def test_cancellation_after_output_execution_discards_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, result = _task_and_result(
        (OutputRequest("field", "node", ("U",)),)
    )
    cancellation = _Cancellation()
    original_execute = workflow_module.execute_output_requests

    def execute_then_cancel(*args: object, **kwargs: object) -> object:
        outcome = original_execute(*args, **kwargs)  # type: ignore[arg-type]
        cancellation.cancelled = True
        return outcome

    monkeypatch.setattr(
        workflow_module,
        "execute_output_requests",
        execute_then_cancel,
    )

    with pytest.raises(_Cancelled, match="cancelled"):
        build_solve_result_bundle(
            task,
            result,
            cancellation=cancellation,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda task, result: setattr(result, "model", object()),
            "task model",
        ),
        (
            lambda task, result: setattr(
                result,
                "step",
                AnalysisStep(task.step_name),
            ),
            "task model step",
        ),
    ),
)
def test_bundle_rejects_result_outside_task_snapshot(
    mutate: Callable[[SolveTaskSnapshot, ModelResult], None],
    message: str,
) -> None:
    task, result = _task_and_result()
    mutate(task, result)

    with pytest.raises(ValueError, match=message):
        build_solve_result_bundle(task, result)


def test_result_model_identity_matches_detached_artifact_inputs() -> None:
    task, result = _task_and_result()
    expected_model = deepcopy(task.model)

    validate_solve_result_model_identity(
        result,
        expected_model,
        task.step_name,
    )


def test_result_model_identity_rejects_foreign_topology_and_step() -> None:
    task, result = _task_and_result()
    foreign_model = deepcopy(task.model)
    foreign_model.mesh.nodes[1].x = 3.0

    with pytest.raises(ValueError, match="topology"):
        validate_solve_result_model_identity(
            result,
            foreign_model,
            task.step_name,
        )

    wrong_step_model = deepcopy(task.model)
    wrong_step_model.steps = [
        AnalysisStep(task.step_name, procedure="dynamic")
    ]
    with pytest.raises(ValueError, match="artifact step"):
        validate_solve_result_model_identity(
            result,
            wrong_step_model,
            task.step_name,
        )
