from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

import fem.application.results.execution as execution_module
from fem.application.results.data import (
    FieldState,
    ResultDiagnostic,
    ResultMaterializationPatch,
)
from fem.application.results.execution import (
    OutputExecutionOutcome,
    OutputExecutionStatus,
    OutputRequestExecution,
    OutputVariableExecution,
    ResultExecutionReport,
    execute_output_requests,
)
from fem.application.results.fields import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
)
from fem.application.results.output_requests import (
    ExecutableOutputRequest,
    OutputRequestProjection,
    OutputVariableProjection,
)
from fem.application.results.provider import (
    ResultProvider,
    build_result_provider,
)
from fem.core.model import OutputRequest, OutputSourceEvidence
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)


def _source(suffix: str = "1") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result-{suffix}",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=4,
        step_name="Step-1",
        run_id=f"run-{suffix}",
    )


def _truss_provider() -> ResultProvider:
    return build_result_provider(
        _source("truss"),
        make_truss_field_characterization_result(),
    )


def _continuum_provider() -> ResultProvider:
    return build_result_provider(
        _source("continuum"),
        make_continuum_nodal_semantics_result(),
    )


def _status_values(
    outcome: OutputExecutionOutcome,
) -> tuple[OutputExecutionStatus, ...]:
    return tuple(
        request.status for request in outcome.report.requests
    )


def test_status_values_are_exact_and_execution_values_are_frozen() -> None:
    provider = _truss_provider()

    outcome = execute_output_requests(provider, ())

    assert tuple(status.value for status in OutputExecutionStatus) == (
        "executed",
        "unsupported",
        "failed",
        "skipped",
    )
    assert outcome.source == provider.source
    assert outcome.eager_patch == ResultMaterializationPatch(
        provider.source,
        (),
    )
    assert outcome.report == ResultExecutionReport(
        provider.source,
        (),
        (),
    )
    assert outcome.provider_draft is provider
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]


def test_execution_rejects_lookalike_provider_and_outcome_draft() -> None:
    provider = _truss_provider()

    class _Lookalike:
        source = provider.source
        profile = provider.profile

        resolve_request = provider.resolve_request
        field_status = provider.field_status
        materialize = provider.materialize
        apply = provider.apply

    with pytest.raises(TypeError, match="exactly ResultProvider"):
        execute_output_requests(_Lookalike(), ())  # type: ignore[arg-type]

    patch = ResultMaterializationPatch(provider.source, ())
    report = ResultExecutionReport(provider.source, (), ())
    with pytest.raises(TypeError, match="exactly ResultProvider"):
        OutputExecutionOutcome(
            provider.source,
            patch,
            report,
            _Lookalike(),  # type: ignore[arg-type]
        )


def test_outcome_cross_validates_report_patch_and_provider_draft() -> None:
    provider = _truss_provider()
    valid = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("S",)),),
    )
    empty_report = ResultExecutionReport(provider.source, (), ())

    with pytest.raises(ValueError, match="required by an executed request"):
        OutputExecutionOutcome(
            source=provider.source,
            eager_patch=valid.eager_patch,
            report=empty_report,
            provider_draft=valid.provider_draft,
        )
    with pytest.raises(ValueError, match="must be READY"):
        OutputExecutionOutcome(
            source=provider.source,
            eager_patch=ResultMaterializationPatch(
                provider.source,
                (),
            ),
            report=valid.report,
            provider_draft=provider,
        )


def _diagnostic(code: str) -> ResultDiagnostic:
    return ResultDiagnostic(
        code=code,
        severity="error",
        message="Execution diagnostic.",
        path=("steps", "Step-1", "outputs", 0),
        remediation="Review the output request.",
        details={"request_index": 0},
    )


def test_parent_request_rejects_skipped_status() -> None:
    variable = OutputVariableExecution(
        source_variable_indices=(0,),
        canonical_variable=ResultVariable.U,
        field_keys=(),
        status=OutputExecutionStatus.SKIPPED,
        diagnostics=(),
    )

    with pytest.raises(ValueError, match="reserved for variable"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.SKIPPED,
            executable_request=None,
            variables=(variable,),
            diagnostics=(),
        )


def test_failed_request_diagnostics_must_include_variable_diagnostics() -> None:
    provider = _truss_provider()
    request = OutputRequest("field", "element", ("S",))
    field_request = FieldRequest(
        ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
    )
    projection = _synthetic_stress_projection(
        request,
        request_index=0,
        field_requests=(field_request,),
    )
    variable_diagnostic = _diagnostic("output.variable.failed")
    variable = OutputVariableExecution(
        source_variable_indices=(0,),
        canonical_variable=ResultVariable.S,
        field_keys=(provider.resolve_request(field_request),),
        status=OutputExecutionStatus.FAILED,
        diagnostics=(variable_diagnostic,),
    )

    with pytest.raises(ValueError, match="include every variable"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.FAILED,
            executable_request=projection.executable_request,
            variables=(variable,),
            diagnostics=(_diagnostic("output.request.other"),),
        )


def test_unsupported_request_must_include_variable_diagnostics() -> None:
    variable = OutputVariableExecution(
        source_variable_indices=(0,),
        canonical_variable=None,
        field_keys=(),
        status=OutputExecutionStatus.UNSUPPORTED,
        diagnostics=(_diagnostic("output.variable.unsupported"),),
    )

    with pytest.raises(ValueError, match="include every variable"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.UNSUPPORTED,
            executable_request=None,
            variables=(variable,),
            diagnostics=(_diagnostic("output.request.unsupported"),),
        )


def test_failed_diagnostics_sort_by_authoring_occurrence_independently(
) -> None:
    provider = _truss_provider()
    authoring = OutputRequest("field", "node", ("RF", "U"))
    executed = execute_output_requests(
        provider,
        (authoring,),
    ).report.requests[0]
    u_execution, rf_execution = executed.variables
    u_diagnostic = ResultDiagnostic(
        code="output.request.materialization_failed",
        severity="error",
        message="U recovery failed.",
        path=(
            "steps",
            provider.source.step_name,
            "outputs",
            0,
            "variables",
            1,
        ),
        remediation="Retry U recovery.",
        details={
            "request_index": 0,
            "source_indices": [1],
        },
    )
    rf_diagnostic = ResultDiagnostic(
        code="output.request.materialization_failed",
        severity="error",
        message="RF recovery failed.",
        path=(
            "steps",
            provider.source.step_name,
            "outputs",
            0,
            "variables",
            0,
        ),
        remediation="Retry RF recovery.",
        details={
            "request_index": 0,
            "source_indices": [0],
        },
    )
    failed_u = OutputVariableExecution(
        source_variable_indices=(1,),
        canonical_variable=ResultVariable.U,
        field_keys=u_execution.field_keys,
        status=OutputExecutionStatus.FAILED,
        diagnostics=(u_diagnostic,),
    )
    failed_rf = OutputVariableExecution(
        source_variable_indices=(0,),
        canonical_variable=ResultVariable.RF,
        field_keys=rf_execution.field_keys,
        status=OutputExecutionStatus.FAILED,
        diagnostics=(rf_diagnostic,),
    )

    request_execution = OutputRequestExecution(
        request_index=0,
        status=OutputExecutionStatus.FAILED,
        executable_request=executed.executable_request,
        variables=(failed_u, failed_rf),
        diagnostics=(rf_diagnostic, u_diagnostic),
    )
    report = ResultExecutionReport(
        source=provider.source,
        requests=(request_execution,),
        diagnostics=(rf_diagnostic, u_diagnostic),
    )

    assert tuple(
        variable.canonical_variable
        for variable in request_execution.variables
    ) == (ResultVariable.U, ResultVariable.RF)
    assert report.diagnostics == (rf_diagnostic, u_diagnostic)
    with pytest.raises(ValueError, match="canonically ordered"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.FAILED,
            executable_request=executed.executable_request,
            variables=(failed_u, failed_rf),
            diagnostics=(u_diagnostic, rf_diagnostic),
        )
    with pytest.raises(ValueError, match="unique and canonically ordered"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.FAILED,
            executable_request=executed.executable_request,
            variables=(failed_u, failed_rf),
            diagnostics=(
                rf_diagnostic,
                rf_diagnostic,
                u_diagnostic,
            ),
        )
    with pytest.raises(ValueError, match="canonical execution order"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.FAILED,
            executable_request=executed.executable_request,
            variables=(failed_rf, failed_u),
            diagnostics=(rf_diagnostic, u_diagnostic),
        )


def test_variable_execution_rejects_a_key_for_another_variable() -> None:
    provider = _truss_provider()
    rf_key = execute_output_requests(
        provider,
        (OutputRequest("field", "node", ("RF",)),),
    ).report.requests[0].variables[0].field_keys[0]

    with pytest.raises(ValueError, match="match the canonical variable"):
        OutputVariableExecution(
            source_variable_indices=(0,),
            canonical_variable=ResultVariable.U,
            field_keys=(rf_key,),
            status=OutputExecutionStatus.EXECUTED,
            diagnostics=(),
        )


def test_request_execution_rejects_a_different_position_for_same_variable() -> None:
    provider = _continuum_provider()
    request = OutputRequest(
        "field",
        "element",
        ("S",),
        {"position": "centroid"},
    )
    executed = execute_output_requests(
        provider,
        (request,),
    ).report.requests[0]
    wrong_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
            )
        )
    )
    wrong_variable = OutputVariableExecution(
        source_variable_indices=(0,),
        canonical_variable=ResultVariable.S,
        field_keys=(wrong_key,),
        status=OutputExecutionStatus.EXECUTED,
        diagnostics=(),
    )

    with pytest.raises(ValueError, match="executable field requests"):
        OutputRequestExecution(
            request_index=0,
            status=OutputExecutionStatus.EXECUTED,
            executable_request=executed.executable_request,
            variables=(wrong_variable,),
            diagnostics=(),
        )


def test_ready_primary_cache_hits_are_executed_without_a_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()
    evidence = OutputSourceEvidence(
        "abaqus",
        parent_parameters=(("FREQUENCY", "1"),),
    )
    request = OutputRequest(
        "field",
        "node",
        ("RF", "u", "U"),
        source_evidence=evidence,
    )
    preserved = deepcopy(request)

    def unexpected_materialize(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        del self, keys, cancellation
        raise AssertionError("READY fields must not be materialized")

    def unexpected_apply(
        self: ResultProvider,
        patch: ResultMaterializationPatch,
    ) -> ResultProvider:
        del self, patch
        raise AssertionError("an empty eager patch must not be applied")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        unexpected_materialize,
    )
    monkeypatch.setattr(ResultProvider, "apply", unexpected_apply)

    outcome = execute_output_requests(provider, (request,))

    execution = outcome.report.requests[0]
    assert execution.status is OutputExecutionStatus.EXECUTED
    assert tuple(
        variable.canonical_variable for variable in execution.variables
    ) == (ResultVariable.U, ResultVariable.RF)
    assert tuple(
        variable.source_variable_indices for variable in execution.variables
    ) == ((1, 2), (0,))
    assert all(
        variable.status is OutputExecutionStatus.EXECUTED
        and len(variable.field_keys) == 1
        for variable in execution.variables
    )
    assert outcome.eager_patch.fields == ()
    assert outcome.provider_draft is provider
    assert request == preserved
    assert request.source_evidence is evidence


def test_unsupported_request_is_atomic_and_unrelated_request_continues() -> None:
    provider = _truss_provider()
    requests = (
        OutputRequest("field", "node", ("U", "FUTURE")),
        OutputRequest("field", "element", ("S",)),
    )
    preserved = deepcopy(requests)

    outcome = execute_output_requests(provider, requests)

    assert _status_values(outcome) == (
        OutputExecutionStatus.UNSUPPORTED,
        OutputExecutionStatus.EXECUTED,
    )
    unsupported = outcome.report.requests[0]
    assert tuple(
        variable.status for variable in unsupported.variables
    ) == (
        OutputExecutionStatus.SKIPPED,
        OutputExecutionStatus.UNSUPPORTED,
    )
    assert all(not variable.field_keys for variable in unsupported.variables)
    assert tuple(
        diagnostic.code for diagnostic in unsupported.diagnostics
    ) == ("output.request.variable_unsupported",)
    diagnostic = unsupported.diagnostics[0]
    assert diagnostic.path == (
        "steps",
        provider.source.step_name,
        "outputs",
        0,
        "variables",
        1,
    )
    assert diagnostic.details["request_index"] == 0
    assert unsupported.variables[1].diagnostics[0] is diagnostic
    assert len(outcome.eager_patch.fields) == 1
    assert (
        outcome.eager_patch.fields[0].key.request.field_id.variable
        is ResultVariable.S
    )
    assert requests == preserved


def test_shared_lazy_key_is_materialized_once_and_satisfies_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()
    stress_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
        )
    )
    lazy_field = provider.materialize((stress_key,)).fields[0]
    requests = (
        OutputRequest("field", "element", ("S",)),
        OutputRequest("field", "element", ("s", "S")),
    )
    original_resolve = ResultProvider.resolve_request
    original_materialize = ResultProvider.materialize
    resolved: list[FieldRequest] = []
    calls: list[tuple[object, ...]] = []

    def resolve_spy(
        self: ResultProvider,
        request: FieldRequest,
    ) -> FieldMaterializationKey:
        resolved.append(request)
        return original_resolve(self, request)

    def materialize_spy(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        requested = tuple(keys)  # type: ignore[arg-type]
        calls.append(requested)
        return original_materialize(
            self,
            requested,
            cancellation=cancellation,
        )

    monkeypatch.setattr(ResultProvider, "resolve_request", resolve_spy)
    monkeypatch.setattr(ResultProvider, "materialize", materialize_spy)

    outcome = execute_output_requests(provider, requests)

    assert _status_values(outcome) == (
        OutputExecutionStatus.EXECUTED,
        OutputExecutionStatus.EXECUTED,
    )
    assert resolved == [stress_key.request]
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert len(outcome.eager_patch.fields) == 1
    eager_field = outcome.eager_patch.fields[0]
    shared_key = eager_field.key
    assert eager_field.descriptor == lazy_field.descriptor
    assert eager_field.locations == lazy_field.locations
    assert eager_field.values == pytest.approx(lazy_field.values)
    assert all(
        request.variables[0].field_keys == (shared_key,)
        for request in outcome.report.requests
    )
    assert provider.field_status(shared_key).state is FieldState.LAZY
    assert (
        outcome.provider_draft.field_status(shared_key).state
        is FieldState.READY
    )
    assert (
        outcome.provider_draft.snapshot.generation
        == provider.snapshot.generation
    )


def test_one_materialization_failure_does_not_block_unrelated_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _continuum_provider()
    requests = (
        OutputRequest(
            "field",
            "element",
            ("S",),
            {"position": "centroid"},
        ),
        OutputRequest(
            "field",
            "element",
            ("S",),
            {"position": "integration_point"},
        ),
    )
    original_materialize = ResultProvider.materialize
    calls: list[FieldPosition] = []

    def fail_centroid(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        requested = tuple(keys)  # type: ignore[arg-type]
        position = requested[0].request.field_id.position
        calls.append(position)
        if position is FieldPosition.CENTROID:
            raise RuntimeError("centroid kernel fault")
        return original_materialize(
            self,
            requested,
            cancellation=cancellation,
        )

    monkeypatch.setattr(ResultProvider, "materialize", fail_centroid)

    outcome = execute_output_requests(provider, requests)

    assert _status_values(outcome) == (
        OutputExecutionStatus.FAILED,
        OutputExecutionStatus.EXECUTED,
    )
    assert set(calls) == {
        FieldPosition.CENTROID,
        FieldPosition.INTEGRATION_POINT,
    }
    assert tuple(
        field.key.request.field_id.position
        for field in outcome.eager_patch.fields
    ) == (FieldPosition.INTEGRATION_POINT,)
    failed = outcome.report.requests[0]
    assert failed.variables[0].status is OutputExecutionStatus.FAILED
    diagnostic = failed.diagnostics[0]
    assert diagnostic.code == "output.request.materialization_failed"
    assert diagnostic.severity == "error"
    assert diagnostic.path == (
        "steps",
        provider.source.step_name,
        "outputs",
        0,
        "variables",
        0,
    )
    assert diagnostic.details["field_position"] == "centroid"
    assert diagnostic.details["error_type"] == "RuntimeError"
    assert diagnostic.details["error_message"] == "centroid kernel fault"
    assert outcome.report.diagnostics == failed.diagnostics


def test_shared_materialization_failure_is_attempted_once_and_reported_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()
    requests = (
        OutputRequest("field", "element", ("S",)),
        OutputRequest("field", "element", ("s",)),
        OutputRequest("field", "node", ("U",)),
    )
    calls = 0

    def fail_stress(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        del self, cancellation
        requested = tuple(keys)  # type: ignore[arg-type]
        assert (
            requested[0].request.field_id.variable
            is ResultVariable.S
        )
        nonlocal calls
        calls += 1
        raise ArithmeticError("stress recovery failed")

    monkeypatch.setattr(ResultProvider, "materialize", fail_stress)

    outcome = execute_output_requests(provider, requests)

    assert calls == 1
    assert _status_values(outcome) == (
        OutputExecutionStatus.FAILED,
        OutputExecutionStatus.FAILED,
        OutputExecutionStatus.EXECUTED,
    )
    assert outcome.eager_patch.fields == ()
    assert outcome.provider_draft is provider
    assert tuple(
        request.diagnostics[0].path
        for request in outcome.report.requests[:2]
    ) == (
        (
            "steps",
            provider.source.step_name,
            "outputs",
            0,
            "variables",
            0,
        ),
        (
            "steps",
            provider.source.step_name,
            "outputs",
            1,
            "variables",
            0,
        ),
    )
    assert outcome.report.diagnostics == (
        outcome.report.requests[0].diagnostics[0],
        outcome.report.requests[1].diagnostics[0],
    )


def _synthetic_stress_projection(
    request: OutputRequest,
    *,
    request_index: int,
    field_requests: tuple[FieldRequest, ...],
) -> OutputRequestProjection:
    variable = OutputVariableProjection(
        source_variable_indices=(0,),
        source_variables=(request.variables[0],),
        canonical_variable=ResultVariable.S,
        field_requests=field_requests,
    )
    executable = ExecutableOutputRequest(
        request_index=request_index,
        kind="field",
        target="element",
        frequency=1,
        variables=(variable,),
        field_requests=field_requests,
    )
    return OutputRequestProjection(
        request_index=request_index,
        authoring_request=request,
        variables=(variable,),
        executable_request=executable,
        diagnostics=(),
    )


def _install_multi_field_projection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_success: bool,
) -> None:
    integration_point = FieldRequest(
        ResultFieldId(
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
        )
    )
    centroid = FieldRequest(
        ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
    )

    def project(
        request: OutputRequest,
        capabilities: object,
        *,
        request_index: int,
    ) -> OutputRequestProjection:
        del capabilities
        fields = (
            (integration_point, centroid)
            if request_index == 0
            else (integration_point,)
        )
        if request_index > 0 and not shared_success:
            raise AssertionError("unexpected second request")
        return _synthetic_stress_projection(
            request,
            request_index=request_index,
            field_requests=fields,
        )

    monkeypatch.setattr(execution_module, "project_output_request", project)


def _fail_centroid_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> list[FieldPosition]:
    original_materialize = ResultProvider.materialize
    calls: list[FieldPosition] = []

    def selective_materialize(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        requested = tuple(keys)  # type: ignore[arg-type]
        position = requested[0].request.field_id.position
        calls.append(position)
        if position is FieldPosition.CENTROID:
            raise RuntimeError("centroid recovery fault")
        return original_materialize(
            self,
            requested,
            cancellation=cancellation,
        )

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        selective_materialize,
    )
    return calls


def test_recovered_derived_field_of_failed_request_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _continuum_provider()
    _install_multi_field_projection(
        monkeypatch,
        shared_success=False,
    )
    calls = _fail_centroid_materialization(monkeypatch)

    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("S",)),),
    )

    assert set(calls) == {
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
    }
    assert _status_values(outcome) == (OutputExecutionStatus.FAILED,)
    assert outcome.eager_patch.fields == ()
    assert outcome.provider_draft is provider


def test_recovered_derived_field_is_installed_when_shared_by_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _continuum_provider()
    _install_multi_field_projection(
        monkeypatch,
        shared_success=True,
    )
    calls = _fail_centroid_materialization(monkeypatch)

    outcome = execute_output_requests(
        provider,
        (
            OutputRequest("field", "element", ("S",)),
            OutputRequest("field", "element", ("S",)),
        ),
    )

    assert calls.count(FieldPosition.INTEGRATION_POINT) == 1
    assert calls.count(FieldPosition.CENTROID) == 1
    assert _status_values(outcome) == (
        OutputExecutionStatus.FAILED,
        OutputExecutionStatus.EXECUTED,
    )
    assert tuple(
        field.key.request.field_id.position
        for field in outcome.eager_patch.fields
    ) == (FieldPosition.INTEGRATION_POINT,)
    failed, succeeded = outcome.report.requests
    assert failed.variables[0].status is OutputExecutionStatus.FAILED
    assert succeeded.variables[0].status is OutputExecutionStatus.EXECUTED
    shared_key = outcome.eager_patch.fields[0].key
    assert succeeded.variables[0].field_keys == (shared_key,)
    assert (
        outcome.provider_draft.field_status(shared_key).state
        is FieldState.READY
    )


class _Cancelled(RuntimeError):
    pass


class _Cancellation:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def checkpoint(self) -> None:
        if self.cancelled:
            raise _Cancelled("cancelled")


def test_cancellation_before_projection_propagates_without_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()

    def unexpected_materialize(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        del self, keys, cancellation
        raise AssertionError("cancelled execution must not materialize")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        unexpected_materialize,
    )

    with pytest.raises(_Cancelled, match="cancelled"):
        execute_output_requests(
            provider,
            (OutputRequest("field", "element", ("S",)),),
            cancellation=_Cancellation(cancelled=True),
        )


def test_cancellation_raised_inside_materialization_is_not_a_field_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()
    token = _Cancellation()
    original_materialize = ResultProvider.materialize

    def cancel_during_materialization(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        token.cancelled = True
        return original_materialize(
            self,
            tuple(keys),  # type: ignore[arg-type]
            cancellation=cancellation,
        )

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        cancel_during_materialization,
    )

    with pytest.raises(_Cancelled, match="cancelled"):
        execute_output_requests(
            provider,
            (OutputRequest("field", "element", ("S",)),),
            cancellation=token,
        )

    stress_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
        )
    )
    assert provider.field_status(stress_key).state is FieldState.LAZY


def test_cancellation_after_recovery_prevents_provider_draft_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _truss_provider()
    token = _Cancellation()
    original_materialize = ResultProvider.materialize
    apply_called = False

    def recover_then_cancel(
        self: ResultProvider,
        keys: object,
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        patch = original_materialize(
            self,
            tuple(keys),  # type: ignore[arg-type]
            cancellation=cancellation,
        )
        token.cancelled = True
        return patch

    def unexpected_apply(
        self: ResultProvider,
        patch: ResultMaterializationPatch,
    ) -> ResultProvider:
        del self, patch
        nonlocal apply_called
        apply_called = True
        raise AssertionError("cancelled execution must not apply its patch")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        recover_then_cancel,
    )
    monkeypatch.setattr(ResultProvider, "apply", unexpected_apply)

    with pytest.raises(_Cancelled, match="cancelled"):
        execute_output_requests(
            provider,
            (OutputRequest("field", "element", ("S",)),),
            cancellation=token,
        )

    assert not apply_called
    stress_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
        )
    )
    assert provider.field_status(stress_key).state is FieldState.LAZY
