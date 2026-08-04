"""Atomic OutputRequest execution over the public immutable provider API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fem.core.model import OutputRequest

from ._materializers import check_cancellation
from .data import (
    FieldAvailability,
    FieldData,
    FieldState,
    ResultDiagnostic,
    ResultMaterializationPatch,
)
from .fields import (
    FieldMaterializationKey,
    FieldRequest,
    ResultSourceKey,
    ResultVariable,
    field_materialization_sort_key,
)
from .output_requests import (
    ExecutableOutputRequest,
    OutputRequestProjection,
    OutputVariableProjection,
    ResultCapabilityCatalog,
    project_output_request,
)
from .provider import ResultProvider
from .registry import ElementResultProfile


class OutputExecutionStatus(str, Enum):
    """Lifecycle state of one projected output request or variable."""

    EXECUTED = "executed"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"


_OUTPUT_VARIABLE_ORDER = (
    ResultVariable.U,
    ResultVariable.UR,
    ResultVariable.RF,
    ResultVariable.RM,
    ResultVariable.S,
)


@dataclass(frozen=True, slots=True)
class OutputVariableExecution:
    """Execution result for every occurrence of one canonical variable."""

    source_variable_indices: tuple[int, ...]
    canonical_variable: ResultVariable | None
    field_keys: tuple[FieldMaterializationKey, ...]
    status: OutputExecutionStatus
    diagnostics: tuple[ResultDiagnostic, ...]

    def __post_init__(self) -> None:
        _validate_source_indices(self.source_variable_indices)
        if (
            self.canonical_variable is not None
            and type(self.canonical_variable) is not ResultVariable
        ):
            raise TypeError(
                "canonical_variable must be ResultVariable or None"
            )
        if (
            self.canonical_variable is not None
            and self.canonical_variable not in _OUTPUT_VARIABLE_ORDER
        ):
            raise ValueError(
                "canonical_variable must be an executable output variable"
            )
        _validate_field_keys(self.field_keys)
        if type(self.status) is not OutputExecutionStatus:
            raise TypeError("status must be OutputExecutionStatus")
        _validate_diagnostics(self.diagnostics)
        if (
            self.canonical_variable is not None
            and any(
                key.request.field_id.variable
                is not self.canonical_variable
                for key in self.field_keys
            )
        ):
            raise ValueError(
                "field keys must match the canonical variable"
            )

        if self.status is OutputExecutionStatus.EXECUTED:
            if self.canonical_variable is None:
                raise ValueError(
                    "executed variables require a canonical variable"
                )
            if not self.field_keys:
                raise ValueError(
                    "executed variables require at least one field key"
                )
            if self.diagnostics:
                raise ValueError(
                    "executed variables cannot contain diagnostics"
                )
        elif self.status is OutputExecutionStatus.FAILED:
            if self.canonical_variable is None:
                raise ValueError(
                    "failed variables require a canonical variable"
                )
            if not self.field_keys:
                raise ValueError(
                    "failed variables require attempted field keys"
                )
            if not self.diagnostics:
                raise ValueError(
                    "failed variables require diagnostics"
                )
        elif self.status is OutputExecutionStatus.UNSUPPORTED:
            if self.field_keys:
                raise ValueError(
                    "unsupported variables cannot contain field keys"
                )
            if not self.diagnostics:
                raise ValueError(
                    "unsupported variables require diagnostics"
                )
        elif self.status is OutputExecutionStatus.SKIPPED:
            if self.field_keys:
                raise ValueError(
                    "skipped variables cannot contain field keys"
                )
            if self.diagnostics:
                raise ValueError(
                    "skipped variables cannot contain diagnostics"
                )


@dataclass(frozen=True, slots=True)
class OutputRequestExecution:
    """Atomic execution status for one preserved authoring request."""

    request_index: int
    status: OutputExecutionStatus
    executable_request: ExecutableOutputRequest | None
    variables: tuple[OutputVariableExecution, ...]
    diagnostics: tuple[ResultDiagnostic, ...]

    def __post_init__(self) -> None:
        _validate_request_index(self.request_index)
        if type(self.status) is not OutputExecutionStatus:
            raise TypeError("status must be OutputExecutionStatus")
        if (
            self.executable_request is not None
            and type(self.executable_request) is not ExecutableOutputRequest
        ):
            raise TypeError(
                "executable_request must be ExecutableOutputRequest or None"
            )
        if type(self.variables) is not tuple:
            raise TypeError("variables must be a tuple")
        if any(
            type(variable) is not OutputVariableExecution
            for variable in self.variables
        ):
            raise TypeError(
                "variables must contain OutputVariableExecution values"
            )
        _validate_diagnostics(self.diagnostics)
        if (
            self.executable_request is not None
            and self.executable_request.request_index != self.request_index
        ):
            raise ValueError(
                "executable request index must match request execution"
            )
        _validate_variable_execution_order(self.variables)
        if self.executable_request is not None:
            projected_identities = tuple(
                (
                    variable.source_variable_indices,
                    variable.canonical_variable,
                )
                for variable in self.executable_request.variables
            )
            execution_identities = tuple(
                (
                    variable.source_variable_indices,
                    variable.canonical_variable,
                )
                for variable in self.variables
            )
            if not all(
                identity in execution_identities
                for identity in projected_identities
            ):
                raise ValueError(
                    "executable variables must appear in the request execution"
                )
            if projected_identities != tuple(
                identity
                for identity in execution_identities
                if identity in projected_identities
            ):
                raise ValueError(
                    "executable variables must retain canonical execution order"
                )
            for projected in self.executable_request.variables:
                executed = next(
                    variable
                    for variable in self.variables
                    if (
                        variable.source_variable_indices,
                        variable.canonical_variable,
                    )
                    == (
                        projected.source_variable_indices,
                        projected.canonical_variable,
                    )
                )
                if executed.status not in {
                    OutputExecutionStatus.EXECUTED,
                    OutputExecutionStatus.FAILED,
                }:
                    continue
                actual_requests = tuple(
                    key.request for key in executed.field_keys
                )
                if (
                    len(actual_requests) != len(projected.field_requests)
                    or set(actual_requests) != set(projected.field_requests)
                ):
                    raise ValueError(
                        "executed field keys must match the executable "
                        "field requests"
                    )

        statuses = tuple(variable.status for variable in self.variables)
        variable_diagnostics = tuple(
            diagnostic
            for variable in self.variables
            for diagnostic in variable.diagnostics
        )
        if self.status is OutputExecutionStatus.EXECUTED:
            if self.executable_request is None:
                raise ValueError(
                    "executed requests require an executable projection"
                )
            if not statuses or any(
                status is not OutputExecutionStatus.EXECUTED
                for status in statuses
            ):
                raise ValueError(
                    "executed requests require all variables to be executed"
                )
            if self.diagnostics:
                raise ValueError(
                    "executed requests cannot contain diagnostics"
                )
        elif self.status is OutputExecutionStatus.UNSUPPORTED:
            if not self.diagnostics:
                raise ValueError(
                    "unsupported requests require diagnostics"
                )
            allowed = {
                OutputExecutionStatus.UNSUPPORTED,
                OutputExecutionStatus.SKIPPED,
            }
            if self.executable_request is not None:
                allowed.add(OutputExecutionStatus.EXECUTED)
            if any(status not in allowed for status in statuses):
                raise ValueError(
                    "unsupported requests allow only executable, unsupported, "
                    "or skipped variables"
                )
            if (
                self.executable_request is not None
                and OutputExecutionStatus.UNSUPPORTED not in statuses
            ):
                raise ValueError(
                    "partially executable requests require an unsupported variable"
                )
        elif self.status is OutputExecutionStatus.FAILED:
            if self.executable_request is None:
                raise ValueError(
                    "failed requests require an executable projection"
                )
            if OutputExecutionStatus.FAILED not in statuses:
                raise ValueError(
                    "failed requests require at least one failed variable"
                )
            if any(
                status
                not in {
                    OutputExecutionStatus.FAILED,
                    OutputExecutionStatus.SKIPPED,
                    OutputExecutionStatus.UNSUPPORTED,
                }
                for status in statuses
            ):
                raise ValueError(
                    "failed requests allow only failed, unsupported, or skipped variables"
                )
            if not self.diagnostics:
                raise ValueError("failed requests require diagnostics")
        elif self.status is OutputExecutionStatus.SKIPPED:
            raise ValueError(
                "SKIPPED is reserved for variable execution status"
            )
        if any(
            diagnostic not in self.diagnostics
            for diagnostic in variable_diagnostics
        ):
            raise ValueError(
                "request diagnostics must include every variable diagnostic"
            )
        if self.diagnostics != _normalized_request_diagnostics(
            self.request_index,
            self.diagnostics,
        ):
            raise ValueError(
                "request diagnostics must be unique and canonically ordered"
            )


@dataclass(frozen=True, slots=True)
class ResultExecutionReport:
    """Source-bound ordered report for all authoring output requests."""

    source: ResultSourceKey
    requests: tuple[OutputRequestExecution, ...]
    diagnostics: tuple[ResultDiagnostic, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.requests) is not tuple:
            raise TypeError("requests must be a tuple")
        if any(
            type(request) is not OutputRequestExecution
            for request in self.requests
        ):
            raise TypeError(
                "requests must contain OutputRequestExecution values"
            )
        indices = tuple(request.request_index for request in self.requests)
        if indices != tuple(range(len(self.requests))):
            raise ValueError(
                "request executions must be ordered by contiguous source index"
            )
        _validate_diagnostics(self.diagnostics)
        expected = tuple(
            diagnostic
            for request in self.requests
            for diagnostic in request.diagnostics
        )
        if self.diagnostics != expected:
            raise ValueError(
                "report diagnostics must flatten request diagnostics in order"
            )
        if any(
            diagnostic.path[1] != self.source.step_name
            for diagnostic in self.diagnostics
        ):
            raise ValueError(
                "report diagnostic paths must match the source step"
            )


@dataclass(frozen=True, slots=True)
class OutputExecutionOutcome:
    """Detached worker outcome without accepted generation or Session mutation."""

    source: ResultSourceKey
    eager_patch: ResultMaterializationPatch
    report: ResultExecutionReport
    provider_draft: ResultProvider

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.eager_patch) is not ResultMaterializationPatch:
            raise TypeError("eager_patch must be ResultMaterializationPatch")
        if type(self.report) is not ResultExecutionReport:
            raise TypeError("report must be ResultExecutionReport")
        if self.eager_patch.source != self.source:
            raise ValueError("eager patch source must match outcome source")
        if self.report.source != self.source:
            raise ValueError("report source must match outcome source")
        if self.eager_patch.diagnostics:
            raise ValueError(
                "successful combined eager patch cannot contain diagnostics"
            )
        if type(self.provider_draft) is not ResultProvider:
            raise TypeError("provider_draft must be exactly ResultProvider")
        if self.provider_draft.source != self.source:
            raise ValueError("provider draft source must match outcome source")
        executed_keys = {
            key
            for request in self.report.requests
            for variable in request.variables
            if variable.status is OutputExecutionStatus.EXECUTED
            for key in variable.field_keys
        }
        eager_keys = {
            field_data.key for field_data in self.eager_patch.fields
        }
        if not eager_keys.issubset(executed_keys):
            raise ValueError(
                "eager patch fields must be required by an executed request"
            )
        draft_fields = {
            field_data.key: field_data
            for field_data in self.provider_draft.snapshot.fields
        }
        if any(
            draft_fields.get(field_data.key) is not field_data
            for field_data in self.eager_patch.fields
        ):
            raise ValueError(
                "eager patch fields must be installed unchanged in the draft"
            )
        for key in executed_keys:
            try:
                availability = self.provider_draft.field_status(key)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "executed field keys must resolve in the provider draft"
                ) from error
            if availability.state is not FieldState.READY:
                raise ValueError(
                    "executed field keys must be READY in the provider draft"
                )


def execute_output_requests(
    provider: ResultProvider,
    requests: tuple[OutputRequest, ...],
    cancellation: object | None = None,
) -> OutputExecutionOutcome:
    """Project and atomically execute authoring requests over one provider."""

    source, profile = _validate_provider(provider)
    if type(requests) is not tuple:
        raise TypeError("requests must be a tuple")
    if any(type(request) is not OutputRequest for request in requests):
        raise TypeError("requests must contain exact OutputRequest values")

    check_cancellation(cancellation)
    capabilities = ResultCapabilityCatalog.from_profile(profile)
    projections: list[OutputRequestProjection] = []
    for request_index, request in enumerate(requests):
        check_cancellation(cancellation)
        projections.append(
            project_output_request(
                request,
                capabilities,
                request_index=request_index,
            )
        )

    resolved_requests: dict[
        FieldRequest,
        FieldMaterializationKey,
    ] = {}
    request_plan_list: list[_RequestPlan | None] = []
    for projection in projections:
        check_cancellation(cancellation)
        request_plan_list.append(
            _resolve_request_plan(
                provider,
                projection,
                resolved_requests=resolved_requests,
            )
            if projection.executable_request is not None
            else None
        )
    request_plans = tuple(request_plan_list)
    unique_keys = tuple(
        sorted(
            {
                key
                for plan in request_plans
                if plan is not None
                for key in plan.required_keys
            },
            key=field_materialization_sort_key,
        )
    )

    ready_keys: set[FieldMaterializationKey] = set()
    pending_keys: list[FieldMaterializationKey] = []
    failures: dict[FieldMaterializationKey, Exception] = {}
    for key in unique_keys:
        check_cancellation(cancellation)
        availability = provider.field_status(key)
        if type(availability) is not FieldAvailability:
            raise TypeError(
                "provider.field_status() must return FieldAvailability"
            )
        if availability.key != key:
            raise ValueError(
                "provider.field_status() returned a different field key"
            )
        if availability.state is FieldState.READY:
            ready_keys.add(key)
        elif availability.state is FieldState.LAZY:
            pending_keys.append(key)
        else:
            failures[key] = RuntimeError(
                "provider marks the executable field unavailable"
            )

    recovered: dict[FieldMaterializationKey, FieldData] = {}
    for key in pending_keys:
        check_cancellation(cancellation)
        probe = _CancellationProbe(cancellation)
        try:
            patch = provider.materialize(
                (key,),
                cancellation=(None if cancellation is None else probe),
            )
            field_data = _single_materialized_field(
                patch,
                source=source,
                key=key,
            )
        except Exception as error:
            if probe.cancelled_error is not None:
                raise
            check_cancellation(cancellation)
            failures[key] = error
            continue
        recovered[key] = field_data
        check_cancellation(cancellation)

    executions: list[OutputRequestExecution] = []
    successful_required_keys: set[FieldMaterializationKey] = set()
    for projection, plan in zip(projections, request_plans, strict=True):
        if plan is None:
            executions.append(
                _unsupported_execution(
                    projection,
                    step_name=source.step_name,
                )
            )
            continue
        execution = _completed_execution(
            plan,
            failures,
            step_name=source.step_name,
        )
        executions.append(execution)
        successful_required_keys.update(
            key
            for variable in execution.variables
            if variable.status is OutputExecutionStatus.EXECUTED
            for key in variable.field_keys
        )

    eager_fields = tuple(
        sorted(
            (
                recovered[key]
                for key in successful_required_keys
                if key not in ready_keys
            ),
            key=lambda field_data: field_materialization_sort_key(
                field_data.key
            ),
        )
    )
    eager_patch = ResultMaterializationPatch(
        source=source,
        fields=eager_fields,
    )
    report_requests = tuple(executions)
    report = ResultExecutionReport(
        source=source,
        requests=report_requests,
        diagnostics=tuple(
            diagnostic
            for request_execution in report_requests
            for diagnostic in request_execution.diagnostics
        ),
    )

    check_cancellation(cancellation)
    provider_draft = (
        provider if not eager_patch.fields else provider.apply(eager_patch)
    )
    if type(provider_draft) is not ResultProvider:
        raise RuntimeError(
            "provider.apply() must return exactly ResultProvider"
        )
    if provider_draft.source != source:
        raise RuntimeError(
            "provider.apply() must return a same-source immutable draft"
        )
    check_cancellation(cancellation)
    return OutputExecutionOutcome(
        source=source,
        eager_patch=eager_patch,
        report=report,
        provider_draft=provider_draft,
    )


@dataclass(frozen=True, slots=True)
class _RequestPlan:
    projection: OutputRequestProjection
    variable_keys: tuple[
        tuple[OutputVariableProjection, tuple[FieldMaterializationKey, ...]],
        ...,
    ]
    required_keys: tuple[FieldMaterializationKey, ...]


class _CancellationProbe:
    """Record cancellation exceptions raised inside provider materialization."""

    __slots__ = ("_cancellation", "cancelled_error")

    def __init__(self, cancellation: object | None) -> None:
        self._cancellation = cancellation
        self.cancelled_error: BaseException | None = None

    def checkpoint(self) -> None:
        try:
            check_cancellation(self._cancellation)
        except BaseException as error:
            self.cancelled_error = error
            raise


def _validate_provider(
    provider: ResultProvider,
) -> tuple[ResultSourceKey, ElementResultProfile]:
    if type(provider) is not ResultProvider:
        raise TypeError("provider must be exactly ResultProvider")
    return provider.source, provider.profile


def _resolve_request_plan(
    provider: ResultProvider,
    projection: OutputRequestProjection,
    *,
    resolved_requests: dict[
        FieldRequest,
        FieldMaterializationKey,
    ],
) -> _RequestPlan:
    executable = projection.executable_request
    if executable is None:
        raise ValueError("cannot resolve an unsupported request projection")
    variable_keys: list[
        tuple[OutputVariableProjection, tuple[FieldMaterializationKey, ...]]
    ] = []
    for variable in executable.variables:
        keys: list[FieldMaterializationKey] = []
        for field_request in variable.field_requests:
            key = resolved_requests.get(field_request)
            if key is None:
                key = provider.resolve_request(field_request)
                if type(key) is not FieldMaterializationKey:
                    raise TypeError(
                        "provider.resolve_request() must return "
                        "FieldMaterializationKey"
                    )
                if key.request != field_request:
                    raise ValueError(
                        "resolved key request must match the field request"
                    )
                resolved_requests[field_request] = key
            keys.append(key)
        ordered_keys = tuple(
            sorted(set(keys), key=field_materialization_sort_key)
        )
        variable_keys.append((variable, ordered_keys))
    required_keys = tuple(
        sorted(
            {
                key
                for _variable, keys in variable_keys
                for key in keys
            },
            key=field_materialization_sort_key,
        )
    )
    return _RequestPlan(
        projection=projection,
        variable_keys=tuple(variable_keys),
        required_keys=required_keys,
    )


def _single_materialized_field(
    patch: object,
    *,
    source: ResultSourceKey,
    key: FieldMaterializationKey,
) -> FieldData:
    if type(patch) is not ResultMaterializationPatch:
        raise TypeError(
            "provider.materialize() must return ResultMaterializationPatch"
        )
    if patch.source != source:
        raise ValueError(
            "provider.materialize() patch source must match provider source"
        )
    if patch.diagnostics:
        raise RuntimeError(
            "provider.materialize() returned diagnostics instead of a field"
        )
    if len(patch.fields) != 1 or patch.fields[0].key != key:
        raise RuntimeError(
            "provider.materialize() must return exactly the requested lazy field"
        )
    return patch.fields[0]


def _unsupported_execution(
    projection: OutputRequestProjection,
    *,
    step_name: str,
) -> OutputRequestExecution:
    source_diagnostics = tuple(
        _qualify_report_diagnostic(
            diagnostic,
            step_name=step_name,
        )
        for diagnostic in projection.diagnostics
    )
    qualified_diagnostics = _normalized_request_diagnostics(
        projection.request_index,
        source_diagnostics,
    )
    variables = tuple(
        OutputVariableExecution(
            source_variable_indices=variable.source_variable_indices,
            canonical_variable=variable.canonical_variable,
            field_keys=(),
            status=(
                OutputExecutionStatus.UNSUPPORTED
                if variable.diagnostics
                else OutputExecutionStatus.SKIPPED
            ),
            diagnostics=tuple(
                _matching_qualified_diagnostic(
                    diagnostic,
                    source_diagnostics=projection.diagnostics,
                    qualified_diagnostics=source_diagnostics,
                )
                for diagnostic in variable.diagnostics
            ),
        )
        for variable in projection.variables
    )
    return OutputRequestExecution(
        request_index=projection.request_index,
        status=OutputExecutionStatus.UNSUPPORTED,
        executable_request=None,
        variables=variables,
        diagnostics=qualified_diagnostics,
    )


def _completed_execution(
    plan: _RequestPlan,
    failures: dict[FieldMaterializationKey, Exception],
    *,
    step_name: str,
) -> OutputRequestExecution:
    projection = plan.projection
    source_diagnostics = projection.diagnostics
    qualified_diagnostics = tuple(
        _qualify_report_diagnostic(
            diagnostic,
            step_name=step_name,
        )
        for diagnostic in source_diagnostics
    )
    failed_variables = {
        id(variable)
        for variable, keys in plan.variable_keys
        if any(key in failures for key in keys)
    }
    request_failed = bool(failed_variables)
    keys_by_variable = {
        id(variable): keys
        for variable, keys in plan.variable_keys
    }
    variable_executions: list[OutputVariableExecution] = []
    request_diagnostics: list[ResultDiagnostic] = list(
        qualified_diagnostics
    )
    for variable in projection.variables:
        if variable.diagnostics:
            variable_executions.append(
                OutputVariableExecution(
                    source_variable_indices=variable.source_variable_indices,
                    canonical_variable=variable.canonical_variable,
                    field_keys=(),
                    status=OutputExecutionStatus.UNSUPPORTED,
                    diagnostics=tuple(
                        _matching_qualified_diagnostic(
                            diagnostic,
                            source_diagnostics=source_diagnostics,
                            qualified_diagnostics=qualified_diagnostics,
                        )
                        for diagnostic in variable.diagnostics
                    ),
                )
            )
            continue

        try:
            keys = keys_by_variable[id(variable)]
        except KeyError as error:
            raise ValueError(
                "executable projection variable has no execution plan"
            ) from error
        if not request_failed:
            variable_executions.append(
                OutputVariableExecution(
                    source_variable_indices=variable.source_variable_indices,
                    canonical_variable=variable.canonical_variable,
                    field_keys=keys,
                    status=OutputExecutionStatus.EXECUTED,
                    diagnostics=(),
                )
            )
            continue
        if id(variable) not in failed_variables:
            variable_executions.append(
                OutputVariableExecution(
                    source_variable_indices=variable.source_variable_indices,
                    canonical_variable=variable.canonical_variable,
                    field_keys=(),
                    status=OutputExecutionStatus.SKIPPED,
                    diagnostics=(),
                )
            )
            continue

        diagnostics = tuple(
            _materialization_failure_diagnostic(
                step_name,
                plan.projection.request_index,
                variable,
                key,
                failures[key],
            )
            for key in keys
            if key in failures
        )
        request_diagnostics.extend(diagnostics)
        variable_executions.append(
            OutputVariableExecution(
                source_variable_indices=variable.source_variable_indices,
                canonical_variable=variable.canonical_variable,
                field_keys=keys,
                status=OutputExecutionStatus.FAILED,
                diagnostics=diagnostics,
            )
        )

    return OutputRequestExecution(
        request_index=projection.request_index,
        status=(
            OutputExecutionStatus.FAILED
            if request_failed
            else (
                OutputExecutionStatus.UNSUPPORTED
                if any(
                    variable.diagnostics
                    for variable in projection.variables
                )
                else OutputExecutionStatus.EXECUTED
            )
        ),
        executable_request=projection.executable_request,
        variables=tuple(variable_executions),
        diagnostics=_normalized_request_diagnostics(
            projection.request_index,
            tuple(request_diagnostics),
        ),
    )


def _qualify_report_diagnostic(
    diagnostic: ResultDiagnostic,
    *,
    step_name: str,
) -> ResultDiagnostic:
    if (
        len(diagnostic.path) < 2
        or diagnostic.path[0] != "outputs"
        or type(diagnostic.path[1]) is not int
    ):
        raise ValueError(
            "projected output diagnostics require an output-index path"
        )
    return ResultDiagnostic(
        code=diagnostic.code,
        severity=diagnostic.severity,
        message=diagnostic.message,
        path=("steps", step_name, *diagnostic.path),
        remediation=diagnostic.remediation,
        details=diagnostic.details,
    )


def _matching_qualified_diagnostic(
    diagnostic: ResultDiagnostic,
    *,
    source_diagnostics: tuple[ResultDiagnostic, ...],
    qualified_diagnostics: tuple[ResultDiagnostic, ...],
) -> ResultDiagnostic:
    for index, candidate in enumerate(source_diagnostics):
        if candidate is diagnostic:
            return qualified_diagnostics[index]
    for index, candidate in enumerate(source_diagnostics):
        if candidate == diagnostic:
            return qualified_diagnostics[index]
    raise ValueError(
        "variable diagnostic must appear in projection diagnostics"
    )


def _materialization_failure_diagnostic(
    step_name: str,
    request_index: int,
    variable: OutputVariableProjection,
    key: FieldMaterializationKey,
    error: Exception,
) -> ResultDiagnostic:
    field_id = key.request.field_id
    error_message = str(error).strip() or type(error).__name__
    return ResultDiagnostic(
        code="output.request.materialization_failed",
        severity="error",
        message=(
            f"Failed to materialize {field_id.variable.value} at "
            f"{field_id.position.value}."
        ),
        path=(
            "steps",
            step_name,
            "outputs",
            request_index,
            "variables",
            variable.source_variable_indices[0],
        ),
        remediation=(
            "Keep the solved result and retry this field as a lazy recovery."
        ),
        details={
            "request_index": request_index,
            "source_indices": list(variable.source_variable_indices),
            "canonical_variable": field_id.variable.value,
            "field_position": field_id.position.value,
            "recovery_contract": key.recovery_contract,
            "error_type": type(error).__name__,
            "error_message": error_message,
        },
    )


def _validate_request_index(value: object) -> None:
    if type(value) is not int:
        raise TypeError("request_index must be an integer")
    if value < 0:
        raise ValueError("request_index must be non-negative")


def _validate_variable_execution_order(
    variables: tuple[OutputVariableExecution, ...],
) -> None:
    canonical = tuple(
        variable.canonical_variable
        for variable in variables
        if variable.canonical_variable is not None
    )
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            "request variables cannot repeat a canonical variable"
        )
    source_indices = tuple(
        source_index
        for variable in variables
        for source_index in variable.source_variable_indices
    )
    if len(set(source_indices)) != len(source_indices):
        raise ValueError(
            "request variables cannot overlap source occurrences"
        )
    expected = tuple(
        sorted(
            variables,
            key=lambda variable: (
                (
                    _OUTPUT_VARIABLE_ORDER.index(
                        variable.canonical_variable
                    )
                    if variable.canonical_variable is not None
                    else len(_OUTPUT_VARIABLE_ORDER)
                ),
                variable.source_variable_indices[0],
            ),
        )
    )
    if variables != expected:
        raise ValueError(
            "request variables must follow canonical execution order"
        )


def _normalized_request_diagnostics(
    request_index: int,
    diagnostics: tuple[ResultDiagnostic, ...],
) -> tuple[ResultDiagnostic, ...]:
    unique: list[ResultDiagnostic] = []
    for diagnostic in diagnostics:
        _validate_report_diagnostic_path(
            diagnostic,
            request_index=request_index,
        )
        if diagnostic not in unique:
            unique.append(diagnostic)
    return tuple(
        sorted(
            unique,
            key=lambda diagnostic: _report_diagnostic_sort_key(
                request_index,
                diagnostic,
            ),
        )
    )


def _validate_report_diagnostic_path(
    diagnostic: ResultDiagnostic,
    *,
    request_index: int,
) -> None:
    path = diagnostic.path
    if (
        len(path) < 4
        or path[0] != "steps"
        or type(path[1]) is not str
        or not path[1].strip()
        or path[2] != "outputs"
        or type(path[3]) is not int
        or path[3] != request_index
    ):
        raise ValueError(
            "request diagnostics require a matching Step/output-index path"
        )
    if diagnostic.details.get("request_index") != request_index:
        raise ValueError(
            "request diagnostic details must include its request index"
        )


def _report_diagnostic_sort_key(
    request_index: int,
    diagnostic: ResultDiagnostic,
) -> tuple[int, int, str, tuple[object, ...]]:
    source_indices = diagnostic.details.get("source_indices")
    if (
        type(source_indices) is tuple
        and source_indices
        and all(type(index) is int for index in source_indices)
    ):
        minimum_source_index = min(source_indices)
    else:
        minimum_source_index = -1
    return (
        request_index,
        minimum_source_index,
        diagnostic.code,
        diagnostic.path,
    )


def _validate_source_indices(value: object) -> None:
    if type(value) is not tuple or not value:
        raise ValueError(
            "source_variable_indices must be a nonempty tuple"
        )
    if any(type(index) is not int for index in value):
        raise TypeError("source variable indices must be integers")
    if value != tuple(sorted(set(value))) or value[0] < 0:
        raise ValueError(
            "source variable indices must be unique increasing non-negative"
        )


def _validate_field_keys(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("field_keys must be a tuple")
    if any(type(key) is not FieldMaterializationKey for key in value):
        raise TypeError(
            "field_keys must contain FieldMaterializationKey values"
        )
    if value != tuple(
        sorted(set(value), key=field_materialization_sort_key)
    ):
        raise ValueError("field_keys must be unique and canonically ordered")


def _validate_diagnostics(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("diagnostics must be a tuple")
    if any(type(item) is not ResultDiagnostic for item in value):
        raise TypeError("diagnostics must contain ResultDiagnostic values")


__all__ = [
    "OutputExecutionOutcome",
    "OutputExecutionStatus",
    "OutputRequestExecution",
    "OutputVariableExecution",
    "ResultExecutionReport",
    "execute_output_requests",
]
