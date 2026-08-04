from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from fem.io.inp import read
from fem.application.results import (
    OutputExecutionStatus,
    ResultCapabilityCatalog,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    execute_output_requests,
    project_output_request,
)
from fem.core.model import OutputRequest
from fem.solvers import static_linear


_STANDARD = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-imported-truss",
        session_id="session-integration",
        artifact_id="artifact-imported-truss",
        model_revision=1,
        step_name="Tension",
        run_id="run-imported-truss",
    )


def test_imported_abaqus_outputs_match_native_projection_and_execution() -> None:
    model = read(_STANDARD / "truss2_tension.inp")
    step = next(item for item in model.steps if item.name == "Tension")
    imported_requests = tuple(step.outputs)
    preserved_requests = deepcopy(imported_requests)
    native_requests = tuple(
        OutputRequest(
            request.kind,
            request.target,
            request.variables,
            dict(request.metadata),
        )
        for request in imported_requests
    )
    result = static_linear.solve(model, step)
    source = _source()
    imported_provider = build_result_provider(source, result)
    native_provider = build_result_provider(source, result)
    imported_catalog = ResultCapabilityCatalog.from_profile(
        imported_provider.profile
    )
    native_catalog = ResultCapabilityCatalog.from_profile(
        native_provider.profile
    )

    assert imported_catalog == native_catalog
    for request_index, (imported, native) in enumerate(
        zip(imported_requests, native_requests, strict=True)
    ):
        imported_projection = project_output_request(
            imported,
            imported_catalog,
            request_index=request_index,
        )
        native_projection = project_output_request(
            native,
            native_catalog,
            request_index=request_index,
        )
        assert (
            imported_projection.executable_request
            == native_projection.executable_request
        )
        assert imported_projection.variables == native_projection.variables
        assert (
            imported_projection.diagnostics
            == native_projection.diagnostics
            == ()
        )

    imported_outcome = execute_output_requests(
        imported_provider,
        imported_requests,
    )
    native_outcome = execute_output_requests(
        native_provider,
        native_requests,
    )

    assert imported_requests == preserved_requests
    assert all(
        request.source_evidence is not None
        and request.source_evidence.source_kind == "abaqus"
        for request in imported_requests
    )
    assert imported_outcome.report == native_outcome.report
    assert tuple(
        request.status for request in imported_outcome.report.requests
    ) == (
        OutputExecutionStatus.EXECUTED,
        OutputExecutionStatus.EXECUTED,
    )
    assert len(imported_outcome.eager_patch.fields) == 1
    eager_stress = imported_outcome.eager_patch.fields[0]
    assert (
        eager_stress.key.request.field_id.variable
        is ResultVariable.S
    )

    stress_key = (
        imported_outcome.report.requests[1]
        .variables[0]
        .field_keys[0]
    )
    lazy_stress = imported_provider.materialize((stress_key,)).fields[0]
    assert eager_stress.descriptor == lazy_stress.descriptor
    assert eager_stress.locations == lazy_stress.locations
    np.testing.assert_allclose(eager_stress.values, lazy_stress.values)
    np.testing.assert_allclose(eager_stress.values[:, 0], (2.10e8,))
