from __future__ import annotations

import fem.application.results as results
from fem.application.results.execution import (
    ResultExecutionReport,
    execute_output_requests,
)
from fem.application.results.output_requests import (
    ResultCapabilityCatalog,
    project_output_request,
)
from fem.application.results.provider import (
    ResultProvider,
    build_result_provider,
    restore_result_provider,
)
from fem.application.results.query import ResultQuery
from fem.application.results.topology import (
    ResultCellKind,
    ResultFieldTopology,
    ResultValueLayout,
    project_scalar_field_topology,
)
from fem.application.results.registry import (
    ElementResultProfile,
    catalog_entries,
    classify_result_model,
)
from fem.application.results.workflow import (
    SolveResultBundle,
    build_solve_result_bundle,
    validate_solve_result_model_identity,
)


def test_result_package_exports_provider_query_projection_and_execution() -> None:
    expected = {
        "ElementResultProfile": ElementResultProfile,
        "ResultCapabilityCatalog": ResultCapabilityCatalog,
        "ResultCellKind": ResultCellKind,
        "ResultExecutionReport": ResultExecutionReport,
        "ResultFieldTopology": ResultFieldTopology,
        "ResultProvider": ResultProvider,
        "ResultQuery": ResultQuery,
        "ResultValueLayout": ResultValueLayout,
        "SolveResultBundle": SolveResultBundle,
        "build_result_provider": build_result_provider,
        "build_solve_result_bundle": build_solve_result_bundle,
        "catalog_entries": catalog_entries,
        "classify_result_model": classify_result_model,
        "execute_output_requests": execute_output_requests,
        "project_output_request": project_output_request,
        "project_scalar_field_topology": project_scalar_field_topology,
        "restore_result_provider": restore_result_provider,
        "validate_solve_result_model_identity": (
            validate_solve_result_model_identity
        ),
    }

    assert set(expected).issubset(results.__all__)
    assert {
        name: getattr(results, name)
        for name in expected
    } == expected
