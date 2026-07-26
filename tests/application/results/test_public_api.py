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
from fem.application.results.registry import (
    ElementResultProfile,
    catalog_entries,
    classify_result_model,
)


def test_result_package_exports_provider_query_projection_and_execution() -> None:
    expected = {
        "ElementResultProfile": ElementResultProfile,
        "ResultCapabilityCatalog": ResultCapabilityCatalog,
        "ResultExecutionReport": ResultExecutionReport,
        "ResultProvider": ResultProvider,
        "ResultQuery": ResultQuery,
        "build_result_provider": build_result_provider,
        "catalog_entries": catalog_entries,
        "classify_result_model": classify_result_model,
        "execute_output_requests": execute_output_requests,
        "project_output_request": project_output_request,
        "restore_result_provider": restore_result_provider,
    }

    assert set(expected).issubset(results.__all__)
    assert {
        name: getattr(results, name)
        for name in expected
    } == expected
