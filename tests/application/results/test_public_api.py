from __future__ import annotations

import fem.results as results
from fem.results.execution import (
    ResultExecutionReport,
    execute_output_requests,
)
from fem.results.output_requests import (
    ResultCapabilityCatalog,
    project_output_request,
)
from fem.results.provider import (
    ResultProvider,
    build_result_provider,
    restore_result_provider,
)
from fem.results.inspection import (
    ElementResultInspectionRequest,
    NodeResultInspectionRequest,
    ResultInspectionField,
    ResultInspectionRequest,
    ResultInspectionResult,
    inspect_result_snapshot,
)
from fem.results.query import ResultQuery
from fem.results.topology import (
    ResultCellKind,
    ResultFieldTopology,
    ResultValueLayout,
    project_scalar_field_topology,
)
from fem.results.registry import (
    ElementResultProfile,
    catalog_entries,
    classify_result_model,
)
def test_result_package_exports_domain_services_only() -> None:
    expected = {
        "ElementResultProfile": ElementResultProfile,
        "ElementResultInspectionRequest": ElementResultInspectionRequest,
        "NodeResultInspectionRequest": NodeResultInspectionRequest,
        "ResultCapabilityCatalog": ResultCapabilityCatalog,
        "ResultCellKind": ResultCellKind,
        "ResultExecutionReport": ResultExecutionReport,
        "ResultFieldTopology": ResultFieldTopology,
        "ResultInspectionField": ResultInspectionField,
        "ResultInspectionRequest": ResultInspectionRequest,
        "ResultInspectionResult": ResultInspectionResult,
        "ResultProvider": ResultProvider,
        "ResultProbeKind": results.ResultProbeKind,
        "ResultProbeRequest": results.ResultProbeRequest,
        "ResultProbeResult": results.ResultProbeResult,
        "ResultProbeTarget": results.ResultProbeTarget,
        "ResultQuery": ResultQuery,
        "ResultValueLayout": ResultValueLayout,
        "build_result_provider": build_result_provider,
        "catalog_entries": catalog_entries,
        "classify_result_model": classify_result_model,
        "execute_output_requests": execute_output_requests,
        "inspect_result_snapshot": inspect_result_snapshot,
        "project_output_request": project_output_request,
        "project_scalar_field_topology": project_scalar_field_topology,
        "probe_result": results.probe_result,
        "probe_result_from_query_result": (
            results.probe_result_from_query_result
        ),
        "result_query_for_probe": results.result_query_for_probe,
        "restore_result_provider": restore_result_provider,
    }

    assert set(expected).issubset(results.__all__)
    assert {
        name: getattr(results, name)
        for name in expected
    } == expected
    assert "SolveResultBundle" not in results.__all__
    assert "build_solve_result_bundle" not in results.__all__
    assert "validate_solve_result_model_identity" not in results.__all__
