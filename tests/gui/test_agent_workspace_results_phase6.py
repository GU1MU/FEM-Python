from __future__ import annotations

from dataclasses import replace
import math

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.results import FieldState
from fem_agent.result_authoring import (
    AcceptedResultReference,
    AgentResultAggregation,
    AgentResultComparisonQuery,
    AgentResultQuery,
    AgentResultQueryBridge,
    AgentResultQueryIdentity,
    AgentResultVariable,
    FakeAgentResultQueryPort,
    ResultAuthoringError,
)
from fem_agent.workspace_catalog import WorkspaceCatalogBridge, WorkspaceDocumentIdentity
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_workspace_catalog import FEMWorkspaceCatalogPort
from fem_gui.workspace import FEMWorkspace
from tests.gui.test_agent_result_query_phase_a7 import _solved_session
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)
from tests.io.test_result_archive_v1 import _snapshot


def _target(document) -> WorkspaceDocumentIdentity:
    return WorkspaceDocumentIdentity(
        str(document.document_id),
        document.session.session_id,
    )


def _maximum_displacement(catalog) -> AgentResultQuery:
    return AgentResultQuery(
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
        expected_source=catalog.source,
        expected_materialization_generation=(
            catalog.materialization_generation
        ),
    )


def _comparison_request(baseline_catalog, candidate_catalog):
    return AgentResultComparisonQuery(
        AgentResultQueryIdentity(
            AgentResultVariable.DISPLACEMENT,
            "Magnitude",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
        ),
        AcceptedResultReference(
            baseline_catalog.source,
            baseline_catalog.materialization_generation,
        ),
        AcceptedResultReference(
            candidate_catalog.source,
            candidate_catalog.materialization_generation,
        ),
    )


def _two_document_result_context():
    baseline_session = _solved_session()
    candidate_session = _solved_session()
    workspace = FEMWorkspace()
    baseline_document = workspace.add_model(
        baseline_session,
        baseline_session.projection_snapshot(),
        display_name="Baseline",
    )
    candidate_document = workspace.add_model(
        candidate_session,
        candidate_session.projection_snapshot(),
        display_name="Candidate",
    )
    workspace.activate(candidate_document)
    port = SessionResultQueryPort(candidate_session, workspace)
    baseline_response = port.catalog(target=_target(baseline_document))
    candidate_response = port.catalog(target=_target(candidate_document))
    assert baseline_response.ok and baseline_response.catalog is not None
    assert candidate_response.ok and candidate_response.catalog is not None
    return (
        workspace,
        baseline_document,
        candidate_document,
        port,
        baseline_response.catalog,
        candidate_response.catalog,
    )


def test_workspace_catalog_routes_result_reads_and_cross_document_compare() -> None:
    baseline_session = _solved_session()
    candidate_session = _solved_session()
    workspace = FEMWorkspace()
    baseline_document = workspace.add_model(
        baseline_session,
        baseline_session.projection_snapshot(),
        display_name="Baseline",
    )
    candidate_document = workspace.add_model(
        candidate_session,
        candidate_session.projection_snapshot(),
        display_name="Candidate",
    )
    workspace.activate(candidate_document)
    active_before = workspace.active_document_id
    baseline_before = baseline_session.projection_snapshot()
    candidate_before = candidate_session.projection_snapshot()

    catalog_bridge = WorkspaceCatalogBridge(FEMWorkspaceCatalogPort(workspace))
    workspace_catalog = catalog_bridge.catalog()
    summaries = {item.target: item for item in workspace_catalog.documents}
    assert summaries[_target(baseline_document)].run_count == 1
    assert summaries[_target(baseline_document)].result_count == 1
    assert summaries[_target(candidate_document)].run_count == 1
    assert summaries[_target(candidate_document)].result_count == 1

    port = SessionResultQueryPort(candidate_session, workspace)
    baseline_runs = port.analysis_runs(target=_target(baseline_document))
    candidate_runs = port.analysis_runs(target=_target(candidate_document))
    assert baseline_runs.document_id == str(baseline_document.document_id)
    assert candidate_runs.document_id == str(candidate_document.document_id)

    baseline_catalog_response = port.catalog(
        baseline_runs.runs[0].run_id,
        target=_target(baseline_document),
    )
    candidate_catalog_response = port.catalog(
        candidate_runs.runs[0].run_id,
        target=_target(candidate_document),
    )
    assert baseline_catalog_response.ok
    assert candidate_catalog_response.ok
    baseline_catalog = baseline_catalog_response.catalog
    candidate_catalog = candidate_catalog_response.catalog
    assert baseline_catalog is not None
    assert candidate_catalog is not None

    baseline_scalar = port.query(_maximum_displacement(baseline_catalog))
    candidate_scalar = port.query(_maximum_displacement(candidate_catalog))
    assert baseline_scalar.ok
    assert candidate_scalar.ok

    comparison = port.compare(
        _comparison_request(baseline_catalog, candidate_catalog)
    )
    assert comparison.ok
    assert comparison.comparison is not None
    assert comparison.comparison.delta == math.fsum(
        (
            comparison.comparison.candidate.value,
            -comparison.comparison.baseline.value,
        )
    )
    assert comparison.comparison.baseline.source.session_id == baseline_session.session_id
    assert comparison.comparison.candidate.source.session_id == candidate_session.session_id

    assert workspace.active_document_id == active_before
    assert baseline_session.projection_snapshot() == baseline_before
    assert candidate_session.projection_snapshot() == candidate_before


def test_exact_target_is_strict_and_omission_reads_bound_session() -> None:
    (
        _workspace,
        baseline_document,
        candidate_document,
        port,
        baseline_catalog,
        candidate_catalog,
    ) = _two_document_result_context()
    bridge = AgentResultQueryBridge(port)

    exact = bridge.catalog(target=_target(baseline_document).to_dict())
    assert exact.ok and exact.catalog is not None
    assert exact.catalog.source.session_id == baseline_catalog.source.session_id
    fake = FakeAgentResultQueryPort(catalog_response=exact)
    fake_bridge = AgentResultQueryBridge(fake)
    fake_bridge.catalog(target=_target(baseline_document).to_dict())
    fake_bridge.catalog()
    assert fake.catalog_targets == [_target(baseline_document), None]

    crossed = bridge.catalog(
        target={
            "document_id": str(baseline_document.document_id),
            "session_id": candidate_document.session.session_id,
        }
    )
    assert not crossed.ok
    assert crossed.diagnostics[0].code == "result.catalog.target_unavailable"

    current = bridge.catalog()
    assert current.ok and current.catalog is not None
    assert current.catalog.source.session_id == candidate_catalog.source.session_id

    invalid_values = ("", " target", "target ", "bad\x00target", "x" * 129)
    for invalid in invalid_values:
        with pytest.raises(ResultAuthoringError):
            bridge.catalog(
                target={
                    "document_id": invalid,
                    "session_id": candidate_document.session.session_id,
                }
            )
        with pytest.raises(ResultAuthoringError):
            bridge.catalog(
                target={
                    "document_id": str(candidate_document.document_id),
                    "session_id": invalid,
                }
            )


def test_duplicate_workspace_source_session_fails_query_closed() -> None:
    session = _solved_session()
    workspace = FEMWorkspace()
    first = workspace.add_model(session, session.projection_snapshot())
    port = SessionResultQueryPort(session, workspace)
    response = port.catalog(target=_target(first))
    assert response.ok and response.catalog is not None
    workspace.add_model(session, session.projection_snapshot())

    query = port.query(_maximum_displacement(response.catalog))

    assert not query.ok
    assert query.diagnostics[0].code == "result.query.source_unavailable"


def test_cross_session_same_run_id_is_valid_but_same_pair_is_rejected() -> None:
    (
        _workspace,
        _baseline_document,
        _candidate_document,
        _port,
        baseline_catalog,
        candidate_catalog,
    ) = _two_document_result_context()
    same_run_candidate = replace(
        candidate_catalog.source,
        run_id=baseline_catalog.source.run_id,
    )

    request = AgentResultComparisonQuery(
        AgentResultQueryIdentity(
            AgentResultVariable.DISPLACEMENT,
            "Magnitude",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
        ),
        AcceptedResultReference(
            baseline_catalog.source,
            baseline_catalog.materialization_generation,
        ),
        AcceptedResultReference(
            same_run_candidate,
            candidate_catalog.materialization_generation,
        ),
    )
    assert request.baseline.expected_source.run_id == request.candidate.expected_source.run_id
    with pytest.raises(ResultAuthoringError):
        replace(request, candidate=request.baseline)


@pytest.mark.parametrize("change", ["identity", "membership"])
def test_cross_session_comparison_fails_stale_on_toctou(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        _baseline_document,
        candidate_document,
        port,
        baseline_catalog,
        candidate_catalog,
    ) = _two_document_result_context()
    request = _comparison_request(baseline_catalog, candidate_catalog)
    session = candidate_document.session
    original_identity = session.result_identity_for
    calls = 0

    def changing_identity(run_id: str):
        nonlocal calls
        calls += 1
        identity = original_identity(run_id)
        if calls == 4:
            if change == "identity":
                return None
            workspace.remove(candidate_document)
        return identity

    monkeypatch.setattr(session, "result_identity_for", changing_identity)
    response = port.compare(request)

    assert not response.ok
    assert response.diagnostics[0].code == "result.comparison.stale"


def test_cross_session_comparison_requires_exact_unit_string(tmp_path) -> None:
    baseline_archive = _snapshot(
        make_continuum_nodal_semantics_result,
        "phase6-unit-baseline",
    )
    candidate_archive = _snapshot(
        make_continuum_nodal_semantics_result,
        "phase6-unit-candidate",
    )
    candidate_archive = replace(
        candidate_archive,
        model_projection=replace(
            candidate_archive.model_projection,
            unit_context=UnitContext("mm", "N", "Pa"),
        ),
        unit_context=UnitContext("mm", "N", "Pa"),
    )
    baseline_session = ModelSession()
    candidate_session = ModelSession()
    assert baseline_session.replace_from_result_archive(
        baseline_archive,
        path=tmp_path / "baseline.femres",
    ).accepted
    assert candidate_session.replace_from_result_archive(
        candidate_archive,
        path=tmp_path / "candidate.femres",
    ).accepted
    workspace = FEMWorkspace()
    baseline_document = workspace.add_result(
        baseline_session,
        baseline_session.projection_snapshot(),
    )
    candidate_document = workspace.add_result(
        candidate_session,
        candidate_session.projection_snapshot(),
    )
    port = SessionResultQueryPort(candidate_session, workspace)
    baseline = port.catalog(target=_target(baseline_document)).catalog
    candidate = port.catalog(target=_target(candidate_document)).catalog
    assert baseline is not None and candidate is not None

    response = port.compare(_comparison_request(baseline, candidate))

    assert not response.ok
    assert response.diagnostics[0].code == "result.comparison.not_comparable"


def test_active_result_only_document_keeps_global_result_reads_available(
    tmp_path,
) -> None:
    result_session = ModelSession()
    archive = _snapshot(make_continuum_nodal_semantics_result, "phase6-result")
    assert result_session.replace_from_result_archive(
        archive,
        path=tmp_path / "phase6-result.femres",
    ).accepted
    workspace = FEMWorkspace()
    result_document = workspace.add_result(
        result_session,
        result_session.projection_snapshot(),
        display_name="Detached result",
    )
    workspace.activate(result_document)

    workspace_bridge = WorkspaceCatalogBridge(FEMWorkspaceCatalogPort(workspace))
    result_bridge = AgentResultQueryBridge(
        SessionResultQueryPort(result_session, workspace)
    )
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(result_session, lambda: None)
    )
    controller = create_session_authoring_workflow_controller(
        result_session,
        authoring_bridge,
        result_bridge,
        workspace_catalog_bridge=workspace_bridge,
    )

    names = {item.name for item in controller.definitions}
    assert names == {
        "read_workspace_documents",
        "read_analysis_run_catalog",
        "read_accepted_result_catalog",
        "query_accepted_result",
    }
    summary = workspace_bridge.catalog().documents[0]
    assert summary.source_kind == "result"
    assert (summary.run_count, summary.result_count) == (1, 1)

    runs = result_bridge.analysis_runs(target=_target(result_document))
    response = result_bridge.catalog(
        runs.runs[0].run_id,
        target=_target(result_document),
    )
    assert response.ok
    assert response.catalog is not None
    scalar = result_bridge.query(_maximum_displacement(response.catalog).to_dict())
    assert scalar.ok
    assert scalar.scalar is not None
    assert scalar.scalar.source.session_id == result_session.session_id
    assert workspace.active_document() is result_document


def test_result_only_catalog_regions_and_queries_do_not_materialize(tmp_path) -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "phase6-regions")
    topology = archive.model_projection.topology
    archive = replace(
        archive,
        model_projection=replace(
            archive.model_projection,
            named_region_node_ids={
                **archive.model_projection.named_region_node_ids,
                "archive-node-region": (topology.node_ids[0],),
            },
            named_region_element_ids={
                **archive.model_projection.named_region_element_ids,
                "archive-element-region": (topology.element_ids[0],),
            },
        ),
    )
    session = ModelSession()
    assert session.replace_from_result_archive(
        archive,
        path=tmp_path / "regions.femres",
    ).accepted
    workspace = FEMWorkspace()
    document = workspace.add_result(session, session.projection_snapshot())
    port = SessionResultQueryPort(session, workspace)
    identity_before = session.current_result_identity()

    response = port.catalog(target=_target(document))
    assert response.ok and response.catalog is not None
    catalog = response.catalog
    provider = session.current_result_provider()
    assert provider is not None
    expected_variables = {
        item.descriptor.field_id.variable.value
        for item in provider.catalog().fields
        if item.state is FieldState.READY
        and item.descriptor.field_id.variable.value in {"U", "RF", "S"}
    }
    assert {item.variable.value for item in catalog.fields} == expected_variables
    assert "archive-node-region" in catalog.nodal_regions
    assert "archive-element-region" in catalog.element_regions

    node_query = replace(
        _maximum_displacement(catalog),
        region="archive-node-region",
    )
    element_query = AgentResultQuery(
        AgentResultVariable.STRESS,
        "Mises",
        "element_nodal",
        "archive-element-region",
        AgentResultAggregation.MAXIMUM,
        catalog.source,
        catalog.materialization_generation,
    )
    assert port.query(node_query).ok
    assert port.query(element_query).ok
    assert session.current_result_identity() == identity_before
