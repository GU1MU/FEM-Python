from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fem.application import ModelSession, run_static_preflight
from fem.application.results import build_solve_result_bundle
from fem.solvers import static_linear
from fem_agent.authoring_runtime import provider_safe_authoring_payload
from fem_agent.result_authoring import (
    RESULT_QUERY_SCHEMA_VERSION,
    AcceptedResultReference,
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultComparisonQuery,
    AgentResultLocation,
    AgentResultQueryBridge,
    AgentResultQueryIdentity,
    AgentResultQueryResponse,
    AgentResultScalar,
    AgentResultVariable,
    FakeAgentResultQueryPort,
    ResultAuthoringError,
    result_comparison_tool_schema,
)
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.gui.test_agent_result_query_phase_a7 import STEP_NAME, _solved_session
from tests.test_agent_authoring_phase_a5 import _analysis, _session


def _reference(session: ModelSession, run_id: str) -> AcceptedResultReference:
    source, generation = session.result_identity_for(run_id)
    return AcceptedResultReference(
        AcceptedResultSource(
            result_id=source.result_id,
            session_id=source.session_id,
            artifact_id=source.artifact_id,
            model_revision=source.model_revision,
            step_name=source.step_name,
            run_id=source.run_id,
        ),
        generation,
    )


def _comparison_query(
    baseline: AcceptedResultReference,
    candidate: AcceptedResultReference,
    *,
    component: str = "Magnitude",
) -> AgentResultComparisonQuery:
    return AgentResultComparisonQuery(
        AgentResultQueryIdentity(
            AgentResultVariable.DISPLACEMENT,
            component,
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
        ),
        baseline,
        candidate,
    )


def _solve_again_with_double_load(
    session: ModelSession,
) -> tuple[AcceptedResultReference, AcceptedResultReference]:
    first_run_id = session.snapshot().displayed_result_run_id
    assert first_run_id is not None
    baseline = _reference(session, first_run_id)
    snapshot = session.snapshot()
    analysis = _analysis()
    doubled = replace(
        analysis,
        loads=(replace(analysis.loads[0], vector=(20.0, 0.0)),),
    )
    session.replace_model_definitions(
        snapshot.materials,
        snapshot.sections,
        snapshot.assignments,
        (doubled.to_step(),),
    )
    validation = session.prepare_validation(STEP_NAME)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert session.accept_validation(validation.token, report).accepted
    task = session.prepare_solve(STEP_NAME, "作业-比较候选")
    assert session.begin_run(task.token).accepted
    result = static_linear.solve(task.model, task.step_name, name="作业-比较候选")
    assert session.accept_run_succeeded(
        task.token,
        build_solve_result_bundle(task, result),
    ).accepted
    return baseline, _reference(session, task.run_id)


def _controller(session: ModelSession):
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot())
    return create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )


def test_comparison_query_and_schema_are_closed_and_round_trip_provider_safe() -> None:
    session = _solved_session()
    baseline, candidate = _solve_again_with_double_load(session)
    request = _comparison_query(baseline, candidate)

    assert AgentResultComparisonQuery.from_dict(
        json.loads(request.to_json())
    ) == request
    widened = request.to_dict()
    widened["unknown"] = True
    with pytest.raises(ResultAuthoringError):
        AgentResultComparisonQuery.from_dict(widened)
    same_run = request.to_dict()
    same_run["candidate"] = request.baseline.to_dict()
    with pytest.raises(ResultAuthoringError):
        AgentResultComparisonQuery.from_dict(same_run)
    foreign_session = request.to_dict()
    foreign_session["candidate"]["expected_source"]["session_id"] = "foreign"
    cross_session = AgentResultComparisonQuery.from_dict(foreign_session)
    assert cross_session.candidate.expected_source.session_id == "foreign"

    schema = result_comparison_tool_schema()
    assert schema["name"] == "compare_accepted_results"
    assert schema["input_schema"]["additionalProperties"] is False
    assert (
        schema["input_schema"]["properties"]["baseline"][
            "additionalProperties"
        ]
        is False
    )
    response = SessionResultQueryPort(session).compare(request)
    assert response.ok
    assert json.loads(response.to_json()) == provider_safe_authoring_payload(
        response.to_dict()
    )


def test_two_accepted_runs_compare_in_order_with_signed_delta_and_no_mutation() -> None:
    session = _solved_session()
    baseline, candidate = _solve_again_with_double_load(session)
    port = SessionResultQueryPort(session)
    before = (
        session.session_revision,
        session.snapshot().selected_run_id,
        session.snapshot().displayed_result_run_id,
    )

    forward = port.compare(_comparison_query(baseline, candidate))
    reverse = port.compare(_comparison_query(candidate, baseline))

    assert forward.ok and reverse.ok
    assert forward.comparison.baseline.source.run_id == baseline.expected_source.run_id
    assert forward.comparison.candidate.source.run_id == candidate.expected_source.run_id
    assert forward.comparison.delta > 0.0
    assert forward.comparison.direction == "increased"
    assert reverse.comparison.delta == pytest.approx(-forward.comparison.delta)
    assert reverse.comparison.direction == "decreased"
    assert (
        session.session_revision,
        session.snapshot().selected_run_id,
        session.snapshot().displayed_result_run_id,
    ) == before


def test_zero_baseline_has_null_relative_change_and_finite_delta() -> None:
    baseline_source = AcceptedResultSource("r0", "s", "a", 1, "Step", "run-0")
    candidate_source = AcceptedResultSource("r1", "s", "a", 1, "Step", "run-1")
    baseline_ref = AcceptedResultReference(baseline_source, 0)
    candidate_ref = AcceptedResultReference(candidate_source, 0)
    request = _comparison_query(baseline_ref, candidate_ref, component="U1")

    def scalar(source: AcceptedResultSource, value: float) -> AgentResultScalar:
        return AgentResultScalar(
            AgentResultVariable.DISPLACEMENT,
            "U1",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
            value,
            "mm",
            source,
            0,
            AgentResultLocation("node", node_id=1),
        )

    baseline_query = request.result_query(baseline_ref)
    candidate_query = request.result_query(candidate_ref)
    port = FakeAgentResultQueryPort(
        {
            baseline_query: AgentResultQueryResponse.success(
                scalar(baseline_source, 0.0)
            ),
            candidate_query: AgentResultQueryResponse.success(
                scalar(candidate_source, 5.0)
            ),
        }
    )

    comparison = AgentResultQueryBridge(port).compare(request).comparison

    assert comparison.delta == 5.0
    assert comparison.absolute_delta == 5.0
    assert comparison.relative_change_percent is None
    assert comparison.baseline_is_zero is True
    assert comparison.direction == "increased"


def test_bridge_preserves_query_only_ports_and_rejects_wrong_provenance() -> None:
    baseline_source = AcceptedResultSource("r0", "s", "a", 1, "Step", "run-0")
    candidate_source = AcceptedResultSource("r1", "s", "a", 1, "Step", "run-1")
    request = _comparison_query(
        AcceptedResultReference(baseline_source, 0),
        AcceptedResultReference(candidate_source, 0),
        component="U1",
    )

    class QueryOnlyPort:
        def catalog(self, run_id=None):
            return FakeAgentResultQueryPort().catalog(run_id)

        def query(self, result_query):
            return AgentResultQueryResponse.failure(
                "result.query.unavailable",
                "Unavailable.",
                retryable=False,
                clarification_required=True,
            )

    unsupported = AgentResultQueryBridge(QueryOnlyPort()).compare(request)
    assert unsupported.diagnostics[0].code == "result.comparison.unsupported"

    def scalar(source: AcceptedResultSource) -> AgentResultScalar:
        return AgentResultScalar(
            AgentResultVariable.DISPLACEMENT,
            "U1",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
            1.0,
            "mm",
            source,
            0,
            AgentResultLocation("node", node_id=1),
        )

    wrong_source = replace(baseline_source, result_id="wrong")
    port = FakeAgentResultQueryPort(
        {
            request.result_query(request.baseline): (
                AgentResultQueryResponse.success(scalar(wrong_source))
            ),
            request.result_query(request.candidate): (
                AgentResultQueryResponse.success(scalar(candidate_source))
            ),
        }
    )
    stale = AgentResultQueryBridge(port).compare(request)
    assert stale.comparison is None
    assert stale.diagnostics[0].code == "result.comparison.stale"
    assert stale.diagnostics[0].retryable


def test_comparison_fails_closed_for_foreign_stale_and_unavailable_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _solved_session()
    baseline, candidate = _solve_again_with_double_load(session)
    port = SessionResultQueryPort(session)

    foreign_baseline = replace(baseline.expected_source, session_id="foreign")
    foreign_candidate = replace(candidate.expected_source, session_id="foreign")
    foreign = port.compare(
        _comparison_query(
            AcceptedResultReference(
                foreign_baseline,
                baseline.expected_materialization_generation,
            ),
            AcceptedResultReference(
                foreign_candidate,
                candidate.expected_materialization_generation,
            ),
        )
    )
    stale = port.compare(
        _comparison_query(
            replace(
                baseline,
                expected_materialization_generation=(
                    baseline.expected_materialization_generation + 1
                ),
            ),
            candidate,
        )
    )
    stale_source = port.compare(
        _comparison_query(
            AcceptedResultReference(
                replace(baseline.expected_source, result_id="stale-result"),
                baseline.expected_materialization_generation,
            ),
            candidate,
        )
    )
    unavailable = port.compare(
        _comparison_query(baseline, candidate, component="U9")
    )
    historical_region = AgentResultComparisonQuery(
        replace(
            _comparison_query(baseline, candidate).query,
            region="边-固定端",
        ),
        baseline,
        candidate,
    )
    unpublished = port.compare(historical_region)

    assert foreign.comparison is None
    assert foreign.diagnostics[0].code == "result.comparison.source_unavailable"
    assert foreign.diagnostics[0].clarification_required
    assert stale.comparison is None
    assert stale.diagnostics[0].code == "result.comparison.stale"
    assert stale.diagnostics[0].retryable
    assert stale_source.comparison is None
    assert stale_source.diagnostics[0].code == "result.comparison.stale"
    assert unavailable.comparison is None
    assert unavailable.diagnostics[0].code == "result.comparison.not_comparable"
    assert unavailable.diagnostics[0].clarification_required
    assert unpublished.comparison is None
    assert unpublished.diagnostics[0].code == "result.comparison.not_comparable"
    assert unpublished.diagnostics[0].clarification_required

    original_identity = session.result_identity_for
    calls = 0

    def changing_identity(run_id: str):
        nonlocal calls
        calls += 1
        identity = original_identity(run_id)
        return None if calls == 8 else identity

    monkeypatch.setattr(session, "result_identity_for", changing_identity)
    changed = port.compare(_comparison_query(baseline, candidate))
    assert changed.comparison is None
    assert changed.diagnostics[0].code == "result.comparison.stale"


def test_comparison_tool_visibility_tracks_retained_result_count() -> None:
    empty = _session()
    one = _solved_session()
    two = _solved_session()
    _solve_again_with_double_load(two)

    assert "compare_accepted_results" not in {
        item.name for item in _controller(empty).definitions
    }
    assert "compare_accepted_results" not in {
        item.name for item in _controller(one).definitions
    }
    two_names = {item.name for item in _controller(two).definitions}
    assert {
        "compare_accepted_results",
        "run_native_preflight",
        "prepare_solve_proposal",
    }.issubset(two_names)

    snapshot = two.snapshot()
    two.replace_model_definitions(
        snapshot.materials,
        snapshot.sections,
        snapshot.assignments,
        snapshot.steps,
    )
    cleared = _controller(two)
    assert cleared.stage.value == "preflight_ready"
    assert "compare_accepted_results" in {
        item.name for item in cleared.definitions
    }

    outcome = cleared.dispatch(
        "compare_accepted_results",
        _comparison_query(
            _reference(two, snapshot.runs[0].run_id),
            _reference(two, snapshot.runs[1].run_id),
        ).to_dict(),
        ToolExecutionContext(
            two.session_id,
            two.session_revision,
            "compare-cleared",
        ),
    )
    assert outcome.ok
    assert outcome.data["schema_version"] == RESULT_QUERY_SCHEMA_VERSION
