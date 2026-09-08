from __future__ import annotations

from fem.application import ScopedDefinitionBatch, run_static_preflight
from fem.application.results import ResultSourceKey, build_solve_result_bundle
from fem.solvers import static_linear
from fem_agent.result_authoring import (
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultQuery,
    AgentResultVariable,
)
from tests.helpers.agent_session_fixtures import (
    _a5_analysis as _analysis,
    _a5_session as _session,
)


STATIC_STEP_NAME = "分析步-静力"


def make_solved_session():
    session = _session()
    snapshot = session.snapshot()
    delta = session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            session.session_revision,
            tuple(snapshot.named_regions.values()),
            snapshot.materials,
            snapshot.sections,
            snapshot.assignments,
            (_analysis().to_step(),),
        )
    )
    assert delta.accepted

    validation = session.prepare_validation(STATIC_STEP_NAME)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert report.passed
    assert session.accept_validation(validation.token, report).accepted

    solve_task = session.prepare_solve(STATIC_STEP_NAME, "作业-A7")
    assert session.begin_run(solve_task.token).accepted
    result = static_linear.solve(
        solve_task.model,
        solve_task.step_name,
        name="作业-A7",
    )
    assert session.accept_run_succeeded(
        solve_task.token,
        build_solve_result_bundle(solve_task, result),
    ).accepted
    return session


def make_accepted_result_source(value: ResultSourceKey) -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id=value.result_id,
        session_id=value.session_id,
        artifact_id=value.artifact_id,
        model_revision=value.model_revision,
        step_name=value.step_name,
        run_id=value.run_id,
    )


def build_result_query(
    session,
    *,
    variable: AgentResultVariable,
    component: str,
    position: str,
    region: str,
    aggregation: AgentResultAggregation,
) -> AgentResultQuery:
    source, generation = session.current_result_identity()
    return AgentResultQuery(
        variable=variable,
        component=component,
        position=position,
        region=region,
        aggregation=aggregation,
        expected_source=make_accepted_result_source(source),
        expected_materialization_generation=generation,
    )
