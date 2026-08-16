from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import (
    ScopedDefinitionBatch,
    run_static_preflight,
)
from fem.application.results import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultQuery,
    ResultQueryValidationError,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    build_solve_result_bundle,
)
from fem.core.model import Edge, ElementEdge, ElementSet, NodeSet
from fem.solvers import static_linear
from fem_agent.result_authoring import (
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultQuery,
    AgentResultVariable,
)
from fem_gui.agent_authoring import SessionResultQueryPort
from fem_gui.main_window import FEMMainWindow
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)
from tests.helpers.agent_session_fixtures import (
    _a5_analysis as _analysis,
    _a5_session as _session,
)


STEP_NAME = "分析步-静力"


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _solved_session():
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

    validation = session.prepare_validation(STEP_NAME)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert report.passed
    assert session.accept_validation(validation.token, report).accepted

    solve_task = session.prepare_solve(STEP_NAME, "作业-A7")
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


def _source(value: ResultSourceKey) -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id=value.result_id,
        session_id=value.session_id,
        artifact_id=value.artifact_id,
        model_revision=value.model_revision,
        step_name=value.step_name,
        run_id=value.run_id,
    )


def _query(
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
        expected_source=_source(source),
        expected_materialization_generation=generation,
    )


def _native_records(session, request: AgentResultQuery):
    provider = session.current_result_provider()
    availability = next(
        item
        for item in provider.catalog().fields
        if (
            item.descriptor.field_id.variable.value
            == request.variable.value
            and item.descriptor.field_id.position.value == request.position
        )
    )
    if request.variable is AgentResultVariable.STRESS:
        ids = (
            ()
            if request.region == "all_elements"
            else provider.named_region_element_ids(request.region)
        )
        query = ResultQuery(
            availability.key,
            request.component,
            element_ids=ids,
        )
    else:
        ids = (
            ()
            if request.region == "all_nodes"
            else provider.named_region_node_ids(request.region)
        )
        query = ResultQuery(
            availability.key,
            request.component,
            node_ids=ids,
        )
    return provider.query(query).records


def test_a7_catalog_exposes_only_bounded_ready_identity_and_units() -> None:
    session = _solved_session()

    response = SessionResultQueryPort(session).catalog()

    assert response.ok
    catalog = response.catalog
    source, generation = session.current_result_identity()
    assert catalog.source == _source(source)
    assert catalog.materialization_generation == generation == 0
    assert tuple(
        (
            item.variable.value,
            item.position,
            item.components,
            item.unit,
        )
        for item in catalog.fields
    ) == (
        ("U", "node", ("U1", "U2", "Magnitude"), "mm"),
        ("RF", "node", ("RF1", "RF2", "Magnitude"), "N"),
        (
            "S",
            "element_nodal",
            (
                "S11",
                "S22",
                "S33",
                "S12",
                "Mises",
                "MaxPrincipal",
                "MidPrincipal",
                "MinPrincipal",
            ),
            "MPa",
        ),
    )
    assert catalog.nodal_regions == (
        "all_nodes",
        "边-固定端",
        "边-加载端",
    )
    assert catalog.element_regions == ("all_elements", "域-板体")
    encoded = str(response.to_dict()).casefold()
    for forbidden in (
        "coordinates",
        "connectivity",
        "records",
        "values",
        "vtk",
        "absolute_path",
    ):
        assert forbidden not in encoded


def test_a7_reads_three_milestone_result_types_with_units_and_identity() -> None:
    session = _solved_session()
    port = SessionResultQueryPort(session)
    displacement = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    stress = _query(
        session,
        variable=AgentResultVariable.STRESS,
        component="Mises",
        position="element_nodal",
        region="域-板体",
        aggregation=AgentResultAggregation.ABSOLUTE_EXTREME,
    )
    reaction = _query(
        session,
        variable=AgentResultVariable.REACTION_FORCE,
        component="RF1",
        position="node",
        region="边-固定端",
        aggregation=AgentResultAggregation.SUM,
    )

    displacement_response = port.query(displacement)
    stress_response = port.query(stress)
    reaction_response = port.query(reaction)

    assert displacement_response.ok
    assert displacement_response.scalar.unit == "mm"
    assert displacement_response.scalar.location.node_id is not None
    assert stress_response.ok
    assert stress_response.scalar.unit == "MPa"
    assert stress_response.scalar.location.element_id in {1, 2}
    assert reaction_response.ok
    assert reaction_response.scalar.unit == "N"
    assert reaction_response.scalar.location is None
    for response in (
        displacement_response,
        stress_response,
        reaction_response,
    ):
        scalar = response.scalar
        assert scalar.source == displacement.expected_source
        assert scalar.materialization_generation == 0
        assert scalar.source.run_id
        assert scalar.source.step_name == STEP_NAME


@pytest.mark.parametrize(
    "aggregation",
    (
        AgentResultAggregation.MINIMUM,
        AgentResultAggregation.MAXIMUM,
        AgentResultAggregation.ABSOLUTE_EXTREME,
    ),
)
def test_a7_stress_extrema_match_native_records_and_keep_signed_value(
    aggregation: AgentResultAggregation,
) -> None:
    session = _solved_session()
    request = _query(
        session,
        variable=AgentResultVariable.STRESS,
        component="S11",
        position="element_nodal",
        region="域-板体",
        aggregation=aggregation,
    )
    records = _native_records(session, request)
    expected = {
        AgentResultAggregation.MINIMUM: lambda: min(
            records,
            key=lambda item: item.value,
        ),
        AgentResultAggregation.MAXIMUM: lambda: max(
            records,
            key=lambda item: item.value,
        ),
        AgentResultAggregation.ABSOLUTE_EXTREME: lambda: max(
            records,
            key=lambda item: abs(item.value),
        ),
    }[aggregation]()

    scalar = SessionResultQueryPort(session).query(request).scalar

    assert scalar.value == pytest.approx(expected.value)
    assert scalar.location.node_id == expected.location.node_id
    assert scalar.location.element_id == expected.location.element_id


def test_a7_fixed_region_reaction_sum_deduplicates_each_node() -> None:
    session = _solved_session()
    provider = session.current_result_provider()
    assert provider.named_region_node_ids("边-固定端") == (4, 1)
    request = _query(
        session,
        variable=AgentResultVariable.REACTION_FORCE,
        component="RF1",
        position="node",
        region="边-固定端",
        aggregation=AgentResultAggregation.SUM,
    )
    expected = math.fsum(item.value for item in _native_records(session, request))

    response = SessionResultQueryPort(session).query(request)

    assert response.scalar.value == pytest.approx(expected)
    assert len(_native_records(session, request)) == 2


def test_a7_public_provider_region_resolution_is_exact_and_fail_closed() -> None:
    result = make_continuum_nodal_semantics_result()
    result.model.node_sets["重复"] = NodeSet("重复", (1, 2, 1))
    result.model.node_sets["空节点"] = NodeSet("空节点", ())
    result.model.element_sets["域"] = ElementSet("域", (1, 2, 1))
    result.model.edges["歧义"] = Edge(
        "歧义",
        (ElementEdge(1, 0, (1, 2)),),
    )
    result.model.node_sets["歧义"] = NodeSet("歧义", (1, 2))
    provider = build_result_provider(
        ResultSourceKey(
            result_id="result-region",
            session_id="session-region",
            artifact_id="artifact-region",
            model_revision=1,
            step_name="Step-1",
            run_id="run-region",
        ),
        result,
    )

    assert provider.named_region_node_ids("重复") == (1, 2)
    assert provider.named_region_element_ids("域") == (1, 2)
    with pytest.raises(ResultQueryValidationError) as ambiguous:
        provider.named_region_node_ids("歧义")
    assert ambiguous.value.code == "result.query.region_ambiguous"
    with pytest.raises(ResultQueryValidationError) as empty:
        provider.named_region_node_ids("空节点")
    assert empty.value.code == "result.query.region_empty"
    with pytest.raises(ResultQueryValidationError) as wrong_entity:
        provider.named_region_node_ids("域")
    assert wrong_entity.value.code == "result.query.region_entity_unsupported"


def test_a7_no_result_component_region_and_position_fail_without_value() -> None:
    empty = SessionResultQueryPort(_session())
    stale_source = AcceptedResultSource(
        "result-none",
        "session-none",
        "artifact-none",
        0,
        STEP_NAME,
        "run-none",
    )
    no_result = empty.query(
        AgentResultQuery(
            AgentResultVariable.DISPLACEMENT,
            "Magnitude",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
            stale_source,
            0,
        )
    )
    assert no_result.scalar is None
    assert no_result.diagnostics[0].code == "result.query.source_unavailable"

    session = _solved_session()
    port = SessionResultQueryPort(session)
    bad_component = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="U9",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    bad_region = _query(
        session,
        variable=AgentResultVariable.REACTION_FORCE,
        component="RF1",
        position="node",
        region="域-板体",
        aggregation=AgentResultAggregation.SUM,
    )
    unpublished_region = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="内部-未发布",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    bad_position = _query(
        session,
        variable=AgentResultVariable.STRESS,
        component="Mises",
        position="centroid",
        region="域-板体",
        aggregation=AgentResultAggregation.MAXIMUM,
    )

    assert (
        port.query(bad_component).diagnostics[0].code
        == "result.query.component_not_available"
    )
    assert (
        port.query(bad_region).diagnostics[0].code
        == "result.query.region_entity_unsupported"
    )
    assert (
        port.query(unpublished_region).diagnostics[0].code
        == "result.query.region_not_published"
    )
    assert (
        port.query(bad_position).diagnostics[0].code
        == "result.query.field_not_available"
    )


def test_a7_rejects_stale_source_generation_and_keeps_historical_run_addressable() -> None:
    session = _solved_session()
    port = SessionResultQueryPort(session)
    request = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    wrong_source = AgentResultQuery(
        request.variable,
        request.component,
        request.position,
        request.region,
        request.aggregation,
        AcceptedResultSource(
            result_id="result-foreign",
            session_id=request.expected_source.session_id,
            artifact_id=request.expected_source.artifact_id,
            model_revision=request.expected_source.model_revision,
            step_name=request.expected_source.step_name,
            run_id=request.expected_source.run_id,
        ),
        request.expected_materialization_generation,
    )
    wrong_generation = AgentResultQuery(
        request.variable,
        request.component,
        request.position,
        request.region,
        request.aggregation,
        request.expected_source,
        request.expected_materialization_generation + 1,
    )

    assert port.query(wrong_source).diagnostics[0].code == "result.query.stale"
    assert (
        port.query(wrong_generation).diagnostics[0].code
        == "result.query.stale"
    )

    solve_task = session.prepare_solve(STEP_NAME, "作业-A7-2")
    assert session.begin_run(solve_task.token).accepted
    result = static_linear.solve(
        solve_task.model,
        solve_task.step_name,
        name="作业-A7-2",
    )
    assert session.accept_run_succeeded(
        solve_task.token,
        build_solve_result_bundle(solve_task, result),
    ).accepted
    historical = port.query(request)
    assert historical.ok
    assert historical.scalar is not None
    assert historical.scalar.source.run_id == request.expected_source.run_id


def test_a7_generation_advance_stales_old_query_without_materializing_it() -> None:
    session = _solved_session()
    provider = session.current_result_provider()
    request = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    lazy_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
        )
    )
    task = session.prepare_result_materialization(
        provider.source.run_id,
        (lazy_key,),
    )
    patch = provider.materialize((lazy_key,))
    assert session.accept_result_materialization(task.token, patch).accepted
    assert session.current_result_identity()[1] == 1

    response = SessionResultQueryPort(session).query(request)

    assert response.scalar is None
    assert response.diagnostics[0].code == "result.query.stale"


def test_a7_rechecks_source_and_generation_after_native_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _solved_session()
    request = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    original_identity = session.result_identity_for
    calls = 0

    def changing_identity(run_id: str):
        nonlocal calls
        calls += 1
        return original_identity(run_id) if calls == 1 else None

    monkeypatch.setattr(
        session,
        "result_identity_for",
        changing_identity,
    )

    response = SessionResultQueryPort(session).query(request)

    assert response.scalar is None
    assert response.diagnostics[0].code == "result.query.stale"
    assert calls == 2


def test_a7_main_window_query_does_not_touch_viewport_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    window = FEMMainWindow()
    try:
        session = _solved_session()
        window.session = session
        window.agent_result_query_bridge = type(
            window.agent_result_query_bridge
        )(SessionResultQueryPort(session))
        request = _query(
            session,
            variable=AgentResultVariable.DISPLACEMENT,
            component="Magnitude",
            position="node",
            region="all_nodes",
            aggregation=AgentResultAggregation.MAXIMUM,
        )
        before = (
            window.result_provider,
            window.result_selection,
            window.viewport._result_render_payload,
            window.viewport._selection_mode,
        )
        calls: list[str] = []
        for method_name in (
            "set_result_render_payload",
            "clear_selection",
            "set_selection_mode",
            "_reset_camera_to_fit",
        ):
            monkeypatch.setattr(
                window.viewport,
                method_name,
                lambda *_args, _name=method_name, **_kwargs: calls.append(
                    _name
                ),
            )

        response = window.agent_result_query_bridge.query(request)

        assert response.ok
        assert calls == []
        assert (
            window.result_provider,
            window.result_selection,
            window.viewport._result_render_payload,
            window.viewport._selection_mode,
        ) == before
    finally:
        window.close()
