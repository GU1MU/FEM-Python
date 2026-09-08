import numpy as np
import pytest

from fem.core.model import (
    AnalysisStep,
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
)
from fem.core.result import ModelResult
from fem.post.stress import beam, field, invariants

from fem_agent.diagnostics import DiagnosticCode
from fem_agent.schemas import (
    ResultQuery,
    ResultQueryKind,
    UnitContext,
)
from fem_agent.tools.results import MAX_PROVIDER_SCALARS, query_results
from tests.helpers.mesh_builders import make_tri3_stiffness_mesh
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
)


UNITS = UnitContext(
    length="mm",
    force="N",
    stress="MPa",
    density="tonne/mm^3",
    acceleration="mm/s^2",
)


def _truss_result():
    mesh = make_simple_truss_mesh()
    model = FEMModel(
        mesh=mesh,
        name="query_bar",
        node_sets={
            "fixed": NodeSet("fixed", (1,)),
            "all": NodeSet("all", (1, 2)),
        },
        element_sets={"bar": ElementSet("bar", (1,))},
    )
    step = AnalysisStep("pull")
    displacement = np.zeros(mesh.num_dofs)
    reactions = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 0)] = -0.5
    displacement[mesh.global_dof(2, 1)] = 0.4
    displacement[mesh.global_dof(2, 2)] = 0.3
    reactions[mesh.global_dof(1, 0)] = -100.0
    reactions[mesh.global_dof(2, 0)] = 20.0
    return ModelResult(model, step, displacement, reactions)


def test_query_results_matches_direct_displacement_and_reaction_apis():
    result = _truss_result()
    queries = (
        ResultQuery(
            ResultQueryKind.DISPLACEMENT_COMPONENT,
            node_id=2,
            component=1,
        ),
        ResultQuery(ResultQueryKind.DISPLACEMENT_MAGNITUDE, node_id=2),
        ResultQuery(
            ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
            component=1,
            node_set="all",
        ),
        ResultQuery(
            ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
            node_set="all",
        ),
        ResultQuery(
            ResultQueryKind.REACTION_COMPONENT,
            node_id=1,
            component=1,
        ),
        ResultQuery(
            ResultQueryKind.REACTION_SUM,
            component=1,
            node_set="all",
        ),
    )

    summary = query_results(
        result,
        queries,
        run_id="run-1",
        unit_context=UNITS,
    )

    assert summary.finite_vectors is True
    assert summary.diagnostics == ()
    assert len(summary.scalars) == len(queries)
    assert summary.scalars[0].value == result.nodal_displacement(2, 1)
    assert summary.scalars[0].unit == "mm"
    assert summary.scalars[0].node_id == 2
    assert summary.scalars[1].value == pytest.approx(
        np.linalg.norm(
            [result.nodal_displacement(2, component) for component in (1, 2, 3)]
        )
    )
    assert summary.scalars[2].value == -0.5
    assert summary.scalars[2].node_id == 2
    assert summary.scalars[3].node_id == 2
    assert summary.scalars[4].value == result.nodal_reaction(1, 1)
    assert summary.scalars[4].unit == "N"
    assert summary.scalars[5].value == pytest.approx(-80.0)
    assert summary.scalars[5].region == "all"
    assert all(scalar.step == "pull" for scalar in summary.scalars)
    assert all(scalar.run_id == "run-1" for scalar in summary.scalars)


def test_query_results_normalizes_missing_regions_per_query():
    summary = query_results(
        _truss_result(),
        (
            ResultQuery(
                ResultQueryKind.REACTION_SUM,
                component=1,
                node_set="missing",
            ),
        ),
        run_id="run-1",
        unit_context=UNITS,
    )

    assert summary.scalars == ()
    assert summary.diagnostics[0].code == DiagnosticCode.RESULT_QUERY_FAILED.value
    assert "node set 'missing'" in summary.diagnostics[0].message


def test_max_displacement_query_resolves_edge_nodes():
    result = _truss_result()
    result.model.edges["free"] = Edge(
        "free",
        (
            ElementEdge(1, 0, (2,)),
        ),
    )

    summary = query_results(
        result,
        (
            ResultQuery(
                ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                edge="free",
            ),
        ),
        run_id="run-edge",
        unit_context=UNITS,
    )

    assert summary.diagnostics == ()
    assert summary.scalars[0].node_id == 2
    assert summary.scalars[0].region == "free"
    assert summary.scalars[0].value == pytest.approx(
        np.linalg.norm(
            [result.nodal_displacement(2, component) for component in (1, 2, 3)]
        )
    )


def test_reaction_sum_deduplicates_nodes_shared_by_surface_faces():
    result = _truss_result()
    result.model.surfaces["loaded"] = Surface(
        "loaded",
        (
            ElementFace(1, 0, (1, 2)),
            ElementFace(1, 1, (2,)),
        ),
    )

    summary = query_results(
        result,
        (
            ResultQuery(
                ResultQueryKind.REACTION_SUM,
                component=1,
                surface="loaded",
            ),
        ),
        run_id="run-surface",
        unit_context=UNITS,
    )

    assert summary.diagnostics == ()
    assert summary.scalars[0].value == pytest.approx(-80.0)
    assert summary.scalars[0].region == "loaded"


def test_displacement_region_failure_has_targeted_remediation():
    summary = query_results(
        _truss_result(),
        (
            ResultQuery(
                ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                edge="missing",
            ),
        ),
        run_id="run-missing-edge",
        unit_context=UNITS,
    )

    diagnostic = summary.diagnostics[0]
    assert diagnostic.code == DiagnosticCode.RESULT_QUERY_FAILED.value
    assert "edge 'missing'" in diagnostic.message
    assert "displacement component" in diagnostic.remediation
    assert "stress output" not in diagnostic.remediation


def test_query_results_enforces_provider_scalar_bound_before_evaluation():
    query = ResultQuery(
        ResultQueryKind.DISPLACEMENT_COMPONENT,
        node_id=2,
        component=1,
    )

    summary = query_results(
        _truss_result(),
        (query,) * (MAX_PROVIDER_SCALARS + 1),
        run_id="run-1",
        unit_context=UNITS,
    )

    assert summary.scalars == ()
    assert len(summary.diagnostics) == 1
    assert "configured limit" in summary.diagnostics[0].message


def test_query_results_returns_diagnostic_for_unavailable_truss_stress():
    summary = query_results(
        _truss_result(),
        (ResultQuery(ResultQueryKind.STRESS_EXTREMA),),
        run_id="run-1",
        unit_context=UNITS,
    )

    assert summary.scalars == ()
    assert summary.diagnostics[0].code == DiagnosticCode.RESULT_QUERY_FAILED.value
    assert "stress" in summary.diagnostics[0].message.casefold()


def test_beam_stress_queries_reuse_one_canonical_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_beam_field_characterization_result()
    original = beam.recover_integration_point_stress
    calls: list[object] = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(beam, "recover_integration_point_stress", counted)

    summary = query_results(
        result,
        (
            ResultQuery(
                ResultQueryKind.STRESS_EXTREMA,
                measure="axial_stress_max",
            ),
            ResultQuery(
                ResultQueryKind.STRESS_EXTREMA,
                measure="axial_stress_min",
            ),
        ),
        run_id="run-beam",
        unit_context=UNITS,
    )

    assert summary.diagnostics == ()
    assert len(summary.scalars) == 4
    assert calls == [result]


def test_stress_extrema_match_existing_nodal_stress_postprocessing():
    mesh = make_tri3_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        name="plate",
        element_sets={"plate": ElementSet("plate", (1,))},
    )
    displacement = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = 0.01 * node.x
        displacement[mesh.global_dof(node.id, 1)] = -0.002 * node.y
    result = ModelResult(
        model,
        AnalysisStep("pull"),
        displacement,
        np.zeros(mesh.num_dofs),
    )

    summary = query_results(
        result,
        (
            ResultQuery(
                ResultQueryKind.STRESS_EXTREMA,
                element_set="plate",
                measure="von_mises",
            ),
        ),
        run_id="run-plate",
        unit_context=UNITS,
    )

    direct_field = field.collect(mesh, displacement)
    direct_values = [
        invariants.von_mises_plane(
            *contribution.components,
            plane_type=contribution.plane_type or "stress",
            nu=contribution.poisson_ratio or 0.0,
        )
        for contributions in direct_field.contributions_by_node.values()
        for contribution in contributions
    ]
    assert summary.diagnostics == ()
    assert len(summary.scalars) == 2
    assert summary.scalars[0].value == pytest.approx(min(direct_values))
    assert summary.scalars[1].value == pytest.approx(max(direct_values))
    assert summary.scalars[0].measure == "von_mises_minimum"
    assert summary.scalars[1].measure == "von_mises_maximum"
    assert all(scalar.unit == "MPa" for scalar in summary.scalars)
    assert all(scalar.element_id == 1 for scalar in summary.scalars)
    assert all(scalar.region == "plate" for scalar in summary.scalars)
