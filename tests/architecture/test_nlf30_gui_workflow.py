from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fem.model import authoring as steps
import fem.application.analysis_execution as analysis_execution
from fem.application import (
    AnalysisConvergenceError,
    analysis_solver_kind,
    AttemptStatus,
    RunDiagnostics,
    run_static_preflight,
    solve_analysis,
)
from fem.results import (
    FieldPosition,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
)
from fem.results._ownership import deep_owned_result
from fem.model import (
    ElementSet,
    DynamicStepControls,
    FEMModel,
    GeometryMode,
    MaterialDefinition,
    NodeSet,
    SectionAssignment,
    StaticFormulation,
)
from fem.analysis import apply_sections
from fem.analysis import (
    AnalysisCancelled,
    compile_analysis,
    resolve_analysis_request,
)
from fem.post.stress.field import (
    FiniteStrainStatePosition,
    FiniteStrainStateRecovery,
    FiniteStrainStressRecovery,
    StressPosition,
)
from fem.analysis import incremental, linear_static as static_linear
from fem.state import StateKey
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh
from tests.helpers.model_builders import make_static_pull_truss_model


def _make_nonlinear_workflow_model(
    *,
    yield_stress: float | None = 1.0e6,
    hardening_modulus: float | None = None,
    load: float = 1.0e-4,
) -> FEMModel:
    model = FEMModel(
        mesh=make_quad4_stiffness_mesh(),
        name="nlf30_gui_workflow",
        node_sets={
            "fixed": NodeSet("fixed", (1, 4)),
            "loaded": NodeSet("loaded", (2, 3)),
        },
        element_sets={"plate": ElementSet("plate", (1,))},
        materials={
            "steel": MaterialDefinition(
                "steel",
                {
                    "E": 210.0,
                    "nu": 0.3,
                    **(
                        {"yield_stress": yield_stress}
                        if yield_stress is not None
                        else {}
                    ),
                    **(
                        {"hardening_modulus": hardening_modulus}
                        if hardening_modulus is not None
                        else {}
                    ),
                },
                constitutive_model=(
                    "j2_plasticity"
                    if yield_stress is not None
                    else "linear_elastic"
                ),
            )
        },
        sections=[
            SectionAssignment(
                "plate",
                "steel",
                "solid",
                {"thickness": 1.0, "plane_type": "stress"},
            )
        ],
    )
    step = steps.static(
        "nonlinear",
        formulation=StaticFormulation.NONLINEAR,
    )
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=load)
    steps.add(model, step)
    return model


def test_application_dispatch_solves_supported_nonlinear_gui_path():
    model = _make_nonlinear_workflow_model()

    report = run_static_preflight(model, "nonlinear")
    assert report.passed
    assert report.numerical_stability_checked
    assert analysis_solver_kind(model, "nonlinear") == "nonlinear_static"

    result = solve_analysis(model, "nonlinear")

    assert result.outputs["material"] == "j2_plasticity"
    assert result.outputs["material_algorithm"] == "hencky"
    assert result.iterations is not None
    assert np.all(np.isfinite(result.U))
    assert np.all(np.isfinite(result.reactions))


def test_application_dispatch_stops_nonlinear_solver_at_cooperative_checkpoint():
    model = _make_nonlinear_workflow_model()

    with pytest.raises(AnalysisCancelled):
        solve_analysis(
            model,
            "nonlinear",
            should_cancel=lambda: True,
        )


def test_application_dispatch_publishes_each_converged_quad4_frame():
    model = _make_nonlinear_workflow_model()
    controls = steps.StaticStepControls(
        initial_increment=0.5,
        maximum_increments=2,
        newton_max_iterations=20,
        residual_tolerance=1.0e-9,
    )
    model.steps[0].controls = controls

    result = solve_analysis(model, "nonlinear")

    assert [frame.frame_index for frame in result.frames] == [1, 2]
    assert [frame.load_factor for frame in result.frames] == pytest.approx(
        [0.5, 1.0]
    )
    assert all(frame.converged for frame in result.frames)
    assert result.frames[-1].U.flags.writeable is False
    assert result.frames[-1].outputs["step"] is model.steps[0]
    for frame in result.frames:
        integration_points = frame.outputs["integration_points"]
        assert integration_points["green_lagrange_strain"].shape == (4, 3, 3)
        assert integration_points["cauchy_stress"].shape == (4, 3, 3)
        assert integration_points["equivalent_plastic_strain"].shape == (4,)
        assert np.all(np.isfinite(integration_points["cauchy_stress"]))
    np.testing.assert_allclose(result.U, result.frames[-1].U)

    owned = deep_owned_result(result)
    assert len(owned.frames) == 2
    assert owned.frames[0].model is owned.model
    assert owned.frames[0].step is owned.step
    assert owned.frames[0].outputs is not result.frames[0].outputs


def test_application_dispatch_uses_cutback_when_newton_limit_is_tight():
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=10.0,
    )
    controls = steps.StaticStepControls(
        initial_increment=0.1,
        maximum_increments=100,
        newton_max_iterations=4,
        residual_tolerance=1.0e-8,
    )
    model.steps[0].controls = controls

    result = solve_analysis(model, "nonlinear")

    assert result.load_factor == pytest.approx(1.0)
    assert len(result.frames) > controls.required_increments
    assert result.frames[-1].residual_norm <= controls.residual_tolerance


def test_finite_strain_frame_stress_recovery_does_not_recompute_linear_stress():
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    result = solve_analysis(model, "nonlinear")
    frame = result.frames[-1]

    recovery = FiniteStrainStressRecovery(frame)
    integration = recovery.collect(StressPosition.INTEGRATION_POINT)
    centroid = recovery.collect(StressPosition.CENTROID)
    element_nodal = recovery.collect(StressPosition.ELEMENT_NODAL)

    assert len(integration.records) == 4
    assert len(centroid.records) == 1
    assert len(element_nodal.records) == 4
    assert integration.records[0].components[0] != pytest.approx(0.0)
    assert all(np.isfinite(record.invariants.mises) for record in integration.records)


def test_result_provider_uses_captured_finite_strain_stress_for_quad4():
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    result = solve_analysis(model, "nonlinear")
    source = ResultSourceKey("result", "session", "artifact", 0, "nonlinear", "run")
    provider = build_result_provider(source, result)
    key = next(
        availability.key
        for availability in provider.catalog().fields
        if availability.descriptor.field_id
        == ResultFieldId(ResultVariable.S, FieldPosition.INTEGRATION_POINT)
    )

    patch = provider.materialize((key,))
    field = patch.fields[0]

    assert len(field.locations) == 4
    assert field.values.shape[1] == 8
    assert np.all(np.isfinite(field.values))


def test_linear_dynamic_result_recovery_uses_compiled_effective_properties():
    model = _make_nonlinear_workflow_model(
        yield_stress=None,
        load=1.0e-4,
    )
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3, "rho": 1.0},
    )
    element = model.mesh.elements[0]
    element.props.pop("E", None)
    element.props.pop("nu", None)
    source_step = model.steps[0]
    controls = DynamicStepControls(
        time_period=0.1,
        initial_time_increment=0.1,
        maximum_increments=1,
    )
    model.steps[0] = replace(
        source_step,
        procedure="dynamic",
        formulation=StaticFormulation.LINEAR,
        geometry_mode=GeometryMode.SMALL_STRAIN,
        controls=controls,
        metadata={**source_step.metadata, **controls.to_metadata()},
    )

    result = solve_analysis(model, model.steps[0])

    assert "E" not in result.model.mesh.elements[0].props
    assert result.compiled_model is not None
    assert result.compiled_model.mesh.elements[0].props["E"] == 210.0
    source = ResultSourceKey("result", "session", "artifact", 0, "dynamic", "run")
    provider = build_result_provider(source, result)
    key = next(
        availability.key
        for availability in provider.catalog().fields
        if availability.descriptor.field_id
        == ResultFieldId(ResultVariable.S, FieldPosition.INTEGRATION_POINT)
    )

    patch = provider.materialize((key,))

    assert patch.fields[0].values.shape == (4, 8)
    assert np.all(np.isfinite(patch.fields[0].values))


def test_result_provider_publishes_captured_finite_strain_state_fields():
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    result = solve_analysis(model, "nonlinear")
    source = ResultSourceKey("result", "session", "artifact", 0, "nonlinear", "run")
    provider = build_result_provider(source, result)

    expected_positions = {
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
        FieldPosition.NODE_REGION,
        FieldPosition.RESOLVED_NODAL,
    }
    state_availability = {
        availability.descriptor.field_id
        for availability in provider.catalog().fields
        if availability.descriptor.field_id.variable
        in {ResultVariable.E, ResultVariable.PEEQ}
    }
    assert {
        ResultFieldId(variable, position)
        for variable in (ResultVariable.E, ResultVariable.PEEQ)
        for position in expected_positions
    } == state_availability

    keys = tuple(
        availability.key
        for availability in provider.catalog().fields
        if availability.descriptor.field_id.variable
        in {ResultVariable.E, ResultVariable.PEEQ}
    )
    patch = provider.materialize(keys)
    assert len(patch.fields) == len(keys)
    for field in patch.fields:
        assert np.all(np.isfinite(field.values))
        assert field.values.shape[1] == len(field.descriptor.columns)

    e_recovery = FiniteStrainStateRecovery(
        result.frames[-1],
        "green_lagrange_strain",
    )
    peeq_recovery = FiniteStrainStateRecovery(
        result.frames[-1],
        "equivalent_plastic_strain",
    )
    assert len(
        e_recovery.collect(FiniteStrainStatePosition.INTEGRATION_POINT).records
    ) == 4
    assert len(
        peeq_recovery.collect(FiniteStrainStatePosition.ELEMENT_NODAL).records
    ) == 4


def test_result_provider_builds_display_only_provider_for_one_increment():
    model = _make_nonlinear_workflow_model(load=1.0e-4)
    controls = steps.StaticStepControls(
        initial_increment=0.5,
        maximum_increments=2,
        newton_max_iterations=20,
        residual_tolerance=1.0e-9,
    )
    model.steps[0].controls = controls
    result = solve_analysis(model, "nonlinear")
    source = ResultSourceKey("result", "session", "artifact", 0, "nonlinear", "run")
    provider = build_result_provider(source, result)

    frame_provider = provider.frame_provider(1)

    assert frame_provider.source == provider.source
    assert frame_provider.model_result is not None
    assert frame_provider.model_result.frames == ()
    expected_displacements = np.column_stack(
        (result.frames[0].U.reshape((-1, 2)), np.zeros(4))
    )
    np.testing.assert_allclose(
        frame_provider.snapshot.topology.nodal_displacements,
        expected_displacements,
    )
    e_key = next(
        availability.key
        for availability in frame_provider.catalog().fields
        if availability.descriptor.field_id
        == ResultFieldId(ResultVariable.E, FieldPosition.CENTROID)
    )
    e_patch = frame_provider.materialize((e_key,))
    assert len(e_patch.fields) == 1
    assert np.all(np.isfinite(e_patch.fields[0].values))
    assert provider.snapshot.generation == 0
    assert not any(
        field_data.key == e_key
        for field_data in provider.snapshot.fields
    )


def test_application_dispatch_keeps_linear_solver_path_unchanged():
    model = make_static_pull_truss_model()

    assert analysis_solver_kind(model, "pull") == "linear_static"
    expected = static_linear.solve(model, "pull")
    actual = solve_analysis(model, "pull")

    np.testing.assert_allclose(actual.U, expected.U)
    np.testing.assert_allclose(actual.reactions, expected.reactions)


def test_application_dispatch_publishes_linear_path_through_common_run_contract():
    model = make_static_pull_truss_model()
    monitor = RunDiagnostics("linear-run", "Job-linear", "pull")

    result = solve_analysis(model, "pull", monitor=monitor)
    snapshot = monitor.snapshot()

    assert snapshot.status == "succeeded"
    assert len(snapshot.attempts) == 1
    assert snapshot.attempts[0].status is AttemptStatus.CONVERGED
    assert snapshot.attempts[0].result_frame == 1
    assert len(result.frames) == 1
    assert result.frames[0].outputs["increment_number"] == 1
    assert result.outputs["attempt_count"] == 1


def test_application_dispatch_preserves_failed_increment_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_nonlinear_workflow_model()
    completed = incremental.IncrementalSolveResult(())
    cause = RuntimeError("synthetic Newton failure")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise incremental.IncrementalConvergenceError(
            "synthetic incremental failure",
            failed_load_factor=0.75,
            completed=completed,
            cause=cause,
        )

    monkeypatch.setattr(analysis_execution.incremental, "solve", fail)

    with pytest.raises(AnalysisConvergenceError) as caught:
        solve_analysis(model, "nonlinear")

    error = caught.value
    assert error.failed_load_factor == pytest.approx(0.75)
    assert error.completed is completed
    assert error.partial_result is None
    assert "failed_load_factor=0.75" in str(error)
    assert "尚无可用的 Newton 收敛历史" in str(error)


def test_nonlinear_preflight_accepts_linear_elastic_nlgeom_material():
    report = run_static_preflight(
        _make_nonlinear_workflow_model(yield_stress=None),
        "nonlinear",
    )

    assert report.passed
    assert report.numerical_stability_checked


def test_public_j2_material_definition_runs_quad4_incremental_loading_unloading():
    model = _make_nonlinear_workflow_model(
        yield_stress=0.01,
        hardening_modulus=0.1,
        load=0.1,
    )
    apply_sections(model)
    problem = compile_analysis(
        model,
        resolve_analysis_request(model.steps[0]),
    ).problem

    result = incremental.solve(
        problem,
        (0.25, 0.5, 0.75, 1.0, 0.5, 0.0),
        max_iterations=30,
        residual_tolerance=1.0e-9,
    )

    assert len(result.increments) == 6
    assert result.final_load_factor == 0.0
    assert np.all(np.isfinite(result.final_solution.values))
    histories = [
        problem.assembly.state.committed(StateKey.material_point(1, point_id))[
            "equivalent_plastic_strain"
        ]
        for point_id in range(1, 5)
    ]
    assert all(value > 0.0 for value in histories)


def test_public_static_controls_define_fixed_factors_and_newton_settings():
    controls = steps.StaticStepControls(
        initial_increment=0.3,
        maximum_increments=4,
        newton_max_iterations=18,
        residual_tolerance=1.0e-9,
    )
    step = steps.static(
        "controlled",
        controls=controls,
        formulation=StaticFormulation.NONLINEAR,
    )

    assert controls.load_factors == pytest.approx((0.3, 0.6, 0.9, 1.0))
    assert steps.StaticStepControls.from_metadata(step.metadata) == controls
    assert step.metadata["NLGEOM"] is True
    assert step.metadata["newton_max_iterations"] == 18


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("initial_increment", 0.0),
        ("initial_increment", 1.1),
        ("maximum_increment", 0.0),
        ("maximum_increments", 0),
        ("newton_max_iterations", 0),
        ("residual_tolerance", 0.0),
    ),
)
def test_public_static_controls_reject_unsupported_values(field, value):
    with pytest.raises(ValueError):
        steps.StaticStepControls(**{field: value})
