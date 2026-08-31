from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from fem.constraints import ConstraintSet
from fem.application import prepare_static_preflight
from fem.analysis import (
    AnalysisPath,
    DEFAULT_ANALYSIS_EXECUTOR,
    ProcedureContext,
    analysis_path_for,
    compile_analysis,
    resolve_analysis_request,
)
from fem.assembly import assemble_mass_matrix
from fem.model import (
    DofSpace,
    DynamicProcedureKind,
    DynamicStepControls,
    Element3D,
    ElementSet,
    FEMModel,
    GeometryMode,
    Mesh3D,
    Node3D,
    NodeSet,
    StaticFormulation,
    TimeAmplitude,
    transient_dynamic,
)
from fem.model import authoring as steps
from fem.problem import LinearDynamicProblem
from fem.solver import newmark


def _truss_model(*, material: str = "linear_elastic") -> FEMModel:
    model = FEMModel(
        mesh=Mesh3D(
            nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
            elements=[
                Element3D(
                    1,
                    [1, 2],
                    "Truss2",
                    {"E": 100.0, "area": 1.0, "rho": 1.0,
                     "constitutive_model": material},
                )
            ],
        ),
        node_sets={
            "fixed": NodeSet("fixed", (1,)),
            "tip": NodeSet("tip", (2,)),
        },
        element_sets={"bar": ElementSet("bar", (1,))},
    )
    step = transient_dynamic(
        controls=DynamicStepControls(
            time_period=0.5,
            initial_time_increment=0.05,
            maximum_increments=10,
            amplitude=TimeAmplitude(((0.0, 0.0), (0.1, 1.0))),
        )
    )
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.displacement(step, "tip", components=(2, 3))
    steps.nodal_load(step, "tip", component=1, value=1.0)
    steps.add(model, step)
    return model


def test_dynamic_controls_are_typed_and_produce_target_times() -> None:
    controls = DynamicStepControls(
        time_period=1.0,
        initial_time_increment=0.3,
        maximum_increments=4,
    )
    np.testing.assert_allclose(controls.time_grid, (0.3, 0.6, 0.9, 1.0))
    assert TimeAmplitude(((0.0, 0.0), (1.0, 2.0))).value_at(0.25) == 0.5


def test_truss_mass_matrix_is_consistent_and_has_expected_total_mass() -> None:
    model = _truss_model()
    mass = assemble_mass_matrix(model.mesh)
    np.testing.assert_allclose(mass.toarray(), mass.toarray().T)
    np.testing.assert_allclose(np.sum(mass.toarray()[::3, ::3]), 1.0)


def test_newmark_single_dof_constant_force_matches_harmonic_solution() -> None:
    dof_space = DofSpace.single_field("U", 1)
    problem = LinearDynamicProblem(
        dof_space=dof_space,
        stiffness=csr_matrix([[4.0]]),
        mass=csr_matrix([[1.0]]),
        damping=csr_matrix([[0.0]]),
        constraints=ConstraintSet(),
        reference_load=np.array([1.0]),
    )
    result = newmark.solve(problem, np.arange(0.05, 1.0001, 0.05))
    expected = 0.25 * (1.0 - math.cos(2.0 * result.final.time))
    np.testing.assert_allclose(
        result.final.solution.values[0],
        expected,
        rtol=0.0,
        atol=2.0e-3,
    )


def test_linear_dynamic_is_registered_and_publishes_time_derivatives() -> None:
    model = _truss_model()
    step = model.steps[0]
    request = resolve_analysis_request(step)
    assert analysis_path_for(model, request) is AnalysisPath.LINEAR_DYNAMIC
    compiled = compile_analysis(model, request)
    numerical = DEFAULT_ANALYSIS_EXECUTOR.run(
        model,
        step,
        request,
        compiled,
        name=None,
        context=ProcedureContext(),
    )
    result = DEFAULT_ANALYSIS_EXECUTOR.materialize(
        model,
        step,
        request,
        numerical,
        name=None,
    )
    assert len(result.frames) == len(request.controls.time_grid)
    assert np.isclose(
        result.frames[-1].outputs["time"],
        request.controls.time_period,
    )
    assert result.frames[-1].outputs["velocity"].shape == (model.mesh.num_dofs,)
    assert result.frames[-1].outputs["acceleration"].shape == (model.mesh.num_dofs,)


def test_linear_dynamic_rejects_non_linear_material_behavior() -> None:
    model = _truss_model(material="j2_plasticity")
    try:
        resolve_analysis_request(model.steps[0])
        compile_analysis(model, resolve_analysis_request(model.steps[0]))
    except ValueError as error:
        assert "linear_elastic" in str(error) or "material" in str(error)
    else:
        raise AssertionError("nonlinear material must not enter linear dynamics")


def test_explicit_linear_dynamic_uses_central_difference_and_lumped_mass() -> None:
    model = _truss_model()
    source = model.steps[0]
    controls = replace(
        source.controls,
        procedure_kind=DynamicProcedureKind.EXPLICIT,
        maximum_increments=20,
    )
    model.steps[0] = replace(
        source,
        controls=controls,
        metadata={**source.metadata, **controls.to_metadata()},
    )

    request = resolve_analysis_request(model.steps[0])
    assert request.controls.procedure_kind is DynamicProcedureKind.EXPLICIT
    assert request.controls.mass_matrix.value == "lumped"
    assert request.controls.integration_method.value == "central_difference"
    assert analysis_path_for(model, request) is AnalysisPath.DYNAMIC_EXPLICIT
    compiled = compile_analysis(model, request)
    numerical = DEFAULT_ANALYSIS_EXECUTOR.run(
        model,
        model.steps[0],
        request,
        compiled,
        name=None,
        context=ProcedureContext(),
    )
    result = DEFAULT_ANALYSIS_EXECUTOR.materialize(
        model,
        model.steps[0],
        request,
        numerical,
        name=None,
    )
    assert result.frames
    assert result.outputs["procedure"] == "dynamic_explicit"
    assert np.isclose(
        result.frames[-1].outputs["time"],
        request.controls.time_period,
    )


def test_dynamic_preflight_does_not_install_compiled_system_as_static_cache() -> None:
    from fem.io.inp import read

    fixture = (
        Path(__file__).parents[1]
        / "helpers"
        / "fixtures"
        / "inp"
        / "abaqus_standard"
        / "truss2_tension.inp"
    )
    model = read(fixture)
    model.mesh.elements[0].props["rho"] = 7800.0
    source_step = next(step for step in model.steps if step.name == "Tension")
    initial_step = next(step for step in model.steps if step.name == "Initial")
    dynamic_step = replace(
        source_step,
        procedure="dynamic",
        boundaries=initial_step.boundaries,
        controls=DynamicStepControls(
            time_period=0.1,
            initial_time_increment=0.1,
            maximum_increments=1,
        ),
        formulation=StaticFormulation.LINEAR,
        geometry_mode=GeometryMode.SMALL_STRAIN,
    )
    model.steps = [dynamic_step]

    evaluation = prepare_static_preflight(model, dynamic_step.name)

    assert evaluation.report.passed, evaluation.report.diagnostics
    assert evaluation.prepared_system is None
