from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from fem.model import authoring as steps
from fem.analysis import AnalysisRequest, compile_analysis, resolve_analysis_request
from fem.assembly import SparseAssembler
from fem.physics.mechanics import (
    ContinuumMechanicsOperator,
    TotalLagrangianKinematics,
)
from fem.materials import J2PlasticityMaterial
from fem.problem import StaticEquilibriumProblem
from fem.analysis import incremental
from fem.state import StateKey, TransactionalStateManager
from fem.materials import resolve_j2_algorithm
from fem.model import ElementSet, FEMModel, NodeSet, StaticFormulation
from fem.elements import (
    Beam2Definition,
    Truss2Definition,
    get_element_definition,
)
from fem.analysis.compilation.capabilities import nonlinear_static_capability_for
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh


def _model(load: float = 1.0e-4) -> FEMModel:
    model = FEMModel(
        mesh=make_quad4_stiffness_mesh(),
        node_sets={
            "fixed": NodeSet("fixed", (1, 4)),
            "loaded": NodeSet("loaded", (2, 3)),
        },
        element_sets={"plate": ElementSet("plate", (1,))},
    )
    step = steps.static(
        "nlgeom",
        formulation=StaticFormulation.NONLINEAR,
    )
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=load)
    steps.add(model, step)
    return model


def _problem(model: FEMModel, algorithm: str | None = None):
    step = model.steps[0]
    if algorithm is not None:
        model.mesh.elements[0].props["algorithm"] = algorithm
    return compile_analysis(model, resolve_analysis_request(step)).problem


def test_v2_problem_is_composed_without_physics_specific_problem_type():
    model = _model(load=1.0e-4)
    problem = _problem(model, "hencky")

    assert isinstance(problem, StaticEquilibriumProblem)
    assert isinstance(problem.assembly, SparseAssembler)
    assert isinstance(problem.assembly.bindings[0].operator, ContinuumMechanicsOperator)
    assert {binding.entity.id for binding in problem.assembly.bindings} == {1}
    assert isinstance(
        problem.assembly.bindings[0].operator.kinematics,
        TotalLagrangianKinematics,
    )
    assert problem.assembly.state.committed(StateKey.material_point(1, 1)) == {}

    result = incremental.solve(
        problem,
        [1.0],
        max_iterations=20,
        residual_tolerance=1.0e-10,
    )
    assert np.all(np.isfinite(result.final_solution.values))
    assert result.increments[-1].outputs["integration_points"]["element_id"].shape == (4,)
    evaluation = problem.evaluate()
    assert issparse(evaluation.tangent)


def test_v2_j2_material_owns_parameters_but_not_history():
    material = J2PlasticityMaterial(210.0, 0.3, 0.01, 0.1)
    assert material.initial_state()["equivalent_plastic_strain"] == 0.0
    assert resolve_j2_algorithm("hencky") == "hencky"


@pytest.mark.parametrize("algorithm", ("hencky", "multiplicative"))
def test_v2_quad4_j2_path_solves_and_exports_integration_point_history(algorithm):
    model = _model(load=0.1)
    model.mesh.elements[0].props.update(
        constitutive_model="j2_plasticity",
        yield_stress=0.01,
        hardening_modulus=0.1,
    )
    problem = _problem(model, algorithm)

    result = incremental.solve(
        problem,
        [1.0],
        max_iterations=20,
        residual_tolerance=1.0e-8,
    )
    points = result.increments[-1].outputs["integration_points"]

    assert np.all(np.isfinite(result.final_solution.values))
    assert points["equivalent_plastic_strain"].shape == (4,)
    assert np.all(points["equivalent_plastic_strain"] > 0.0)
    assert all(
        state["equivalent_plastic_strain"] > 0.0
        for state in points["history"]
    )


def test_compiled_request_excludes_authored_metadata():
    model = _model()
    model.mesh.elements[0].props.update(
        constitutive_model="j2_plasticity",
        yield_stress=0.01,
        hardening_modulus=0.1,
    )
    step = model.steps[0]
    model.mesh.elements[0].props.update(
        constitutive_model="j2_plasticity",
        algorithm="multiplicative",
    )
    compiled = compile_analysis(model, resolve_analysis_request(step))
    assert compiled.problem.assembly.bindings[0].resources["material"].algorithm == "multiplicative"
    assert not hasattr(compiled.request.step, "metadata")


def test_v2_state_manager_separates_trial_from_committed_state():
    state = TransactionalStateManager()
    key = StateKey.material_point(1, 1)
    state.register(key, {"alpha": 0.0})
    state.stage(key, {"alpha": 1.0})
    assert state.committed(key)["alpha"] == 0.0
    state.rollback()
    assert state.committed(key)["alpha"] == 0.0
    state.stage(key, {"alpha": 2.0})
    state.commit()
    assert state.committed(key)["alpha"] == 2.0


def test_reference_element_definitions_have_one_registry_owner():
    assert isinstance(get_element_definition("CPS4"), type(get_element_definition("Quad4")))
    assert isinstance(get_element_definition("Beam2"), Beam2Definition)
    assert isinstance(get_element_definition("Truss2"), Truss2Definition)

    for element_type in ("Quad4", "Tri3", "Quad8", "Tri6", "Hex8", "Tet4", "Hex20", "Tet10"):
        capability = nonlinear_static_capability_for(element_type)
        assert type(capability.definition_factory()) is type(
            get_element_definition(element_type)
        )


def test_analysis_request_is_the_single_typed_static_execution_snapshot():
    controls = steps.StaticStepControls(
        initial_increment=0.25,
        maximum_increments=4,
        newton_max_iterations=31,
        residual_tolerance=1.0e-9,
    )
    step = steps.static("typed", controls=controls)

    request = resolve_analysis_request(step)

    assert request.controls is controls
    assert request.controls.load_factors == controls.load_factors
    assert step.metadata["newton_max_iterations"] == 31
    assert request.time_values == ()


def test_analysis_request_decodes_persistence_metadata_only_at_the_boundary():
    step = steps.static(
        "legacy",
        initial_increment=0.5,
        maximum_increments=2,
        newton_max_iterations=17,
    )

    request = resolve_analysis_request(step)

    assert request.controls.newton_max_iterations == 17
    assert request.controls.load_factors == (0.5, 1.0)
    with pytest.raises(TypeError, match="StaticStepControls"):
        AnalysisRequest(step=request.step, controls={})
