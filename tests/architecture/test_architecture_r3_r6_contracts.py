from __future__ import annotations

import numpy as np
import pytest

from fem.model import authoring as steps
from fem.application import RunDiagnostics, solve_analysis
from fem.results import (
    FieldAssociation,
    FieldLocation,
    ResultSourceKey,
    frame_catalog_from_model_result,
)
from fem.application.run_monitor import AttemptStatus
from fem.analysis.compilation.boundary.compiled import CompiledBoundary, compile_boundary
from fem.analysis.compilation.boundary.loads import build_load_vector
from fem.assembly import Assembly, SparseAssembler
from fem.elements.quad4 import Quad4Definition
from fem.model.beam_section import BeamSectionPoint
from fem.physics.mechanics import (
    Quad4LinearOperator,
    ContinuumMechanicsOperator,
    TotalLagrangianKinematics,
    get_mechanical_operator,
    get_recovery_service,
)
from fem.physics.mechanics.operators.plane import PlaneProperties
from fem.analysis import (
    compile_analysis,
    compile_material_assignments,
    resolve_analysis_request,
)
from fem.materials import (
    KinematicMeasure,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)
from fem.problem import ProblemContext, StaticEquilibriumProblem
from fem.analysis import incremental
from fem.state import EvaluationContext, SolutionState, StateKey
from fem.state.manager import TransactionalStateManager
from tests.architecture.test_nlf20_incremental_loading import _RampProblem
from tests.architecture.test_nlf30_gui_workflow import (
    _make_nonlinear_workflow_model,
)
from tests.helpers.mesh_builders import (
    make_quad8_stiffness_mesh,
    make_tri3_stiffness_mesh,
)


def test_r3_compiled_material_assignment_does_not_mutate_element_props():
    model = _make_nonlinear_workflow_model(yield_stress=None)
    before = [dict(element.props) for element in model.mesh.elements]

    compiled = compile_material_assignments(model)

    assignment = compiled.for_element(1)
    assert assignment.source == "section"
    assert assignment.properties["E"] == pytest.approx(210.0)
    assert assignment.properties["thickness"] == pytest.approx(1.0)
    assert [dict(element.props) for element in model.mesh.elements] == before


def test_quad4_properties_are_one_shared_contract_for_both_solution_paths():
    properties = PlaneProperties.from_mapping(
        {
            "E": 210.0,
            "nu": 0.3,
            "thickness": 2.0,
            "plane_type": "plane_stress",
        },
        element_id=1,
    )

    assert properties.E == pytest.approx(210.0)
    assert properties.nu == pytest.approx(0.3)
    assert properties.thickness == pytest.approx(2.0)
    assert properties.plane_type == "stress"
    assert properties.as_mapping() == {
        "E": 210.0,
        "nu": 0.3,
        "thickness": 2.0,
        "plane_type": "stress",
    }
    with pytest.raises(KeyError, match="missing property E"):
        PlaneProperties.from_mapping({"nu": 0.3}, element_id=1)


def test_r7_quad4_linear_path_uses_v2_assembly_and_explicit_properties():
    model = _make_nonlinear_workflow_model(yield_stress=None)
    mesh = model.mesh
    element = mesh.elements[0]
    operator = Quad4LinearOperator()
    properties = {
        1: {
            "E": 210.0,
            "nu": 0.3,
            "thickness": 2.0,
            "plane_type": "stress",
        }
    }
    assembly = SparseAssembler.from_displacement_mesh(
        mesh,
        operator,
        element_properties_by_element=properties,
    )
    assert isinstance(assembly, Assembly)

    displacement = np.arange(mesh.num_dofs, dtype=float) * 1.0e-4
    solution = SolutionState(assembly.dof_space, displacement)
    result = assembly.assemble(
        solution,
        context=EvaluationContext(load_factor=0.5),
    )
    binding = assembly.bindings[0]
    local = operator.evaluate(
        binding.entity,
        binding.reference_coordinates,
        binding.local_fields(solution),
        binding.dofs,
        binding.resources,
        assembly.state,
        binding.properties,
        context=EvaluationContext(load_factor=0.5),
        state_namespace=binding.state_namespace,
    )

    assert np.allclose(result.tangent.toarray(), local.tangent)
    assert np.allclose(result.residual, local.residual)
    assert element.props["thickness"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "builder",
    [make_tri3_stiffness_mesh, make_quad8_stiffness_mesh],
    ids=["tri3", "quad8"],
)
def test_r7_migrated_plane_linear_operators_match_existing_stiffness(
    builder,
):
    mesh = builder()
    element = mesh.elements[0]
    kernel = get_recovery_service(element.type)
    assembly = SparseAssembler.from_displacement_mesh(
        mesh,
        get_mechanical_operator(mesh, element.type),
    )
    displacement = np.linspace(1.0e-5, 2.0e-5, mesh.num_dofs)

    result = assembly.assemble(
        SolutionState(assembly.dof_space, displacement),
        context=EvaluationContext(load_factor=1.0),
    )
    expected = kernel.stiffness(mesh, element)

    assert np.allclose(result.tangent.toarray(), expected)
    assert np.allclose(result.residual, expected @ displacement)
    assert len(result.local_outputs) == 1
    points = result.local_outputs[0].points
    expected_points = 1 if element.type == "Tri3" else 9
    assert len(points) == expected_points
    assert all(
        point.fields["engineering_strain"].shape == (3,)
        and point.fields["cauchy_stress"].shape == (3,)
        for point in points
    )


def test_r4_problem_exposes_typed_constraints_and_element_outputs():
    model = _make_nonlinear_workflow_model(yield_stress=None)
    step = model.steps[0]
    boundary = compile_boundary(model, step)
    assert isinstance(boundary, CompiledBoundary)
    assert boundary.constraints.prescribed_values
    assert boundary.loads.nodal_forces

    problem = compile_analysis(
        model,
        resolve_analysis_request(step),
    ).problem
    evaluation = problem.evaluate()
    # Compilation owns the authoring-to-runtime boundary conversion.  The
    # mathematical Problem consumes only its typed equation data.
    assert problem.constraints == boundary.constraints
    assert not hasattr(problem, "boundary")
    assert len(evaluation.local_outputs) == 1
    assert len(evaluation.local_outputs[0].points) == 4
    assert evaluation.outputs["integration_points"]["element_id"].tolist() == [
        1,
        1,
        1,
        1,
    ]


def test_r5_attempt_record_is_shared_by_result_frame_and_monitor():
    monitor = RunDiagnostics("run-1", "Job-1", "Step-1")
    monitor.start()
    solved = incremental.solve(
        _RampProblem(),
        [0.5, 1.0],
        monitor=monitor,
        residual_tolerance=1.0e-12,
    )
    snapshot = monitor.snapshot()

    assert len(solved.attempts) == 2
    assert all(record.status == "converged" for record in solved.attempts)
    assert [
        attempt.duration_seconds for attempt in snapshot.attempts
    ] == pytest.approx(
        [record.duration_seconds for record in solved.attempts]
    )
    assert all(item.record is not None for item in solved.increments)


def test_r5_result_frame_catalog_reads_increment_facts_from_frame_outputs():
    model = _make_nonlinear_workflow_model(yield_stress=None)
    result = solve_analysis(model, model.steps[0])
    source = ResultSourceKey("result", "session", "artifact", 1, "Step-1", "run")

    catalog = frame_catalog_from_model_result(source, result)

    assert catalog.last is not None
    last = catalog.last
    assert last.increment_number == result.frames[-1].outputs["increment_number"]
    assert last.attempt_number == result.frames[-1].outputs["attempt_number"]
    assert last.duration_seconds is not None


def test_r6_field_location_identity_includes_section_point_and_label_is_central():
    first = FieldLocation(
        association=FieldAssociation.INTEGRATION_POINT,
        coordinates=(0.0, 0.0, 0.0),
        displacement=None,
        element_id=1,
        integration_point=1,
        section_point=BeamSectionPoint(1, 1.0, 1.0),
    )
    second = FieldLocation(
        association=FieldAssociation.INTEGRATION_POINT,
        coordinates=(0.0, 0.0, 0.0),
        displacement=None,
        element_id=1,
        integration_point=1,
        section_point=BeamSectionPoint(2, -1.0, 1.0),
    )

    assert first.identity_key() != second.identity_key()
    assert "截面位置 右上" in first.identity_label()
    assert "截面位置 左上" in second.identity_label()


def test_material_contract_owns_nested_state_and_readonly_response_arrays():
    committed = {"plastic_strain": np.array([1.0, 2.0])}
    update = MaterialPointInput(kinematics=object(), committed_state=committed)
    response_state = {"plastic_strain": np.array([3.0, 4.0])}
    response_outputs = {"stress_measure": {"name": "P"}}
    response = MaterialResponse(
        stress=np.eye(3),
        tangent=np.eye(9),
        stress_measure=StressMeasure.FIRST_PIOLA,
        tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
        trial_state=response_state,
        outputs=response_outputs,
    )

    committed["plastic_strain"][0] = 99.0
    response_state["plastic_strain"][0] = 88.0
    response_outputs["stress_measure"]["name"] = "S"

    assert update.committed_state["plastic_strain"][0] == pytest.approx(1.0)
    assert response.trial_state["plastic_strain"][0] == pytest.approx(3.0)
    assert response.outputs["stress_measure"]["name"] == "P"
    assert response.stress.flags.writeable is False
    assert response.tangent.flags.writeable is False


def test_transactional_state_manager_detaches_stage_and_rollback():
    manager = TransactionalStateManager()
    committed = {"Fp": np.eye(3)}
    key = StateKey.material_point(1, 1)
    manager.register(key, committed)
    committed["Fp"][0, 0] = 9.0

    trial = manager.committed(key)
    assert trial["Fp"][0, 0] == pytest.approx(1.0)

    staged = {"Fp": np.diag([2.0, 1.0, 1.0])}
    manager.stage(key, staged)
    staged["Fp"][0, 0] = 7.0
    assert manager.committed(key)["Fp"][0, 0] == pytest.approx(1.0)

    manager.rollback()
    assert manager.committed(key)["Fp"][0, 0] == pytest.approx(1.0)


def test_analysis_context_reaches_material_without_becoming_element_property():
    class RecordingMaterial:
        def __init__(self):
            self.contexts = []

        def initial_state(self):
            return {}

        def evaluate(self, point):
            self.contexts.append(
                (point.context.time, point.fields["temperature"])
            )
            return MaterialResponse(
                stress=np.zeros((3, 3)),
                tangent=np.eye(9),
                stress_measure=StressMeasure.FIRST_PIOLA,
                tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
            )

    model = _make_nonlinear_workflow_model(yield_stress=None)
    material = RecordingMaterial()
    assembly = SparseAssembler.from_displacement_mesh(
        model.mesh,
        ContinuumMechanicsOperator(
            kinematics=TotalLagrangianKinematics(),
            definition=Quad4Definition(),
        ),
        state=TransactionalStateManager(),
        material_by_element={1: material},
        element_properties_by_element={
            1: {"E": 210.0, "nu": 0.3, "thickness": 1.0},
        },
    )
    boundary = compile_boundary(model, model.steps[0])
    problem = StaticEquilibriumProblem(
        assembly=assembly,
        constraints=boundary.constraints,
        reference_load=build_load_vector(model.mesh, boundary.loads),
    )

    problem.evaluate(
        ProblemContext(
            solution=SolutionState.zeros(assembly.dof_space),
            evaluation=EvaluationContext(
                time=2.5,
                load_factor=0.0,
                parameters={"temperature": 55.0},
            ),
        )
    )

    assert material.contexts == [(pytest.approx(2.5), pytest.approx(55.0))] * 4


@pytest.mark.parametrize("field", ["time", "load_factor"])
def test_evaluation_context_rejects_nonfinite_coordinates(field):
    kwargs = {field: np.nan}
    with pytest.raises(ValueError, match="evaluation"):
        EvaluationContext(**kwargs)


def test_cutback_attempts_and_converged_frames_share_one_increment_history():
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
    model.steps[0].metadata.update(controls.to_metadata())
    monitor = RunDiagnostics("r-cutback", "Job-cutback", "nonlinear")
    monitor.start()

    result = solve_analysis(model, "nonlinear", monitor=monitor)
    snapshot = monitor.snapshot()
    converged = [
        attempt
        for attempt in snapshot.attempts
        if attempt.status is AttemptStatus.CONVERGED
    ]

    assert len(converged) == len(result.frames)
    assert any(
        attempt.status is AttemptStatus.CUTBACK
        for attempt in snapshot.attempts
    )
    assert [attempt.result_frame for attempt in converged] == list(
        range(1, len(result.frames) + 1)
    )
    assert [frame.outputs["increment_number"] for frame in result.frames] == [
        attempt.increment for attempt in converged
    ]
    assert [frame.outputs["attempt_number"] for frame in result.frames] == [
        attempt.attempt for attempt in converged
    ]
    assert result.outputs["attempt_count"] == len(snapshot.attempts)
    assert result.outputs["cutback_count"] == sum(
        attempt.status is AttemptStatus.CUTBACK
        for attempt in snapshot.attempts
    )
    source = ResultSourceKey("result", "session", "artifact", 1, "nonlinear", "run")
    catalog = frame_catalog_from_model_result(source, result)
    assert catalog.frames[0].cutbacks >= 0
    assert any(frame.cutbacks > 0 for frame in catalog.frames)
