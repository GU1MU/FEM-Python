from __future__ import annotations

import numpy as np
import pytest

from fem.assembly import SparseAssembler
from fem.elements import Quad4Definition
from fem.materials import (
    GreenLagrangeElasticMaterial,
    J2PlasticityMaterial,
)
from fem.materials.contracts import MaterialPointInput
from fem.model import Element2D, Mesh2D, Node2D
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.physics.mechanics.kinematics import KinematicPoint
from fem.state import (
    EvaluationContext,
    SolutionState,
    StateKey,
    TransactionalStateManager,
)


def _quad4_mesh() -> Mesh2D:
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 2.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[Element2D(1, [1, 2, 3, 4], "Quad4")],
    )


def _quad4_patch_mesh() -> Mesh2D:
    nodes = [
        Node2D(node_id, float(x), float(y))
        for node_id, (x, y) in enumerate(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (0.0, 2.0),
                (1.0, 2.0),
                (2.0, 2.0),
            ),
            start=1,
        )
    ]
    elements = [
        Element2D(1, [1, 2, 5, 4], "Quad4"),
        Element2D(2, [2, 3, 6, 5], "Quad4"),
        Element2D(3, [4, 5, 8, 7], "Quad4"),
        Element2D(4, [5, 6, 9, 8], "Quad4"),
    ]
    return Mesh2D(nodes=nodes, elements=elements)


def _assembler(mesh: Mesh2D, material) -> SparseAssembler:
    properties = {
        int(element.id): {
            "E": material.E,
            "nu": material.nu,
            "thickness": 1.0,
            "plane_type": "stress",
        }
        for element in mesh.elements
    }
    materials = {int(element.id): material for element in mesh.elements}
    return SparseAssembler.from_displacement_mesh(
        mesh,
        ContinuumMechanicsOperator(
            kinematics=TotalLagrangianKinematics(),
            definition=Quad4Definition(),
        ),
        state=TransactionalStateManager(),
        material_by_element=materials,
        element_properties_by_element=properties,
    )


def _affine_values(mesh: Mesh2D, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        displacement = gradient @ np.array([node.x, node.y], dtype=float)
        values[mesh.global_dof(node.id, 0)] = displacement[0]
        values[mesh.global_dof(node.id, 1)] = displacement[1]
    return values


def _assemble(assembler: SparseAssembler, values: np.ndarray):
    assembler.begin_increment()
    return assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )


def test_quad4_affine_patch_has_no_interior_node_residual():
    mesh = _quad4_patch_mesh()
    assembler = _assembler(mesh, GreenLagrangeElasticMaterial(210.0, 0.3))
    gradient = np.array(
        [[0.02, 0.01], [-0.015, 0.025]],
        dtype=float,
    )

    result = _assemble(assembler, _affine_values(mesh, gradient))

    np.testing.assert_allclose(
        result.residual[list(mesh.node_dofs(5))],
        0.0,
        rtol=0.0,
        atol=2.0e-10,
    )


def test_quad4_elastic_internal_force_tangent_matches_global_finite_difference():
    mesh = _quad4_mesh()
    assembler = _assembler(mesh, GreenLagrangeElasticMaterial(210.0, 0.3))
    values = _affine_values(
        mesh,
        np.array([[0.03, 0.02], [-0.01, 0.015]], dtype=float),
    )
    base = _assemble(assembler, values)
    step = 1.0e-7
    numerical = np.column_stack(
        [
            (
                _assemble(assembler, values + step * np.eye(mesh.num_dofs)[:, column]).residual
                - _assemble(assembler, values - step * np.eye(mesh.num_dofs)[:, column]).residual
            )
            / (2.0 * step)
            for column in range(mesh.num_dofs)
        ]
    )

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=3.0e-6,
        atol=3.0e-7,
    )


def _kinematics(F: np.ndarray) -> KinematicPoint:
    return KinematicPoint(
        reference_shape_gradients=np.zeros((2, 4), dtype=float),
        deformation_gradient=F,
        green_lagrange_strain=0.5 * (F.T @ F - np.eye(2)),
        reference_jacobian_determinant=1.0,
    )


@pytest.mark.parametrize("algorithm", ("hencky", "multiplicative"))
def test_quad4_j2_material_tangent_matches_fixed_state_finite_difference(algorithm):
    material = J2PlasticityMaterial(
        210.0,
        0.3,
        yield_stress=0.5,
        hardening_modulus=10.0,
        algorithm=algorithm,
    )
    F = np.array([[1.04, 0.03], [0.01, 0.97]], dtype=float)
    committed = material.initial_state()
    reference = material.evaluate(
        MaterialPointInput(kinematics=_kinematics(F), committed_state=committed)
    )
    step = 1.0e-7
    numerical = np.zeros((9, 4), dtype=float)
    columns = ((0, 0), (0, 1), (1, 0), (1, 1))
    for column, index in enumerate(columns):
        plus = F.copy()
        minus = F.copy()
        plus[index] += step
        minus[index] -= step
        stress_plus = material.evaluate(
            MaterialPointInput(kinematics=_kinematics(plus), committed_state=committed)
        ).stress
        stress_minus = material.evaluate(
            MaterialPointInput(kinematics=_kinematics(minus), committed_state=committed)
        ).stress
        numerical[:, column] = ((stress_plus - stress_minus) / (2.0 * step)).reshape(9)

    tangent_columns = [0, 1, 3, 4]
    np.testing.assert_allclose(
        numerical,
        reference.tangent[:, tangent_columns],
        rtol=3.0e-4,
        atol=3.0e-5,
    )


def test_quad4_j2_trial_history_rolls_back_and_repeats_deterministically():
    mesh = _quad4_mesh()
    material = J2PlasticityMaterial(
        210.0,
        0.3,
        yield_stress=0.5,
        hardening_modulus=10.0,
    )
    assembler = _assembler(mesh, material)
    values = _affine_values(
        mesh,
        np.array([[0.08, 0.02], [0.01, -0.05]], dtype=float),
    )

    trial = _assemble(assembler, values)
    key = StateKey.material_point(1, 1)
    assert assembler.state is not None
    assert assembler.state.committed(key)["equivalent_plastic_strain"] == 0.0
    assert trial.outputs["integration_points"]["history"][0][
        "equivalent_plastic_strain"
    ] > 0.0

    assembler.rollback()
    repeat = _assemble(assembler, values)
    np.testing.assert_allclose(repeat.residual, trial.residual)
    assert assembler.state.committed(key)["equivalent_plastic_strain"] == 0.0

    assembler.commit()
    assert assembler.state.committed(key)["equivalent_plastic_strain"] > 0.0
