from __future__ import annotations

import numpy as np

from fem.application import solve_analysis
from fem.assembly import SparseAssembler
from fem.elements import Tet4Definition
from fem.materials import FiniteStrainElasticMaterial
from fem.model import (
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodeSet,
    SectionAssignment,
)
from fem.model import authoring as steps
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.state import EvaluationContext, SolutionState, TransactionalStateManager
from tests.helpers.mesh_builders import make_tet4_stiffness_mesh


def _affine_values(mesh, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        coordinates = np.array([node.x, node.y, node.z], dtype=float)
        displacement = gradient @ coordinates
        for component, value in enumerate(displacement):
            values[mesh.global_dof(node.id, component)] = value
    return values


def _assemble(mesh, values: np.ndarray):
    material = FiniteStrainElasticMaterial(210.0, 0.3)
    assembler = SparseAssembler.from_displacement_mesh(
        mesh,
        ContinuumMechanicsOperator(
            kinematics=TotalLagrangianKinematics(),
            definition=Tet4Definition(),
        ),
        state=TransactionalStateManager(),
        material_by_element={1: material},
        element_properties_by_element={1: {"E": 210.0, "nu": 0.3}},
    )
    assembler.begin_increment()
    result = assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )
    return result


def test_tet4_definition_has_partition_of_unity_and_reference_volume_rule():
    definition = Tet4Definition()
    point = definition.gauss_points()[0]
    shape = definition.shape_functions(*point[:-1])

    np.testing.assert_allclose(shape.sum(), 1.0)
    np.testing.assert_allclose(
        definition.shape_gradients(*point[:-1]).sum(axis=1),
        0.0,
    )
    assert point == (0.25, 0.25, 0.25, 1.0 / 6.0)


def test_tet4_continuum_internal_force_tangent_matches_finite_difference():
    mesh = make_tet4_stiffness_mesh()
    values = _affine_values(
        mesh,
        np.array(
            [
                [0.02, 0.01, 0.0],
                [-0.005, 0.015, 0.004],
                [0.003, -0.002, 0.025],
            ],
            dtype=float,
        ),
    )
    base = _assemble(mesh, values)
    step = 1.0e-7
    columns = []
    for column in range(mesh.num_dofs):
        direction = np.zeros(mesh.num_dofs, dtype=float)
        direction[column] = step
        plus = _assemble(mesh, values + direction)
        minus = _assemble(mesh, values - direction)
        columns.append((plus.residual - minus.residual) / (2.0 * step))

    np.testing.assert_allclose(
        np.column_stack(columns),
        base.tangent.toarray(),
        rtol=5.0e-6,
        atol=5.0e-7,
    )


def test_tet4_elastic_model_runs_through_compiled_nonlinear_static_path():
    mesh = make_tet4_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        name="tet4_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 3, 4)),
            "loaded": NodeSet("loaded", (2,)),
        },
        element_sets={"part": ElementSet("part", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[SectionAssignment("part", "steel", "solid", {})],
    )
    step = steps.static("tetra-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    result = solve_analysis(model, "tetra-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (1, 3, 3)
    assert np.all(np.isfinite(result.U))
