from __future__ import annotations

import numpy as np

from fem.assembly import SparseAssembler
from fem.analysis.compilation import nonlinear_static_capability_for
from fem.application import analysis_solver_kind, solve_analysis
from fem.elements import Hex20Definition, Tet10Definition
from fem.materials import FiniteStrainElasticMaterial
from fem.model import (
    ElementSet,
    FEMModel,
    MaterialDefinition,
    Mesh3D,
    NodeSet,
    SectionAssignment,
    authoring as steps,
)
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.state import EvaluationContext, SolutionState, TransactionalStateManager
from tests.helpers.mesh_builders import make_hex20_stiffness_mesh, make_tet10_stiffness_mesh


def _assemble(
    mesh: Mesh3D,
    definition,
    material: FiniteStrainElasticMaterial,
    values: np.ndarray,
):
    properties = {
        int(element.id): dict(getattr(element, "props", {}))
        for element in mesh.elements
    }
    materials = {int(element.id): material for element in mesh.elements}
    assembler = SparseAssembler.from_displacement_mesh(
        mesh,
        ContinuumMechanicsOperator(
            kinematics=TotalLagrangianKinematics(),
            definition=definition,
        ),
        state=TransactionalStateManager(),
        material_by_element=materials,
        element_properties_by_element=properties,
    )
    assembler.begin_increment()
    return assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )


def _affine_values(mesh: Mesh3D, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        coordinates = np.array([node.x, node.y, node.z], dtype=float)
        displacement = gradient @ coordinates
        for component, value in enumerate(displacement):
            values[mesh.global_dof(node.id, component)] = value
    return values


def _finite_difference_tangent(
    mesh: Mesh3D,
    definition,
    material: FiniteStrainElasticMaterial,
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for column in range(mesh.num_dofs):
        direction = np.zeros(mesh.num_dofs, dtype=float)
        direction[column] = step
        plus = _assemble(mesh, definition, material, values + direction)
        minus = _assemble(mesh, definition, material, values - direction)
        columns.append((plus.residual - minus.residual) / (2.0 * step))
    return np.column_stack(columns)


def test_high_order_solid_definitions_preserve_interpolation_identities():
    for definition in (Tet10Definition(), Hex20Definition()):
        for point in definition.gauss_points():
            coordinates = point[:-1]
            np.testing.assert_allclose(
                definition.shape_functions(*coordinates).sum(),
                1.0,
                atol=1.0e-14,
            )
            np.testing.assert_allclose(
                definition.shape_gradients(*coordinates).sum(axis=1),
                0.0,
                atol=1.0e-14,
            )

    assert len(Tet10Definition().gauss_points()) == 4
    assert len(Hex20Definition().gauss_points()) == 27


def test_tet10_continuum_tangent_matches_finite_difference():
    mesh = make_tet10_stiffness_mesh()
    definition = Tet10Definition()
    material = FiniteStrainElasticMaterial(210.0, 0.3)
    values = _affine_values(
        mesh,
        np.array(
            [
                [0.02, 0.01, 0.003],
                [-0.005, 0.015, 0.004],
                [0.003, -0.002, 0.025],
            ],
            dtype=float,
        ),
    )
    base = _assemble(mesh, definition, material, values)
    numerical = _finite_difference_tangent(mesh, definition, material, values, 1.0e-7)

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=6.0e-6,
        atol=6.0e-7,
    )
    assert base.outputs["integration_points"]["element_id"].shape == (4,)


def test_hex20_continuum_tangent_matches_finite_difference():
    mesh = make_hex20_stiffness_mesh()
    definition = Hex20Definition()
    material = FiniteStrainElasticMaterial(210.0, 0.3)
    values = _affine_values(
        mesh,
        np.array(
            [
                [0.02, 0.01, 0.003],
                [-0.005, 0.015, 0.004],
                [0.003, -0.002, 0.025],
            ],
            dtype=float,
        ),
    )
    base = _assemble(mesh, definition, material, values)
    numerical = _finite_difference_tangent(mesh, definition, material, values, 1.0e-7)

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=7.0e-6,
        atol=7.0e-7,
    )
    assert base.outputs["integration_points"]["element_id"].shape == (27,)


def test_high_order_solid_aliases_resolve_to_one_common_operator():
    tet10 = nonlinear_static_capability_for("C3D10")
    hex20 = nonlinear_static_capability_for("C3D20")

    assert tet10.canonical_type == "Tet10"
    assert hex20.canonical_type == "Hex20"
    assert tet10.build_operator({}).__class__ is hex20.build_operator({}).__class__
    assert tet10.build_operator({}).definition.canonical_type == "Tet10"
    assert hex20.build_operator({}).definition.canonical_type == "Hex20"


def test_tet10_elastic_model_runs_through_compiled_nonlinear_static_path():
    model = FEMModel(
        mesh=make_tet10_stiffness_mesh(),
        name="tet10_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 2, 3, 5, 6, 7)),
            "loaded": NodeSet("loaded", (4, 8, 9, 10)),
        },
        element_sets={"part": ElementSet("part", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[SectionAssignment("part", "steel", "solid", {})],
    )
    step = steps.static("tet10-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.005)
    steps.add(model, step)

    assert analysis_solver_kind(model, "tet10-load") == "nonlinear_static"
    result = solve_analysis(model, "tet10-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (4, 3, 3)
    assert np.all(np.isfinite(result.U))


def test_hex20_elastic_model_runs_through_compiled_nonlinear_static_path():
    model = FEMModel(
        mesh=make_hex20_stiffness_mesh(),
        name="hex20_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 2, 3, 4, 9, 10, 11, 12)),
            "loaded": NodeSet("loaded", (2, 3, 6, 7, 10, 14, 18, 19)),
        },
        element_sets={"part": ElementSet("part", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[SectionAssignment("part", "steel", "solid", {})],
    )
    step = steps.static("hex20-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.005)
    steps.add(model, step)

    assert analysis_solver_kind(model, "hex20-load") == "nonlinear_static"
    result = solve_analysis(model, "hex20-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (27, 3, 3)
    assert np.all(np.isfinite(result.U))
