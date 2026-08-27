from __future__ import annotations

import numpy as np

from fem.assembly import SparseAssembler
from fem.analysis.compilation import nonlinear_static_capability_for
from fem.application import analysis_solver_kind, solve_analysis
from fem.elements import Quad8Definition, Tri6Definition
from fem.materials import GreenLagrangeElasticMaterial
from fem.model import (
    ElementSet,
    FEMModel,
    MaterialDefinition,
    Mesh2D,
    NodeSet,
    SectionAssignment,
    authoring as steps,
)
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.state import EvaluationContext, SolutionState, TransactionalStateManager
from tests.helpers.mesh_builders import make_quad8_stiffness_mesh, make_tri6_stiffness_mesh


def _assemble(
    mesh: Mesh2D,
    definition,
    material: GreenLagrangeElasticMaterial,
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


def _affine_values(mesh: Mesh2D, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        displacement = gradient @ np.array([node.x, node.y], dtype=float)
        for component, value in enumerate(displacement):
            values[mesh.global_dof(node.id, component)] = value
    return values


def _finite_difference_tangent(
    mesh: Mesh2D,
    definition,
    material: GreenLagrangeElasticMaterial,
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


def test_high_order_plane_definitions_preserve_interpolation_identities():
    for definition in (Quad8Definition(), Tri6Definition()):
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

    assert len(Quad8Definition().gauss_points()) == 9
    assert len(Tri6Definition().gauss_points()) == 3


def test_quad8_continuum_tangent_matches_finite_difference():
    mesh = make_quad8_stiffness_mesh()
    definition = Quad8Definition()
    material = GreenLagrangeElasticMaterial(210.0, 0.3, "stress")
    values = _affine_values(
        mesh,
        np.array([[0.03, 0.02], [-0.01, 0.015]], dtype=float),
    )
    base = _assemble(mesh, definition, material, values)
    numerical = _finite_difference_tangent(mesh, definition, material, values, 1.0e-7)

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=4.0e-6,
        atol=4.0e-7,
    )
    assert base.outputs["integration_points"]["element_id"].shape == (9,)


def test_tri6_continuum_tangent_matches_finite_difference():
    mesh = make_tri6_stiffness_mesh()
    definition = Tri6Definition()
    material = GreenLagrangeElasticMaterial(210.0, 0.3, "stress")
    values = _affine_values(
        mesh,
        np.array([[0.03, 0.02], [-0.01, 0.015]], dtype=float),
    )
    base = _assemble(mesh, definition, material, values)
    numerical = _finite_difference_tangent(mesh, definition, material, values, 1.0e-7)

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=4.0e-6,
        atol=4.0e-7,
    )
    assert base.outputs["integration_points"]["element_id"].shape == (3,)


def test_high_order_plane_aliases_resolve_to_one_common_operator():
    quad8 = nonlinear_static_capability_for("CPS8")
    tri6 = nonlinear_static_capability_for("CPE6")

    assert quad8.canonical_type == "Quad8"
    assert tri6.canonical_type == "Tri6"
    assert quad8.build_operator({}).__class__ is tri6.build_operator({}).__class__
    assert quad8.build_operator({}).definition.canonical_type == "Quad8"
    assert tri6.build_operator({}).definition.canonical_type == "Tri6"


def test_quad8_elastic_model_runs_through_compiled_nonlinear_static_path():
    model = FEMModel(
        mesh=make_quad8_stiffness_mesh(),
        name="quad8_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 4, 8)),
            "loaded": NodeSet("loaded", (2, 3, 6)),
        },
        element_sets={"part": ElementSet("part", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[
            SectionAssignment(
                "part",
                "steel",
                "solid",
                {"thickness": 1.0, "plane_type": "stress"},
            )
        ],
    )
    step = steps.static("quad8-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    assert analysis_solver_kind(model, "quad8-load") == "nonlinear_static"
    result = solve_analysis(model, "quad8-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (9, 3, 3)
    assert np.all(np.isfinite(result.U))


def test_tri6_elastic_model_runs_through_compiled_nonlinear_static_path():
    model = FEMModel(
        mesh=make_tri6_stiffness_mesh(),
        name="tri6_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 3, 6)),
            "loaded": NodeSet("loaded", (2, 4)),
        },
        element_sets={"part": ElementSet("part", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[
            SectionAssignment(
                "part",
                "steel",
                "solid",
                {"thickness": 1.0, "plane_type": "stress"},
            )
        ],
    )
    step = steps.static("tri6-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    assert analysis_solver_kind(model, "tri6-load") == "nonlinear_static"
    result = solve_analysis(model, "tri6-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (3, 3, 3)
    assert np.all(np.isfinite(result.U))
