from __future__ import annotations

import numpy as np

from fem.assembly import SparseAssembler
from fem.analysis.compilation import nonlinear_static_capability_for
from fem.application import analysis_solver_kind, run_static_preflight, solve_analysis
from fem.elements import Hex8Definition, Tri3Definition
from fem.materials import (
    FiniteStrainElasticMaterial,
    GreenLagrangeElasticMaterial,
)
from fem.model import (
    ElementSet,
    FEMModel,
    MaterialDefinition,
    Mesh2D,
    Mesh3D,
    NodeSet,
    SectionAssignment,
    authoring as steps,
)
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.state import EvaluationContext, SolutionState, TransactionalStateManager
from tests.helpers.mesh_builders import (
    make_tri3_stiffness_mesh,
    make_unit_hex8_mesh,
)


def _assemble(
    mesh: Mesh2D | Mesh3D,
    definition,
    material,
    values: np.ndarray,
) -> tuple[SparseAssembler, object]:
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
    result = assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )
    return assembler, result


def _affine_values(mesh: Mesh2D | Mesh3D, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        coordinates = np.array(
            [getattr(node, name) for name in ("x", "y", "z") if hasattr(node, name)],
            dtype=float,
        )
        displacement = gradient @ coordinates
        for component, value in enumerate(displacement):
            values[mesh.global_dof(node.id, component)] = value
    return values


def _finite_difference_tangent(
    mesh: Mesh2D | Mesh3D,
    definition,
    material,
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for column in range(mesh.num_dofs):
        direction = np.zeros(mesh.num_dofs, dtype=float)
        direction[column] = step
        _, plus = _assemble(mesh, definition, material, values + direction)
        _, minus = _assemble(mesh, definition, material, values - direction)
        columns.append((plus.residual - minus.residual) / (2.0 * step))
    return np.column_stack(columns)


def test_tri3_definition_has_partition_of_unity_and_full_area_rule():
    definition = Tri3Definition()
    shape = definition.shape_functions(1.0 / 3.0, 1.0 / 3.0)

    np.testing.assert_allclose(shape.sum(), 1.0)
    np.testing.assert_allclose(
        definition.shape_gradients(0.2, 0.3).sum(axis=1),
        0.0,
    )
    assert definition.gauss_points() == ((1.0 / 3.0, 1.0 / 3.0, 0.5),)


def test_hex8_definition_has_partition_of_unity_and_unit_reference_volume_rule():
    definition = Hex8Definition()

    for point in definition.gauss_points():
        shape = definition.shape_functions(*point[:-1])
        np.testing.assert_allclose(shape.sum(), 1.0)
        np.testing.assert_allclose(
            definition.shape_gradients(*point[:-1]).sum(axis=1),
            0.0,
        )
    assert sum(point[-1] for point in definition.gauss_points()) == 8.0


def test_tri3_continuum_internal_force_tangent_matches_finite_difference():
    mesh = make_tri3_stiffness_mesh()
    values = _affine_values(
        mesh,
        np.array([[0.03, 0.02], [-0.01, 0.015]], dtype=float),
    )
    material = GreenLagrangeElasticMaterial(210.0, 0.3)
    _, base = _assemble(mesh, Tri3Definition(), material, values)
    numerical = _finite_difference_tangent(
        mesh,
        Tri3Definition(),
        material,
        values,
        1.0e-7,
    )

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=3.0e-6,
        atol=3.0e-7,
    )


def test_hex8_continuum_internal_force_tangent_matches_finite_difference():
    mesh = make_unit_hex8_mesh()
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
    material = FiniteStrainElasticMaterial(210.0, 0.3)
    _, base = _assemble(mesh, Hex8Definition(), material, values)
    numerical = _finite_difference_tangent(
        mesh,
        Hex8Definition(),
        material,
        values,
        1.0e-7,
    )

    np.testing.assert_allclose(
        numerical,
        base.tangent.toarray(),
        rtol=5.0e-6,
        atol=5.0e-7,
    )
    integration = base.outputs["integration_points"]
    assert integration["element_id"].shape == (8,)
    assert integration["deformation_gradient"].shape == (8, 3, 3)


def test_low_order_continuum_definitions_use_expected_dof_sizes():
    tri_mesh = make_tri3_stiffness_mesh()
    hex_mesh = make_unit_hex8_mesh()

    assert tri_mesh.num_dofs == 6
    assert hex_mesh.num_dofs == 24
    assert Tri3Definition.node_count == 3
    assert Hex8Definition.node_count == 8


def test_nonlinear_capability_registry_binds_low_order_elements_to_one_operator():
    tri_capability = nonlinear_static_capability_for("CPS3")
    hex_capability = nonlinear_static_capability_for("C3D8")

    assert tri_capability.canonical_type == "Tri3"
    assert hex_capability.canonical_type == "Hex8"
    assert tri_capability.build_operator({}).definition.canonical_type == "Tri3"
    assert hex_capability.build_operator({}).definition.canonical_type == "Hex8"


def test_hex8_elastic_model_runs_through_compiled_nonlinear_static_path():
    mesh = make_unit_hex8_mesh()
    model = FEMModel(
        mesh=mesh,
        name="hex8_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 2, 3, 4)),
            "loaded": NodeSet("loaded", (5, 6, 7, 8)),
        },
        element_sets={"block": ElementSet("block", (1,))},
        materials={
            "steel": MaterialDefinition("steel", {"E": 210.0, "nu": 0.3})
        },
        sections=[SectionAssignment("block", "steel", "solid", {})],
    )
    step = steps.static("block-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    report = run_static_preflight(model, "block-load")
    assert report.passed
    assert analysis_solver_kind(model, "block-load") == "nonlinear_static"

    result = solve_analysis(model, "block-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (8, 3, 3)
    assert np.all(np.isfinite(result.U))


def test_tri3_elastic_model_runs_through_compiled_nonlinear_static_path():
    mesh = make_tri3_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        name="tri3_nonlinear_foundation",
        node_sets={
            "fixed": NodeSet("fixed", (1, 3)),
            "loaded": NodeSet("loaded", (2,)),
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
    step = steps.static("triangle-load", NLGEOM=True)
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=0.01)
    steps.add(model, step)

    result = solve_analysis(model, "triangle-load")

    assert result.outputs["material"] == "linear_elastic"
    assert result.frames[-1].outputs["integration_points"][
        "deformation_gradient"
    ].shape == (1, 3, 3)
    assert np.all(np.isfinite(result.U))
