from __future__ import annotations

import numpy as np
import pytest

from fem.assembly import SparseAssembler
from fem.analysis.compilation import nonlinear_static_capability_for
from fem.elements import (
    Hex20Definition,
    Hex8Definition,
    Quad4Definition,
    Quad8Definition,
    Tet10Definition,
    Tet4Definition,
    Tri3Definition,
    Tri6Definition,
)
from fem.materials import J2PlasticityMaterial
from fem.model import Mesh2D, Mesh3D
from fem.physics.mechanics import ContinuumMechanicsOperator, TotalLagrangianKinematics
from fem.state import (
    EvaluationContext,
    SolutionState,
    StateKey,
    TransactionalStateManager,
)
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_tri6_stiffness_mesh,
)


_PLANE_GRADIENT = np.array(
    [[0.08, 0.02], [0.01, -0.05]],
    dtype=float,
)
_SOLID_GRADIENT = np.array(
    [
        [0.02, 0.01, 0.003],
        [-0.005, 0.015, 0.004],
        [0.003, -0.002, 0.025],
    ],
    dtype=float,
)


def _affine_values(mesh: Mesh2D | Mesh3D, gradient: np.ndarray) -> np.ndarray:
    values = np.zeros(mesh.num_dofs, dtype=float)
    for node in mesh.nodes:
        coordinates = np.array(
            [
                getattr(node, name)
                for name in ("x", "y", "z")
                if hasattr(node, name)
            ],
            dtype=float,
        )
        displacement = gradient @ coordinates
        for component, value in enumerate(displacement):
            values[mesh.global_dof(node.id, component)] = value
    return values


def _build_assembler(mesh, definition, material):
    properties = {
        int(element.id): {
            **dict(getattr(element, "props", {})),
            "E": material.E,
            "nu": material.nu,
        }
        for element in mesh.elements
    }
    materials = {int(element.id): material for element in mesh.elements}
    return SparseAssembler.from_displacement_mesh(
        mesh,
        ContinuumMechanicsOperator(
            kinematics=TotalLagrangianKinematics(),
            definition=definition,
        ),
        state=TransactionalStateManager(),
        material_by_element=materials,
        element_properties_by_element=properties,
    )


@pytest.mark.parametrize(
    ("mesh_factory", "definition", "gradient", "point_count"),
    (
        (make_quad4_stiffness_mesh, Quad4Definition(), _PLANE_GRADIENT, 4),
        (make_tri3_stiffness_mesh, Tri3Definition(), _PLANE_GRADIENT, 1),
        (make_quad8_stiffness_mesh, Quad8Definition(), _PLANE_GRADIENT, 9),
        (make_tri6_stiffness_mesh, Tri6Definition(), _PLANE_GRADIENT, 3),
        (make_hex8_stiffness_mesh, Hex8Definition(), _SOLID_GRADIENT, 8),
        (make_tet4_stiffness_mesh, Tet4Definition(), _SOLID_GRADIENT, 1),
        (make_hex20_stiffness_mesh, Hex20Definition(), _SOLID_GRADIENT, 27),
        (make_tet10_stiffness_mesh, Tet10Definition(), _SOLID_GRADIENT, 4),
    ),
    ids=("quad4", "tri3", "quad8", "tri6", "hex8", "tet4", "hex20", "tet10"),
)
def test_j2_trial_commit_rollback_is_identical_across_continuum_elements(
    mesh_factory,
    definition,
    gradient: np.ndarray,
    point_count: int,
):
    mesh = mesh_factory()
    material = J2PlasticityMaterial(
        210.0,
        0.3,
        yield_stress=0.5,
        hardening_modulus=10.0,
        algorithm="multiplicative",
    )
    assembler = _build_assembler(mesh, definition, material)
    values = _affine_values(mesh, gradient)

    assembler.begin_increment()
    trial = assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )
    points = trial.outputs["integration_points"]

    assert points["element_id"].shape == (point_count,)
    assert points["equivalent_plastic_strain"].shape == (point_count,)
    assert np.all(points["equivalent_plastic_strain"] > 0.0)
    assert np.all(np.isfinite(trial.residual))
    assert np.all(np.isfinite(trial.tangent.data))

    key_values = []
    for point_id in range(1, point_count + 1):
        key = StateKey.material_point(1, point_id)
        assert assembler.state is not None
        key_values.append(assembler.state.committed(key))
        assert key_values[-1]["equivalent_plastic_strain"] == pytest.approx(0.0)

    assembler.rollback()
    assembler.begin_increment()
    repeat = assembler.assemble(
        SolutionState(assembler.dof_space, values),
        context=EvaluationContext(load_factor=1.0),
    )
    np.testing.assert_allclose(repeat.residual, trial.residual)
    np.testing.assert_allclose(repeat.tangent.toarray(), trial.tangent.toarray())

    assembler.commit()
    assert assembler.state is not None
    for point_id in range(1, point_count + 1):
        committed = assembler.state.committed(StateKey.material_point(1, point_id))
        assert committed["equivalent_plastic_strain"] > 0.0


@pytest.mark.parametrize(
    "element_type",
    ("Quad4", "Tri3", "Quad8", "Tri6", "Hex8", "Tet4", "Hex20", "Tet10"),
)
def test_all_continuum_capabilities_build_the_same_j2_material_contract(element_type):
    capability = nonlinear_static_capability_for(element_type)
    properties = {
        "E": 210.0,
        "nu": 0.3,
        "constitutive_model": "j2_plasticity",
        "yield_stress": 0.5,
        "hardening_modulus": 10.0,
        "thickness": 1.0,
        "plane_type": "strain",
    }

    material = capability.build_material(
        properties,
        {"material_algorithm": "multiplicative"},
        element_id=1,
        element_type=element_type,
    )

    assert isinstance(material, J2PlasticityMaterial)
    assert material.algorithm == "multiplicative"
    assert material.initial_state()["equivalent_plastic_strain"] == 0.0
