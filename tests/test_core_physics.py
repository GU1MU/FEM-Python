from pathlib import Path
import tomllib

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.core.model import AnalysisStep, DisplacementConstraint, FEMModel, NodalLoad, NodeSet
from fem.elements import get_element_kernel
from fem.materials import linear_elastic
from fem.solvers import linear, static_linear
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_tri6_stiffness_mesh,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "builder",
    [
        make_tri3_stiffness_mesh,
        make_tri6_stiffness_mesh,
        make_quad4_stiffness_mesh,
        make_quad8_stiffness_mesh,
    ],
    ids=["tri3", "tri6", "quad4", "quad8"],
)
def test_plane_elements_reproduce_affine_constant_stress(builder):
    mesh = builder()
    elem = mesh.elements[0]
    U = np.zeros(mesh.num_dofs, dtype=float)

    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = 0.3 + 0.01 * node.x + 0.02 * node.y
        U[mesh.global_dof(node.id, 1)] = -0.2 - 0.03 * node.x + 0.04 * node.y

    strain = np.array([0.01, 0.04, -0.01])
    expected = linear_elastic.plane_stress_matrix(
        elem.props["E"], elem.props["nu"]
    ) @ strain
    recovered, plane_type, poisson_ratio = get_element_kernel(elem.type).nodal_stress(
        mesh, elem, U
    )

    assert plane_type == "stress"
    assert poisson_ratio == pytest.approx(elem.props["nu"])
    assert np.allclose(recovered, expected, rtol=1e-10, atol=1e-10)


def test_cpe4_reproduces_nonzero_affine_plane_strain_stress():
    mesh = make_quad4_stiffness_mesh()
    elem = mesh.elements[0]
    elem.type = "CPE4"
    elem.props.pop("plane_type", None)
    U = np.zeros(mesh.num_dofs, dtype=float)

    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = 0.3 + 0.01 * node.x + 0.02 * node.y
        U[mesh.global_dof(node.id, 1)] = -0.2 - 0.03 * node.x + 0.04 * node.y

    strain = np.array([0.01, 0.04, -0.01])
    expected = linear_elastic.plane_strain_matrix(
        elem.props["E"], elem.props["nu"]
    ) @ strain
    recovered, plane_type, _ = get_element_kernel(elem.type).nodal_stress(
        mesh, elem, U
    )

    assert plane_type == "strain"
    assert np.linalg.norm(expected) > 0.0
    assert np.allclose(recovered, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("builder", "element_type", "expected_plane_type"),
    [
        (make_tri3_stiffness_mesh, "CPS3", "stress"),
        (make_tri3_stiffness_mesh, "CPE3", "strain"),
        (make_tri6_stiffness_mesh, "CPS6", "stress"),
        (make_tri6_stiffness_mesh, "CPE6", "strain"),
        (make_quad4_stiffness_mesh, "CPS4", "stress"),
        (make_quad4_stiffness_mesh, "CPE4", "strain"),
        (make_quad8_stiffness_mesh, "CPS8", "stress"),
        (make_quad8_stiffness_mesh, "CPE8", "strain"),
    ],
)
def test_plane_aliases_infer_formulation_without_explicit_property(
    builder, element_type, expected_plane_type
):
    mesh = builder()
    elem = mesh.elements[0]
    elem.type = element_type
    elem.props.pop("plane_type", None)

    _, plane_type, _ = get_element_kernel(element_type).nodal_stress(
        mesh, elem, np.zeros(mesh.num_dofs)
    )

    assert plane_type == expected_plane_type


@pytest.mark.parametrize(
    "builder",
    [
        make_tri3_stiffness_mesh,
        make_tri6_stiffness_mesh,
        make_quad4_stiffness_mesh,
        make_quad8_stiffness_mesh,
    ],
    ids=["tri3", "tri6", "quad4", "quad8"],
)
@pytest.mark.parametrize("thickness", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_plane_elements_reject_invalid_thickness(builder, thickness):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props["thickness"] = thickness

    with pytest.raises(ValueError, match="thickness must be finite and > 0"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


@pytest.mark.parametrize(
    "builder",
    [
        make_hex8_stiffness_mesh,
        make_hex20_stiffness_mesh,
        make_tet4_stiffness_mesh,
        make_tet10_stiffness_mesh,
    ],
    ids=["hex8", "hex20", "tet4", "tet10"],
)
def test_solid_elements_reproduce_affine_constant_stress(builder):
    mesh = builder()
    elem = mesh.elements[0]
    U = np.zeros(mesh.num_dofs, dtype=float)

    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = (
            0.3 + 0.01 * node.x + 0.02 * node.y + 0.03 * node.z
        )
        U[mesh.global_dof(node.id, 1)] = (
            -0.2 - 0.04 * node.x + 0.05 * node.y + 0.06 * node.z
        )
        U[mesh.global_dof(node.id, 2)] = (
            0.1 + 0.07 * node.x - 0.08 * node.y + 0.09 * node.z
        )

    strain = np.array([0.01, 0.05, 0.09, -0.02, -0.02, 0.10])
    expected = linear_elastic.solid_3d_matrix(
        elem.props["E"], elem.props["nu"]
    ) @ strain
    recovered = get_element_kernel(elem.type).nodal_stress(mesh, elem, U)

    assert np.allclose(recovered, expected, rtol=1e-9, atol=1e-9)


def test_nonzero_dirichlet_constraint_preserves_coupled_solution():
    K = csr_matrix([[2.0, -2.0], [-2.0, 2.0]])
    F = np.zeros(2, dtype=float)
    boundary = BoundaryCondition(prescribed_displacements={0: 0.25})

    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    U = linear.solve(K_mod, F_mod)

    assert U == pytest.approx([0.25, 0.25])
    assert K @ U - F == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("mesh_builder", "fixed_nodes", "loaded_node", "components"),
    [
        (make_mixed_tri3_quad4_mesh, (1, 4), 5, (1, 2)),
        (make_mixed_hex8_tet4_mesh, (1, 4, 5, 8), 9, (1, 2, 3)),
    ],
    ids=["connected_plane", "connected_solid"],
)
def test_connected_mixed_models_preserve_global_force_balance(
    mesh_builder, fixed_nodes, loaded_node, components
):
    mesh = mesh_builder()
    model = FEMModel(
        mesh=mesh,
        node_sets={
            "fixed": NodeSet("fixed", fixed_nodes),
            "loaded": NodeSet("loaded", (loaded_node,)),
        },
        steps=[
            AnalysisStep(
                "pull",
                boundaries=(
                    DisplacementConstraint(
                        "fixed", min(components), max(components), 0.0
                    ),
                ),
                cloads=(NodalLoad("loaded", 1, 1.0),),
            )
        ],
    )

    result = static_linear.solve(model, "pull")

    assert np.all(np.isfinite(result.U))
    assert float(result.reactions[0::mesh.dofs_per_node].sum()) == pytest.approx(-1.0)
    for component in range(1, mesh.dofs_per_node):
        assert float(result.reactions[component::mesh.dofs_per_node].sum()) == pytest.approx(
            0.0, abs=1e-10
        )


def test_project_declares_tested_numerical_runtime_dependencies():
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert {"numpy>=2.3,<3", "scipy>=1.16,<2"}.issubset(project["dependencies"])


def test_registry_rejects_unsupported_coupled_temperature_element():
    with pytest.raises(NotImplementedError, match="Unsupported element type: C3D4T"):
        get_element_kernel("C3D4T")
