import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.boundary.condition import BoundaryCondition
from fem.boundary.constraints import apply_dirichlet
from fem.core.model import AnalysisStep, DisplacementConstraint, FEMModel, NodalLoad, NodeSet
from fem.elements import get_element_kernel
from fem.solvers import linear, static_linear
from tests.helpers.mesh_builders import (
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_tri6_stiffness_mesh,
)


@pytest.mark.parametrize(
    "builder",
    [
        make_tri3_stiffness_mesh,
        make_quad4_stiffness_mesh,
        make_quad8_stiffness_mesh,
    ],
    ids=["shared_triangle", "quad4", "quad8"],
)
@pytest.mark.parametrize("thickness", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_plane_thickness_validators_reject_invalid_equivalence_classes(
    builder,
    thickness,
):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props["thickness"] = thickness

    with pytest.raises(ValueError, match="thickness must be finite and > 0"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_tri6_stiffness_consumer_rejects_invalid_shared_thickness():
    mesh = make_tri6_stiffness_mesh()
    elem = mesh.elements[0]
    elem.props["thickness"] = 0.0

    with pytest.raises(ValueError, match="thickness must be finite and > 0"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_nonzero_dirichlet_constraint_preserves_coupled_solution():
    K = csr_matrix([[2.0, -2.0], [-2.0, 2.0]])
    F = np.zeros(2, dtype=float)
    boundary = BoundaryCondition(prescribed_displacements={0: 0.25})

    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    U = linear.solve(K_mod, F_mod)

    np.testing.assert_allclose(K_mod.toarray(), [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(F_mod, [0.25, 0.5])
    assert U == pytest.approx([0.25, 0.25])
    assert K @ U - F == pytest.approx([0.0, 0.0])
    np.testing.assert_array_equal(K.toarray(), [[2.0, -2.0], [-2.0, 2.0]])
    np.testing.assert_array_equal(F, [0.0, 0.0])
    assert boundary.prescribed_displacements == {0: 0.25}


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


def test_registry_rejects_unsupported_coupled_temperature_element():
    with pytest.raises(NotImplementedError, match="Unsupported element type: C3D4T"):
        get_element_kernel("C3D4T")
