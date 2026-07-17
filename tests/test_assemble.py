import numpy as np
import pytest

from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from fem.core.mesh import Element3D
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_mixed_hex20_tet10_mesh,
    make_mixed_hex8_hex20_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_mixed_tri6_quad8_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_tri6_stiffness_mesh,
    make_truss_stiffness_mesh,
)


@pytest.mark.parametrize(
    "mesh_builder",
    [
        make_truss_stiffness_mesh,
        make_beam_stiffness_mesh,
        make_tri3_stiffness_mesh,
        make_tri6_stiffness_mesh,
        make_quad4_stiffness_mesh,
        make_quad8_stiffness_mesh,
        make_hex8_stiffness_mesh,
        make_hex20_stiffness_mesh,
        make_tet4_stiffness_mesh,
        make_tet10_stiffness_mesh,
    ],
)
def test_dense_and_sparse_assembly_agree_for_supported_single_elements(mesh_builder):
    mesh = mesh_builder()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse_dense = assemble_global_stiffness_sparse(mesh).toarray()

    expected_shape = (mesh.num_dofs, mesh.num_dofs)
    assert K_dense.shape == expected_shape
    assert K_sparse_dense.shape == expected_shape
    assert np.all(np.isfinite(K_dense))
    assert np.all(np.isfinite(K_sparse_dense))
    assert np.allclose(K_dense, K_dense.T)
    assert np.allclose(K_sparse_dense, K_sparse_dense.T)
    assert np.allclose(K_dense, K_sparse_dense)


@pytest.mark.parametrize(
    "mesh_builder",
    [
        make_mixed_tri3_quad4_mesh,
        make_mixed_tri6_quad8_mesh,
        make_mixed_hex8_tet4_mesh,
        make_mixed_hex8_hex20_mesh,
        make_mixed_hex20_tet10_mesh,
    ],
)
def test_dense_and_sparse_assembly_agree_for_supported_mixed_meshes(mesh_builder):
    mesh = mesh_builder()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse_dense = assemble_global_stiffness_sparse(mesh).toarray()

    expected_shape = (mesh.num_dofs, mesh.num_dofs)
    assert K_dense.shape == expected_shape
    assert K_sparse_dense.shape == expected_shape
    assert np.all(np.isfinite(K_dense))
    assert np.all(np.isfinite(K_sparse_dense))
    assert np.allclose(K_dense, K_dense.T)
    assert np.allclose(K_sparse_dense, K_sparse_dense.T)
    assert np.allclose(K_dense, K_sparse_dense)


def test_assembly_reports_unsupported_element_type_in_mixed_mesh():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements.append(Element3D(3, [1, 2, 3, 5], "UnsupportedSolid", {}))

    with pytest.raises(NotImplementedError, match="Unsupported element type: UnsupportedSolid"):
        assemble_global_stiffness_sparse(mesh)
