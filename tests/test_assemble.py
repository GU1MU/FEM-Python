import importlib
import sys

import numpy as np
import pytest

from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from fem.core.mesh import Element3D
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_stiffness_mesh,
    make_truss_stiffness_mesh,
)


def test_sparse_assembly_accepts_mesh_for_truss_and_beam():
    for mesh in (make_truss_stiffness_mesh(), make_beam_stiffness_mesh()):
        K = assemble_global_stiffness_sparse(mesh)

        assert K.shape == (mesh.num_dofs, mesh.num_dofs)
        assert np.allclose(K.toarray(), K.toarray().T)


def test_dense_and_sparse_assembly_accept_mesh():
    for mesh in (make_truss_stiffness_mesh(), make_beam_stiffness_mesh()):
        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        assert np.allclose(K_dense, K_sparse.toarray())


def test_assembly_requires_mesh_only():
    mesh = make_truss_stiffness_mesh()

    with pytest.raises(TypeError):
        assemble_global_stiffness()
    with pytest.raises(TypeError):
        assemble_global_stiffness_sparse()
    with pytest.raises(TypeError):
        assemble_global_stiffness(mesh, elements=mesh.elements)
    with pytest.raises(TypeError):
        assemble_global_stiffness_sparse(mesh.num_dofs, num_elements=1)
    with pytest.raises(TypeError):
        assemble_global_stiffness(num_dofs=mesh.num_dofs)
    with pytest.raises(TypeError):
        assemble_global_stiffness_sparse(
            mesh,
            get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
        )


def test_assemble_package_exposes_stiffness_module():
    import fem.assemble as assemble
    from fem.assemble import stiffness

    assert hasattr(assemble, "__path__")
    assert assemble.assemble_global_stiffness_sparse is stiffness.assemble_global_stiffness_sparse
    assert assemble.assemble_global_stiffness is stiffness.assemble_global_stiffness


def test_sparse_assembly_accepts_mesh_for_quad4():
    mesh = make_quad4_stiffness_mesh()

    K = assemble_global_stiffness_sparse(mesh)

    assert K.shape == (mesh.num_dofs, mesh.num_dofs)
    assert np.allclose(K.toarray(), K.toarray().T)


def test_dense_assembly_matches_sparse_for_quad4():
    mesh = make_quad4_stiffness_mesh()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse = assemble_global_stiffness_sparse(mesh)

    assert np.allclose(K_dense, K_sparse.toarray())


def test_sparse_assembly_accepts_mesh_for_tri3_and_quad8():
    for mesh in (make_tri3_stiffness_mesh(), make_quad8_stiffness_mesh()):
        K = assemble_global_stiffness_sparse(mesh)

        assert K.shape == (mesh.num_dofs, mesh.num_dofs)
        assert np.allclose(K.toarray(), K.toarray().T)


def test_dense_assembly_matches_sparse_for_tri3_and_quad8():
    for mesh in (make_tri3_stiffness_mesh(), make_quad8_stiffness_mesh()):
        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        assert np.allclose(K_dense, K_sparse.toarray())


def test_sparse_assembly_accepts_mesh_for_hex8():
    mesh = make_hex8_stiffness_mesh()

    K = assemble_global_stiffness_sparse(mesh)

    assert K.shape == (mesh.num_dofs, mesh.num_dofs)
    assert np.allclose(K.toarray(), K.toarray().T)


def test_dense_assembly_matches_sparse_for_hex8():
    mesh = make_hex8_stiffness_mesh()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse = assemble_global_stiffness_sparse(mesh)

    assert np.allclose(K_dense, K_sparse.toarray())


def test_sparse_assembly_accepts_mesh_for_tet4_and_tet10():
    for mesh in (make_tet4_stiffness_mesh(), make_tet10_stiffness_mesh()):
        K = assemble_global_stiffness_sparse(mesh)

        assert K.shape == (mesh.num_dofs, mesh.num_dofs)
        assert np.allclose(K.toarray(), K.toarray().T)


def test_dense_assembly_matches_sparse_for_tet4_and_tet10():
    for mesh in (make_tet4_stiffness_mesh(), make_tet10_stiffness_mesh()):
        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        assert np.allclose(K_dense, K_sparse.toarray())


def test_sparse_and_dense_assembly_accept_mixed_solid_mesh():
    mesh = make_mixed_hex8_tet4_mesh()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse = assemble_global_stiffness_sparse(mesh)

    assert K_dense.shape == (mesh.num_dofs, mesh.num_dofs)
    assert K_sparse.shape == (mesh.num_dofs, mesh.num_dofs)
    assert np.allclose(K_dense, K_dense.T)
    assert np.allclose(K_dense, K_sparse.toarray())


def test_sparse_and_dense_assembly_accept_mixed_plane_mesh():
    mesh = make_mixed_tri3_quad4_mesh()

    K_dense = assemble_global_stiffness(mesh)
    K_sparse = assemble_global_stiffness_sparse(mesh)

    assert K_dense.shape == (mesh.num_dofs, mesh.num_dofs)
    assert K_sparse.shape == (mesh.num_dofs, mesh.num_dofs)
    assert np.allclose(K_dense, K_dense.T)
    assert np.allclose(K_dense, K_sparse.toarray())


def test_assembly_reports_unsupported_element_type_in_mixed_mesh():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements.append(Element3D(3, [1, 2, 3, 5], "UnsupportedSolid", {}))

    with pytest.raises(NotImplementedError, match="Unsupported element type: UnsupportedSolid"):
        assemble_global_stiffness_sparse(mesh)


def test_stiffness_module_is_removed_in_favor_of_element_kernels():
    sys.modules.pop("fem.stiffness", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.stiffness")
