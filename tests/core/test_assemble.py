from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from fem.assemble import stiffness as stiffness_module
from fem.core.mesh import Element3D, Mesh3D, Node3D
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


ASSEMBLERS = [
    pytest.param(assemble_global_stiffness, id="dense"),
    pytest.param(assemble_global_stiffness_sparse, id="sparse"),
]


@pytest.mark.parametrize(
    "mesh_builder",
    [
        pytest.param(make_truss_stiffness_mesh, id="truss2"),
        pytest.param(make_beam_stiffness_mesh, id="beam2"),
        pytest.param(make_tri3_stiffness_mesh, id="tri3"),
        pytest.param(make_tri6_stiffness_mesh, id="tri6"),
        pytest.param(make_quad4_stiffness_mesh, id="quad4"),
        pytest.param(make_quad8_stiffness_mesh, id="quad8"),
        pytest.param(make_hex8_stiffness_mesh, id="hex8"),
        pytest.param(make_hex20_stiffness_mesh, id="hex20"),
        pytest.param(make_tet4_stiffness_mesh, id="tet4"),
        pytest.param(make_tet10_stiffness_mesh, id="tet10"),
        pytest.param(make_mixed_tri3_quad4_mesh, id="tri3-quad4"),
        pytest.param(make_mixed_tri6_quad8_mesh, id="tri6-quad8"),
        pytest.param(make_mixed_hex8_tet4_mesh, id="hex8-tet4"),
        pytest.param(make_mixed_hex8_hex20_mesh, id="hex8-hex20"),
        pytest.param(make_mixed_hex20_tet10_mesh, id="hex20-tet10"),
    ],
)
def test_dense_and_sparse_assembly_agree_for_supported_meshes(mesh_builder):
    mesh = mesh_builder()
    dense = assemble_global_stiffness(mesh)
    sparse = assemble_global_stiffness_sparse(mesh).toarray()

    for matrix in (dense, sparse):
        assert matrix.shape == (mesh.num_dofs, mesh.num_dofs)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
    np.testing.assert_allclose(dense, sparse, atol=1e-10)


def test_connected_trusses_scatter_and_sum_in_global_dof_order():
    mesh = Mesh3D(
        nodes=[
            Node3D(70, 3.0, 1.0, 0.0),
            Node3D(30, 0.0, 0.0, 0.0),
            Node3D(10, 1.0, 1.0, 0.0),
        ],
        elements=[
            Element3D(8, [30, 10], "Truss2", {"E": np.sqrt(8), "area": 1.0}),
            Element3D(3, [70, 10], "Truss2", {"E": 6.0, "area": 1.0}),
        ],
    )
    # DOFs are (10x, 10y, 10z, 30x, 30y, 30z, 70x, 70y, 70z).
    # The diagonal rod has EA/L = 2 and cx = cy = 1/sqrt(2),
    # giving unit x/y coefficients; the horizontal rod has EA/L = 3.
    expected = np.array([
        [4, 1, 0, -1, -1, 0, -3, 0, 0],
        [1, 1, 0, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 1, 1, 0, 0, 0, 0],
        [-1, -1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    reordered = Mesh3D(list(reversed(mesh.nodes)), list(reversed(mesh.elements)))
    for candidate in (mesh, reordered):
        for assembler in (assemble_global_stiffness, assemble_global_stiffness_sparse):
            original_nodes = deepcopy(candidate.nodes)
            original_elements = deepcopy(candidate.elements)
            assembled = assembler(candidate)
            if assembler is assemble_global_stiffness_sparse:
                assembled = assembled.toarray()
            np.testing.assert_allclose(assembled, expected, atol=1e-12)
            assert candidate.nodes == original_nodes
            assert candidate.elements == original_elements


def test_assembly_reports_unsupported_element_type_in_mixed_mesh():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements.append(Element3D(3, [1, 2, 3, 5], "UnsupportedSolid", {}))

    with pytest.raises(NotImplementedError, match="Unsupported element type: UnsupportedSolid"):
        assemble_global_stiffness_sparse(mesh)


@pytest.fixture
def kernel_output(monkeypatch):
    output = SimpleNamespace(matrix=np.eye(6))
    kernel = SimpleNamespace(stiffness=lambda *args, **kwargs: output.matrix)
    monkeypatch.setattr(stiffness_module, "get_element_kernel", lambda _: kernel)
    return output


@pytest.mark.parametrize("assembler", ASSEMBLERS)
@pytest.mark.parametrize(
    ("failure", "strict", "message"),
    [
        pytest.param("shape", True, "stiffness shape", id="shape-strict"),
        pytest.param("nonfinite", True, "contains non-finite values", id="nonfinite-strict"),
        pytest.param("asymmetric", True, "stiffness is not symmetric", id="asymmetric-strict"),
        pytest.param("shape", False, "stiffness shape", id="shape-nonstrict"),
        pytest.param("nonfinite", False, "contains non-finite values", id="nonfinite-nonstrict"),
    ],
)
def test_assembly_rejects_invalid_kernel_output(assembler, kernel_output, failure, strict, message):
    if failure == "shape":
        kernel_output.matrix = kernel_output.matrix[:-1, :-1]
    elif failure == "nonfinite":
        kernel_output.matrix[0, 0] = np.nan
    else:
        kernel_output.matrix[0, 1] = 1.0

    with pytest.raises(ValueError, match=message):
        assembler(make_truss_stiffness_mesh(), strict=strict)


@pytest.mark.parametrize("assembler", ASSEMBLERS)
def test_assembly_allows_asymmetric_kernel_when_not_strict(assembler, kernel_output):
    kernel_output.matrix[0, 1] = 1.0
    assembled = assembler(make_truss_stiffness_mesh(), strict=False)
    if assembler is assemble_global_stiffness_sparse:
        assembled = assembled.toarray()
    np.testing.assert_array_equal(assembled, kernel_output.matrix)


@pytest.mark.parametrize(
    ("invalid_dof", "expected_exception", "message"),
    [
        pytest.param(-1, IndexError, "out of bounds", id="negative"),
        pytest.param(6, IndexError, "out of bounds", id="past-end"),
        pytest.param(1.5, TypeError, "DOF index must be an integer", id="fraction"),
        pytest.param(True, TypeError, "DOF index must be an integer", id="boolean"),
    ],
)
def test_assembly_validates_custom_element_dof_indices(invalid_dof, expected_exception, message):
    mesh = make_truss_stiffness_mesh()
    dofs = list(mesh.element_dofs(mesh.elements[0]))
    dofs[-1] = invalid_dof
    mesh.element_dofs = lambda elem: tuple(dofs)

    with pytest.raises(expected_exception, match=message):
        assemble_global_stiffness_sparse(mesh)


@pytest.mark.parametrize("assembler", ASSEMBLERS)
def test_assembly_requires_boolean_strict_option(assembler):
    with pytest.raises(TypeError, match="strict must be bool"):
        assembler(make_truss_stiffness_mesh(), strict=1)
