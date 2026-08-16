import gc
import tracemalloc
from time import perf_counter

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from fem.assemble import assemble_global_stiffness, assemble_global_stiffness_sparse
from fem.assemble import stiffness as stiffness_module
from fem.core.mesh import Element2D, Element3D, Mesh2D, Node2D
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


def test_sparse_assembly_plan_uses_only_flat_exact_preallocations():
    mesh = make_mixed_tri3_quad4_mesh()

    plan = stiffness_module._build_assembly_plan(mesh)
    expected_dof_count = sum(len(mesh.element_dofs(elem)) for elem in mesh.elements)
    expected_entry_count = sum(
        len(mesh.element_dofs(elem)) ** 2 for elem in mesh.elements
    )

    assert plan.dof_offsets.shape == (len(mesh.elements) + 1,)
    assert plan.entry_offsets.shape == (len(mesh.elements) + 1,)
    assert int(plan.dof_offsets[-1]) == expected_dof_count
    assert plan.rows.shape == (expected_entry_count,)
    assert plan.cols.shape == (expected_entry_count,)
    assert all(
        isinstance(values, np.ndarray) and values.ndim == 1
        for values in (
            plan.dof_offsets,
            plan.entry_offsets,
            plan.rows,
            plan.cols,
        )
    )
    assert not hasattr(plan, "__dict__")


def test_sparse_assembly_sums_repeated_coo_entries(monkeypatch):
    mesh = make_truss_stiffness_mesh()
    first = mesh.elements[0]
    mesh.elements.append(
        Element3D(
            id=2,
            node_ids=list(first.node_ids),
            type=first.type,
            props=dict(first.props),
        )
    )

    class Kernel:
        def stiffness(self, mesh, elem, node_lookup=None):
            dof_count = len(mesh.element_dofs(elem))
            return np.ones((dof_count, dof_count))

    monkeypatch.setattr(
        stiffness_module,
        "get_element_kernel",
        lambda element_type: Kernel(),
    )

    assembled = assemble_global_stiffness_sparse(mesh).toarray()

    assert np.array_equal(assembled, np.full(assembled.shape, 2.0))


def test_sparse_assembly_preserves_zero_dof_kernel_shape_diagnostic(
    monkeypatch,
):
    mesh = make_truss_stiffness_mesh()
    mesh.elements[0].node_ids = []
    mesh.rebuild_dof_map()

    class Kernel:
        def stiffness(self, mesh, elem, node_lookup=None):
            return np.ones((1, 1))

    monkeypatch.setattr(
        stiffness_module,
        "get_element_kernel",
        lambda element_type: Kernel(),
    )

    with pytest.raises(
        ValueError,
        match=r"stiffness shape \(1, 1\) does not match 0 DOFs",
    ):
        assemble_global_stiffness_sparse(mesh)


@pytest.mark.parametrize(
    ("failure", "expected_exception", "message"),
    [
        ("shape", ValueError, "stiffness shape"),
        ("nonfinite", ValueError, "contains non-finite values"),
        ("asymmetric", ValueError, "stiffness is not symmetric"),
    ],
)
def test_sparse_assembly_preserves_kernel_output_diagnostics(
    monkeypatch,
    failure,
    expected_exception,
    message,
):
    mesh = make_truss_stiffness_mesh()
    dof_count = len(mesh.element_dofs(mesh.elements[0]))
    stiffness = np.eye(dof_count)
    if failure == "shape":
        stiffness = stiffness[:-1, :-1]
    elif failure == "nonfinite":
        stiffness[0, 0] = np.inf
    else:
        stiffness[0, 1] = 1.0

    class Kernel:
        def stiffness(self, mesh, elem, node_lookup=None):
            return stiffness

    monkeypatch.setattr(
        stiffness_module,
        "get_element_kernel",
        lambda element_type: Kernel(),
    )

    with pytest.raises(expected_exception, match=message):
        assemble_global_stiffness_sparse(mesh)


@pytest.mark.parametrize(
    ("invalid_dof", "expected_exception", "message"),
    [
        (6, IndexError, r"out of bounds \[0, 6\)"),
        (1.5, TypeError, "DOF index must be an integer"),
        (True, TypeError, "DOF index must be an integer"),
    ],
)
def test_sparse_assembly_validates_custom_element_dof_indices(
    invalid_dof,
    expected_exception,
    message,
):
    mesh = make_truss_stiffness_mesh()
    valid_dofs = list(mesh.element_dofs(mesh.elements[0]))
    valid_dofs[-1] = invalid_dof
    mesh.element_dofs = lambda elem: tuple(valid_dofs)

    with pytest.raises(expected_exception, match=message):
        assemble_global_stiffness_sparse(mesh)


def test_sparse_assembly_rejects_element_dof_mapping_length_changes():
    mesh = make_truss_stiffness_mesh()
    calls = 0

    def changing_element_dofs(elem):
        nonlocal calls
        calls += 1
        dofs = tuple(range(mesh.num_dofs))
        return dofs if calls == 1 else dofs[:-1]

    mesh.element_dofs = changing_element_dofs

    with pytest.raises(ValueError, match="DOF mapping changed"):
        assemble_global_stiffness_sparse(mesh)


@pytest.mark.parametrize(
    "assembler",
    [assemble_global_stiffness, assemble_global_stiffness_sparse],
)
@pytest.mark.parametrize("strict", [None, 1, "true"])
def test_assembly_requires_boolean_strict_option(assembler, strict):
    with pytest.raises(TypeError, match="strict must be bool"):
        assembler(make_truss_stiffness_mesh(), strict=strict)


def test_sparse_assembly_fast_path_only_skips_symmetry(monkeypatch):
    mesh = make_truss_stiffness_mesh()
    dof_count = len(mesh.element_dofs(mesh.elements[0]))
    asymmetric = np.eye(dof_count)
    asymmetric[0, 1] = 1.0

    class Kernel:
        def stiffness(self, mesh, elem, node_lookup=None):
            return asymmetric

    monkeypatch.setattr(
        stiffness_module,
        "get_element_kernel",
        lambda element_type: Kernel(),
    )

    assembled = assemble_global_stiffness_sparse(mesh, strict=False).toarray()
    assert np.array_equal(assembled, asymmetric)

    asymmetric[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        assemble_global_stiffness_sparse(mesh, strict=False)


def test_medium_mixed_plane_assembly_matches_legacy_oracle_and_records_cost(
    record_property,
):
    mesh = _make_medium_mixed_plane_mesh(cell_count=12)
    assemble_global_stiffness_sparse(mesh)
    _legacy_sparse_assembly_oracle(mesh)

    current, current_seconds, current_peak = _measure_assembly(
        lambda: assemble_global_stiffness_sparse(mesh)
    )
    legacy, legacy_seconds, legacy_peak = _measure_assembly(
        lambda: _legacy_sparse_assembly_oracle(mesh)
    )

    assert np.allclose(current.toarray(), legacy.toarray())
    record_property("current_assembly_seconds", current_seconds)
    record_property("legacy_assembly_seconds", legacy_seconds)
    record_property("current_tracemalloc_peak_bytes", current_peak)
    record_property("legacy_tracemalloc_peak_bytes", legacy_peak)


def _make_medium_mixed_plane_mesh(cell_count):
    nodes = [
        Node2D(
            id=row * (cell_count + 1) + column + 1,
            x=float(column),
            y=float(row),
        )
        for row in range(cell_count + 1)
        for column in range(cell_count + 1)
    ]
    elements = []
    element_id = 1
    properties = {
        "E": 210.0,
        "nu": 0.3,
        "thickness": 1.0,
        "plane_type": "stress",
    }
    for row in range(cell_count):
        for column in range(cell_count):
            lower_left = row * (cell_count + 1) + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + cell_count + 1
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                elements.append(
                    Element2D(
                        element_id,
                        [
                            lower_left,
                            lower_right,
                            upper_right,
                            upper_left,
                        ],
                        "Quad4",
                        dict(properties),
                    )
                )
                element_id += 1
            else:
                elements.extend(
                    (
                        Element2D(
                            element_id,
                            [lower_left, lower_right, upper_right],
                            "Tri3",
                            dict(properties),
                        ),
                        Element2D(
                            element_id + 1,
                            [lower_left, upper_right, upper_left],
                            "Tri3",
                            dict(properties),
                        ),
                    )
                )
                element_id += 2
    return Mesh2D(nodes, elements)


def _legacy_sparse_assembly_oracle(mesh):
    stiffness_module._validate_mesh(mesh)
    node_lookup = {node.id: node for node in mesh.nodes}
    row_blocks = []
    col_blocks = []
    data_blocks = []
    for elem in mesh.elements:
        element_stiffness = stiffness_module.get_element_kernel(
            elem.type
        ).stiffness(mesh, elem, node_lookup=node_lookup)
        dofs = stiffness_module._validated_element_dofs(mesh, elem)
        element_stiffness = stiffness_module._validate_element_stiffness(
            element_stiffness,
            len(dofs),
            elem,
            strict=True,
        )
        dof_array = np.asarray(dofs, dtype=np.int64)
        row_blocks.append(np.repeat(dof_array, dof_array.size))
        col_blocks.append(np.tile(dof_array, dof_array.size))
        data_blocks.append(element_stiffness.reshape(-1))
    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    data = np.concatenate(data_blocks)
    return coo_matrix(
        (data, (rows, cols)),
        shape=(mesh.num_dofs, mesh.num_dofs),
    ).tocsr()


def _measure_assembly(callback):
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    result = callback()
    elapsed = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak
