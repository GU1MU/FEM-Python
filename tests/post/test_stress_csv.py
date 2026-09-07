import csv

import numpy as np
import pytest

from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.elements import get_element_kernel
from fem.post import (
    displacement,
    stress,
)
from fem.post.stress import dispatch
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_mixed_hex20_tet10_mesh,
    make_mixed_hex8_hex20_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_mixed_tri6_quad8_mesh,
    make_unit_hex8_mesh,
)


from tests.helpers.post_builders import (
    _affine_solid_displacement, _write_current_element_stress, _write_current_nodal_stress,
)


def _make_beam_dispatch_mesh():
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2")],
        dofs_per_node=6,
    )


def _assert_csv_values_equal(actual, expected):
    with actual.open(newline="", encoding="utf-8") as stream:
        actual_rows = list(csv.reader(stream))
    with expected.open(newline="", encoding="utf-8") as stream:
        expected_rows = list(csv.reader(stream))
    assert actual_rows[0] == expected_rows[0]
    assert len(actual_rows) == len(expected_rows)
    continuous_columns = {
        "x", "y", "z", "xi", "eta", "zeta",
        "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises",
        "S11", "S22", "S33", "S12", "S13", "S23",
        "Mises", "MaxPrincipal", "MidPrincipal", "MinPrincipal",
    }
    for actual_row, expected_row in zip(actual_rows[1:], expected_rows[1:]):
        assert len(actual_row) == len(expected_row)
        for column, actual_value, expected_value in zip(actual_rows[0], actual_row, expected_row):
            if column in continuous_columns and expected_value != "":
                assert float(actual_value) == pytest.approx(float(expected_value), abs=1e-12)
            else:
                assert actual_value == expected_value


def test_recovered_plane_stress_writers_match_legacy_exports(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    mesh.elements[0].props["plane_type"] = "strain"
    mesh.elements[1].props = dict(mesh.elements[0].props)
    displacement = np.asarray([
        0.0,
        0.001,
        0.01,
        0.002,
        0.013,
        0.018,
        -0.002,
        0.016,
        0.021,
        -0.004,
    ])
    recovered = stress.StressRecovery(mesh, displacement).collect(
        stress.StressPosition.ELEMENT_NODAL
    )
    csv_recovered = stress.collect_plane_element_nodal(mesh, displacement)

    expected_element = tmp_path / "expected_element.csv"
    actual_element = tmp_path / "actual_element.csv"
    stress.element.mixed(
        ("tri3", "quad4"),
        mesh,
        displacement,
        expected_element,
    )
    stress.element.write_plane_element_nodal(
        mesh,
        recovered,
        actual_element,
    )

    expected_nodal = tmp_path / "expected_nodal.csv"
    actual_nodal = tmp_path / "actual_nodal.csv"
    legacy_raw = stress.nodal_from_stress_field(mesh, recovered)
    legacy_resolved = stress.field.resolve(legacy_raw, threshold=100.0)
    stress.nodal._write_resolved(
        mesh,
        legacy_resolved,
        expected_nodal,
    )
    stress.nodal.write_recovered(
        mesh,
        csv_recovered,
        actual_nodal,
        threshold=100.0,
    )

    expected_canonical = tmp_path / "expected_canonical.csv"
    actual_canonical = tmp_path / "actual_canonical.csv"
    stress.export.csv(
        mesh,
        displacement,
        expected_canonical,
        position=stress.StressPosition.ELEMENT_NODAL,
    )
    stress.export.write_csv(recovered, actual_canonical)

    _assert_csv_values_equal(actual_element, expected_element)
    _assert_csv_values_equal(actual_nodal, expected_nodal)
    _assert_csv_values_equal(actual_canonical, expected_canonical)


def test_recovered_solid_nodal_writer_matches_legacy_resolution(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[1].props = dict(mesh.elements[0].props)
    displacement = np.arange(mesh.num_dofs, dtype=float) ** 2 * 0.001
    recovered = stress.StressRecovery(mesh, displacement).collect(
        stress.StressPosition.ELEMENT_NODAL
    )
    expected = tmp_path / "expected_solid_nodal.csv"
    actual = tmp_path / "actual_solid_nodal.csv"

    legacy_raw = stress.nodal_from_stress_field(mesh, recovered)
    legacy_resolved = stress.field.resolve(legacy_raw, threshold=100.0)
    stress.nodal._write_resolved(mesh, legacy_resolved, expected)
    stress.nodal.write_recovered(
        mesh,
        recovered,
        actual,
        threshold=100.0,
    )

    _assert_csv_values_equal(actual, expected)


def test_canonical_csv_defaults_to_integration_points_and_writes_s33(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    output = tmp_path / "stress.csv"

    stress.export.csv(mesh, np.zeros(mesh.num_dofs), output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["position"] for row in rows} == {"integration_point"}
    assert all(row["integration_point"] for row in rows)
    assert all(row["S33"] for row in rows)


def test_deprecated_stress_export_wrappers_emit_explicit_warnings(tmp_path):
    mesh = make_unit_hex8_mesh()
    displacement = np.zeros(mesh.num_dofs)

    with pytest.warns(
        DeprecationWarning,
        match=r"stress\.export\.element\(\) is deprecated",
    ):
        stress.export.element(
            mesh,
            displacement,
            tmp_path / "compat-element.csv",
        )
    with pytest.warns(
        DeprecationWarning,
        match=r"stress\.export\.nodal\(\) is deprecated",
    ):
        stress.export.nodal(
            mesh,
            displacement,
            tmp_path / "compat-nodal.csv",
        )


def test_nodal_stress_csv_preserves_material_boundary_contributions(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(
                1,
                [1, 2, 3],
                "Tri3",
                {"E": 100.0, "nu": 0.25, "plane_type": "stress", "thickness": 1.0},
            ),
            Element2D(
                2,
                [3, 4, 5],
                "Tri3",
                {"E": 200.0, "nu": 0.3, "plane_type": "stress", "thickness": 1.0},
            ),
        ],
    )
    csv_path = tmp_path / "material_boundary.csv"

    _write_current_nodal_stress(
        mesh,
        np.zeros(mesh.num_dofs),
        csv_path,
        threshold=100.0,
    )

    with csv_path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == [
        "node_id", "x", "y", "elem_id", "local_node", "averaged",
        "sig_x", "sig_y", "tau_xy", "mises",
    ]
    assert [(int(row["node_id"]), int(row["elem_id"]), int(row["local_node"])) for row in rows] == [
        (1, 1, 1), (2, 1, 2), (3, 1, 3), (3, 2, 1), (4, 2, 2), (5, 2, 3),
    ]
    assert all(row["averaged"] == "false" for row in rows)
    np.testing.assert_allclose(
        [[float(row[key]) for key in ("x", "y")] for row in rows],
        [[0, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [0, -1]],
    )
    assert all(float(row[key]) == 0.0 for row in rows for key in ("sig_x", "sig_y", "tau_xy", "mises"))


def test_dispatch_rejects_reduced_integration_hex20():
    element_type = "C3D20R"
    mesh = make_hex20_stiffness_mesh()
    mesh.elements[0].type = element_type

    assert dispatch.type_key_from_name(element_type) is None
    with pytest.raises(
        ValueError,
        match=rf"Unsupported stress element type: '{element_type}'",
    ):
        dispatch.resolve_type_keys(mesh, None)


def test_hex20_stress_exports_write_one_element_and_twenty_nodes(tmp_path):
    mesh = make_hex20_stiffness_mesh(curved=True)
    mesh.elements[0].type = "C3D20"
    U = _affine_solid_displacement(mesh)
    elem_path = tmp_path / "hex20_element_stress.csv"
    nodal_path = tmp_path / "hex20_nodal_stress.csv"

    _write_current_element_stress(mesh, U, elem_path)
    _write_current_nodal_stress(mesh, U, nodal_path)

    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert len(elem_rows) == 2
    assert len(nodal_rows) == 21
    rows_by_node = {int(row[0]): row for row in nodal_rows[1:]}
    expected = get_element_kernel(mesh.elements[0].type).nodal_stress(
        mesh,
        mesh.elements[0],
        U,
    )
    for local_index in (0, 8):
        node_id = mesh.elements[0].node_ids[local_index]
        exported = np.array([float(value) for value in rows_by_node[node_id][7:13]])
        assert not np.allclose(expected[local_index], 0.0)
        assert np.allclose(exported, expected[local_index])


def test_mixed_solid_element_export_uses_hex20_and_tet4_centroids(tmp_path):
    hex20_mesh = make_hex20_stiffness_mesh(curved=True)
    mesh = Mesh3D(
        nodes=[*hex20_mesh.nodes, Node3D(21, 2.0, 0.0, 0.0)],
        elements=[
            hex20_mesh.elements[0],
            Element3D(
                2,
                [2, 21, 3, 6],
                "Tet4",
                {"E": 120.0, "nu": 0.25},
            ),
        ],
    )
    U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
    csv_path = tmp_path / "mixed_hex20_tet4_element_stress.csv"

    _write_current_element_stress(mesh, U, csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    expected = [
        get_element_kernel(mesh.elements[0].type).stress_at(
            mesh, mesh.elements[0], U, 0.0, 0.0, 0.0
        ),
        get_element_kernel(mesh.elements[1].type).stress_at(
            mesh, mesh.elements[1], U, 0.25, 0.25, 0.25
        ),
    ]

    assert [row[0] for row in rows[1:]] == ["1", "2"]
    for row, expected_stress in zip(rows[1:], expected):
        assert np.allclose([float(value) for value in row[1:7]], expected_stress)


def test_direct_post_exports_create_parent_dirs_and_beam_uses_six_components(tmp_path):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2")],
        dofs_per_node=6,
    )
    output_path = tmp_path / "nested" / "beam_displacement.csv"

    displacement.export.nodal(mesh, np.arange(12, dtype=float), output_path)

    with output_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["node_id"]) for row in rows] == [1, 2]
    np.testing.assert_allclose(
        [[float(row[key]) for key in ("ux", "uy", "uz", "rx", "ry", "rz")] for row in rows],
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
    )


@pytest.mark.parametrize(
    (
        "mesh_builder",
        "expected_keys",
        "expected_group",
        "element_supported",
        "nodal_supported",
        "gauss_order",
    ),
    (
        (make_hex20_stiffness_mesh, ("hex20",), "solid", True, True, 3),
        (make_mixed_hex8_tet4_mesh, ("hex8", "tet4"), "solid", True, True, None),
        (make_mixed_hex8_hex20_mesh, ("hex8", "hex20"), "solid", True, True, None),
        (make_mixed_hex20_tet10_mesh, ("hex20", "tet10"), "solid", True, True, None),
        (make_mixed_tri3_quad4_mesh, ("tri3", "quad4"), "plane", True, True, None),
        (make_mixed_tri6_quad8_mesh, ("tri6", "quad8"), "plane", True, True, None),
        (_make_beam_dispatch_mesh, ("beam2",), "line", False, True, None),
    ),
)
def test_post_stress_dispatch_supports_current_type_groups(
    mesh_builder,
    expected_keys,
    expected_group,
    element_supported,
    nodal_supported,
    gauss_order,
):
    mesh = mesh_builder()

    assert dispatch.resolve_type_keys(mesh, None) == expected_keys
    assert dispatch.stress_group_for_keys(expected_keys) == expected_group
    assert dispatch.element_stress_supported(expected_keys) is element_supported
    assert dispatch.nodal_stress_supported(expected_keys) is nodal_supported
    if gauss_order is not None:
        assert dispatch.default_gauss_order(expected_keys[0]) == gauss_order


@pytest.mark.parametrize(
    ("mesh_builder", "name", "element_row_count"),
    (
        (make_mixed_hex8_tet4_mesh, "mixed_solid", 3),
        (make_mixed_tri3_quad4_mesh, "mixed_plane", 8),
    ),
)
def test_mixed_stress_exports_write_element_and_nodal_rows(
    tmp_path,
    mesh_builder,
    name,
    element_row_count,
):
    mesh = mesh_builder()
    elem_path = tmp_path / f"{name}_element_stress.csv"
    nodal_path = tmp_path / f"{name}_nodal_stress.csv"

    _write_current_element_stress(mesh, np.zeros(mesh.num_dofs), elem_path)
    _write_current_nodal_stress(mesh, np.zeros(mesh.num_dofs), nodal_path)
    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == element_row_count
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == sum(len(elem.node_ids) for elem in mesh.elements) + 1
    assert {row[0] for row in nodal_rows[1:]} == {str(node.id) for node in mesh.nodes}
