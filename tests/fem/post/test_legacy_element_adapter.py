from __future__ import annotations

import csv

import numpy as np
import pytest

from fem.elements import get_element_kernel
from fem.post.stress import element
from fem.post.stress.field import StressPosition, collect_stress
from tests.helpers.mesh_builders import (
    make_mixed_tri3_quad4_mesh, make_mixed_tri6_quad8_mesh,
    make_quad8_stiffness_mesh, make_unit_hex8_mesh,
)


@pytest.mark.parametrize(
    ("builder", "type_keys", "plane_type", "expected_values"),
    [
        (make_quad8_stiffness_mesh, ("quad8",), "strain", (2.4, 3.36, 1.44, np.sqrt(8.9856))),
        (make_mixed_tri3_quad4_mesh, ("tri3", "quad4"), "stress", (1.92, 2.88, 1.44, np.sqrt(12.672))),
        (make_mixed_tri6_quad8_mesh, ("tri6", "quad8"), "stress", (1.92, 2.88, 1.44, np.sqrt(12.672))),
    ],
    ids=["single-plane-strain", "mixed-linear", "mixed-quadratic"],
)
def test_legacy_plane_csv_preserves_schema_order_and_analytic_stress(
    tmp_path, builder, type_keys, plane_type, expected_values,
):
    mesh = builder()
    for elem in mesh.elements:
        elem.props.update(E=120.0, nu=0.25, plane_type=plane_type)
    displacement = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = 0.01 * node.x + 0.03 * node.y
        displacement[mesh.global_dof(node.id, 1)] = 0.02 * node.y
    target = tmp_path / "plane-element.csv"

    if len(type_keys) == 1:
        element.quad8(mesh, displacement, target, gauss_order=3)
    else:
        element.mixed(type_keys, mesh, displacement, target)

    with target.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises",
        ]
        rows = list(reader)
    assert [(int(row["elem_id"]), int(row["node_id"]), int(row["local_node"])) for row in rows] == [
        (elem.id, node_id, local_node)
        for elem in mesh.elements
        for local_node, node_id in enumerate(elem.node_ids, start=1)
    ]
    # Strains (.01, .02, .03 engineering shear), G=48. Plane strain adds sigma_z=1.44.
    # Mises squared = half the sum of squared normal differences + 3*tau_xy**2.
    expected = np.tile(expected_values, (len(rows), 1))
    np.testing.assert_allclose(
        [[float(row[name]) for name in ("sig_x", "sig_y", "tau_xy", "mises")] for row in rows],
        expected, atol=1e-12,
    )


def test_distorted_solid_legacy_representative_point_is_not_centroid_field(
    tmp_path,
):
    mesh = make_unit_hex8_mesh()
    mesh.nodes[2].x = 1.4
    mesh.nodes[2].y = 1.2
    mesh.nodes[5].z = 1.3
    mesh.nodes[6].x = 0.8
    mesh.nodes[7].y = 0.7
    displacement = np.arange(mesh.num_dofs, dtype=float) ** 2 * 0.001
    target = tmp_path / "distorted-hex8-element.csv"

    element.hex8(mesh, displacement, target)

    with target.open(newline="", encoding="utf-8") as stream:
        exported = next(csv.DictReader(stream))
    exported_components = np.asarray([
        float(exported[name])
        for name in ("sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx")
    ])
    representative = np.asarray(
        get_element_kernel("Hex8").stress_at(
            mesh,
            mesh.elements[0],
            displacement,
            0.0,
            0.0,
            0.0,
        )
    )
    centroid = np.asarray(
        collect_stress(
            mesh,
            displacement,
            position=StressPosition.CENTROID,
        ).records[0].components
    )

    assert exported_components == pytest.approx(representative)
    assert np.max(np.abs(representative - centroid)) > 0.4
