from __future__ import annotations

from math import hypot, isfinite, pi

import pytest

from fem.application.preprocessing import generate_fem_model
from fem.mesh.settings import MeshSettings
from tests.helpers.fixtures.profile_transform_baseline import (
    RING_EXTRUSION_HEIGHT,
    RING_INNER_RADIUS,
    RING_OUTER_RADIUS,
    concentric_ring_fixture,
)


def _signed_tetrahedron_volume(points):
    a, b, c, d = points
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    w = tuple(d[i] - a[i] for i in range(3))
    determinant = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return determinant / 6.0


def test_ring_extrusion_mesh_preserves_hole_and_analytic_volume(real_gmsh) -> None:
    fixture = concentric_ring_fixture()
    generated = generate_fem_model(
        fixture.extrusion,
        MeshSettings(20.0, cell_shape="tetrahedron"),
    )
    mesh = generated.mesh
    nodes = {node.id: (node.x, node.y, node.z) for node in mesh.nodes}
    assert nodes and mesh.elements
    assert len(nodes) == len(mesh.nodes)
    assert len({element.id for element in mesh.elements}) == len(mesh.elements)
    for x, y, z in nodes.values():
        assert all(isfinite(value) for value in (x, y, z))
        assert RING_INNER_RADIUS - 1e-8 <= hypot(x, y) <= RING_OUTER_RADIUS + 1e-8
        assert -1e-8 <= z <= RING_EXTRUSION_HEIGHT + 1e-8
    assert min(point[2] for point in nodes.values()) == pytest.approx(0.0)
    assert max(point[2] for point in nodes.values()) == pytest.approx(
        RING_EXTRUSION_HEIGHT
    )

    volumes = []
    used_nodes = set()
    for element in mesh.elements:
        assert element.type == "Tet4"
        assert len(element.node_ids) == len(set(element.node_ids)) == 4
        used_nodes.update(element.node_ids)
        points = [nodes[node_id] for node_id in element.node_ids]
        centroid_x = sum(point[0] for point in points) / 4.0
        centroid_y = sum(point[1] for point in points) / 4.0
        # Allow coarse circular faceting, but reject cells spanning the hole.
        assert hypot(centroid_x, centroid_y) >= 0.95 * RING_INNER_RADIUS
        volume = _signed_tetrahedron_volume(points)
        assert volume > 0.0
        volumes.append(volume)
    assert used_nodes == set(nodes)
    expected_volume = (
        pi * (RING_OUTER_RADIUS**2 - RING_INNER_RADIUS**2) * RING_EXTRUSION_HEIGHT
    )
    # Linear tetrahedra facet the circular boundaries at a coarse size of 20 mm.
    assert sum(volumes) == pytest.approx(expected_volume, rel=0.05)
