from __future__ import annotations

import numpy as np

from fem.physics.mechanics import get_recovery_service
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh


def test_quad4_reference_dof_order_and_linear_internal_force_tangent() -> None:
    mesh = make_quad4_stiffness_mesh()
    element = mesh.elements[0]
    kernel = get_recovery_service(element.type)
    tangent = kernel.stiffness(mesh, element)
    displacement = np.array(
        [0.001, -0.002, 0.004, 0.001, 0.003, -0.001, -0.002, 0.002],
        dtype=float,
    )
    internal_force = tangent @ displacement
    step = 1.0e-7
    numerical_tangent = np.column_stack(
        [
            (
                tangent @ (displacement + step * np.eye(8)[column])
                - tangent @ (displacement - step * np.eye(8)[column])
            )
            / (2.0 * step)
            for column in range(8)
        ]
    )

    assert tuple(mesh.element_dofs(element)) == tuple(range(8))
    assert internal_force.shape == (8,)
    np.testing.assert_allclose(numerical_tangent, tangent, rtol=1.0e-9, atol=1.0e-9)


def test_quad4_linear_reference_kernel_has_zero_rigid_body_internal_force() -> None:
    mesh = make_quad4_stiffness_mesh()
    element = mesh.elements[0]
    tangent = get_recovery_service(element.type).stiffness(mesh, element)

    translation_x = np.tile([1.0, 0.0], 4)
    translation_y = np.tile([0.0, 1.0], 4)
    rotation = np.concatenate(
        [
            np.array([-node.y, node.x], dtype=float)
            for node in mesh.nodes
        ]
    )

    np.testing.assert_allclose(tangent @ translation_x, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(tangent @ translation_y, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(tangent @ rotation, 0.0, atol=1.0e-10)
