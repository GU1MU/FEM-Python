
import numpy as np

from fem.post import (
    displacement,
    path,
    stress,
)
from fem.post.stress import dispatch


def _affine_solid_displacement(mesh):
    U = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = 0.01 * node.x + 0.02 * node.y + 0.03 * node.z
        U[mesh.global_dof(node.id, 1)] = -0.02 * node.x + 0.04 * node.y + 0.01 * node.z
        U[mesh.global_dof(node.id, 2)] = 0.03 * node.x - 0.01 * node.y + 0.05 * node.z
    return U


def _write_current_element_stress(
    mesh,
    displacement,
    path,
    element_type=None,
    gauss_order=None,
):
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        stress.element.by_type(
            type_keys[0],
            mesh,
            displacement,
            path,
            gauss_order,
        )
        return
    stress.element.mixed(
        type_keys,
        mesh,
        displacement,
        path,
        gauss_order,
    )


def _write_current_nodal_stress(
    mesh,
    displacement,
    path,
    element_type=None,
    gauss_order=None,
    threshold=75.0,
):
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        stress.nodal.by_type(
            type_keys[0],
            mesh,
            displacement,
            path,
            gauss_order,
            threshold,
        )
        return
    stress.nodal.mixed(
        type_keys,
        mesh,
        displacement,
        path,
        gauss_order,
        threshold,
    )
