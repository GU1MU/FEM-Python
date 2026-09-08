import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition, ElementGravityLoad, ElementLoad
from fem.boundary.loads import build_load_vector
from tests.helpers.mesh_builders import make_truss_stiffness_mesh


def _line_mesh(**properties):
    mesh = make_truss_stiffness_mesh()
    mesh.elements[0].props.update(properties)
    return mesh


def test_body_force_assembly_rejects_nonfinite_vector_components():
    mesh = _line_mesh()
    bc = BoundaryCondition(body_forces=[ElementLoad(1, (np.nan, 0.0, 0.0))])

    with pytest.raises(ValueError, match=r"body force vector components must be finite"):
        build_load_vector(mesh, bc)


def test_gravity_assembly_consumer_rejects_nonfinite_vector_components():
    mesh = _line_mesh()
    bc = BoundaryCondition(gravity=(0.0, np.nan, 0.0))

    with pytest.raises(ValueError, match=r"gravity vector components must be finite"):
        build_load_vector(mesh, bc)


@pytest.mark.parametrize("rho", [-1.0, np.nan, np.inf, -np.inf])
def test_gravity_rejects_invalid_density_stored_directly_on_element(rho):
    mesh = _line_mesh(rho=rho)
    bc = BoundaryCondition()
    bc.set_gravity(0.0, -9.81, 0.0)

    with pytest.raises(ValueError, match=r"Element 1 rho must be finite and >= 0"):
        build_load_vector(mesh, bc)


def test_targeted_gravity_requires_stamped_density_but_global_gravity_skips_it():
    mesh = _line_mesh()
    targeted = BoundaryCondition(
        element_gravities=[ElementGravityLoad(1, (0.0, -9.81, 0.0))]
    )

    with pytest.raises(ValueError, match="rho is required for targeted gravity"):
        build_load_vector(mesh, targeted)

    global_bc = BoundaryCondition()
    global_bc.set_gravity(0.0, -9.81, 0.0)
    assert np.allclose(build_load_vector(mesh, global_bc), 0.0)


@pytest.mark.parametrize(
    ("acceleration", "message"),
    [
        ((0.0, -1.0), "must have 3 components"),
        ((0.0, np.nan, 0.0), "components must be finite"),
    ],
)
def test_targeted_gravity_assembly_revalidates_acceleration(acceleration, message):
    mesh = _line_mesh(rho=1.0)
    bc = BoundaryCondition(
        element_gravities=[ElementGravityLoad(1, acceleration)]
    )

    with pytest.raises(ValueError, match=message):
        build_load_vector(mesh, bc)


def test_targeted_gravity_rejects_unknown_element():
    mesh = _line_mesh(rho=1.0)
    bc = BoundaryCondition(
        element_gravities=[ElementGravityLoad(99, (0.0, -1.0, 0.0))]
    )

    with pytest.raises(KeyError, match="Element 99 not found"):
        build_load_vector(mesh, bc)


def test_targeted_gravity_allows_zero_density_and_produces_zero_load():
    mesh = _line_mesh(rho=0.0)
    bc = BoundaryCondition()
    bc.add_gravity_element(1, 0.0, -9.81, 0.0)

    assert np.allclose(build_load_vector(mesh, bc), 0.0)


@pytest.mark.parametrize(
    ("method", "arguments", "message"),
    [
        ("add_displacement_dof", (0, np.nan), "prescribed displacement must be finite"),
        ("add_nodal_force_dof", (0, np.inf), "nodal force must be finite"),
        ("add_body_force_element", (1, -np.inf, 0.0), "load vector component must be finite"),
    ],
    ids=["displacement", "nodal-force", "body-force"],
)
def test_boundary_condition_rejects_nonfinite_scalar_values(method, arguments, message):
    boundary = BoundaryCondition()

    with pytest.raises(ValueError, match=message):
        getattr(boundary, method)(*arguments)


def test_boundary_condition_rejects_overflow_during_force_accumulation():
    boundary = BoundaryCondition()
    boundary.add_nodal_force_dof(0, np.finfo(float).max)

    with pytest.raises(ValueError, match="accumulated nodal force at DOF 0 must be finite"):
        boundary.add_nodal_force_dof(0, np.finfo(float).max)


def test_load_assembly_rejects_nonfinite_direct_nodal_force_maps():
    mesh = _line_mesh()
    boundary = BoundaryCondition(nodal_forces={0: np.nan})

    with pytest.raises(ValueError, match="nodal force at DOF 0 must be finite"):
        build_load_vector(mesh, boundary)
