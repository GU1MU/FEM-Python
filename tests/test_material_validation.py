import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition, ElementGravityLoad, ElementLoad
from fem.boundary.loads import build_load_vector
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel
from fem.materials import linear_elastic


ELASTIC_MATRIX_BUILDERS = (
    linear_elastic.plane_stress_matrix,
    linear_elastic.plane_strain_matrix,
    linear_elastic.solid_3d_matrix,
)


def _line_mesh(element_type="Truss2", **properties):
    defaults = {"E": 210.0}
    if element_type == "Beam2":
        defaults.update({
            "nu": 0.3,
            "section_type": "rectangle",
            "height": 1.0,
            "width": 0.5,
        })
    else:
        defaults["area"] = 0.5
    defaults.update(properties)
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], element_type, defaults)],
        dofs_per_node=6 if element_type == "Beam2" else 3,
    )


@pytest.mark.parametrize("E", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_material_rejects_nonpositive_or_nonfinite_elastic_modulus(E):
    with pytest.raises(ValueError, match=r"E must be finite and > 0"):
        linear_elastic.material("bad", E=E, nu=0.3)


@pytest.mark.parametrize("matrix_builder", ELASTIC_MATRIX_BUILDERS)
def test_constitutive_matrix_consumers_reject_invalid_elastic_modulus(matrix_builder):
    with pytest.raises(ValueError, match=r"E must be finite and > 0"):
        matrix_builder(0.0, 0.3)


@pytest.mark.parametrize("nu", [-1.0, 0.5, np.nan, np.inf, -np.inf])
def test_material_rejects_out_of_range_or_nonfinite_poisson_ratio(nu):
    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        linear_elastic.material("bad", E=210.0, nu=nu)


@pytest.mark.parametrize("matrix_builder", ELASTIC_MATRIX_BUILDERS)
def test_constitutive_matrix_consumers_reject_invalid_poisson_ratio(matrix_builder):
    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        matrix_builder(210.0, -1.0)


@pytest.mark.parametrize("rho", [-1.0, np.nan, np.inf, -np.inf])
def test_material_rejects_negative_or_nonfinite_density(rho):
    with pytest.raises(ValueError, match=r"rho must be finite and >= 0"):
        linear_elastic.material("bad", E=210.0, nu=0.3, rho=rho)


def test_material_accepts_admissible_boundary_nearby_values():
    material = linear_elastic.material(
        "valid",
        E=210.0,
        nu=np.nextafter(-1.0, 0.0),
        rho=0.0,
    )

    assert material.properties == {
        "E": 210.0,
        "nu": np.nextafter(-1.0, 0.0),
        "rho": 0.0,
    }


@pytest.mark.parametrize(
    ("element_type", "property_name", "value"),
    [
        ("Truss2", "area", 0.0),
        ("Truss2", "area", -1.0),
        ("Truss2", "area", np.nan),
        ("Truss2", "area", np.inf),
        ("Truss2", "area", -np.inf),
        ("Truss2", "E", 0.0),
        ("Beam2", "E", 0.0),
        ("Beam2", "height", 0.0),
        ("Beam2", "height", -1.0),
        ("Beam2", "height", np.nan),
        ("Beam2", "height", np.inf),
        ("Beam2", "height", -np.inf),
        ("Beam2", "width", 0.0),
    ],
)
def test_line_positive_property_validators_cover_invalid_equivalence_classes(
    element_type,
    property_name,
    value,
):
    mesh = _line_mesh(element_type, **{property_name: value})
    elem = mesh.elements[0]

    with pytest.raises(
        ValueError,
        match=rf"{property_name}.*finite and > 0",
    ):
        get_element_kernel(element_type).stiffness(mesh, elem)


def test_line_body_force_consumer_rejects_invalid_area():
    mesh = _line_mesh(area=0.0)
    elem = mesh.elements[0]

    with pytest.raises(ValueError, match=r"property area must be finite and > 0"):
        get_element_kernel(elem.type).body_force(mesh, elem, (0.0, -9.81, 0.0))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_body_force_assembly_rejects_nonfinite_vector_components(bad_value):
    mesh = _line_mesh()
    bc = BoundaryCondition(body_forces=[ElementLoad(1, (bad_value, 0.0, 0.0))])

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


def test_line_kernel_rejects_nonfinite_body_vector_when_called_directly():
    mesh = _line_mesh()
    elem = mesh.elements[0]

    with pytest.raises(ValueError, match=r"body force components must be finite"):
        get_element_kernel(elem.type).body_force(mesh, elem, (np.nan, 0.0, 0.0))


@pytest.mark.parametrize("nu", [-1.0, 0.5, np.nan, np.inf, -np.inf])
def test_beam_stiffness_rejects_invalid_poisson_ratio(nu):
    mesh = _line_mesh("Beam2", nu=nu)

    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])


@pytest.mark.parametrize(
    ("element_type", "rho"),
    [
        ("Truss2", -1.0),
        ("Truss2", np.nan),
        ("Truss2", np.inf),
        ("Truss2", -np.inf),
        ("Beam2", -1.0),
    ],
)
def test_line_stiffness_rejects_invalid_optional_density(element_type, rho):
    mesh = _line_mesh(element_type, rho=rho)

    with pytest.raises(ValueError, match=r"rho must be finite and >= 0"):
        get_element_kernel(element_type).stiffness(mesh, mesh.elements[0])
