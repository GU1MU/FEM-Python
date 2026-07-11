import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition, ElementLoad
from fem.boundary.loads import build_load_vector
from fem.core.mesh import BeamMesh2D, Element2D, Node2D, TrussMesh2D
from fem.elements import get_element_kernel
from fem.materials import linear_elastic


ELASTIC_MATRIX_BUILDERS = (
    linear_elastic.plane_stress_matrix,
    linear_elastic.plane_strain_matrix,
    linear_elastic.solid_3d_matrix,
)


def _line_mesh(element_type="Truss2D", **properties):
    mesh_type = BeamMesh2D if element_type == "Beam2D" else TrussMesh2D
    defaults = {"E": 210.0, "area": 0.5}
    if element_type == "Beam2D":
        defaults["Izz"] = 0.25
    defaults.update(properties)
    return mesh_type(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 2.0, 0.0)],
        elements=[Element2D(1, [1, 2], element_type, defaults)],
    )


@pytest.mark.parametrize("E", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_material_rejects_nonpositive_or_nonfinite_elastic_modulus(E):
    with pytest.raises(ValueError, match=r"E must be finite and > 0"):
        linear_elastic.material("bad", E=E, nu=0.3)


@pytest.mark.parametrize("matrix_builder", ELASTIC_MATRIX_BUILDERS)
@pytest.mark.parametrize("E", [0.0, -1.0, np.nan, np.inf])
def test_constitutive_matrices_reject_invalid_elastic_modulus(matrix_builder, E):
    with pytest.raises(ValueError, match=r"E must be finite and > 0"):
        matrix_builder(E, 0.3)


@pytest.mark.parametrize("nu", [-1.0, 0.5, np.nan, np.inf, -np.inf])
def test_material_rejects_out_of_range_or_nonfinite_poisson_ratio(nu):
    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        linear_elastic.material("bad", E=210.0, nu=nu)


@pytest.mark.parametrize("matrix_builder", ELASTIC_MATRIX_BUILDERS)
@pytest.mark.parametrize("nu", [-1.0, 0.5, np.nan, np.inf])
def test_constitutive_matrices_reject_invalid_poisson_ratio(matrix_builder, nu):
    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        matrix_builder(210.0, nu)


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
    ("element_type", "property_name"),
    [
        ("Truss2D", "E"),
        ("Truss2D", "area"),
        ("Beam2D", "area"),
        ("Beam2D", "Izz"),
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_line_stiffness_rejects_invalid_positive_properties(
    element_type,
    property_name,
    value,
):
    mesh = _line_mesh(element_type, **{property_name: value})
    elem = mesh.elements[0]

    with pytest.raises(
        ValueError,
        match=rf"property {property_name} must be finite and > 0",
    ):
        get_element_kernel(element_type).stiffness(mesh, elem)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_line_body_force_rejects_invalid_area(value):
    mesh = _line_mesh(area=value)
    elem = mesh.elements[0]

    with pytest.raises(ValueError, match=r"property area must be finite and > 0"):
        get_element_kernel(elem.type).body_force(mesh, elem, (0.0, -9.81))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_body_force_assembly_rejects_nonfinite_vector_components(bad_value):
    mesh = _line_mesh()
    bc = BoundaryCondition(body_forces=[ElementLoad(1, (bad_value, 0.0))])

    with pytest.raises(ValueError, match=r"body force vector components must be finite"):
        build_load_vector(mesh, bc)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_gravity_assembly_rejects_nonfinite_vector_components(bad_value):
    mesh = _line_mesh()
    bc = BoundaryCondition(gravity=(0.0, bad_value))

    with pytest.raises(ValueError, match=r"gravity vector components must be finite"):
        build_load_vector(mesh, bc)


@pytest.mark.parametrize("rho", [-1.0, np.nan, np.inf, -np.inf])
def test_gravity_rejects_invalid_density_stored_directly_on_element(rho):
    mesh = _line_mesh(rho=rho)
    bc = BoundaryCondition()
    bc.set_gravity(0.0, -9.81)

    with pytest.raises(ValueError, match=r"Element 1 rho must be finite and >= 0"):
        build_load_vector(mesh, bc)


def test_line_kernel_rejects_nonfinite_body_vector_when_called_directly():
    mesh = _line_mesh()
    elem = mesh.elements[0]

    with pytest.raises(ValueError, match=r"body force components must be finite"):
        get_element_kernel(elem.type).body_force(mesh, elem, (np.nan, 0.0))
