import numpy as np
import pytest

from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import make_beam_stiffness_mesh, make_truss_stiffness_mesh


def _line_mesh(element_type="Truss2", **properties):
    mesh = make_beam_stiffness_mesh() if element_type == "Beam2" else make_truss_stiffness_mesh()
    mesh.elements[0].props.update(properties)
    return mesh


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
        ("Beam2", "width", 0.0),
    ],
)
def test_line_stiffness_rejects_invalid_positive_properties(
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


def test_line_kernel_rejects_nonfinite_body_vector_when_called_directly():
    mesh = _line_mesh()
    elem = mesh.elements[0]

    with pytest.raises(ValueError, match=r"body force components must be finite"):
        get_element_kernel(elem.type).body_force(mesh, elem, (np.nan, 0.0, 0.0))


@pytest.mark.parametrize("nu", [-1.0, 0.5])
def test_beam_stiffness_rejects_invalid_poisson_ratio(nu):
    mesh = _line_mesh("Beam2", nu=nu)

    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])


@pytest.mark.parametrize(
    ("element_type", "rho"),
    [
        ("Truss2", -1.0),
        ("Beam2", -1.0),
    ],
)
def test_line_stiffness_rejects_invalid_optional_density(element_type, rho):
    mesh = _line_mesh(element_type, rho=rho)

    with pytest.raises(ValueError, match=r"rho must be finite and >= 0"):
        get_element_kernel(element_type).stiffness(mesh, mesh.elements[0])
