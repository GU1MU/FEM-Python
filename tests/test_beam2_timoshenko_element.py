import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel, resolve_beam_frame
from fem.elements.beam_frame import BeamFrameField
from fem.elements.beam_section import parse_beam2_section
from fem.elements.line import (
    _beam2_integrated_line_load,
    _beam2_variable_stiffness,
)


def _beam_mesh(
    *,
    length: float = 0.6,
    height: float = 0.24,
    width: float = 0.18,
) -> Mesh3D:
    return Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, length, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Beam2",
                {
                    "E": 70.0e9,
                    "nu": 0.3,
                    "section_type": "rectangle",
                    "height": height,
                    "width": width,
                    "rho": 2700.0,
                },
            )
        ],
        dofs_per_node=6,
    )


def _section_rigidities(mesh: Mesh3D) -> tuple[float, float, float, float]:
    element = mesh.elements[0]
    section = parse_beam2_section(element.props)
    elastic_modulus = element.props["E"]
    poisson_ratio = element.props["nu"]
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    kGA_y, kGA_z = section.effective_shear_rigidities(
        shear_modulus,
        poisson_ratio,
    )
    return elastic_modulus, shear_modulus, kGA_y, kGA_z


def test_free_timoshenko_beam_has_exactly_six_rigid_body_modes() -> None:
    mesh = _beam_mesh()
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    eigenvalues = np.linalg.eigvalsh((stiffness + stiffness.T) / 2.0)
    tolerance = eigenvalues[-1] * 1.0e-10

    assert stiffness == pytest.approx(stiffness.T, abs=1.0e-6)
    assert np.count_nonzero(np.abs(eigenvalues) <= tolerance) == 6
    assert np.count_nonzero(eigenvalues > tolerance) == 6


@pytest.mark.parametrize(
    ("transverse_dof", "inertia_name", "shear_axis"),
    [
        (1, "Izz", "y"),
        (2, "Iyy", "z"),
    ],
)
def test_thick_cantilever_tip_deflection_includes_bending_and_shear(
    transverse_dof: int,
    inertia_name: str,
    shear_axis: str,
) -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    section = parse_beam2_section(element.props)
    elastic_modulus, _, kGA_y, kGA_z = _section_rigidities(mesh)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, element)
    tip_force = 1800.0
    free_load = np.zeros(6)
    free_load[transverse_dof] = tip_force

    free_displacement = np.linalg.solve(stiffness[6:, 6:], free_load)

    compensated_y, compensated_z = section.abaqus_b31_shear_rigidities(
        elastic_modulus / (2.0 * (1.0 + element.props["nu"])),
        element.props["nu"],
        mesh.nodes[1].x,
    )
    shear_rigidity = compensated_y if shear_axis == "y" else compensated_z
    expected = (
        tip_force
        * mesh.nodes[1].x**3
        / (4.0 * elastic_modulus * getattr(section, inertia_name))
        + tip_force * mesh.nodes[1].x / shear_rigidity
    )
    assert free_displacement[transverse_dof] == pytest.approx(expected)


def test_slender_cantilever_matches_abaqus_b31_one_point_limit() -> None:
    length = 20.0
    mesh = _beam_mesh(length=length, height=0.08, width=0.06)
    element = mesh.elements[0]
    section = parse_beam2_section(element.props)
    elastic_modulus, shear_modulus, _, _ = _section_rigidities(mesh)
    load = 100.0
    free_load = np.zeros(6)
    free_load[1] = load

    displacement = np.linalg.solve(
        get_element_kernel("Beam2").stiffness(mesh, element)[6:, 6:],
        free_load,
    )[1]

    euler_bernoulli = load * length**3 / (
        3.0 * elastic_modulus * section.Izz
    )
    compensated_y, _ = section.abaqus_b31_shear_rigidities(
        shear_modulus,
        element.props["nu"],
        length,
    )
    b31 = load * (
        length**3 / (4.0 * elastic_modulus * section.Izz)
        + length / compensated_y
    )
    assert displacement == pytest.approx(b31)
    assert displacement < euler_bernoulli


def test_constant_and_zero_variation_frame_paths_match() -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    frame = resolve_beam_frame(mesh, element)
    field = BeamFrameField.constant(frame)
    section = parse_beam2_section(element.props)
    elastic_modulus, shear_modulus, _, _ = _section_rigidities(mesh)
    kGA_y, kGA_z = section.abaqus_b31_shear_rigidities(
        shear_modulus,
        element.props["nu"],
        field.length,
    )

    closed_form = kernel.stiffness(mesh, element)
    integrated = _beam2_variable_stiffness(
        field,
        elastic_modulus,
        section.area,
        section.Iyy,
        section.Izz,
        shear_modulus,
        section.J,
        kGA_y,
        kGA_z,
    )

    assert integrated == pytest.approx(closed_form, rel=1.0e-12, abs=1.0e-5)


def test_uniform_line_load_preserves_balance_and_b31_discrete_response() -> None:
    length = 0.6
    mesh = _beam_mesh(length=length)
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    section = parse_beam2_section(element.props)
    elastic_modulus, shear_modulus, _, _ = _section_rigidities(mesh)
    kGA_y, _ = section.abaqus_b31_shear_rigidities(
        shear_modulus,
        element.props["nu"],
        length,
    )
    load_per_length = 2400.0
    local_load = kernel.local_line_load(
        mesh,
        element,
        (0.0, load_per_length, 0.0),
        "local",
    )
    displacement = np.zeros(12)
    displacement[6:] = np.linalg.solve(
        kernel.stiffness(mesh, element)[6:, 6:],
        local_load[6:],
    )
    reaction = kernel.stiffness(mesh, element) @ displacement - local_load
    start_forces, end_forces = kernel.local_section_end_actions(
        mesh,
        element,
        displacement,
        local_load,
    )

    expected_tip = (
        load_per_length * length**4 / (8.0 * elastic_modulus * section.Izz)
        + load_per_length * length**2 / (2.0 * kGA_y)
    )
    assert local_load[1] + local_load[7] == pytest.approx(
        load_per_length * length
    )
    assert (
        local_load[5] + length * local_load[7] + local_load[11]
    ) == pytest.approx(load_per_length * length**2 / 2.0)
    assert displacement[7] == pytest.approx(expected_tip)
    assert reaction[:6] == pytest.approx(
        (0.0, -load_per_length * length, 0.0, 0.0, 0.0, -load_per_length * length**2 / 2.0),
        abs=1.0e-9,
    )
    assert reaction[6:] == pytest.approx(np.zeros(6), abs=1.0e-9)
    assert (start_forces.Vy, start_forces.Mz) == pytest.approx(
        (load_per_length * length, load_per_length * length**2 / 2.0)
    )
    assert (end_forces.Vy, end_forces.Mz) == pytest.approx(
        (0.0, 0.0),
        abs=1.0e-9,
    )


def test_body_force_and_line_load_share_b31_interpolation() -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    section = parse_beam2_section(element.props)
    body_vector = (1.25, -2.5, 0.75)

    body_force = kernel.body_force(mesh, element, body_vector)
    equivalent_line_force = kernel.line_load(
        mesh,
        element,
        tuple(section.area * value for value in body_vector),
        "global",
    )

    assert body_force == pytest.approx(equivalent_line_force)


def test_constant_load_and_zero_variation_integration_paths_match() -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    field = BeamFrameField.constant(resolve_beam_frame(mesh, element))
    vector = (3.0, -4.0, 2.0)

    closed_form = kernel.line_load(mesh, element, vector, "local")
    integrated = _beam2_integrated_line_load(
        field,
        vector,
        "local",
    )

    assert integrated == pytest.approx(closed_form, rel=1.0e-12, abs=1.0e-12)
