import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY, resolve_beam_frame
from fem.elements.beam_section import parse_beam2_section
from fem.elements.beam_frame import BeamFrameField
from fem.elements.line import _beam2_integrated_line_load, _beam2_variable_stiffness
from tests.helpers.mesh_builders import make_beam_stiffness_mesh


def _beam_mesh(*, end=(4.0, 0.0, 0.0), props=None):
    properties = {
        "E": 210.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 3.0,
        "width": 2.0,
        "rho": 4.0,
    }
    if props:
        properties.update(props)
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, *end)],
        elements=[Element3D(1, [1, 2], "Beam2", properties)],
        dofs_per_node=6,
    )


def test_beam2_automatic_frame_maps_rectangle_width_to_local_y():
    mesh = _beam_mesh(props={"height": 4.0, "width": 1.0})
    frame = resolve_beam_frame(mesh, mesh.elements[0])
    section = parse_beam2_section(mesh.elements[0].props)

    assert frame.local_y == pytest.approx((0.0, 1.0, 0.0))
    assert frame.local_z == pytest.approx((0.0, 0.0, 1.0))
    assert section.Iyy == pytest.approx(1.0 * 4.0**3 / 12.0)
    assert section.Izz == pytest.approx(4.0 * 1.0**3 / 12.0)


@pytest.mark.parametrize("mode", range(6))
def test_beam2_six_rigid_body_modes_have_zero_internal_force(mode):
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    elem = mesh.elements[0]
    stiffness = get_element_kernel("Beam2").stiffness(mesh, elem)
    displacement = np.zeros(12)

    if mode < 3:
        displacement[mode] = 1.0
        displacement[6 + mode] = 1.0
    else:
        omega = np.eye(3)[mode - 3]
        end_position = np.array([2.0, 3.0, 6.0])
        displacement[3:6] = omega
        displacement[6:9] = np.cross(omega, end_position)
        displacement[9:12] = omega

    assert stiffness @ displacement == pytest.approx(np.zeros(12), abs=1e-10)


def test_beam2_rectangle_dimension_swap_exchanges_cantilever_bending_response():
    tall = _beam_mesh(props={"height": 4.0, "width": 1.0})
    wide = _beam_mesh(props={"height": 1.0, "width": 4.0})
    kernel = get_element_kernel("Beam2")

    tall_compliance = np.linalg.inv(kernel.stiffness(tall, tall.elements[0])[6:, 6:])
    wide_compliance = np.linalg.inv(kernel.stiffness(wide, wide.elements[0])[6:, 6:])

    assert tall_compliance[1, 1] == pytest.approx(wide_compliance[2, 2])
    assert tall_compliance[2, 2] == pytest.approx(wide_compliance[1, 1])
    assert tall_compliance[1, 1] > tall_compliance[2, 2]


def test_beam2_explicit_orientation_rotates_rectangle_bending_axes() -> None:
    local_y_global_y = _beam_mesh(
        props={BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0)}
    )
    local_y_global_z = _beam_mesh(
        props={BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 1.0)}
    )
    kernel = get_element_kernel("Beam2")

    first = np.linalg.inv(
        kernel.stiffness(
            local_y_global_y,
            local_y_global_y.elements[0],
        )[6:, 6:]
    )
    rotated = np.linalg.inv(
        kernel.stiffness(
            local_y_global_z,
            local_y_global_z.elements[0],
        )[6:, 6:]
    )

    assert first[1, 1] == pytest.approx(rotated[2, 2])
    assert first[2, 2] == pytest.approx(rotated[1, 1])
    assert first[1, 1] > first[2, 2]
    assert rotated[2, 2] > rotated[1, 1]


def test_beam2_explicit_orientation_removes_near_global_z_axis_swap() -> None:
    properties = {
        "height": 0.1,
        "width": 0.02,
        BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
    }
    vertical = _beam_mesh(end=(0.0, 0.0, 4.0), props=properties)
    perturbed = _beam_mesh(end=(1e-11, 0.0, 4.0), props=properties)
    kernel = get_element_kernel("Beam2")

    vertical_compliance = np.linalg.inv(
        kernel.stiffness(vertical, vertical.elements[0])[6:, 6:]
    )
    perturbed_compliance = np.linalg.inv(
        kernel.stiffness(perturbed, perturbed.elements[0])[6:, 6:]
    )

    assert perturbed_compliance[0, 0] == pytest.approx(
        vertical_compliance[0, 0],
        rel=1e-9,
    )


def test_beam2_circular_stiffness_is_invariant_to_roll_about_beam_axis():
    mesh = _beam_mesh()
    elem = mesh.elements[0]
    elem.props.pop("height")
    elem.props.pop("width")
    elem.props.update({"section_type": "solid_circle", "radius": 1.5})
    stiffness = get_element_kernel("Beam2").stiffness(mesh, elem)
    angle = np.deg2rad(37.0)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )
    transformation = np.zeros((12, 12))
    for start in (0, 3, 6, 9):
        transformation[start : start + 3, start : start + 3] = roll

    assert transformation.T @ stiffness @ transformation == pytest.approx(
        stiffness, abs=1e-10
    )


def test_beam2_body_force_preserves_resultant_and_moment():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    elem = mesh.elements[0]
    body_vector = np.array([1.5, -2.0, 0.25])
    element_force = get_element_kernel("Beam2").body_force(
        mesh, elem, tuple(body_vector)
    )
    total_force = body_vector * parse_beam2_section(elem.props).area * 7.0
    end = np.array([2.0, 3.0, 6.0])
    assembled_moment = (
        np.cross(np.zeros(3), element_force[:3])
        + element_force[3:6]
        + np.cross(end, element_force[6:9])
        + element_force[9:12]
    )

    assert element_force[:3] + element_force[6:9] == pytest.approx(total_force)
    assert assembled_moment == pytest.approx(np.cross(end / 2.0, total_force))


def test_beam2_explicit_orientation_body_force_reversal_only_permutes_nodes():
    properties = {BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0)}
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), props=properties)
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0), props=properties)
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    body_vector = (1.5, -2.0, 0.25)

    force = kernel.body_force(mesh, mesh.elements[0], body_vector)
    reversed_force = kernel.body_force(
        reversed_mesh,
        reversed_mesh.elements[0],
        body_vector,
    )

    assert reversed_force == pytest.approx(permutation @ force, abs=1e-12)


def _aluminum_beam_mesh(*, length=0.6, height=0.24, width=0.18):
    return _beam_mesh(
        end=(length, 0.0, 0.0),
        props={"E": 70.0e9, "nu": 0.3, "height": height, "width": width, "rho": 2700.0},
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
    mesh = _aluminum_beam_mesh()
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    eigenvalues = np.linalg.eigvalsh((stiffness + stiffness.T) / 2.0)
    tolerance = eigenvalues[-1] * 1.0e-10

    assert stiffness == pytest.approx(stiffness.T, abs=1.0e-6)
    assert np.count_nonzero(np.abs(eigenvalues) <= tolerance) == 6
    assert np.count_nonzero(eigenvalues > tolerance) == 6


def test_constant_and_zero_variation_frame_paths_match() -> None:
    mesh = _aluminum_beam_mesh()
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
    mesh = _aluminum_beam_mesh(length=length)
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    inertia = 0.24 * 0.18**3 / 12.0
    area = 0.24 * 0.18
    kGA_y = 0.85 * (70.0e9 / 2.6) * area / (1.0 + 0.25 * length**2 * area / (12.0 * inertia))
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
        load_per_length * length**4 / (8.0 * 70.0e9 * inertia)
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
    mesh = _aluminum_beam_mesh()
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
    mesh = _aluminum_beam_mesh()
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


def test_beam_rejects_zero_length():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    node_lookup = {node.id: node for node in mesh.nodes}
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    nj.x = ni.x
    nj.y = ni.y
    nj.z = ni.z

    with pytest.raises(ValueError, match="zero length"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_beam_reports_missing_node():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    elem.node_ids[1] = 999

    with pytest.raises(KeyError, match=r"Element 1 references missing node 999"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_beam_reports_missing_required_property():
    missing_property = "width"
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    elem.props.pop(missing_property)

    with pytest.raises(
        KeyError,
        match=rf"missing property {missing_property}",
    ):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_beam_stiffness_supports_default_and_explicit_node_lookup():
    mesh = make_beam_stiffness_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel(element.type)
    expected = kernel.stiffness(mesh, element)
    actual = kernel.stiffness(mesh, element, {node.id: node for node in mesh.nodes})
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("end", "reference"),
    [((2.0, 3.0, 6.0), None), ((2.0, 3.0, 6.0), (0.0, 1.0, 0.0)),
     ((0.0, 0.0, 4.0), None), ((1e-14, 0.0, 4.0), None)],
    ids=["inclined-automatic", "inclined-explicit", "vertical", "near-vertical"],
)
def test_beam_connectivity_reversal_permutes_stiffness_without_changing_response(end, reference):
    properties = {} if reference is None else {BEAM_LOCAL_Y_REFERENCE_KEY: reference}
    mesh = _beam_mesh(end=end, props=properties)
    reversed_mesh = _beam_mesh(end=end, props=properties)
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    order = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]

    stiffness = kernel.stiffness(mesh, mesh.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-12)
    np.testing.assert_allclose(reversed_stiffness, stiffness[np.ix_(order, order)], atol=1e-10)


@pytest.mark.parametrize("dof", [0, 3], ids=["axial", "torsion"])
def test_circular_beam_axial_and_torsional_compliance_matches_analytic_rigidities(dof):
    mesh = _beam_mesh(props={"section_type": "solid_circle", "radius": 1.5})
    mesh.elements[0].props.pop("height")
    mesh.elements[0].props.pop("width")
    force = np.zeros(6)
    force[dof] = 12.0
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])

    tip = np.linalg.solve(stiffness[6:, 6:], force)

    rigidity = 210.0 * np.pi * 1.5**2 if dof == 0 else (210.0 / 2.5) * np.pi * 1.5**4 / 2.0
    assert tip[dof] == pytest.approx(12.0 * 4.0 / rigidity)


@pytest.mark.parametrize(
    ("length", "height", "width", "force", "dof"),
    [(0.6, 0.24, 0.18, 1800.0, 1), (0.6, 0.24, 0.18, 1800.0, 2),
     (20.0, 0.08, 0.06, 100.0, 1)],
    ids=["thick-bend-y", "thick-bend-z", "slender-bend-y"],
)
def test_cantilever_matches_b31_single_point_bending_and_compensated_shear(length, height, width, force, dof):
    mesh = _aluminum_beam_mesh(length=length, height=height, width=width)
    load = np.zeros(6)
    load[dof] = force
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    tip = np.linalg.solve(stiffness[6:, 6:], load)

    # Independent rectangular-section geometry and B31 constants: kappa=.85,
    # slenderness compensation=.25; one-point bending uses L^3/(4 E I).
    area = height * width
    inertia = height * width**3 / 12.0 if dof == 1 else width * height**3 / 12.0
    elastic_modulus = 70.0e9
    shear_modulus = elastic_modulus / 2.6
    shear_rigidity = 0.85 * shear_modulus * area / (1.0 + 0.25 * length**2 * area / (12.0 * inertia))
    bending = force * length**3 / (4.0 * elastic_modulus * inertia)
    shear = force * length / shear_rigidity
    assert tip[dof] == pytest.approx(bending + shear)
    assert tip[dof] > bending
    if length == 20.0:
        assert tip[dof] < force * length**3 / (3.0 * elastic_modulus * inertia)


def test_inclined_beam_tip_displacement_and_rotation_match_b31_response():
    mesh = _beam_mesh(end=(3.0, 4.0, 0.0))
    # Automatic frame: local x=(.6,.8,0), y=(-.8,.6,0), z=(0,0,1).
    load = np.array([-0.8, 0.6, 0.0, 0.0, 0.0, 0.0])
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])

    tip = np.linalg.solve(stiffness[6:, 6:], load)

    # E=210, G=84, rectangle A=6, Izz=2, length=5; unit force along local y.
    shear_rigidity = 0.85 * 84.0 * 6.0 / (1.0 + 0.25 * 25.0 * 6.0 / 24.0)
    transverse = 125.0 / (4.0 * 210.0 * 2.0) + 5.0 / shear_rigidity
    rotation = 25.0 / (2.0 * 210.0 * 2.0)
    np.testing.assert_allclose(tip, [-0.8*transverse, 0.6*transverse, 0.0, 0.0, 0.0, rotation], atol=1e-12)
