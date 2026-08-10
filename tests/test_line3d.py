import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    get_element_kernel,
    resolve_beam_frame,
)
from fem.elements.beam_section import axial_stress_extrema, parse_beam2_section
from fem.post.stress import beam as beam_stress


def _truss_mesh(*, reversed_nodes=False, props=None):
    nodes = [Node3D(10, 1.0, -2.0, 0.5), Node3D(20, 3.0, 1.0, 6.5)]
    node_ids = [20, 10] if reversed_nodes else [10, 20]
    return Mesh3D(
        nodes=nodes,
        elements=[
            Element3D(
                1,
                node_ids,
                "Truss2",
                props or {"E": 210.0, "area": 2.5, "rho": 4.0},
            )
        ],
    )


def test_truss2_spatial_stiffness_matches_outer_product_contract():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    delta = np.array([2.0, 3.0, 6.0])
    length = np.linalg.norm(delta)
    direction = delta / length
    block = np.outer(direction, direction)
    expected = elem.props["E"] * elem.props["area"] / length * np.block(
        [[block, -block], [-block, block]]
    )

    stiffness = get_element_kernel("Truss2").stiffness(mesh, elem)

    assert stiffness == pytest.approx(expected)
    assert stiffness == pytest.approx(stiffness.T)
    assert np.linalg.matrix_rank(stiffness, tol=1e-10) == 1


def test_truss2_node_reversal_only_permutes_element_stiffness():
    forward = _truss_mesh()
    reversed_mesh = _truss_mesh(reversed_nodes=True)
    kernel = get_element_kernel("Truss2")
    permutation = np.eye(6)[[3, 4, 5, 0, 1, 2]]

    forward_stiffness = kernel.stiffness(forward, forward.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert reversed_stiffness == pytest.approx(
        permutation @ forward_stiffness @ permutation.T
    )


def test_truss2_rigid_translation_and_axial_extension_results():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel("Truss2")
    direction = np.array([2.0, 3.0, 6.0]) / 7.0
    rigid = np.tile([0.4, -0.2, 0.7], 2)

    assert kernel.stiffness(mesh, elem) @ rigid == pytest.approx(np.zeros(6), abs=1e-12)
    assert kernel.element_stress(mesh, elem, rigid) == pytest.approx((0.0, 0.0, 0.0))

    extension = 0.14
    displacement = np.concatenate([np.zeros(3), extension * direction])
    strain, stress, mises = kernel.element_stress(mesh, elem, displacement)

    assert strain == pytest.approx(extension / 7.0)
    assert stress == pytest.approx(elem.props["E"] * extension / 7.0)
    assert mises == pytest.approx(abs(stress))


def test_truss2_body_force_and_gravity_preserve_total_force():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel("Truss2")
    body_vector = np.array([1.5, -2.0, 0.25])
    expected_total = body_vector * elem.props["area"] * 7.0

    element_force = kernel.body_force(mesh, elem, tuple(body_vector))

    assert element_force[:3] == pytest.approx(expected_total / 2.0)
    assert element_force[3:] == pytest.approx(expected_total / 2.0)

    boundary = BoundaryCondition()
    boundary.set_gravity(0.0, 0.0, -9.81)
    gravity_force = build_load_vector(mesh, boundary)

    assert gravity_force.reshape(2, 3).sum(axis=0) == pytest.approx(
        [0.0, 0.0, -9.81 * elem.props["rho"] * elem.props["area"] * 7.0]
    )


@pytest.mark.parametrize("vector", [(1.0, 2.0), (1.0, 2.0, np.nan)])
def test_truss2_rejects_invalid_body_force_vectors(vector):
    mesh = _truss_mesh()

    with pytest.raises(ValueError, match="Truss2 body force"):
        get_element_kernel("Truss2").body_force(mesh, mesh.elements[0], vector)


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


@pytest.mark.parametrize(
    ("props", "expected"),
    [
        (
            {"section_type": "solid_circle", "radius": 2.0},
            (4.0 * np.pi, 4.0 * np.pi, 4.0 * np.pi, 8.0 * np.pi),
        ),
        (
            {
                "section_type": "hollow_circle",
                "outer_radius": 2.0,
                "inner_radius": 1.0,
            },
            (3.0 * np.pi, 15.0 * np.pi / 4.0, 15.0 * np.pi / 4.0, 15.0 * np.pi / 2.0),
        ),
        (
            {"section_type": "rectangle", "height": 4.0, "width": 1.0},
            (4.0, 16.0 / 3.0, 1.0 / 3.0, 1.1232518332307355),
        ),
    ],
)
def test_beam2_standard_sections_derive_stiffness_properties(props, expected):
    section = parse_beam2_section(props)

    assert (section.area, section.Iyy, section.Izz, section.J) == pytest.approx(expected)


def test_beam2_rectangle_uses_height_and_width_dimensions():
    section = parse_beam2_section(
        {"section_type": "rectangle", "height": 4.0, "width": 1.0}
    )

    assert section.height == 4.0
    assert section.width == 1.0
    assert (section.area, section.Iyy, section.Izz) == pytest.approx(
        (4.0, 16.0 / 3.0, 1.0 / 3.0)
    )


def test_beam2_automatic_frame_maps_rectangle_width_to_local_y():
    mesh = _beam_mesh(props={"height": 4.0, "width": 1.0})
    frame = resolve_beam_frame(mesh, mesh.elements[0])
    section = parse_beam2_section(mesh.elements[0].props)

    assert frame.local_y == pytest.approx((0.0, 1.0, 0.0))
    assert frame.local_z == pytest.approx((0.0, 0.0, 1.0))
    assert section.Iyy == pytest.approx(1.0 * 4.0**3 / 12.0)
    assert section.Izz == pytest.approx(4.0 * 1.0**3 / 12.0)


def test_beam2_rectangle_dimension_swap_swaps_bending_inertias_only():
    tall = parse_beam2_section(
        {"section_type": "rectangle", "height": 4.0, "width": 1.0}
    )
    wide = parse_beam2_section(
        {"section_type": "rectangle", "height": 1.0, "width": 4.0}
    )

    assert wide.area == pytest.approx(tall.area)
    assert wide.J == pytest.approx(tall.J)
    assert wide.Iyy == pytest.approx(tall.Izz)
    assert wide.Izz == pytest.approx(tall.Iyy)


def test_beam2_square_section_torsion_matches_saint_venant_coefficient():
    section = parse_beam2_section(
        {"section_type": "rectangle", "height": 2.0, "width": 2.0}
    )

    assert section.J == pytest.approx(0.14057701495517982 * 2.0**4)


@pytest.mark.parametrize(
    ("props", "message"),
    [
        ({}, "section_type"),
        ({"section_type": "general"}, "section_type"),
        ({"section_type": "solid_circle"}, "radius"),
        ({"section_type": "solid_circle", "radius": 0.0}, "radius"),
        ({"section_type": "solid_circle", "radius": np.inf}, "radius"),
        (
            {"section_type": "hollow_circle", "outer_radius": 1.0, "inner_radius": 1.0},
            "outer_radius",
        ),
        (
            {"section_type": "rectangle", "height": 1.0, "width": -1.0},
            "width",
        ),
        (
            {"section_type": "rectangle", "height": 1.0, "width": 2.0, "radius": 3.0},
            "radius",
        ),
    ],
)
def test_beam2_standard_sections_reject_invalid_contracts(props, message):
    with pytest.raises((KeyError, ValueError), match=message):
        parse_beam2_section(props)


@pytest.mark.parametrize(
    ("props", "forces", "expected"),
    [
        (
            {"section_type": "solid_circle", "radius": 2.0},
            (10.0, 3.0, 4.0),
            (5.0 / np.pi, 0.0, 5.0 / np.pi),
        ),
        (
            {
                "section_type": "hollow_circle",
                "outer_radius": 2.0,
                "inner_radius": 1.0,
            },
            (6.0 * np.pi, 45.0 * np.pi / 4.0, 15.0 * np.pi),
            (12.0, -8.0, 12.0),
        ),
        (
            {"section_type": "rectangle", "height": 4.0, "width": 2.0},
            (16.0, 32.0 / 3.0, 8.0 / 3.0),
            (5.0, -1.0, 5.0),
        ),
    ],
)
def test_beam2_section_axial_stress_extrema_include_axial_and_biaxial_bending(
    props, forces, expected
):
    section = parse_beam2_section(props)

    assert axial_stress_extrema(section, *forces) == pytest.approx(expected)


def _beam_result(mesh, U, step=None):
    model = FEMModel(mesh=mesh)
    selected_step = step or AnalysisStep("result")
    return ModelResult(
        model,
        selected_step,
        np.asarray(U, dtype=float),
        np.zeros(mesh.num_dofs),
    )


def test_beam2_rigid_motion_recovers_zero_nodal_axial_stress():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    rigid = np.tile([0.4, -0.2, 0.7, 0.0, 0.0, 0.0], 2)

    rows = beam_stress.nodal_envelope(_beam_result(mesh, rigid))

    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], abs=1e-10
    )


def test_beam2_pure_tension_recovers_positive_stress_at_both_ends():
    mesh = _beam_mesh()
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 0)] = 0.04
    result = _beam_result(mesh, U)

    rows = beam_stress.nodal_envelope(result)

    expected = 210.0 * 0.04 / 4.0
    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(expected, expected, expected), (expected, expected, expected)]
    )
    assert beam_stress.absolute_maximum(result) == pytest.approx(expected)


def test_beam2_pure_bending_recovers_same_extrema_at_both_ends():
    mesh = _beam_mesh()
    section = parse_beam2_section(mesh.elements[0].props)
    curvature = 0.03
    length = 4.0
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 1)] = 0.5 * curvature * length**2
    U[mesh.global_dof(2, 5)] = curvature * length

    rows = beam_stress.nodal_envelope(_beam_result(mesh, U))

    moment = 210.0 * section.Izz * curvature
    increment = abs(moment / section.Izz) * section.width / 2.0
    assert np.allclose(
        [(row.maximum, row.minimum, row.absolute_maximum) for row in rows],
        [(increment, -increment, increment), (increment, -increment, increment)],
    )


def test_beam2_pure_bending_about_local_y_recovers_section_extrema():
    mesh = _beam_mesh()
    section = parse_beam2_section(mesh.elements[0].props)
    curvature = 0.02
    length = 4.0
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 2)] = -0.5 * curvature * length**2
    U[mesh.global_dof(2, 4)] = curvature * length

    rows = beam_stress.nodal_envelope(_beam_result(mesh, U))

    moment = 210.0 * section.Iyy * curvature
    increment = abs(moment / section.Iyy) * section.height / 2.0
    assert np.allclose(
        [(row.maximum, row.minimum, row.absolute_maximum) for row in rows],
        [(increment, -increment, increment), (increment, -increment, increment)],
    )


def test_beam2_inclined_and_reversed_elements_preserve_physical_extrema():
    inclined = _beam_mesh(end=(2.0, 3.0, 6.0))
    rotation = resolve_beam_frame(
        inclined,
        inclined.elements[0],
    ).rotation
    local_displacement = np.zeros(12)
    local_displacement[6] = 0.07
    local_displacement[7] = 0.02
    local_displacement[11] = 0.01
    global_displacement = np.zeros(12)
    for start in (0, 3, 6, 9):
        global_displacement[start : start + 3] = rotation.T @ local_displacement[start : start + 3]

    forward = beam_stress.nodal_envelope(_beam_result(inclined, global_displacement))
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    reversed_mesh.elements[0].node_ids = [2, 1]
    reversed_rows = beam_stress.nodal_envelope(
        _beam_result(reversed_mesh, global_displacement)
    )

    forward_by_node = {row.node_id: row for row in forward}
    reversed_by_node = {row.node_id: row for row in reversed_rows}
    for node_id in inclined.node_ids:
        assert (
            reversed_by_node[node_id].maximum,
            reversed_by_node[node_id].minimum,
            reversed_by_node[node_id].absolute_maximum,
        ) == pytest.approx(
            (
                forward_by_node[node_id].maximum,
                forward_by_node[node_id].minimum,
                forward_by_node[node_id].absolute_maximum,
            )
        )


def test_beam2_shared_node_uses_maximum_minimum_envelope_without_averaging():
    props = dict(_beam_mesh().elements[0].props)
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(1, [1, 2], "Beam2", dict(props)),
            Element3D(2, [2, 3], "Beam2", dict(props)),
        ],
        dofs_per_node=6,
    )
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 0)] = 0.1

    rows = beam_stress.nodal_envelope(_beam_result(mesh, U))

    assert [row.node_id for row in rows] == [1, 2, 3]
    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(21.0, 21.0, 21.0), (21.0, -21.0, 21.0), (-21.0, -21.0, 21.0)]
    )


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


@pytest.mark.parametrize(
    ("dof", "section_property"),
    [
        (0, "area"),
        (1, "Izz"),
        (2, "Iyy"),
        (3, "J"),
    ],
    ids=["axial", "bend-local-y", "bend-local-z", "torsion"],
)
def test_beam2_cantilever_matches_closed_form_tip_response(dof, section_property):
    mesh = _beam_mesh()
    section = parse_beam2_section(mesh.elements[0].props)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    free_stiffness = stiffness[6:12, 6:12]
    load = 12.0
    force = np.zeros(6)
    force[dof] = load

    tip = np.linalg.solve(free_stiffness, force)

    if dof == 0:
        expected = load * 4.0 / (210.0 * section.area)
    elif dof in (1, 2):
        shear_modulus = 210.0 / (2.0 * (1.0 + 0.25))
        shear_rigidities = section.effective_shear_rigidities(
            shear_modulus,
            0.25,
        )
        shear_rigidity = shear_rigidities[dof - 1]
        expected = (
            load
            * 4.0**3
            / (3.0 * 210.0 * getattr(section, section_property))
            + load * 4.0 / shear_rigidity
        )
    else:
        expected = load * 4.0 / ((210.0 / 2.5) * section.J)

    assert tip[dof] == pytest.approx(expected)


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


def test_beam2_inclined_stiffness_is_symmetric_and_reversal_invariant():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]

    stiffness = kernel.stiffness(mesh, mesh.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert stiffness == pytest.approx(stiffness.T, abs=1e-12)
    assert reversed_stiffness == pytest.approx(
        permutation @ stiffness @ permutation.T,
        abs=1e-10,
    )


def test_beam2_explicit_orientation_reversal_only_permutes_stiffness():
    properties = {BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0)}
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), props=properties)
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0), props=properties)
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]

    stiffness = kernel.stiffness(mesh, mesh.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert reversed_stiffness == pytest.approx(
        permutation @ stiffness @ permutation.T,
        abs=1e-10,
    )


@pytest.mark.parametrize("end", [(0.0, 0.0, 4.0), (1e-14, 0.0, 4.0)])
def test_beam2_global_z_fallback_is_reversal_invariant(end):
    mesh = _beam_mesh(end=end)
    reversed_mesh = _beam_mesh(end=end)
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]

    stiffness = kernel.stiffness(mesh, mesh.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert reversed_stiffness == pytest.approx(
        permutation @ stiffness @ permutation.T,
        abs=1e-10,
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
