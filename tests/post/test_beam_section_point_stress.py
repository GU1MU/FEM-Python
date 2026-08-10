from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel
from fem.elements.beam_section import (
    BeamSectionEndForces,
    BeamSectionPoint,
    default_section_points,
    parse_beam2_section,
    recover_section_point_stress,
)


def _section(section_type: str):
    dimensions = {
        "rectangle": {"height": 4.0, "width": 2.0},
        "solid_circle": {"radius": 2.0},
        "hollow_circle": {"outer_radius": 2.0, "inner_radius": 1.0},
    }[section_type]
    return parse_beam2_section({"section_type": section_type, **dimensions})


@pytest.mark.parametrize(
    ("section_type", "expected"),
    [
        (
            "rectangle",
            ((1, 1.0, 2.0), (2, -1.0, 2.0), (3, -1.0, -2.0), (4, 1.0, -2.0)),
        ),
        (
            "solid_circle",
            ((1, 2.0, 0.0), (2, 0.0, 2.0), (3, -2.0, 0.0), (4, 0.0, -2.0)),
        ),
        (
            "hollow_circle",
            ((1, 2.0, 0.0), (2, 0.0, 2.0), (3, -2.0, 0.0), (4, 0.0, -2.0)),
        ),
    ],
)
def test_default_section_points_have_stable_identity_and_order(
    section_type: str,
    expected: tuple[tuple[int, float, float], ...],
) -> None:
    points = default_section_points(_section(section_type))

    assert (
        tuple((point.number, point.local_y, point.local_z) for point in points)
        == expected
    )
    assert len({point.number for point in points}) == 4
    assert len({point.local_coordinates for point in points}) == 4
    with pytest.raises(FrozenInstanceError):
        points[0].local_y = 0.0
    with pytest.raises(TypeError, match="integer"):
        BeamSectionPoint(True, 0.0, 0.0)


@pytest.mark.parametrize(
    "section_type",
    ("rectangle", "solid_circle", "hollow_circle"),
)
def test_pure_axial_force_is_uniform_at_all_points(section_type: str) -> None:
    section = _section(section_type)
    result = recover_section_point_stress(
        section,
        BeamSectionEndForces(7.0 * section.area, 0.0, 0.0, 0.0),
    )

    assert [row.s11 for row in result.point_stresses] == pytest.approx([7.0] * 4)
    assert [row.s12 for row in result.point_stresses] == [0.0] * 4
    assert result.section_values() == pytest.approx(
        {
            "S11Max": 7.0,
            "S11Min": 7.0,
            "S11AbsMax": 7.0,
            "S12AbsMax": 0.0,
        }
    )


def test_rectangle_pure_bending_signs_follow_local_y_and_z_axes() -> None:
    section = _section("rectangle")

    about_y = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 2.0 * section.Iyy, 0.0, 0.0),
    )
    about_z = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, 3.0 * section.Izz, 0.0),
    )

    assert [row.s11 for row in about_y.point_stresses] == pytest.approx(
        [-4.0, -4.0, 4.0, 4.0]
    )
    assert [row.s11 for row in about_z.point_stresses] == pytest.approx(
        [3.0, -3.0, -3.0, 3.0]
    )
    assert about_y.s11_max == pytest.approx(4.0)
    assert about_y.s11_min == pytest.approx(-4.0)
    assert about_z.s11_abs_max == pytest.approx(3.0)


@pytest.mark.parametrize("section_type", ("solid_circle", "hollow_circle"))
def test_circular_pure_bending_signs_follow_axis_intersections(
    section_type: str,
) -> None:
    section = _section(section_type)

    about_y = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, section.Iyy, 0.0, 0.0),
    )
    about_z = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, section.Izz, 0.0),
    )

    assert [row.s11 for row in about_y.point_stresses] == pytest.approx(
        [0.0, -2.0, 0.0, 2.0]
    )
    assert [row.s11 for row in about_z.point_stresses] == pytest.approx(
        [2.0, 0.0, -2.0, 0.0]
    )


@pytest.mark.parametrize("section_type", ("solid_circle", "hollow_circle"))
def test_circular_pure_torsion_matches_outer_radius_solution(
    section_type: str,
) -> None:
    section = _section(section_type)
    torque = 5.0 * section.J / 2.0

    result = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, 0.0, torque),
    )

    assert [row.s11 for row in result.point_stresses] == [0.0] * 4
    assert [row.s12 for row in result.point_stresses] == pytest.approx([5.0] * 4)
    assert result.s12_abs_max == pytest.approx(5.0)


def test_rectangle_torsion_has_zero_corner_shear_and_nonzero_true_maximum() -> None:
    section = _section("rectangle")

    result = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, 0.0, 9.0),
    )

    assert [row.s12 for row in result.point_stresses] == [0.0] * 4
    assert result.s12_abs_max > 0.0
    negative = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, 0.0, -9.0),
    )
    assert negative.s12_abs_max == pytest.approx(result.s12_abs_max)


def test_square_torsion_maximum_matches_saint_venant_series_benchmark() -> None:
    section = parse_beam2_section(
        {"section_type": "rectangle", "height": 2.0, "width": 2.0}
    )

    result = recover_section_point_stress(
        section,
        BeamSectionEndForces(0.0, 0.0, 0.0, 8.0),
    )

    # For a square of side a: tau_max = 4.80387553775 T / a^3.
    assert result.s12_abs_max == pytest.approx(4.803875537753306)


def test_combined_stress_invariants_are_derived_independently_per_point() -> None:
    section = _section("solid_circle")
    result = recover_section_point_stress(
        section,
        BeamSectionEndForces(
            3.0 * section.area,
            0.5 * section.Iyy,
            1.25 * section.Izz,
            2.0 * section.J / 2.0,
        ),
    )

    assert len({row.s11 for row in result.point_stresses}) > 1
    for row in result.point_stresses:
        span = np.sqrt(row.s11**2 + 4.0 * row.s12**2)
        assert row.mises == pytest.approx(np.sqrt(row.s11**2 + 3.0 * row.s12**2))
        assert (
            row.max_principal,
            row.mid_principal,
            row.min_principal,
        ) == pytest.approx(((row.s11 + span) / 2.0, 0.0, (row.s11 - span) / 2.0))
        assert row.max_principal >= row.mid_principal >= row.min_principal

    sampled_abs_max = max(abs(row.s11) for row in result.point_stresses)
    assert result.s11_abs_max > sampled_abs_max


def test_circular_section_axis_rotation_preserves_physical_point_stress() -> None:
    section = _section("solid_circle")
    original = recover_section_point_stress(
        section,
        BeamSectionEndForces(2.0, 3.0, 4.0, 5.0),
    )
    rotated = recover_section_point_stress(
        section,
        BeamSectionEndForces(2.0, 4.0, -3.0, 5.0),
    )

    # y' = z and z' = -y maps original points 1..4 to 4,1,2,3.
    for old_number, new_number in ((1, 4), (2, 1), (3, 2), (4, 3)):
        old = original.point_stresses[old_number - 1]
        new = rotated.point_stresses[new_number - 1]
        assert old.values() == pytest.approx(new.values())
    assert original.section_values() == pytest.approx(rotated.section_values())


def _beam_mesh(*, reversed_connectivity: bool = False) -> Mesh3D:
    node_ids = [2, 1] if reversed_connectivity else [1, 2]
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 4.0, 0.0, 0.0)],
        elements=[
            Element3D(
                10,
                node_ids,
                "Beam2",
                {
                    "E": 200.0,
                    "nu": 0.25,
                    "section_type": "solid_circle",
                    "radius": 1.5,
                },
            )
        ],
        dofs_per_node=6,
    )


def test_new_end_force_contract_adds_torque_without_changing_legacy_shape() -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    section = parse_beam2_section(element.props)
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 0)] = 0.04
    displacement[mesh.global_dof(2, 3)] = 0.02
    kernel = get_element_kernel("Beam2")

    forces = kernel.local_section_end_forces(mesh, element, displacement)
    legacy = kernel.local_end_actions(mesh, element, displacement)
    shear_modulus = element.props["E"] / (2.0 * (1.0 + element.props["nu"]))

    assert legacy.shape == (2, 3)
    assert legacy == pytest.approx(
        np.asarray([[row.axial_force, row.moment_y, row.moment_z] for row in forces])
    )
    axial_force = element.props["E"] * section.area * 0.04 / 4.0
    assert [row.N for row in forces] == pytest.approx([axial_force] * 2)
    assert [row.My for row in forces] == [0.0, 0.0]
    assert [row.Mz for row in forces] == [0.0, 0.0]
    assert [row.T for row in forces] == pytest.approx(
        [shear_modulus * section.J * 0.02 / 4.0] * 2
    )
    with pytest.raises(FrozenInstanceError):
        forces[0].torque = 0.0


def test_connectivity_reversal_preserves_resultant_magnitudes_by_node() -> None:
    displacement = np.zeros(12)
    displacement[6] = 0.04
    displacement[9] = 0.02
    kernel = get_element_kernel("Beam2")
    forward_mesh = _beam_mesh()
    reversed_mesh = _beam_mesh(reversed_connectivity=True)

    forward = kernel.local_section_end_forces(
        forward_mesh,
        forward_mesh.elements[0],
        displacement,
    )
    reversed_forces = kernel.local_section_end_forces(
        reversed_mesh,
        reversed_mesh.elements[0],
        displacement,
    )
    forward_by_node = dict(zip((1, 2), forward, strict=True))
    reversed_by_node = dict(zip((2, 1), reversed_forces, strict=True))

    for node_id in (1, 2):
        assert abs(reversed_by_node[node_id].N) == pytest.approx(
            abs(forward_by_node[node_id].N)
        )
        assert abs(reversed_by_node[node_id].T) == pytest.approx(
            abs(forward_by_node[node_id].T)
        )
