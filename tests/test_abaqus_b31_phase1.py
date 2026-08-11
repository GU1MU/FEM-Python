from __future__ import annotations

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel
from fem.elements.beam_frame import BEAM_LOCAL_Y_REFERENCE_KEY
from fem.elements.beam_section import parse_beam2_section
from fem.elements.line import _beam2_b31_interpolation


_E = 210.0e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_TIP_LOAD = np.asarray((1100.0, 1700.0, 2300.0, 700.0, 0.0, 0.0))
_SECTION_CASES = {
    "RECT": {"section_type": "rectangle", "width": 0.2, "height": 0.4},
    "CIRC": {"section_type": "solid_circle", "radius": 0.2},
    "THICK PIPE": {
        "section_type": "hollow_circle",
        "outer_radius": 0.2,
        "inner_radius": 0.1,
    },
}

# Abaqus 2023 B31, one element per case, *BEAM SECTION with default transverse
# shear stiffness and default slenderness compensation.  The source model is
# isolated from repository product data and uses only the dimensions above.
_ABAQUS_TIP_ORACLE = {
    ("RECT", 0.5): (
        3.273809667803107e-08,
        1.345238047179009e-06,
        6.120448006186052e-07,
        5.9673807551556285e-06,
        -1.283482106373413e-06,
        3.794642907450907e-06,
    ),
    ("RECT", 2.0): (
        1.309523867121243e-07,
        7.680952694499865e-05,
        2.660784230101854e-05,
        2.3869523020622514e-05,
        -2.05357137019746e-05,
        6.071428651921451e-05,
    ),
    ("RECT", 12.0): (
        7.857142918510363e-07,
        0.01646085642278194,
        0.005571411922574043,
        0.0001432171381237351,
        -0.0007392857223749161,
        0.002185714198276401,
    ),
    ("CIRC", 0.5): (
        2.084171946137303e-08,
        3.444170317834505e-07,
        4.659759724745527e-07,
        1.724178559925349e-06,
        -1.089453462554957e-06,
        8.052482485254586e-07,
    ),
    ("CIRC", 2.0): (
        8.336687784549213e-08,
        1.639690526644699e-05,
        2.218404733866919e-05,
        6.896714239701396e-06,
        -1.743125540087931e-05,
        1.288397197640734e-05,
    ),
    ("CIRC", 12.0): (
        5.002012244403886e-07,
        0.003462690394371748,
        0.00468481658026576,
        4.138028452871367e-05,
        -0.0006275252089835703,
        0.0004638229729607701,
    ),
    ("THICK PIPE", 0.5): (
        2.778895868971176e-08,
        4.697782287621521e-07,
        6.355822961268132e-07,
        1.839123797253706e-06,
        -1.162083663075464e-06,
        8.589314575147e-07,
    ),
    ("THICK PIPE", 2.0): (
        1.11155834758847e-07,
        1.926388540596236e-05,
        2.606290399853606e-05,
        7.356495189014822e-06,
        -1.859333860920742e-05,
        1.37429033202352e-05,
    ),
    ("THICK PIPE", 12.0): (
        6.66935022763937e-07,
        0.004009772092103958,
        0.005424986127763987,
        4.413897113408893e-05,
        -0.0006693602190352976,
        0.0004947445122525096,
    ),
}

# Abaqus 2023 B31, default integrated RECT section (5x5 Simpson points),
# E=2.6, nu=0.3 (and therefore G=1), unit length and unit end torque.  The
# r=1.37 and r=7.3 pairs are out-of-sample checks that were not used while
# identifying the closed finite-dimensional 5x5 formula in production.
_ABAQUS_RECT_DEFAULT_5X5_TORSION_J_ORACLE = (
    (1.0, 1.0, 0.14083333333333087),
    (1.0, 2.0, 0.45385629716914206),
    (2.0, 1.0, 0.45385629716914206),
    (1.0, 1.37, 0.25207089777347835),
    (1.37, 1.0, 0.25207089777347835),
    (1.0, 7.3, 2.2198688065298127),
    (7.3, 1.0, 2.2198688065298136),
)


def _mesh(
    section: dict[str, float | str],
    length: float,
    *,
    end: tuple[float, float, float] | None = None,
    local_y: tuple[float, float, float] | None = None,
    reversed_connectivity: bool = False,
) -> Mesh3D:
    properties: dict[str, object] = {"E": _E, "nu": _NU, **section}
    if local_y is not None:
        properties[BEAM_LOCAL_Y_REFERENCE_KEY] = local_y
    return Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, *(end or (length, 0.0, 0.0))),
        ),
        elements=(
            Element3D(
                10,
                (2, 1) if reversed_connectivity else (1, 2),
                "Beam2",
                properties,
            ),
        ),
        dofs_per_node=6,
    )


@pytest.mark.parametrize(
    ("profile", "length", "oracle"),
    tuple(
        (profile, length, oracle)
        for (profile, length), oracle in _ABAQUS_TIP_ORACLE.items()
    ),
)
def test_three_sections_and_three_slenderness_ratios_match_abaqus_b31(
    profile: str,
    length: float,
    oracle: tuple[float, ...],
) -> None:
    mesh = _mesh(_SECTION_CASES[profile], length)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])

    tip = np.linalg.solve(stiffness[6:, 6:], _TIP_LOAD)

    # All six components exercise the single B31 owner, including the default
    # integrated-RECT Prandtl/Simpson torsion constant.
    formulation_components = (0, 1, 2, 3, 4, 5)
    np.testing.assert_allclose(
        tip[list(formulation_components)],
        np.asarray(oracle)[list(formulation_components)],
        rtol=5.0e-6,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("profile", tuple(_SECTION_CASES))
def test_axial_and_saint_venant_torsion_remain_exact(profile: str) -> None:
    length = 3.7
    section_properties = _SECTION_CASES[profile]
    mesh = _mesh(section_properties, length)
    section = parse_beam2_section(section_properties)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    load = np.asarray((1100.0, 0.0, 0.0, 700.0, 0.0, 0.0))

    tip = np.linalg.solve(stiffness[6:, 6:], load)

    assert tip[0] == pytest.approx(
        load[0] * length / (_E * section.area), rel=1.0e-12
    )
    assert tip[3] == pytest.approx(
        load[3] * length / (_G * section.J), rel=1.0e-12
    )
    np.testing.assert_allclose(tip[[1, 2, 4, 5]], 0.0, atol=1.0e-15)


@pytest.mark.parametrize(
    ("width", "height", "oracle_j"),
    _ABAQUS_RECT_DEFAULT_5X5_TORSION_J_ORACLE,
)
def test_default_integrated_rect_torsion_matches_multi_aspect_abaqus_oracle(
    width: float,
    height: float,
    oracle_j: float,
) -> None:
    section_properties = {
        "section_type": "rectangle",
        "width": width,
        "height": height,
    }
    section = parse_beam2_section(section_properties)
    mesh = _mesh(section_properties, 1.0)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    tip_load = np.asarray((0.0, 0.0, 0.0, 700.0, 0.0, 0.0))

    tip = np.linalg.solve(stiffness[6:, 6:], tip_load)
    displacement = np.concatenate((np.zeros(6), tip))
    internal_force = stiffness @ displacement

    assert section.J == pytest.approx(oracle_j, rel=1.0e-12)
    assert tip[3] == pytest.approx(700.0 / (_G * oracle_j), rel=1.0e-12)
    np.testing.assert_allclose(tip[[0, 1, 2, 4, 5]], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        internal_force,
        np.asarray((0.0, 0.0, 0.0, -700.0, 0.0, 0.0, *tip_load)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_default_slenderness_compensation_is_directional() -> None:
    section = parse_beam2_section(_SECTION_CASES["RECT"])
    length = 2.0
    actual_y, actual_z = section.effective_shear_rigidities(_G, _NU)

    compensated_y, compensated_z = section.abaqus_b31_shear_rigidities(
        _G,
        _NU,
        length,
    )

    expected_y = actual_y / (
        1.0 + 0.25 * length**2 * section.area / (12.0 * section.Izz)
    )
    expected_z = actual_z / (
        1.0 + 0.25 * length**2 * section.area / (12.0 * section.Iyy)
    )
    assert compensated_y == pytest.approx(expected_y)
    assert compensated_z == pytest.approx(expected_z)
    assert compensated_y != pytest.approx(compensated_z)


def test_b31_generalized_displacements_use_linear_nodal_interpolation() -> None:
    shape, _ = _beam2_b31_interpolation(0.25, 3.0)
    nodal = np.arange(12.0)

    interpolated = shape @ nodal

    np.testing.assert_allclose(interpolated, 0.75 * nodal[:6] + 0.25 * nodal[6:])


def test_b31_interpolation_annihilates_all_six_rigid_body_modes() -> None:
    length = 2.4
    _, strain = _beam2_b31_interpolation(0.5, length)
    rigid_modes = np.zeros((12, 6))
    rigid_modes[(0, 6), 0] = 1.0
    rigid_modes[(1, 7), 1] = 1.0
    rigid_modes[(2, 8), 2] = 1.0
    rigid_modes[(3, 9), 3] = 1.0
    rigid_modes[(4, 10), 4] = 1.0
    rigid_modes[8, 4] = -length
    rigid_modes[(5, 11), 5] = 1.0
    rigid_modes[7, 5] = length

    np.testing.assert_allclose(strain @ rigid_modes, 0.0, atol=1.0e-15)


def test_free_b31_stiffness_is_symmetric_with_exactly_six_rigid_modes() -> None:
    mesh = _mesh(_SECTION_CASES["RECT"], 1.3)
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    scale = float(np.max(np.abs(stiffness)))
    eigenvalues = np.linalg.eigvalsh((stiffness + stiffness.T) / 2.0)

    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0.0, atol=scale * 1e-12)
    assert np.count_nonzero(np.abs(eigenvalues) <= eigenvalues[-1] * 1e-10) == 6
    assert np.count_nonzero(eigenvalues > eigenvalues[-1] * 1e-10) == 6


def test_rotated_and_reversed_connectivity_stiffness_is_covariant() -> None:
    angle_z = np.deg2rad(31.0)
    angle_y = np.deg2rad(-23.0)
    rotate_z = np.asarray(
        (
            (np.cos(angle_z), -np.sin(angle_z), 0.0),
            (np.sin(angle_z), np.cos(angle_z), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotate_y = np.asarray(
        (
            (np.cos(angle_y), 0.0, np.sin(angle_y)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle_y), 0.0, np.cos(angle_y)),
        )
    )
    rotation = rotate_z @ rotate_y
    length = 2.7
    baseline = _mesh(
        _SECTION_CASES["RECT"], length, local_y=(0.0, 1.0, 0.0)
    )
    end = tuple(rotation @ np.asarray((length, 0.0, 0.0)))
    local_y = tuple(rotation @ np.asarray((0.0, 1.0, 0.0)))
    rotated = _mesh(_SECTION_CASES["RECT"], length, end=end, local_y=local_y)
    reversed_mesh = _mesh(
        _SECTION_CASES["RECT"],
        length,
        end=end,
        local_y=local_y,
        reversed_connectivity=True,
    )
    kernel = get_element_kernel("Beam2")
    stiffness = kernel.stiffness(baseline, baseline.elements[0])
    rotated_stiffness = kernel.stiffness(rotated, rotated.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])
    coordinate = np.zeros((12, 12))
    for start in (0, 3, 6, 9):
        coordinate[start : start + 3, start : start + 3] = rotation
    permutation = np.eye(12)[(6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5), :]

    np.testing.assert_allclose(
        rotated_stiffness,
        coordinate @ stiffness @ coordinate.T,
        rtol=1.0e-10,
        atol=np.max(np.abs(stiffness)) * 1.0e-12,
    )
    np.testing.assert_allclose(
        reversed_stiffness,
        permutation @ rotated_stiffness @ permutation.T,
        rtol=1.0e-10,
        atol=np.max(np.abs(stiffness)) * 1.0e-12,
    )


def test_old_exact_static_bending_interpolation_is_not_production_api() -> None:
    from fem.elements import line

    assert not hasattr(line, "_beam2_bending_interpolation")
