from __future__ import annotations

from math import pi, sqrt
from pathlib import Path

import pytest

from fem import abaqus
from fem.application import RegionRef, resolve_effective_beam_frames
from fem.elements.beam_section import parse_beam2_section
from fem.solvers.static_linear import solve
from tests.helpers.file_builders import write_inp


STANDARD = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _rect_cantilever(load_label: str) -> list[str]:
    return [
        "*Heading",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 2.0, 0.0, 0.0",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        "*Nset, nset=FIXED",
        "1",
        "*Material, name=STEEL",
        "*Elastic",
        "210000000000.0, 0.3",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.20, 0.10",
        "0.0, 1.0, 0.0",
        "*Boundary",
        "FIXED, 1, 6, 0.0",
        "*Step, name=LOAD",
        "*Static",
        "*Dload",
        f"BEAM, {load_label}, 120.0",
        "*End Step",
    ]


def test_rect_a_b_axes_match_euler_bernoulli_tip_deflection_oracle(
    tmp_path,
) -> None:
    p1_path = write_inp(
        tmp_path,
        "rect_p1.inp",
        _rect_cantilever("P1"),
    )
    p2_path = write_inp(
        tmp_path,
        "rect_p2.inp",
        _rect_cantilever("P2"),
    )

    p1_model = abaqus.read(p1_path)
    p2_model = abaqus.read(p2_path)
    p1_result = solve(p1_model, "LOAD")
    p2_result = solve(p2_model, "LOAD")

    height = 0.20
    width = 0.10
    length = 2.0
    modulus = 210.0e9
    load = 120.0
    i_yy = height * width**3 / 12.0
    i_zz = width * height**3 / 12.0
    expected_y = load * length**4 / (8.0 * modulus * i_zz)
    expected_z = load * length**4 / (8.0 * modulus * i_yy)

    displacement_y = p1_result.nodal_displacement(2, 2)
    displacement_z = p2_result.nodal_displacement(2, 3)
    assert displacement_y == pytest.approx(expected_y)
    assert displacement_z == pytest.approx(expected_z)
    assert displacement_y / displacement_z == pytest.approx(
        i_yy / i_zz
    )
    assert displacement_y / displacement_z == pytest.approx(0.25)


def test_thick_pipe_maps_to_exact_annulus_section_properties() -> None:
    model = abaqus.read(STANDARD / "b31_thick_pipe.inp")
    assignment = model.sections[0]
    section = parse_beam2_section(
        {
            "section_type": assignment.section_type,
            **assignment.properties,
        }
    )
    outer_radius = 0.10
    inner_radius = 0.075
    radius_delta_2 = outer_radius**2 - inner_radius**2
    radius_delta_4 = outer_radius**4 - inner_radius**4

    assert section.outer_radius == pytest.approx(outer_radius)
    assert section.inner_radius == pytest.approx(inner_radius)
    assert section.area == pytest.approx(pi * radius_delta_2)
    assert section.Iyy == pytest.approx(pi * radius_delta_4 / 4.0)
    assert section.Izz == pytest.approx(pi * radius_delta_4 / 4.0)
    assert section.J == pytest.approx(pi * radius_delta_4 / 2.0)


def test_inclined_default_n1_frame_keeps_global_and_local_axes_distinct() -> None:
    model = abaqus.read(STANDARD / "b31_rect_default_n1.inp")
    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAM"),
    )
    frame = report.frames[0]

    assert report.passed
    assert frame.local_x == pytest.approx(
        (1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0)
    )
    assert frame.local_y == pytest.approx((0.0, 0.0, -1.0))
    assert frame.local_z == pytest.approx(
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0)
    )
