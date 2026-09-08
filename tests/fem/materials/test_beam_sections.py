from __future__ import annotations

from math import isfinite, pi

import pytest

from fem.elements.beam_section import parse_beam2_section


@pytest.mark.parametrize(
    ("props", "expected"),
    [
        (
            {"section_type": "solid_circle", "radius": 2.0},
            (4.0 * pi, 4.0 * pi, 4.0 * pi, 8.0 * pi),
        ),
        (
            {
                "section_type": "hollow_circle",
                "outer_radius": 2.0,
                "inner_radius": 1.0,
            },
            (3.0 * pi, 15.0 * pi / 4.0, 15.0 * pi / 4.0, 15.0 * pi / 2.0),
        ),
        (
            {"section_type": "rectangle", "height": 4.0, "width": 1.0},
            (4.0, 16.0 / 3.0, 1.0 / 3.0, 1.1043940392156846),
        ),
    ],
    ids=("solid-circle", "hollow-circle", "rectangle"),
)
def test_standard_sections_match_area_inertia_and_torsion_oracles(props, expected):
    section = parse_beam2_section(props)

    assert (section.area, section.Iyy, section.Izz, section.J) == pytest.approx(expected)


def test_rectangle_dimension_swap_swaps_bending_inertias_only():
    tall = parse_beam2_section(
        {"section_type": "rectangle", "height": 4.0, "width": 1.0}
    )
    wide = parse_beam2_section(
        {"section_type": "rectangle", "height": 1.0, "width": 4.0}
    )

    assert (tall.height, tall.width) == (4.0, 1.0)
    assert (wide.height, wide.width) == (1.0, 4.0)
    assert wide.area == pytest.approx(tall.area)
    assert wide.J == pytest.approx(tall.J)
    assert wide.Iyy == pytest.approx(tall.Izz)
    assert wide.Izz == pytest.approx(tall.Iyy)


def test_square_section_torsion_matches_abaqus_default_rect_coefficient():
    section = parse_beam2_section(
        {"section_type": "rectangle", "height": 2.0, "width": 2.0}
    )

    assert section.J == pytest.approx((169.0 / 1200.0) * 2.0**4)


@pytest.mark.parametrize(
    ("props", "message"),
    [
        ({}, "section_type"),
        ({"section_type": "general"}, "section_type"),
        ({"section_type": "solid_circle"}, "radius"),
        (
            {"section_type": "hollow_circle", "outer_radius": 1.0, "inner_radius": 1.0},
            "outer_radius",
        ),
        (
            {"section_type": "rectangle", "height": 1.0, "width": 2.0, "radius": 3.0},
            "radius",
        ),
    ],
    ids=(
        "missing-section-type",
        "unknown-section-type",
        "missing-radius",
        "equal-inner-outer-radii",
        "extraneous-dimension",
    ),
)
def test_rejects_invalid_section_definitions(props, message):
    with pytest.raises((KeyError, ValueError), match=message):
        parse_beam2_section(props)


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        ("height", 0.0),
        ("height", -1.0),
        ("height", float("nan")),
        ("height", float("inf")),
        ("height", -float("inf")),
        ("width", 0.0),
        ("width", -1.0),
        ("radius", 0.0),
        ("radius", float("inf")),
    ],
)
def test_section_dimensions_reject_nonpositive_and_nonfinite_values(dimension, value):
    props = (
        {"section_type": "solid_circle", "radius": 1.0}
        if dimension == "radius"
        else {"section_type": "rectangle", "height": 1.0, "width": 1.0}
    )
    props[dimension] = value

    with pytest.raises(ValueError, match=dimension):
        parse_beam2_section(props)


def _rectangle():
    return parse_beam2_section(
        {"section_type": "rectangle", "height": 0.8, "width": 0.3}
    )


def _solid_circle():
    return parse_beam2_section({"section_type": "solid_circle", "radius": 0.4})


def _hollow_circle(inner_radius: float):
    return parse_beam2_section(
        {
            "section_type": "hollow_circle",
            "outer_radius": 0.4,
            "inner_radius": inner_radius,
        }
    )


def test_rectangle_factor_matches_abaqus_library_value() -> None:
    poisson_ratio = 0.3
    expected = 0.85

    kappa_y, kappa_z = _rectangle().shear_correction_factors(poisson_ratio)

    assert kappa_y == pytest.approx(expected)
    assert kappa_z == pytest.approx(expected)


def test_solid_circle_factor_matches_abaqus_library_value() -> None:
    poisson_ratio = 0.3
    expected = 0.89

    kappa_y, kappa_z = _solid_circle().shear_correction_factors(poisson_ratio)

    assert kappa_y == pytest.approx(expected)
    assert kappa_z == pytest.approx(expected)


def test_hollow_circle_cowper_factors_match_independent_formula() -> None:
    poisson_ratio = 0.27
    inner_radius = 0.16
    radius_ratio_squared = (inner_radius / 0.4) ** 2
    radius_term = (1.0 + radius_ratio_squared) ** 2
    expected = 6.0 * (1.0 + poisson_ratio) * radius_term / (
        (7.0 + 6.0 * poisson_ratio) * radius_term
        + (20.0 + 12.0 * poisson_ratio) * radius_ratio_squared
    )

    kappa_y, kappa_z = _hollow_circle(inner_radius).shear_correction_factors(
        poisson_ratio
    )

    assert kappa_y == pytest.approx(expected)
    assert kappa_z == pytest.approx(expected)


def test_hollow_circle_converges_to_solid_circle_cowper_limit() -> None:
    poisson_ratio = 0.23
    expected = 6.0 * (1.0 + poisson_ratio) / (7.0 + 6.0 * poisson_ratio)

    kappa_y, kappa_z = _hollow_circle(4.0e-9).shear_correction_factors(
        poisson_ratio
    )

    assert kappa_y == pytest.approx(expected, rel=1.0e-14)
    assert kappa_z == pytest.approx(expected, rel=1.0e-14)


def test_hollow_circle_converges_to_thin_ring_cowper_limit() -> None:
    poisson_ratio = 0.23
    expected = 2.0 * (1.0 + poisson_ratio) / (4.0 + 3.0 * poisson_ratio)

    kappa_y, kappa_z = _hollow_circle(
        0.4 * (1.0 - 1.0e-9)
    ).shear_correction_factors(poisson_ratio)

    assert kappa_y == pytest.approx(expected, rel=1.0e-9)
    assert kappa_z == pytest.approx(expected, rel=1.0e-9)


@pytest.mark.parametrize(
    "section",
    [_rectangle(), _solid_circle(), _hollow_circle(0.2)],
    ids=("rectangle", "solid-circle", "hollow-circle"),
)
@pytest.mark.parametrize("poisson_ratio", [-0.99, 0.0, 0.499])
def test_supported_section_shear_factors_are_finite_positive_and_symmetric(
    section,
    poisson_ratio: float,
) -> None:
    kappa_y, kappa_z = section.shear_correction_factors(poisson_ratio)

    assert isfinite(kappa_y)
    assert isfinite(kappa_z)
    assert kappa_y > 0.0
    assert kappa_z > 0.0
    assert kappa_y == pytest.approx(kappa_z)


@pytest.mark.parametrize(
    "section",
    [_rectangle(), _solid_circle(), _hollow_circle(0.2)],
    ids=("rectangle", "solid-circle", "hollow-circle"),
)
def test_effective_shear_rigidities_apply_section_correction_factors(section) -> None:
    shear_modulus = 79.0e9
    poisson_ratio = 0.3
    kappa_y, kappa_z = section.shear_correction_factors(poisson_ratio)

    rigidity_y, rigidity_z = section.effective_shear_rigidities(
        shear_modulus,
        poisson_ratio,
    )

    assert rigidity_y == pytest.approx(kappa_y * shear_modulus * section.area)
    assert rigidity_z == pytest.approx(kappa_z * shear_modulus * section.area)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, float("nan"), float("inf")])
def test_shear_correction_factors_reject_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValueError, match=r"-1 < nu < 0.5"):
        _rectangle().shear_correction_factors(poisson_ratio)


@pytest.mark.parametrize("shear_modulus", [0.0, -1.0, float("nan"), float("inf")])
def test_effective_shear_rigidities_reject_invalid_modulus(
    shear_modulus: float,
) -> None:
    with pytest.raises(ValueError, match="shear modulus"):
        _solid_circle().effective_shear_rigidities(shear_modulus, 0.3)
