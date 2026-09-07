import numpy as np
import pytest

from fem.materials import linear_elastic


ELASTIC_MATRIX_BUILDERS = (
    linear_elastic.plane_stress_matrix,
    linear_elastic.plane_strain_matrix,
    linear_elastic.solid_3d_matrix,
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


@pytest.mark.parametrize("nu", [np.nextafter(-1.0, 0.0), np.nextafter(0.5, 0.0)])
def test_material_accepts_admissible_boundary_nearby_values(nu):
    material = linear_elastic.material(
        "valid",
        E=210.0,
        nu=nu,
        rho=0.0,
    )

    assert material.properties == {
        "E": 210.0,
        "nu": nu,
        "rho": 0.0,
    }


# E=120 and nu=0.25 give shear modulus 48 and Lame's first parameter 48.
# Independent numeric oracles use engineering strains: xx, yy, (zz), xy, (yz, zx).
PLANE_STRESS_EXPECTED = [
    [128.0, 32.0, 0.0],
    [32.0, 128.0, 0.0],
    [0.0, 0.0, 48.0],
]
PLANE_STRAIN_EXPECTED = [
    [144.0, 48.0, 0.0],
    [48.0, 144.0, 0.0],
    [0.0, 0.0, 48.0],
]
SOLID_EXPECTED = [
    [144.0, 48.0, 48.0, 0.0, 0.0, 0.0],
    [48.0, 144.0, 48.0, 0.0, 0.0, 0.0],
    [48.0, 48.0, 144.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 48.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 48.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 48.0],
]


@pytest.mark.parametrize(
    ("matrix_builder", "expected"),
    [
        (linear_elastic.plane_stress_matrix, PLANE_STRESS_EXPECTED),
        (linear_elastic.plane_strain_matrix, PLANE_STRAIN_EXPECTED),
        (linear_elastic.solid_3d_matrix, SOLID_EXPECTED),
    ],
    ids=["plane-stress", "plane-strain", "solid-3d"],
)
def test_constitutive_matrix_matches_independent_engineering_strain_oracle(
    matrix_builder, expected
):
    np.testing.assert_allclose(matrix_builder(120.0, 0.25), expected)


@pytest.mark.parametrize(
    ("plane_type", "expected"),
    [("stress", PLANE_STRESS_EXPECTED), ("strain", PLANE_STRAIN_EXPECTED)],
)
def test_plane_matrix_dispatches_to_the_requested_constitutive_law(plane_type, expected):
    np.testing.assert_allclose(linear_elastic.plane_matrix(120.0, 0.25, plane_type), expected)


@pytest.mark.parametrize(
    ("E", "nu", "message"),
    [(0.0, 0.25, "E must be finite and > 0"), (120.0, 0.5, "-1 < nu < 0.5")],
)
def test_plane_matrix_rejects_invalid_material_constants(E, nu, message):
    with pytest.raises(ValueError, match=message):
        linear_elastic.plane_matrix(E, nu, "stress")


def test_plane_matrix_rejects_unknown_plane_type():
    with pytest.raises(ValueError, match="expected 'stress' or 'strain'"):
        linear_elastic.plane_matrix(120.0, 0.25, "unknown")


@pytest.mark.parametrize("rho", [None, 7.85], ids=["without-density", "with-density"])
def test_material_preserves_name_extra_properties_and_optional_density(rho):
    material = linear_elastic.material("steel", E=120.0, nu=0.25, rho=rho, grade="A")

    expected = {"E": 120.0, "nu": 0.25, "grade": "A"}
    if rho is not None:
        expected["rho"] = rho
    assert material.name == "steel"
    assert material.properties == expected
