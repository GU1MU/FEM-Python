import pytest

from fem.post.stress.beam import recover_integration_point_stress
from fem.solvers import static_linear
from tests.helpers.beam_section_builders import (
    _SECTION_CASES, _POINT_COLUMNS, _inline_cantilever, _independent_integration_point_oracle,
)


@pytest.mark.parametrize(("section_type", "dimensions"), _SECTION_CASES, ids=["rectangle", "solid-circle", "hollow-circle"])
def test_four_section_points_recover_independent_axial_and_bending_stress(section_type, dimensions):
    result = static_linear.solve(_inline_cantilever(section_type, dimensions), "Load")
    recovered = recover_integration_point_stress(result)
    expected = _independent_integration_point_oracle(section_type, dimensions)

    assert tuple(field.point_number for field in recovered.section_points) == (1, 2, 3, 4)
    for field in recovered.section_points:
        assert len(field.rows) == 1
        row = field.rows[0]
        assert (row.element_id, row.integration_point, row.section_point.number) == (10, 1, field.point_number)
        assert tuple(row.values()) == _POINT_COLUMNS
        assert row.s11 == pytest.approx(expected[field.point_number][0], rel=2e-10, abs=1e-8)
