from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from fem.geometry._validation import (
    _finite_float,
    _integer_at_least,
    _nonnegative_float,
    _positive_feature_vector,
    _positive_float,
    _validate_entity_dimension,
    _validate_mesh_dimension,
    _validate_positive_tag,
)


class _IndexValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"IndexValue({self.value})"


class _NotNumeric:
    def __repr__(self) -> str:
        return "NotNumeric()"


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_validate_mesh_dimension_accepts_only_supported_python_integers(
    dimension: int,
) -> None:
    assert _validate_mesh_dimension(dimension) == dimension


@pytest.mark.parametrize(
    "value",
    (False, True, -1, 0, 4, 2.0, "2", _IndexValue(2)),
)
def test_validate_mesh_dimension_rejects_other_equivalence_classes(
    value: Any,
) -> None:
    with pytest.raises(ValueError) as captured:
        _validate_mesh_dimension(value)

    assert str(captured.value) == f"dimension must be 1, 2, or 3, got {value!r}"


@pytest.mark.parametrize("dimension", (0, 1, 2, 3))
def test_validate_entity_dimension_accepts_the_four_entity_dimensions(
    dimension: int,
) -> None:
    assert _validate_entity_dimension(dimension) == dimension


@pytest.mark.parametrize(
    "value",
    (False, True, -1, 4, 1.0, "1", _IndexValue(1)),
)
def test_validate_entity_dimension_rejects_non_python_integer_dimensions(
    value: Any,
) -> None:
    with pytest.raises(ValueError) as captured:
        _validate_entity_dimension(value)

    assert str(captured.value) == (
        "entity dimension must be an integer from 0 through 3, "
        f"got {value!r}"
    )


def test_validate_positive_tag_uses_operator_index() -> None:
    assert _validate_positive_tag(_IndexValue(7), "entity tag") == 7


@pytest.mark.parametrize("value", (False, True, 0, -2))
def test_validate_positive_tag_rejects_boolean_and_nonpositive_values(
    value: Any,
) -> None:
    with pytest.raises(ValueError) as captured:
        _validate_positive_tag(value, "entity tag")

    assert str(captured.value) == (
        f"entity tag must be a positive integer, got {value!r}"
    )
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("value", (1.0, "1", _NotNumeric()))
def test_validate_positive_tag_chains_non_indexable_failures(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _validate_positive_tag(value, "curve loop tag")

    assert str(captured.value) == (
        f"curve loop tag must be a positive integer, got {value!r}"
    )
    assert isinstance(captured.value.__cause__, TypeError)


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0, 0.0), (-3, -3.0), (2.5, 2.5), ("4.25", 4.25)),
)
def test_finite_float_preserves_numeric_coercion(value: Any, expected: float) -> None:
    assert _finite_float(value, "coordinate") == expected


@pytest.mark.parametrize("value", (False, True, float("nan"), float("inf"), -float("inf")))
def test_finite_float_rejects_boolean_and_nonfinite_values(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _finite_float(value, "coordinate")

    assert str(captured.value) == f"coordinate must be finite, got {value!r}"
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("value", ("not-a-number", _NotNumeric()))
def test_finite_float_chains_failed_numeric_coercion(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _finite_float(value, "coordinate")

    assert str(captured.value) == f"coordinate must be finite, got {value!r}"
    assert isinstance(captured.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize(("value", "expected"), ((1, 1.0), ("2.5", 2.5)))
def test_positive_float_accepts_positive_finite_values(
    value: Any,
    expected: float,
) -> None:
    assert _positive_float(value, "radius") == expected


@pytest.mark.parametrize("value", (0, -1.5))
def test_positive_float_rejects_nonpositive_values_after_finite_coercion(
    value: Any,
) -> None:
    with pytest.raises(ValueError) as captured:
        _positive_float(value, "radius")

    assert str(captured.value) == (
        f"radius must be finite and > 0, got {value!r}"
    )


@pytest.mark.parametrize("value", (True, float("nan"), float("inf")))
def test_positive_float_preserves_finite_validation_order(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _positive_float(value, "radius")

    assert str(captured.value) == f"radius must be finite, got {value!r}"


@pytest.mark.parametrize(("value", "expected"), ((0, 0.0), ("2", 2.0)))
def test_nonnegative_float_accepts_zero_and_positive_values(
    value: Any,
    expected: float,
) -> None:
    assert _nonnegative_float(value, "distance") == expected


def test_nonnegative_float_rejects_negative_values_after_finite_coercion() -> None:
    with pytest.raises(ValueError) as captured:
        _nonnegative_float(-0.5, "distance")

    assert str(captured.value) == (
        "distance must be finite and >= 0, got -0.5"
    )


@pytest.mark.parametrize("value", (False, float("nan"), -float("inf")))
def test_nonnegative_float_preserves_finite_validation_order(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _nonnegative_float(value, "distance")

    assert str(captured.value) == f"distance must be finite, got {value!r}"


def _positive_values() -> Iterator[float]:
    yield from (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ((2,), (2.0,)),
        ([1, 2, 3], (1.0, 2.0, 3.0)),
    ),
)
def test_positive_feature_vector_materializes_each_supported_length(
    values: Any,
    expected: tuple[float, ...],
) -> None:
    assert _positive_feature_vector(values, count=3, label="radii") == expected


def test_positive_feature_vector_materializes_a_fresh_generator() -> None:
    assert _positive_feature_vector(
        _positive_values(), count=3, label="radii"
    ) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_positive_feature_vector_chains_noniterable_input() -> None:
    with pytest.raises(TypeError) as captured:
        _positive_feature_vector(3, count=2, label="radii")  # type: ignore[arg-type]

    assert str(captured.value) == "radii must be a sequence of finite values"
    assert isinstance(captured.value.__cause__, TypeError)


def test_positive_feature_vector_validates_length_before_values() -> None:
    with pytest.raises(ValueError) as captured:
        _positive_feature_vector([False, -1.0], count=3, label="radii")

    assert str(captured.value) == (
        "radii must contain one value, one value per target, or two values "
        "per target; got 2"
    )


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ([1.0, 0.0, 2.0], "radii[1] must be finite and > 0, got 0.0"),
        ([1.0, float("nan"), 2.0], "radii[1] must be finite, got nan"),
    ),
)
def test_positive_feature_vector_reports_the_failing_element(
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError) as captured:
        _positive_feature_vector(values, count=3, label="radii")

    assert str(captured.value) == message


def test_integer_at_least_uses_operator_index() -> None:
    assert _integer_at_least(_IndexValue(4), "order", minimum=2) == 4


@pytest.mark.parametrize("value", (False, True, 0, 1))
def test_integer_at_least_rejects_booleans_and_values_below_minimum(
    value: Any,
) -> None:
    with pytest.raises(ValueError) as captured:
        _integer_at_least(value, "order", minimum=2)

    assert str(captured.value) == f"order must be an integer >= 2, got {value!r}"
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("value", (2.0, "2", _NotNumeric()))
def test_integer_at_least_chains_non_indexable_failures(value: Any) -> None:
    with pytest.raises(ValueError) as captured:
        _integer_at_least(value, "order", minimum=2)

    assert str(captured.value) == f"order must be an integer >= 2, got {value!r}"
    assert isinstance(captured.value.__cause__, TypeError)
