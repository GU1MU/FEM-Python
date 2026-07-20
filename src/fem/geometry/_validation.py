"""Backend-free scalar validation shared by scripted geometry modules."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Any, Literal


def _validate_mesh_dimension(value: Any) -> Literal[1, 2, 3]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError(f"dimension must be 1, 2, or 3, got {value!r}")
    return value


def _validate_entity_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
        raise ValueError(
            f"entity dimension must be an integer from 0 through 3, got {value!r}"
        )
    return value


def _validate_positive_tag(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc
    if normalized <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return normalized


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite, got {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return normalized


def _positive_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return normalized


def _nonnegative_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized < 0.0:
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return normalized


def _positive_feature_vector(
    values: Sequence[float],
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence of finite values") from exc
    allowed_lengths = {1, count, 2 * count}
    if len(materialized) not in allowed_lengths:
        raise ValueError(
            f"{label} must contain one value, one value per target, or two "
            f"values per target; got {len(materialized)}"
        )
    return tuple(
        _positive_float(value, f"{label}[{index}]")
        for index, value in enumerate(materialized)
    )


def _integer_at_least(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        ) from exc
    if normalized < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return normalized
