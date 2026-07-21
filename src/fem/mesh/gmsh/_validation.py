"""Backend-free validation for the native Gmsh meshing backend."""

from __future__ import annotations

import math
import operator
from typing import Any, Literal


_FIELD_TYPES = frozenset({"Distance", "Threshold", "Min"})


def _validate_dimension(value: Any) -> Literal[1, 2, 3]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError(f"dimension must be 1, 2, or 3, got {value!r}")
    return value


def _validate_order(value: Any) -> Literal[1, 2]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError(f"order must be integer 1 or 2, got {value!r}")
    return value


def _validate_spec_level(value: Any) -> Literal[1, 2, 3, 4, 5]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in range(1, 6)
    ):
        raise ValueError(
            "level must be a Python integer from 1 through 5, "
            f"got {value!r}"
        )
    return value


def _validate_runtime_level(value: Any) -> Literal[1, 2, 3, 4, 5]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in range(1, 6)
    ):
        raise ValueError(
            "AutoMeshSpec level must be a Python integer from 1 through "
            f"5, got {value!r}"
        )
    return value


def _validate_positive_tag(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"{label} must be a positive integer, got {value!r}"
        ) from exc
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
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be finite and > 0, got {value!r}"
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return normalized


def _nonnegative_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized < 0.0:
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return normalized


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


def _validate_field_type(
    value: Any,
) -> Literal["Distance", "Threshold", "Min"]:
    if not isinstance(value, str) or value not in _FIELD_TYPES:
        raise ValueError(
            "mesh field type must be 'Distance', 'Threshold', or 'Min', "
            f"got {value!r}"
        )
    return value
