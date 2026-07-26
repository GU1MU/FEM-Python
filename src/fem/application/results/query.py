"""Typed result-query values without evaluation or presentation policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from fem.post.fields import ResultRegionKey

from .data import FieldLocation
from .fields import FieldMaterializationKey, ResultSourceKey


@dataclass(frozen=True, slots=True)
class ResultQuery:
    """One exact field/component query with optional FEM identity filters."""

    field_key: FieldMaterializationKey
    component: str
    node_ids: tuple[int, ...] = ()
    element_ids: tuple[int, ...] = ()
    region_keys: tuple[ResultRegionKey, ...] = ()

    def __post_init__(self) -> None:
        if type(self.field_key) is not FieldMaterializationKey:
            raise TypeError("field_key must be FieldMaterializationKey")
        _require_nonblank_string(self.component, label="component")
        object.__setattr__(
            self,
            "node_ids",
            _identity_filter(self.node_ids, label="node_ids"),
        )
        object.__setattr__(
            self,
            "element_ids",
            _identity_filter(self.element_ids, label="element_ids"),
        )
        object.__setattr__(
            self,
            "region_keys",
            _region_filter(self.region_keys),
        )


@dataclass(frozen=True, slots=True)
class ResultQueryRecord:
    """One scalar value at an exact canonical result location."""

    source: ResultSourceKey
    location: FieldLocation
    value: float

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.location) is not FieldLocation:
            raise TypeError("location must be FieldLocation")
        value = self.value
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("value must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", numeric)


@dataclass(frozen=True, slots=True)
class ResultQueryResult:
    """Ordered query records bound to one materialization generation."""

    source: ResultSourceKey
    materialization_generation: int
    query: ResultQuery
    records: tuple[ResultQueryRecord, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError(
                "materialization_generation must be non-negative"
            )
        if type(self.query) is not ResultQuery:
            raise TypeError("query must be ResultQuery")
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        for record in self.records:
            if type(record) is not ResultQueryRecord:
                raise TypeError(
                    "records must contain only ResultQueryRecord values"
                )
            if record.source != self.source:
                raise ValueError(
                    "every query record source must match the result source"
                )


def _identity_filter(
    values: object,
    *,
    label: str,
) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    seen: set[int] = set()
    for value in values:
        if type(value) is not int:
            raise TypeError(f"{label} must contain integers")
        if value <= 0:
            raise ValueError(f"{label} must contain positive FEM IDs")
        if value in seen:
            raise ValueError(f"{label} must not contain duplicate FEM IDs")
        seen.add(value)
    return values


def _region_filter(
    values: object,
) -> tuple[ResultRegionKey, ...]:
    if type(values) is not tuple:
        raise TypeError("region_keys must be a tuple")
    seen: set[ResultRegionKey] = set()
    for value in values:
        if type(value) is not ResultRegionKey:
            raise TypeError(
                "region_keys must contain ResultRegionKey values"
            )
        if value in seen:
            raise ValueError("region_keys must not contain duplicates")
        seen.add(value)
    return values


def _require_nonblank_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


__all__ = [
    "ResultQuery",
    "ResultQueryRecord",
    "ResultQueryResult",
]
