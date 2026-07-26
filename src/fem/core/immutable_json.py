"""Strict, deeply owned immutable JSON values.

The public domain objects that use this module retain authoring data rather
than a mutable view of caller-owned dictionaries and lists.  Freezing accepts
only the concrete containers and scalar types that JSON can represent
losslessly; thawing is the explicit persistence boundary back to ordinary
``dict`` and ``list`` values.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math
from typing import Any


class FrozenJsonMapping(Mapping[str, Any]):
    """An insertion-ordered immutable JSON object.

    Storage is a tuple of immutable key/value pairs rather than a proxy around
    a mutable dictionary.  The value is therefore safe to copy, deepcopy and
    pickle as part of frozen authoring DTOs.
    """

    __slots__ = ("_items",)

    def __init__(self, value: dict[str, Any]) -> None:
        if type(value) is not dict:
            raise TypeError("JSON object must be an exact dict")
        object.__setattr__(
            self,
            "_items",
            _freeze_mapping_items(value, path="$", ancestors=set()),
        )

    @classmethod
    def _from_frozen_items(
        cls,
        items: tuple[tuple[str, Any], ...],
    ) -> FrozenJsonMapping:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_items", items)
        return instance

    def __getitem__(self, key: str) -> Any:
        for stored_key, value in self._items:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({thaw_json_mapping(self)!r})"

    def __copy__(self) -> FrozenJsonMapping:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenJsonMapping:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (type(self), (thaw_json_mapping(self),))


def freeze_json_mapping(
    value: Any,
    *,
    name: str = "JSON object",
) -> FrozenJsonMapping:
    """Deep-own one exact JSON object as recursively immutable values."""

    if type(value) is FrozenJsonMapping:
        return value
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    return FrozenJsonMapping._from_frozen_items(
        _freeze_mapping_items(value, path=name, ancestors=set())
    )


def thaw_json_mapping(
    value: Any,
    *,
    name: str = "JSON object",
) -> dict[str, Any]:
    """Return a detached ordinary JSON object from a frozen mapping."""

    if type(value) is not FrozenJsonMapping:
        raise TypeError(f"{name} must be a FrozenJsonMapping")
    return {
        key: _thaw_json_value(item, path=f"{name}.{key}")
        for key, item in value._items
    }


def _freeze_mapping_items(
    value: dict[str, Any],
    *,
    path: str,
    ancestors: set[int],
) -> tuple[tuple[str, Any], ...]:
    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"{path} contains a cyclic JSON reference")
    ancestors.add(identity)
    try:
        result: list[tuple[str, Any]] = []
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} JSON object keys must be exact strings")
            _require_utf8(key, path=f"{path} object key")
            result.append(
                (
                    key,
                    _freeze_json_value(
                        item,
                        path=f"{path}.{key}",
                        ancestors=ancestors,
                    ),
                )
            )
        return tuple(result)
    finally:
        ancestors.remove(identity)


def _freeze_json_value(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return value
    if value_type is str:
        _require_utf8(value, path=path)
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite JSON number")
        return value
    if value_type is list:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic JSON reference")
        ancestors.add(identity)
        try:
            return tuple(
                _freeze_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                )
                for index, item in enumerate(value)
            )
        finally:
            ancestors.remove(identity)
    if value_type is dict:
        return FrozenJsonMapping._from_frozen_items(
            _freeze_mapping_items(value, path=path, ancestors=ancestors)
        )
    raise TypeError(
        f"{path} contains unsupported JSON value type {value_type.__name__}"
    )


def _thaw_json_value(value: Any, *, path: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str}:
        return value
    if value_type is tuple:
        return [
            _thaw_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value_type is FrozenJsonMapping:
        return {
            key: _thaw_json_value(item, path=f"{path}.{key}")
            for key, item in value._items
        }
    raise TypeError(
        f"{path} contains invalid frozen JSON value type {value_type.__name__}"
    )


def _require_utf8(value: str, *, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path} must be valid UTF-8 text") from error


__all__ = [
    "FrozenJsonMapping",
    "freeze_json_mapping",
    "thaw_json_mapping",
]
