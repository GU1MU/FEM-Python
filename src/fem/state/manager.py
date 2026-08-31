"""Reference transactional state store used by the v2 core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import StateKey


@dataclass(slots=True)
class TransactionalStateManager:
    """Own committed/trial values without embedding history in a material model."""

    _committed: dict[StateKey, "_FrozenState"] = field(default_factory=dict)
    _trial: dict[StateKey, "_FrozenState"] = field(default_factory=dict)

    def register(
        self,
        key: StateKey,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = _key(key)
        if normalized in self._committed:
            raise ValueError(f"state key already registered: {normalized!r}")
        self._committed[normalized] = _freeze_mapping(state or {})

    def committed(self, key: StateKey) -> Mapping[str, Any]:
        normalized = _key(key)
        try:
            state = self._committed[normalized]
        except KeyError as exc:
            raise KeyError(f"unknown state key: {normalized!r}") from exc
        return state

    def stage(self, key: StateKey, state: Mapping[str, Any]) -> None:
        normalized = _key(key)
        if normalized not in self._committed:
            raise KeyError(f"unknown state key: {normalized!r}")
        self._trial[normalized] = _freeze_mapping(state)

    def stage_immutable(self, key: StateKey, state: Mapping[str, Any]) -> None:
        """Stage a caller-owned read-only state without a deep copy.

        This narrow fast path is used by batched constitutive kernels after
        they mark every stored ndarray read-only.  The ordinary ``stage``
        method deliberately keeps its defensive deep-copy behavior for all
        other callers.
        """

        normalized = _key(key)
        if normalized not in self._committed:
            raise KeyError(f"unknown state key: {normalized!r}")
        self._trial[normalized] = _freeze_immutable_mapping(state)

    def stage_immutable_batch(
        self,
        entries: Iterable[tuple[StateKey, Mapping[str, Any]]],
    ) -> None:
        """Stage several already immutable states with one store update.

        The method keeps the same validation rules as ``stage_immutable`` but
        moves the dictionary update out of the element/point hot loop.  It is
        intentionally additive; generic operators can continue using the
        scalar StateManager protocol.
        """

        staged = []
        for key, state in entries:
            normalized = _key(key)
            if normalized not in self._committed:
                raise KeyError(f"unknown state key: {normalized!r}")
            staged.append(
                (normalized, _freeze_immutable_mapping(state))
            )
        self._trial.update(staged)

    def begin_increment(self) -> None:
        self._trial.clear()

    def commit(self) -> None:
        self._committed.update(self._trial)
        self._trial.clear()

    def rollback(self) -> None:
        self._trial.clear()


def _key(value: StateKey) -> StateKey:
    if type(value) is not StateKey:
        raise TypeError("state key must be exactly StateKey")
    return value


class _FrozenState(Mapping[str, Any]):
    """Read-only state snapshot whose arrays cannot mutate committed history."""

    _fem_immutable_mapping = True

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _freeze_mapping(value: Mapping[str, Any]) -> _FrozenState:
    if not isinstance(value, Mapping):
        raise TypeError("state must be a mapping")
    copied = deepcopy(dict(value))
    for item in copied.values():
        if isinstance(item, np.ndarray):
            item.flags.writeable = False
    return _FrozenState(copied)


def _freeze_immutable_mapping(value: Mapping[str, Any]) -> _FrozenState:
    if not isinstance(value, Mapping):
        raise TypeError("state must be a mapping")
    values = dict(value)
    for item in values.values():
        if isinstance(item, np.ndarray):
            if item.flags.writeable:
                raise ValueError(
                    "stage_immutable requires all ndarray state values to be read-only"
                )
            continue
        if isinstance(item, (Mapping, list, set, bytearray)):
            raise TypeError(
                "stage_immutable accepts only scalar values and read-only ndarrays"
            )
    return _FrozenState(values)


__all__ = ["TransactionalStateManager"]
