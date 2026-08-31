"""Local-equation contracts owned by the physics layer.

Physics operators create local residuals, tangents, and output snapshots.
Assembly only scatters those values and element definitions only provide
interpolation data, so neither neighboring subsystem owns this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from fem.state import EvaluationContext, LocalFieldState, StateManager


@dataclass(frozen=True, slots=True)
class LocalPointOutput:
    """Explicit output snapshot at a point of any bound local entity."""

    entity_kind: str
    entity_id: int
    point_id: int
    local_coordinates: tuple[float, ...]
    weight: float
    fields: Mapping[str, Any] = field(default_factory=dict)
    history: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity_kind = str(self.entity_kind).strip()
        entity_id = int(self.entity_id)
        point_id = int(self.point_id)
        if not entity_kind:
            raise ValueError("local output entity_kind must be nonblank")
        if entity_id < 1 or point_id < 1:
            raise ValueError("local output identities must be positive")
        coordinates = tuple(float(value) for value in self.local_coordinates)
        if not coordinates or not np.all(np.isfinite(coordinates)):
            raise ValueError("natural_coordinates must be a non-empty finite tuple")
        weight = float(self.weight)
        if not np.isfinite(weight):
            raise ValueError("integration-point weight must be finite")
        if not isinstance(self.fields, Mapping) or not isinstance(self.history, Mapping):
            raise TypeError("integration-point fields and history must be mappings")
        object.__setattr__(self, "entity_kind", entity_kind)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "point_id", point_id)
        object.__setattr__(self, "local_coordinates", coordinates)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "fields", deepcopy(dict(self.fields)))
        object.__setattr__(self, "history", deepcopy(dict(self.history)))

    def as_record(self) -> dict[str, Any]:
        """Return a flat record, including mesh aliases for element points."""

        record = dict(self.fields)
        record.update(
            {
                "entity_kind": self.entity_kind,
                "entity_id": self.entity_id,
                "point_id": self.point_id,
                "local_coordinates": self.local_coordinates,
                "weight": self.weight,
                "history": deepcopy(dict(self.history)),
            }
        )
        if self.entity_kind == "element":
            record.update(
                {
                    "element_id": self.entity_id,
                    "integration_point": self.point_id,
                    "natural_coordinates": self.local_coordinates,
                }
            )
        return record

    def __deepcopy__(self, memo: dict[int, Any]) -> "LocalPointOutput":
        """Deep-copy private immutable mappings without pickle support."""

        cached = memo.get(id(self))
        if cached is not None:
            return cached
        instance = object.__new__(type(self))
        memo[id(self)] = instance
        object.__setattr__(instance, "entity_kind", self.entity_kind)
        object.__setattr__(instance, "entity_id", self.entity_id)
        object.__setattr__(instance, "point_id", self.point_id)
        object.__setattr__(
            instance,
            "local_coordinates",
            tuple(self.local_coordinates),
        )
        object.__setattr__(instance, "weight", self.weight)
        object.__setattr__(
            instance,
            "fields",
            _immutable_output_mapping(
                {
                    key: deepcopy(value, memo)
                    for key, value in self.fields.items()
                }
            ),
        )
        object.__setattr__(
            instance,
            "history",
            _immutable_output_mapping(
                {
                    key: deepcopy(value, memo)
                    for key, value in self.history.items()
                }
            ),
        )
        return instance

    @classmethod
    def _from_immutable(
        cls,
        *,
        entity_kind: str,
        entity_id: int,
        point_id: int,
        local_coordinates: tuple[float, ...],
        weight: float,
        fields: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> "LocalPointOutput":
        """Adopt a producer-owned immutable output without repeated copies.

        This is intentionally private.  Batched constitutive kernels call it
        only after freezing every ndarray they place in the mappings; the
        public constructor keeps its defensive deep-copy contract.
        """

        instance = object.__new__(cls)
        object.__setattr__(instance, "entity_kind", str(entity_kind).strip())
        object.__setattr__(instance, "entity_id", int(entity_id))
        object.__setattr__(instance, "point_id", int(point_id))
        object.__setattr__(
            instance,
            "local_coordinates",
            tuple(float(value) for value in local_coordinates),
        )
        object.__setattr__(instance, "weight", float(weight))
        object.__setattr__(
            instance,
            "fields",
            _immutable_output_mapping(fields),
        )
        object.__setattr__(
            instance,
            "history",
            _immutable_output_mapping(history),
        )
        return instance

    @classmethod
    def element_point(
        cls,
        element_id: int,
        integration_point: int,
        natural_coordinates: tuple[float, ...],
        weight: float,
        *,
        fields: Mapping[str, Any] | None = None,
        history: Mapping[str, Any] | None = None,
    ) -> "LocalPointOutput":
        return cls(
            "element",
            element_id,
            integration_point,
            natural_coordinates,
            weight,
            fields or {},
            history or {},
        )


@dataclass(frozen=True, slots=True)
class LocalOutputBatch:
    """Point outputs published by one local entity evaluation."""

    entity_kind: str
    entity_id: int
    points: tuple[LocalPointOutput, ...]

    def __post_init__(self) -> None:
        entity_kind = str(self.entity_kind).strip()
        entity_id = int(self.entity_id)
        points = tuple(self.points)
        if not entity_kind or entity_id < 1:
            raise ValueError("local output entity identity is invalid")
        if any(type(point) is not LocalPointOutput for point in points):
            raise TypeError("points must contain only LocalPointOutput values")
        if any(
            point.entity_kind != entity_kind or point.entity_id != entity_id
            for point in points
        ):
            raise ValueError("local output points must match the batch entity")
        point_ids = tuple(point.point_id for point in points)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("element output points must use unique integration points")
        object.__setattr__(self, "entity_kind", entity_kind)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "points", points)

    def as_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(point.as_record() for point in self.points)

    def __deepcopy__(self, memo: dict[int, Any]) -> "LocalOutputBatch":
        """Deep-copy batches containing private immutable point outputs."""

        cached = memo.get(id(self))
        if cached is not None:
            return cached
        instance = object.__new__(type(self))
        memo[id(self)] = instance
        object.__setattr__(instance, "entity_kind", self.entity_kind)
        object.__setattr__(instance, "entity_id", self.entity_id)
        object.__setattr__(
            instance,
            "points",
            tuple(deepcopy(point, memo) for point in self.points),
        )
        return instance

    @classmethod
    def _from_immutable(
        cls,
        entity_id: int,
        points: tuple[LocalPointOutput, ...],
    ) -> "LocalOutputBatch":
        """Adopt already validated immutable point outputs."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "entity_kind", "element")
        object.__setattr__(instance, "entity_id", int(entity_id))
        object.__setattr__(instance, "points", tuple(points))
        return instance

    @classmethod
    def element(
        cls,
        element_id: int,
        points: tuple[LocalPointOutput, ...],
    ) -> "LocalOutputBatch":
        return cls("element", element_id, points)


def _immutable_output_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a private batch-output mapping while retaining its arrays."""

    if not isinstance(value, Mapping):
        raise TypeError("output values must be mappings")
    copied = dict(value)
    for item in copied.values():
        if isinstance(item, np.ndarray):
            item.flags.writeable = False
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class LocalContribution:
    """One element's contribution before global scatter."""

    dofs: tuple[int, ...]
    residual: np.ndarray
    tangent: np.ndarray
    mass: np.ndarray | None = None
    damping: np.ndarray | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    output_batch: LocalOutputBatch | None = None

    def __post_init__(self) -> None:
        dofs = tuple(int(value) for value in self.dofs)
        residual = np.asarray(self.residual, dtype=float)
        tangent = np.asarray(self.tangent, dtype=float)
        if residual.shape != (len(dofs),):
            raise ValueError("element residual shape must match dofs")
        if tangent.shape != (len(dofs), len(dofs)):
            raise ValueError("element tangent shape must match dofs")
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent)):
            raise ValueError("element contribution must be finite")
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "residual", np.array(residual, copy=True))
        object.__setattr__(self, "tangent", np.array(tangent, copy=True))
        for name in ("mass", "damping"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != tangent.shape:
                raise ValueError(f"element {name} shape must match tangent")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"element {name} must be finite")
            object.__setattr__(self, name, np.array(matrix, copy=True))
        object.__setattr__(self, "outputs", dict(self.outputs))
        if self.output_batch is not None and type(self.output_batch) is not LocalOutputBatch:
            raise TypeError("output_batch must be LocalOutputBatch or None")


@dataclass(frozen=True, slots=True)
class LocalContributionBatch:
    """Block-local contributions for an optional vectorized assembly path.

    Numerical arrays are always block-shaped.  Output batches are optional and
    are used only when the caller explicitly requests the accepted-result
    output payload; ordinary Newton trials therefore do not allocate output
    objects.  The arrays are owned by the producer and made read-only in place;
    this avoids one defensive copy per element while keeping the assembly
    boundary immutable to its consumers.
    """

    binding_indices: np.ndarray
    dofs: np.ndarray
    residual: np.ndarray
    tangent: np.ndarray
    mass: np.ndarray | None = None
    damping: np.ndarray | None = None
    output_batches: tuple[LocalOutputBatch, ...] = ()

    def __post_init__(self) -> None:
        binding_indices = np.asarray(self.binding_indices, dtype=int)
        dofs = np.asarray(self.dofs, dtype=int)
        residual = np.asarray(self.residual, dtype=float)
        tangent = np.asarray(self.tangent, dtype=float)
        if binding_indices.ndim != 1:
            raise ValueError("batch binding_indices must be one-dimensional")
        batch_size = int(binding_indices.size)
        if dofs.ndim != 2 or dofs.shape[0] != batch_size:
            raise ValueError("batch dofs must have shape (batch, local_size)")
        if residual.shape != dofs.shape:
            raise ValueError("batch residual must match batch dofs")
        local_size = int(dofs.shape[1])
        if tangent.shape != (batch_size, local_size, local_size):
            raise ValueError(
                "batch tangent must have shape "
                "(batch, local_size, local_size)"
            )
        if (
            not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(tangent))
        ):
            raise ValueError("batch contribution must be finite")
        object.__setattr__(self, "binding_indices", binding_indices)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "tangent", tangent)
        for name in ("binding_indices", "dofs", "residual", "tangent"):
            getattr(self, name).flags.writeable = False
        for name in ("mass", "damping"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != tangent.shape:
                raise ValueError(f"batch {name} must match batch tangent")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"batch {name} must be finite")
            matrix.flags.writeable = False
            object.__setattr__(self, name, matrix)
        output_batches = tuple(self.output_batches)
        if any(type(batch) is not LocalOutputBatch for batch in output_batches):
            raise TypeError(
                "output_batches must contain LocalOutputBatch values"
            )
        object.__setattr__(self, "output_batches", output_batches)


@runtime_checkable
class PhysicsOperator(Protocol):
    """Evaluate one bound element for any supported physical field set."""

    def initialize(
        self,
        entity: Any,
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        state_namespace: str,
    ) -> None:
        """Register local history before the first evaluation."""

    def evaluate(
        self,
        entity: Any,
        reference_coordinates: np.ndarray,
        fields: Mapping[str, LocalFieldState],
        dofs: tuple[int, ...],
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        context: EvaluationContext,
        state_namespace: str,
    ) -> LocalContribution:
        """Return local residual, tangent, optional mass/damping, and outputs."""


__all__ = [
    "LocalContribution",
    "LocalContributionBatch",
    "PhysicsOperator",
    "LocalOutputBatch",
    "LocalPointOutput",
]
