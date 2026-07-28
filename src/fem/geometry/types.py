"""Public value objects for scripted geometry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from ._validation import _validate_entity_dimension, _validate_positive_tag
from .errors import EntityOwnershipError


SweepFrame = Literal[
    "discrete",
    "corrected_frenet",
    "frenet",
    "fixed",
    "constant_normal",
    "darboux",
]
LoftContinuity = Literal["C0", "G1", "C1", "G2", "C2", "C3", "CN"]
LoftParametrization = Literal[
    "chord_length",
    "centripetal",
    "iso_parametric",
]


@dataclass(frozen=True, slots=True)
class EntityRef:
    """Immutable reference to one entity owned by a geometry model."""

    dimension: int
    tag: int
    _owner_token: object = field(repr=False)
    _entity_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_entity_dimension(self.dimension)
        _validate_positive_tag(self.tag, "entity tag")


@dataclass(frozen=True, slots=True)
class OrientedCurveRef:
    """One live curve with an explicit traversal orientation."""

    curve: EntityRef
    reversed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.curve, EntityRef) or self.curve.dimension != 1:
            raise ValueError("curve must be a dimension-one EntityRef")
        if not isinstance(self.reversed, bool):
            raise TypeError(f"reversed must be a boolean, got {self.reversed!r}")


@dataclass(frozen=True, slots=True)
class CurveLoopRef:
    """Owner-local identity for one closed, ordered OCC curve loop."""

    tag: int
    curves: tuple[OrientedCurveRef, ...]
    _owner_token: object = field(repr=False)
    _loop_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_positive_tag(self.tag, "curve loop tag")
        try:
            normalized_curves = tuple(self.curves)
        except TypeError as exc:
            raise TypeError(
                "curves must be an iterable of OrientedCurveRef values"
            ) from exc
        object.__setattr__(self, "curves", normalized_curves)
        if not self.curves:
            raise ValueError("curves must contain at least one oriented curve")
        if any(not isinstance(curve, OrientedCurveRef) for curve in self.curves):
            raise TypeError("curves must contain only OrientedCurveRef values")


@dataclass(frozen=True, slots=True)
class WireRef:
    """Owner-local identity for one ordered open or closed OCC wire."""

    tag: int
    curves: tuple[OrientedCurveRef, ...]
    closed: bool
    _owner_token: object = field(repr=False)
    _wire_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_positive_tag(self.tag, "wire tag")
        try:
            normalized_curves = tuple(self.curves)
        except TypeError as exc:
            raise TypeError(
                "curves must be an iterable of OrientedCurveRef values"
            ) from exc
        object.__setattr__(self, "curves", normalized_curves)
        if not self.curves:
            raise ValueError("curves must contain at least one oriented curve")
        if any(not isinstance(curve, OrientedCurveRef) for curve in self.curves):
            raise TypeError("curves must contain only OrientedCurveRef values")
        if not isinstance(self.closed, bool):
            raise TypeError(f"closed must be a boolean, got {self.closed!r}")


@dataclass(frozen=True, slots=True)
class BooleanResult:
    """Typed outputs and input-to-output mapping from an OCC boolean."""

    outputs: tuple[EntityRef, ...]
    input_map: tuple[tuple[EntityRef, ...], ...]

    def of_dimension(self, dimension: int) -> tuple[EntityRef, ...]:
        """Return boolean outputs having the requested entity dimension."""
        normalized = _validate_entity_dimension(dimension)
        return tuple(
            entity for entity in self.outputs if entity.dimension == normalized
        )


@dataclass(frozen=True, slots=True)
class SurfaceTessellation:
    """Detached display mesh retaining the OCC owner of each display cell."""

    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    face_entities: tuple[EntityRef, ...]
    edges: tuple[tuple[int, ...], ...]
    edge_entities: tuple[EntityRef, ...]
    point_entities: tuple[EntityRef | None, ...]

    def __post_init__(self) -> None:
        if len(self.faces) != len(self.face_entities):
            raise ValueError("face_entities must match tessellated faces")
        if len(self.edges) != len(self.edge_entities):
            raise ValueError("edge_entities must match tessellated edges")
        if len(self.points) != len(self.point_entities):
            raise ValueError("point_entities must match tessellated points")
        if any(entity.dimension != 2 for entity in self.face_entities):
            raise ValueError("face tessellation owners must be surfaces")
        if any(entity.dimension != 1 for entity in self.edge_entities):
            raise ValueError("edge tessellation owners must be curves")
        if any(
            entity is not None and entity.dimension != 0
            for entity in self.point_entities
        ):
            raise ValueError("point tessellation owners must be points")


@dataclass(frozen=True, slots=True)
class StrictBodyBooleanPreview:
    """Detached true OCC result tessellation with local logical identities."""

    target_body_id: str
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    face_logical_ids: tuple[str, ...]
    edge_logical_ids: tuple[str, ...]
    point_logical_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if len(self.faces) != len(self.face_logical_ids):
            raise ValueError("Boolean preview faces and IDs must align")
        if len(self.edges) != len(self.edge_logical_ids):
            raise ValueError("Boolean preview edges and IDs must align")
        if len(self.points) != len(self.point_logical_ids):
            raise ValueError("Boolean preview points and IDs must align")


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Typed topology produced by one geometry feature operation.

    The references describe the geometry model state at creation time; they are
    not persistent topological names. Destructive modifying features may retain
    stale source references in ``inputs`` as operation history.
    """

    operation: str
    inputs: tuple[EntityRef, ...]
    outputs: tuple[EntityRef, ...]
    primary: tuple[EntityRef, ...]
    ends: tuple[EntityRef, ...] = ()
    sides: tuple[EntityRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a nonempty string")

        for field_name in ("inputs", "outputs", "primary", "ends", "sides"):
            raw_entities = getattr(self, field_name)
            try:
                entities = tuple(raw_entities)
            except TypeError as exc:
                raise TypeError(
                    f"{field_name} must be an iterable of EntityRef values"
                ) from exc
            if any(not isinstance(entity, EntityRef) for entity in entities):
                raise TypeError(
                    f"{field_name} must contain only EntityRef values"
                )
            object.__setattr__(self, field_name, entities)

        if not self.inputs:
            raise ValueError("inputs must contain at least one EntityRef")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("inputs must be duplicate-free")
        if not self.outputs:
            raise ValueError("outputs must contain at least one EntityRef")
        if not self.primary:
            raise ValueError("primary must contain at least one EntityRef")

        all_entities = (
            *self.inputs,
            *self.outputs,
            *self.primary,
            *self.ends,
            *self.sides,
        )
        owner_token = self.inputs[0]._owner_token
        if any(entity._owner_token is not owner_token for entity in all_entities):
            raise EntityOwnershipError(
                "feature result references must belong to one geometry model"
            )

        input_dimensions = {entity.dimension for entity in self.inputs}
        if len(input_dimensions) != 1:
            raise ValueError("feature result inputs must have one common dimension")
        input_dimension = next(iter(input_dimensions))
        primary_dimensions = {entity.dimension for entity in self.primary}
        if len(primary_dimensions) != 1:
            raise ValueError("primary entities must have one common dimension")
        primary_dimension = next(iter(primary_dimensions))
        if primary_dimension < input_dimension:
            raise ValueError(
                "feature result primary dimension must not be below its inputs"
            )
        boundary_dimension = primary_dimension - 1

        input_set = set(self.inputs)
        unique_outputs = _unique_first_seen(self.outputs)
        echoed_inputs = tuple(
            entity for entity in unique_outputs if entity in input_set
        )
        generated_outputs = tuple(
            entity for entity in unique_outputs if entity not in input_set
        )

        if primary_dimension == input_dimension:
            if self.ends or self.sides:
                raise ValueError(
                    "same-dimensional modifying features cannot report ends or sides"
                )
            if any(entity.dimension > primary_dimension for entity in self.outputs):
                raise ValueError(
                    "feature result outputs exceed the primary dimension"
                )
            expected_primary = tuple(
                entity
                for entity in unique_outputs
                if entity.dimension == primary_dimension
            )
            if echoed_inputs or self.primary != expected_primary:
                raise ValueError(
                    "modifying-feature primary entities must be all unique "
                    "top-dimensional outputs"
                )
        else:
            if any(
                entity not in input_set
                and entity.dimension not in {boundary_dimension, primary_dimension}
                for entity in self.outputs
            ):
                raise ValueError(
                    "generated feature outputs must have the primary or boundary "
                    "dimension"
                )
            if any(entity.dimension != primary_dimension for entity in self.primary):
                raise ValueError(
                    "primary entities must have the generated topological dimension"
                )
            if any(
                entity.dimension != boundary_dimension
                for entity in (*self.ends, *self.sides)
            ):
                raise ValueError(
                    "end and side entities must match the primary boundary dimension"
                )

        partitions = (self.primary, self.ends, self.sides)
        if any(len(set(partition)) != len(partition) for partition in partitions):
            raise ValueError("feature result semantic fields must be duplicate-free")
        partition_sets = tuple(set(partition) for partition in partitions)
        if any(
            left & right
            for index, left in enumerate(partition_sets)
            for right in partition_sets[index + 1 :]
        ):
            raise ValueError("primary, ends, and sides must be disjoint")

        semantic_entities = set().union(*partition_sets)
        partitioned_outputs = (
            generated_outputs
            if primary_dimension > input_dimension
            else self.primary
        )
        if semantic_entities != set(partitioned_outputs):
            raise ValueError(
                "primary, ends, and sides must partition the generated outputs"
            )
        for field_name, partition, partition_set in zip(
            ("primary", "ends", "sides"),
            partitions,
            partition_sets,
            strict=True,
        ):
            expected_order = tuple(
                entity for entity in partitioned_outputs if entity in partition_set
            )
            if partition != expected_order:
                raise ValueError(
                    f"{field_name} must preserve first-seen output order"
                )

    def of_dimension(self, dimension: int) -> tuple[EntityRef, ...]:
        """Return outputs having the requested dimension, preserving repeats."""
        normalized = _validate_entity_dimension(dimension)
        return tuple(
            entity for entity in self.outputs if entity.dimension == normalized
        )


@dataclass(frozen=True, slots=True)
class LoftResult:
    """A common feature topology result with grouped loft-section history."""

    topology: FeatureResult
    sections: tuple[WireRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topology, FeatureResult):
            raise TypeError("topology must be a FeatureResult")
        if self.topology.operation != "loft":
            raise ValueError("loft topology must use operation 'loft'")
        try:
            normalized_sections = tuple(self.sections)
        except TypeError as exc:
            raise TypeError("sections must be an iterable of WireRef values") from exc
        object.__setattr__(self, "sections", normalized_sections)
        if len(self.sections) < 2:
            raise ValueError("sections must contain at least two WireRef values")
        if any(not isinstance(section, WireRef) for section in self.sections):
            raise TypeError("sections must contain only WireRef values")
        owner_token = self.topology.inputs[0]._owner_token
        if any(section._owner_token is not owner_token for section in self.sections):
            raise EntityOwnershipError(
                "loft sections and topology must belong to one geometry model"
            )
        flattened_inputs = tuple(
            oriented.curve
            for section in self.sections
            for oriented in section.curves
        )
        if self.topology.inputs != flattened_inputs:
            raise ValueError(
                "loft topology inputs must preserve grouped section-curve order"
            )

    @property
    def operation(self) -> str:
        return self.topology.operation

    @property
    def inputs(self) -> tuple[EntityRef, ...]:
        return self.topology.inputs

    @property
    def outputs(self) -> tuple[EntityRef, ...]:
        return self.topology.outputs

    @property
    def primary(self) -> tuple[EntityRef, ...]:
        return self.topology.primary

    @property
    def ends(self) -> tuple[EntityRef, ...]:
        return self.topology.ends

    @property
    def sides(self) -> tuple[EntityRef, ...]:
        return self.topology.sides

    def of_dimension(self, dimension: int) -> tuple[EntityRef, ...]:
        return self.topology.of_dimension(dimension)


def _unique_first_seen(entities: Iterable[EntityRef]) -> tuple[EntityRef, ...]:
    seen: set[EntityRef] = set()
    unique: list[EntityRef] = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique.append(entity)
    return tuple(unique)


__all__ = [
    "BooleanResult",
    "CurveLoopRef",
    "EntityRef",
    "FeatureResult",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "OrientedCurveRef",
    "StrictBodyBooleanPreview",
    "SurfaceTessellation",
    "SweepFrame",
    "WireRef",
]
