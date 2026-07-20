"""Owner-local typed-reference identity for the Gmsh geometry facade."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .._validation import _validate_entity_dimension, _validate_positive_tag
from ..errors import EntityOwnershipError, GeometryError, StaleEntityError
from ..types import CurveLoopRef, EntityRef, OrientedCurveRef, WireRef


_EntityKey = tuple[int, int]
_DependencyResolver = Callable[[CurveLoopRef | WireRef], Iterable[_EntityKey]]
_MemberValidator = Callable[[Iterable[EntityRef], str], None]


class _EntityRegistry:
    """Own stable identities for live ``(dimension, tag)`` entity keys."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._owner_token = object()
        self._tokens: dict[_EntityKey, object] = {}

    @property
    def owner_token(self) -> object:
        """Return the opaque owner identity shared by this model's references."""
        return self._owner_token

    def wrap(self, pair: Any) -> EntityRef:
        """Return a stable typed identity for one normalized native entity key."""
        dimension, tag = _normalize_entity_key(pair)
        key = (dimension, tag)
        token = self._tokens.get(key)
        if token is None:
            token = object()
            self._tokens[key] = token
        return EntityRef(dimension, tag, self._owner_token, token)

    def normalize(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]:
        """Validate iterable, type, ownership, liveness, and uniqueness."""
        try:
            normalized = tuple(entities)
        except TypeError as exc:
            raise TypeError(f"{operation} entities must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one entity")

        seen: set[EntityRef] = set()
        for entity in normalized:
            if not isinstance(entity, EntityRef):
                raise TypeError(
                    f"{operation} requires EntityRef values, got {entity!r}"
                )
            if entity._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self._model_name!r}: {operation} received an "
                    "entity owned by another geometry model"
                )
            current_token = self._tokens.get((entity.dimension, entity.tag))
            if current_token is not entity._entity_token:
                raise StaleEntityError(
                    f"geometry model {self._model_name!r}: {operation} received "
                    f"stale entity ({entity.dimension}, {entity.tag})"
                )
            if entity in seen:
                raise ValueError(f"{operation} entity inputs must be duplicate-free")
            seen.add(entity)
        return normalized

    def normalize_optional(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]:
        """Validate an entity iterable while permitting an empty result."""
        try:
            materialized = tuple(entities)
        except TypeError as exc:
            raise TypeError(f"{operation} {label} must be iterable") from exc
        if not materialized:
            return ()
        return self.normalize(materialized, operation=f"{operation} {label}")

    def is_current(self, entity: EntityRef) -> bool:
        """Return whether an entity carries this registry's current key token."""
        return (
            isinstance(entity, EntityRef)
            and entity._owner_token is self._owner_token
            and self._tokens.get((entity.dimension, entity.tag))
            is entity._entity_token
        )

    def invalidate(self, keys: Iterable[_EntityKey]) -> None:
        """Invalidate only the supplied entity keys."""
        for key in set(keys):
            self._tokens.pop(key, None)

    def clear(self) -> None:
        """Invalidate every entity identity while retaining the owner identity."""
        self._tokens.clear()


class _TopologyReferenceRegistry:
    """Own separate curve-loop and wire identity namespaces."""

    def __init__(self, model_name: str, owner_token: object) -> None:
        self._model_name = model_name
        self._owner_token = owner_token
        self._curve_loop_tokens: dict[int, object] = {}
        self._curve_loop_dependencies: dict[int, frozenset[_EntityKey]] = {}
        self._wire_tokens: dict[int, object] = {}
        self._wire_dependencies: dict[int, frozenset[_EntityKey]] = {}

    def register_curve_loop(
        self,
        raw_tag: Any,
        curves: Iterable[OrientedCurveRef],
        dependencies: Iterable[_EntityKey],
    ) -> CurveLoopRef:
        """Register a native loop identity, failing closed within loop scope."""
        try:
            tag = _validate_positive_tag(raw_tag, "curve loop tag")
        except ValueError as error:
            self.clear_curve_loops()
            raise GeometryError(
                f"geometry model {self._model_name!r}: curve_loop returned an "
                "invalid loop tag; typed loop identities were invalidated"
            ) from error
        if tag in self._curve_loop_tokens:
            self.clear_curve_loops()
            raise GeometryError(
                f"geometry model {self._model_name!r}: curve_loop returned "
                f"duplicate loop tag {tag}; typed loop identities were invalidated"
            )

        token = object()
        reference = CurveLoopRef(
            tag,
            tuple(curves),
            self._owner_token,
            token,
        )
        self._curve_loop_tokens[tag] = token
        self._curve_loop_dependencies[tag] = frozenset(dependencies)
        return reference

    def register_wire(
        self,
        raw_tag: Any,
        curves: Iterable[OrientedCurveRef],
        closed: bool,
        dependencies: Iterable[_EntityKey],
    ) -> WireRef:
        """Register a native wire identity, failing closed within wire scope."""
        try:
            tag = _validate_positive_tag(raw_tag, "wire tag")
        except ValueError as error:
            self.clear_wires()
            raise GeometryError(
                f"geometry model {self._model_name!r}: wire returned an invalid "
                "wire tag; typed wire identities were invalidated"
            ) from error
        if tag in self._wire_tokens:
            self.clear_wires()
            raise GeometryError(
                f"geometry model {self._model_name!r}: wire returned duplicate "
                f"wire tag {tag}; typed wire identities were invalidated"
            )

        token = object()
        reference = WireRef(
            tag,
            tuple(curves),
            closed,
            self._owner_token,
            token,
        )
        self._wire_tokens[tag] = token
        self._wire_dependencies[tag] = frozenset(dependencies)
        return reference

    def normalize_curve_loops(
        self,
        loops: Iterable[CurveLoopRef],
        *,
        operation: str,
        dependency_resolver: _DependencyResolver,
        member_validator: _MemberValidator,
    ) -> tuple[CurveLoopRef, ...]:
        """Validate loop identities and their current recursive dependencies."""
        try:
            normalized = tuple(loops)
        except TypeError as exc:
            raise TypeError(f"{operation} curve loops must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one curve loop")

        seen_tags: set[int] = set()
        member_keys: set[_EntityKey] = set()
        for loop in normalized:
            if not isinstance(loop, CurveLoopRef):
                raise TypeError(
                    f"{operation} requires CurveLoopRef values, got {loop!r}"
                )
            if loop._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self._model_name!r}: {operation} received a "
                    "curve loop owned by another geometry model"
                )
            if self._curve_loop_tokens.get(loop.tag) is not loop._loop_token:
                raise StaleEntityError(
                    f"geometry model {self._model_name!r}: {operation} received "
                    f"stale curve loop {loop.tag}"
                )
            if loop.tag in seen_tags:
                raise ValueError(f"{operation} curve loops must be duplicate-free")
            seen_tags.add(loop.tag)

            dependency_keys = self._curve_loop_dependencies.get(loop.tag)
            expected_keys = frozenset(dependency_resolver(loop))
            if dependency_keys != expected_keys:
                raise StaleEntityError(
                    f"geometry model {self._model_name!r}: {operation} received "
                    f"stale curve loop {loop.tag}"
                )
            overlap = member_keys & set(dependency_keys)
            if overlap:
                raise ValueError(
                    f"{operation} curve loops must not share member curves or "
                    "boundary points"
                )
            member_keys.update(dependency_keys)
            member_validator(
                tuple(item.curve for item in loop.curves),
                f"{operation} curve loop {loop.tag}",
            )
        return normalized

    def normalize_wires(
        self,
        wires: Iterable[WireRef],
        *,
        operation: str,
        dependency_resolver: _DependencyResolver,
        member_validator: _MemberValidator,
    ) -> tuple[WireRef, ...]:
        """Validate wire identities and their current recursive dependencies."""
        try:
            normalized = tuple(wires)
        except TypeError as exc:
            raise TypeError(f"{operation} wires must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one wire")

        seen_tags: set[int] = set()
        for wire in normalized:
            if not isinstance(wire, WireRef):
                raise TypeError(
                    f"{operation} requires WireRef values, got {wire!r}"
                )
            if wire._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self._model_name!r}: {operation} received a "
                    "wire owned by another geometry model"
                )
            if self._wire_tokens.get(wire.tag) is not wire._wire_token:
                raise StaleEntityError(
                    f"geometry model {self._model_name!r}: {operation} received "
                    f"stale wire {wire.tag}"
                )
            if wire.tag in seen_tags:
                raise ValueError(f"{operation} wires must be duplicate-free")
            seen_tags.add(wire.tag)

            member_curves = tuple(item.curve for item in wire.curves)
            dependency_keys = self._wire_dependencies.get(wire.tag)
            expected_keys = frozenset(dependency_resolver(wire))
            if dependency_keys != expected_keys:
                raise StaleEntityError(
                    f"geometry model {self._model_name!r}: {operation} received "
                    f"stale wire {wire.tag}"
                )
            member_validator(member_curves, f"{operation} wire {wire.tag}")
        return normalized

    def invalidate(self, keys: Iterable[_EntityKey]) -> None:
        """Invalidate loop and wire identities intersecting supplied keys."""
        materialized = set(keys)
        invalid_loops = [
            tag
            for tag, dependencies in self._curve_loop_dependencies.items()
            if dependencies & materialized
        ]
        for tag in invalid_loops:
            self._curve_loop_tokens.pop(tag, None)
            self._curve_loop_dependencies.pop(tag, None)

        invalid_wires = [
            tag
            for tag, dependencies in self._wire_dependencies.items()
            if dependencies & materialized
        ]
        for tag in invalid_wires:
            self._wire_tokens.pop(tag, None)
            self._wire_dependencies.pop(tag, None)

    def clear_curve_loops(self) -> None:
        """Invalidate every curve-loop identity without touching wires."""
        self._curve_loop_tokens.clear()
        self._curve_loop_dependencies.clear()

    def clear_wires(self) -> None:
        """Invalidate every wire identity without touching curve loops."""
        self._wire_tokens.clear()
        self._wire_dependencies.clear()

    def clear(self) -> None:
        """Invalidate every curve-loop and wire identity."""
        self.clear_curve_loops()
        self.clear_wires()


class _ReferenceRegistry:
    """Compose entity and topology identities under one model owner token."""

    def __init__(self, model_name: str) -> None:
        self._entities = _EntityRegistry(model_name)
        self._topology = _TopologyReferenceRegistry(
            model_name,
            self._entities.owner_token,
        )

    @property
    def owner_token(self) -> object:
        return self._entities.owner_token

    def wrap_entity(self, pair: Any) -> EntityRef:
        return self._entities.wrap(pair)

    def normalize_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]:
        return self._entities.normalize(entities, operation=operation)

    def normalize_optional_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]:
        return self._entities.normalize_optional(
            entities,
            operation=operation,
            label=label,
        )

    def is_current_entity(self, entity: EntityRef) -> bool:
        return self._entities.is_current(entity)

    def register_curve_loop(
        self,
        raw_tag: Any,
        curves: Iterable[OrientedCurveRef],
        dependencies: Iterable[_EntityKey],
    ) -> CurveLoopRef:
        return self._topology.register_curve_loop(raw_tag, curves, dependencies)

    def register_wire(
        self,
        raw_tag: Any,
        curves: Iterable[OrientedCurveRef],
        closed: bool,
        dependencies: Iterable[_EntityKey],
    ) -> WireRef:
        return self._topology.register_wire(
            raw_tag,
            curves,
            closed,
            dependencies,
        )

    def normalize_curve_loops(
        self,
        loops: Iterable[CurveLoopRef],
        *,
        operation: str,
        dependency_resolver: _DependencyResolver,
    ) -> tuple[CurveLoopRef, ...]:
        return self._topology.normalize_curve_loops(
            loops,
            operation=operation,
            dependency_resolver=dependency_resolver,
            member_validator=self._validate_members,
        )

    def normalize_wires(
        self,
        wires: Iterable[WireRef],
        *,
        operation: str,
        dependency_resolver: _DependencyResolver,
    ) -> tuple[WireRef, ...]:
        return self._topology.normalize_wires(
            wires,
            operation=operation,
            dependency_resolver=dependency_resolver,
            member_validator=self._validate_members,
        )

    def invalidate_entities(self, keys: Iterable[_EntityKey]) -> None:
        materialized = set(keys)
        self._entities.invalidate(materialized)
        self._topology.invalidate(materialized)

    def invalidate_topology(self, keys: Iterable[_EntityKey]) -> None:
        self._topology.invalidate(keys)

    def clear_curve_loops(self) -> None:
        self._topology.clear_curve_loops()

    def clear_wires(self) -> None:
        self._topology.clear_wires()

    def clear(self) -> None:
        self._entities.clear()
        self._topology.clear()

    def _validate_members(
        self,
        entities: Iterable[EntityRef],
        operation: str,
    ) -> None:
        self._entities.normalize(entities, operation=operation)


def _normalize_entity_key(value: Any) -> _EntityKey:
    try:
        dimension, tag = value
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"invalid Gmsh entity reference {value!r}") from exc
    return (
        _validate_entity_dimension(dimension),
        _validate_positive_tag(tag, "entity tag"),
    )


__all__: list[str] = []
