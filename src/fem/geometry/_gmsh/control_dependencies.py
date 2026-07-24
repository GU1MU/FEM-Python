"""Pure topology-dependency guards for committed native mesh controls."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import GeometryStateError


_EntityKey = tuple[int, int]


class _ControlDependencyLedger:
    """Track topology promises made by successful entity-dependent controls."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._entity_dependencies: set[_EntityKey] = set()
        self._transform_unsafe_dependencies: set[_EntityKey] = set()
        self._scope_unknown = False

    @property
    def has_dependencies(self) -> bool:
        return bool(self._entity_dependencies)

    @property
    def has_transform_unsafe_dependencies(self) -> bool:
        return bool(self._transform_unsafe_dependencies)

    @property
    def scope_unknown(self) -> bool:
        return self._scope_unknown

    def register(
        self,
        keys: Iterable[_EntityKey],
        *,
        transform_unsafe: bool,
    ) -> None:
        """Commit dependency keys and their OCC-transform safety class."""
        materialized = set(keys)
        self._entity_dependencies.update(materialized)
        if transform_unsafe:
            self._transform_unsafe_dependencies.update(materialized)

    def check_scope_known(self, operation: str) -> None:
        """Reject an operation that requires a complete dependency scope."""
        if self._scope_unknown:
            raise self._state_error(
                operation,
                "raw access or a failed controlled topology operation made "
                "mesh-control dependencies unknown",
            )

    def check_removal(
        self,
        operation: str,
        removed_keys: Iterable[_EntityKey],
    ) -> None:
        """Reject destructive replacement of committed dependency keys."""
        materialized = set(removed_keys)
        if not materialized:
            return
        self.check_scope_known(operation)
        conflicts = materialized & self._entity_dependencies
        if conflicts:
            dimension, tag = min(conflicts)
            raise self._state_error(
                operation,
                "destructive topology replacement would invalidate an "
                "entity-dependent mesh control on topology "
                f"({dimension}, {tag})",
            )

    def check_transform(
        self,
        operation: str,
        transformed_keys: Iterable[_EntityKey],
    ) -> None:
        """Reject an OCC transform that would discard committed controls."""
        self.check_scope_known(operation)
        if not self._transform_unsafe_dependencies:
            return
        conflicts = set(transformed_keys) & self._transform_unsafe_dependencies
        if conflicts:
            dimension, tag = min(conflicts)
            raise self._state_error(
                operation,
                "the OCC transform would discard an entity-dependent mesh "
                f"control on topology ({dimension}, {tag})",
            )

    def mark_unknown_after_raw_access(self) -> None:
        """Lose dependency scope only when raw access can affect a promise."""
        if self._entity_dependencies:
            self._scope_unknown = True

    def mark_unknown_after_unknown_mutation(self) -> None:
        """Lose dependency scope after a controlled mutation may have started."""
        self._scope_unknown = True

    def snapshot_unknown_scope(self) -> bool:
        """Capture the structured-extrusion rollback value."""
        return self._scope_unknown

    def restore_unknown_scope(self, snapshot: bool) -> None:
        """Restore a previously captured structured-extrusion rollback value."""
        self._scope_unknown = snapshot

    def clear(self) -> None:
        """Reset every dependency promise at model lifecycle reset or close."""
        self._entity_dependencies.clear()
        self._transform_unsafe_dependencies.clear()
        self._scope_unknown = False

    def _state_error(self, operation: str, detail: str) -> GeometryStateError:
        return GeometryStateError(
            f"geometry model {self._model_name!r}: {operation} failed because {detail}"
        )


__all__: list[str] = []
