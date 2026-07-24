"""Restricted geometry host capability for one Gmsh mesh runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, TypeVar

from fem.geometry.types import EntityRef, FeatureResult


_T = TypeVar("_T")


class _NativeModelBorrow(Protocol):
    """Structural geometry-issued authority retained by a generated lease."""

    def borrow(self) -> Any: ...


class _MeshingPortOwner(Protocol):
    """Structural owner services used without importing the concrete model."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def _topology_provenance_unknown(self) -> bool: ...

    def _meshing_validate(self, authority: object, operation: str) -> None: ...

    def _meshing_normalize_entities(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]: ...

    def _meshing_normalize_optional_entities(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]: ...

    def _meshing_assert_entities_live(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> None: ...

    def _meshing_boundary_closure(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[tuple[int, int], ...]: ...

    def _meshing_assert_corners_on_boundary(
        self,
        authority: object,
        target: EntityRef,
        corners: tuple[EntityRef, ...],
        *,
        operation: str,
    ) -> None: ...

    def _meshing_native_query(
        self,
        authority: object,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T: ...

    def _meshing_native_control(
        self,
        authority: object,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T: ...

    def _meshing_commit_generation_attempt(
        self,
        authority: object,
        operation: str,
    ) -> None: ...

    def _meshing_register_control_dependencies(
        self,
        authority: object,
        keys: Iterable[tuple[int, int]],
        *,
        transform_unsafe: bool,
    ) -> None: ...

    def _meshing_has_pending_numeric_options(self, authority: object) -> bool: ...

    def _meshing_apply_numeric_options(
        self,
        authority: object,
        replacements: Iterable[tuple[str, float]],
    ) -> None: ...

    def _meshing_restore_numeric_options(self, authority: object) -> None: ...

    def _meshing_fail_generation(
        self,
        authority: object,
        operation: str,
    ) -> None: ...

    def _structured_extrude(
        self,
        authority: object,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int],
        heights: Sequence[float],
        recombine: bool,
    ) -> FeatureResult: ...

    def _meshing_prepare_native_borrow(
        self,
        authority: object,
        operation: str,
    ) -> _NativeModelBorrow: ...

    def _meshing_complete_generation(
        self,
        authority: object,
        operation: str,
        native_borrow: _NativeModelBorrow,
    ) -> None: ...


class _BoundMeshingPort:
    """Identity-bearing allowlist of geometry services used by meshing."""

    __slots__ = (
        "__dimension",
        "__model_name",
        "__owner",
        "__prepared_native_borrow",
        "__topology_provenance_unknown",
    )

    def __init__(self, owner: _MeshingPortOwner) -> None:
        self.__owner = owner
        self.__model_name = owner.name
        self.__dimension = owner.dimension
        self.__topology_provenance_unknown = owner._topology_provenance_unknown
        self.__prepared_native_borrow: _NativeModelBorrow | None = None

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def dimension(self) -> int:
        return self.__dimension

    @property
    def topology_provenance_unknown(self) -> bool:
        return self.__topology_provenance_unknown

    def validate(self, operation: str) -> None:
        self.__owner._meshing_validate(self, operation)

    def normalize_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]:
        return self.__owner._meshing_normalize_entities(
            self,
            entities,
            operation=operation,
        )

    def normalize_optional_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]:
        return self.__owner._meshing_normalize_optional_entities(
            self,
            entities,
            operation=operation,
            label=label,
        )

    def assert_entities_live(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> None:
        self.__owner._meshing_assert_entities_live(
            self,
            entities,
            operation=operation,
        )

    def boundary_closure(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[tuple[int, int], ...]:
        return self.__owner._meshing_boundary_closure(
            self,
            entities,
            operation=operation,
        )

    def assert_corners_on_boundary(
        self,
        target: EntityRef,
        corners: tuple[EntityRef, ...],
        *,
        operation: str,
    ) -> None:
        self.__owner._meshing_assert_corners_on_boundary(
            self,
            target,
            corners,
            operation=operation,
        )

    def native_query(
        self,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T:
        result = self.__owner._meshing_native_query(self, operation, callback)
        _assert_backend_free(result, operation)
        return result

    def native_control(
        self,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T:
        result = self.__owner._meshing_native_control(self, operation, callback)
        _assert_backend_free(result, operation)
        return result

    def commit_generation_attempt(self, operation: str) -> None:
        self.__owner._meshing_commit_generation_attempt(self, operation)

    def register_control_dependencies(
        self,
        keys: Iterable[tuple[int, int]],
        *,
        transform_unsafe: bool,
    ) -> None:
        self.__owner._meshing_register_control_dependencies(
            self,
            keys,
            transform_unsafe=transform_unsafe,
        )

    @property
    def has_pending_numeric_options(self) -> bool:
        return self.__owner._meshing_has_pending_numeric_options(self)

    def apply_numeric_options(
        self,
        replacements: Iterable[tuple[str, float]],
    ) -> None:
        self.__owner._meshing_apply_numeric_options(self, replacements)

    def restore_numeric_options(self) -> None:
        self.__owner._meshing_restore_numeric_options(self)

    def fail_generation(self, operation: str) -> None:
        self.__owner._meshing_fail_generation(self, operation)

    def structured_extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int],
        heights: Sequence[float],
        recombine: bool,
    ) -> FeatureResult:
        return self.__owner._structured_extrude(
            self,
            entities,
            dx,
            dy,
            dz,
            num_elements=num_elements,
            heights=heights,
            recombine=recombine,
        )

    def prepare_native_borrow(self, operation: str) -> _NativeModelBorrow:
        native_borrow = self.__owner._meshing_prepare_native_borrow(
            self,
            operation,
        )
        self.__prepared_native_borrow = native_borrow
        return native_borrow

    def complete_generation(self, operation: str) -> None:
        native_borrow = self.__prepared_native_borrow
        if native_borrow is None:
            raise RuntimeError(
                "generation completion requires a prepared native model borrow"
            )
        self.__owner._meshing_complete_generation(
            self,
            operation,
            native_borrow,
        )
        self.__prepared_native_borrow = None


def _assert_backend_free(value: Any, operation: str) -> None:
    """Reject a callback result that could retain a native backend object."""
    if value is None or type(value) in {bool, float, int, str}:
        return
    if type(value) is tuple:
        for item in value:
            _assert_backend_free(item, operation)
        return
    raise TypeError(
        f"{operation} native callback must return backend-free scalar or tuple data"
    )


__all__: list[str] = []
