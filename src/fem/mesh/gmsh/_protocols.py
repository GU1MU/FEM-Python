"""Structural contracts at the geometry-to-meshing capability boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, TypeVar

from fem.geometry import EntityRef, FeatureResult


_T = TypeVar("_T")


class _NativeModelBorrow(Protocol):
    """Revocable geometry-issued authority for one native facade model."""

    def borrow(self) -> Any: ...


class _GeometryMeshingPort(Protocol):
    """Restricted host services consumed by one mesh-owned runtime."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def topology_provenance_unknown(self) -> bool: ...

    def validate(self, operation: str) -> None: ...

    def normalize_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]: ...

    def normalize_optional_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]: ...

    def assert_entities_live(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> None: ...

    def boundary_closure(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[tuple[int, int], ...]: ...

    def assert_corners_on_boundary(
        self,
        target: EntityRef,
        corners: tuple[EntityRef, ...],
        *,
        operation: str,
    ) -> None: ...

    def native_query(
        self,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T: ...

    def native_control(
        self,
        operation: str,
        callback: Callable[[Any], _T],
    ) -> _T: ...

    def commit_generation_attempt(self, operation: str) -> None: ...

    def register_control_dependencies(
        self,
        keys: Iterable[tuple[int, int]],
        *,
        transform_unsafe: bool,
    ) -> None: ...

    @property
    def has_pending_numeric_options(self) -> bool: ...

    def apply_numeric_options(
        self,
        replacements: Iterable[tuple[str, float]],
    ) -> None: ...

    def restore_numeric_options(self) -> None: ...

    def fail_generation(self, operation: str) -> None: ...

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
    ) -> FeatureResult: ...

    def prepare_native_borrow(self, operation: str) -> _NativeModelBorrow: ...

    def complete_generation(self, operation: str) -> None: ...
