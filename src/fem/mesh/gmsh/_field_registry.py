"""Backend-free identity and liveness registry for native mesh fields."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

from fem.geometry import GeometryError

from ._validation import _validate_field_type, _validate_positive_tag
from .errors import MeshFieldOwnershipError, StaleMeshFieldError
from .types import MeshFieldRef


_FieldType = Literal["Distance", "Threshold", "Min"]


class _MeshFieldRegistry:
    """Own field authority independently of geometry entity identity."""

    __slots__ = ("_field_tokens", "_field_types", "_model_name", "_owner_token")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._owner_token = object()
        self._field_tokens: dict[int, object] = {}
        self._field_types: dict[int, _FieldType] = {}

    @property
    def owner_token(self) -> object:
        return self._owner_token

    def normalize(
        self,
        fields: Iterable[MeshFieldRef],
        *,
        operation: str,
    ) -> tuple[MeshFieldRef, ...]:
        try:
            normalized = tuple(fields)
        except TypeError as exc:
            raise TypeError(f"{operation} fields must be iterable") from exc
        seen_tags: set[int] = set()
        for mesh_field in normalized:
            if not isinstance(mesh_field, MeshFieldRef):
                raise TypeError(
                    f"{operation} requires MeshFieldRef values, got "
                    f"{mesh_field!r}"
                )
            if mesh_field._owner_token is not self._owner_token:
                raise MeshFieldOwnershipError(
                    f"geometry model {self._model_name!r}: {operation} received a "
                    "mesh field owned by another geometry model"
                )
            current_token = self._field_tokens.get(mesh_field.tag)
            registered_type = self._field_types.get(mesh_field.tag)
            if (
                current_token is not mesh_field._field_token
                or registered_type != mesh_field.field_type
            ):
                raise StaleMeshFieldError(
                    f"geometry model {self._model_name!r}: {operation} received stale "
                    f"mesh field {mesh_field.tag}"
                )
            if mesh_field.tag in seen_tags:
                raise ValueError(f"{operation} field inputs must be duplicate-free")
            seen_tags.add(mesh_field.tag)
        return normalized

    def assert_liveness(
        self,
        fields: Iterable[MeshFieldRef],
        active_tags: frozenset[int],
        *,
        operation: str,
    ) -> None:
        for mesh_field in fields:
            token = self._field_tokens.get(mesh_field.tag)
            registered_type = self._field_types.get(mesh_field.tag)
            if (
                token is not mesh_field._field_token
                or registered_type != mesh_field.field_type
                or mesh_field.tag not in active_tags
            ):
                self._field_tokens.pop(mesh_field.tag, None)
                self._field_types.pop(mesh_field.tag, None)
                raise StaleMeshFieldError(
                    f"geometry model {self._model_name!r}: {operation} mesh field "
                    f"{mesh_field.tag} no longer exists"
                )

    def construct(
        self,
        field_type: _FieldType,
        allocate_and_configure: Callable[[Callable[[Any], int]], None],
        rollback: Callable[[int], None],
    ) -> MeshFieldRef:
        """Allocate, publish, and roll back one field without backend knowledge."""
        normalized_type = _validate_field_type(field_type)
        allocated_tag: int | None = None

        def record_allocated(raw_tag: Any) -> int:
            nonlocal allocated_tag
            allocated_tag = _validate_positive_tag(raw_tag, "mesh field tag")
            return allocated_tag

        try:
            allocate_and_configure(record_allocated)
            if allocated_tag is None:
                raise GeometryError(
                    f"geometry model {self._model_name!r}: native mesh-field "
                    "allocation did not report a field tag"
                )
            token = object()
            reference = MeshFieldRef(
                allocated_tag,
                normalized_type,
                self._owner_token,
                token,
            )
            self._field_tokens[allocated_tag] = token
            self._field_types[allocated_tag] = normalized_type
        except BaseException as error:
            if allocated_tag is not None:
                self._field_tokens.pop(allocated_tag, None)
                self._field_types.pop(allocated_tag, None)
                try:
                    rollback(allocated_tag)
                except BaseException as rollback_error:
                    error.add_note(
                        f"geometry model {self._model_name!r}: mesh-field rollback "
                        f"also failed while removing field {allocated_tag}: "
                        f"{rollback_error}"
                    )
            raise
        return reference


def _normalize_active_tags(raw_tags: Any, model_name: str) -> frozenset[int]:
    try:
        return frozenset(
            _validate_positive_tag(tag, "mesh field tag") for tag in raw_tags
        )
    except (TypeError, ValueError) as exc:
        raise GeometryError(
            f"geometry model {model_name!r}: Gmsh returned an invalid mesh "
            "field list"
        ) from exc
