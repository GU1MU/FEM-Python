"""Mesh-owned mutable workflow configuration and attempt state."""

from __future__ import annotations

from typing import Literal

from fem.geometry import GeometryStateError

from .errors import MeshControlConflictError
from .types import MeshFieldRef


_SizeMode = Literal["none", "point", "background"]


class _MeshConfiguration:
    """Own size precedence, automatic blockers, and the one-attempt rule."""

    __slots__ = (
        "_attempted",
        "_auto_blockers",
        "_background_field",
        "_size_mode",
    )

    def __init__(self) -> None:
        self._size_mode: _SizeMode = "none"
        self._background_field: MeshFieldRef | None = None
        self._auto_blockers: set[str] = set()
        self._attempted = False

    @property
    def size_mode(self) -> _SizeMode:
        return self._size_mode

    @property
    def background_field(self) -> MeshFieldRef | None:
        return self._background_field

    def assert_attempt_available(self, model_name: str, operation: str) -> None:
        if self._attempted:
            raise GeometryStateError(
                f"geometry model {model_name!r}: {operation} failed because "
                "the one mesh attempt was already used"
            )

    def consume_attempt(self) -> None:
        self._attempted = True

    def assert_mesh_size_allowed(self) -> None:
        if self._size_mode == "background":
            raise ValueError(
                "mesh_size cannot be combined with a selected background field"
            )

    def commit_mesh_size(self) -> None:
        self._size_mode = "point"

    def assert_background_allowed(self) -> None:
        if self._background_field is not None:
            raise ValueError("background_field may be selected only once")
        if self._size_mode == "point":
            raise ValueError(
                "background_field cannot be combined with typed point sizes"
            )

    def commit_background(self, field: MeshFieldRef) -> None:
        self._background_field = field
        self._size_mode = "background"

    def assert_uniform_size_allowed(self, size: float | None) -> None:
        if size is not None and self._size_mode != "none":
            raise ValueError(
                "size cannot be supplied after typed point sizes or a typed "
                "background field has been configured"
            )

    def add_auto_blocker(self, operation: str) -> None:
        self._auto_blockers.add(operation)

    def assert_automatic_allowed(
        self,
        model_name: str,
        *,
        topology_provenance_unknown: bool,
    ) -> None:
        if topology_provenance_unknown:
            raise MeshControlConflictError(
                f"geometry model {model_name!r}: AutoMeshSpec generation cannot own "
                "the mesh topology because raw access or a failed controlled "
                "topology operation made the automatic topology scope unknown; "
                "use Mesher.generate(MeshSpec(...)) for this model"
            )
        if self._auto_blockers:
            blockers = ", ".join(sorted(self._auto_blockers))
            raise MeshControlConflictError(
                f"geometry model {model_name!r}: AutoMeshSpec generation conflicts "
                f"with explicit topology controls: {blockers}; use "
                "Mesher.generate(MeshSpec(...)) for this model"
            )
