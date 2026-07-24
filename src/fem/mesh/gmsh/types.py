"""Immutable public references and private generated-mesh authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ._protocols import _NativeModelBorrow
from ._validation import (
    _validate_dimension,
    _validate_field_type,
    _validate_positive_tag,
)
from .errors import StaleGmshMeshError


@dataclass(frozen=True, slots=True)
class MeshFieldRef:
    """Immutable reference to one field owned by one mesh runtime."""

    tag: int
    field_type: Literal["Distance", "Threshold", "Min"]
    _owner_token: object = field(repr=False)
    _field_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_positive_tag(self.tag, "mesh field tag")
        _validate_field_type(self.field_type)


_LEASE_FACTORY_AUTHORITY = object()


def _new_bearer_token() -> object:
    """Allocate one unforgeable bearer identity during reference preparation."""
    return object()


class _GeneratedMeshLease:
    """Nominal bearer lease around one geometry-issued native capability."""

    __slots__ = (
        "__active",
        "__bearer_token",
        "__dimension",
        "__model_name",
        "__native_borrow",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("_GeneratedMeshLease is sealed")

    def __init__(
        self,
        native_borrow: _NativeModelBorrow,
        dimension: Literal[1, 2, 3],
        model_name: str,
        bearer_token: object,
        *,
        _factory_authority: object,
    ) -> None:
        if type(self) is not _GeneratedMeshLease:
            raise TypeError("_GeneratedMeshLease is sealed")
        if _factory_authority is not _LEASE_FACTORY_AUTHORITY:
            raise TypeError("generated mesh leases use a sealed factory")
        if not callable(getattr(native_borrow, "borrow", None)):
            raise TypeError("native model borrow capability is malformed")
        self.__native_borrow = native_borrow
        self.__dimension = dimension
        self.__model_name = model_name
        self.__bearer_token = bearer_token
        self.__active = False

    def _activate(self) -> None:
        """Activate a fully prepared lease through a no-fail assignment."""
        self.__active = True

    def _borrow(
        self,
        bearer_token: object,
        dimension: int,
        model_name: str,
    ) -> Any:
        if (
            type(self) is not _GeneratedMeshLease
            or not self.__active
            or bearer_token is not self.__bearer_token
            or dimension != self.__dimension
            or model_name != self.__model_name
        ):
            raise RuntimeError("generated mesh lease authority is invalid")
        return self.__native_borrow.borrow()


@dataclass(frozen=True, slots=True)
class GmshMeshRef:
    """Read-only reference to a generated mesh in a live Gmsh model context."""

    dimension: Literal[1, 2, 3]
    model_name: str
    _lease: object = field(repr=False)
    _bearer_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimension(self.dimension)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")

    def _borrow_model(self) -> Any:
        """Return the reactivated live Gmsh model for the FEM IO adapter."""
        stale_error = _stale_gmsh_mesh_error(self.model_name)
        lease = self._lease
        if type(lease) is not _GeneratedMeshLease:
            raise stale_error
        try:
            return lease._borrow(
                self._bearer_token,
                self.dimension,
                self.model_name,
            )
        except BaseException as error:
            raise stale_error from error


def _prepare_generated_mesh_reference(
    native_borrow: _NativeModelBorrow,
    *,
    dimension: Literal[1, 2, 3],
    model_name: str,
) -> tuple[GmshMeshRef, _GeneratedMeshLease]:
    """Prepare one dormant nominal lease and its sole bearer reference."""
    bearer_token = _new_bearer_token()
    lease = _GeneratedMeshLease(
        native_borrow,
        dimension,
        model_name,
        bearer_token,
        _factory_authority=_LEASE_FACTORY_AUTHORITY,
    )
    reference = GmshMeshRef(
        dimension,
        model_name,
        lease,
        bearer_token,
    )
    return reference, lease


def _stale_gmsh_mesh_error(model_name: str) -> StaleGmshMeshError:
    return StaleGmshMeshError(
        f"generated Gmsh mesh for model {model_name!r} is stale; import it "
        "with fem.io.gmsh.read() inside the owning geometry model context"
    )


__all__ = ["GmshMeshRef", "MeshFieldRef"]
