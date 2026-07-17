from __future__ import annotations

from typing import Any, Iterable

from ..core.model import ElementSet
from ..elements import get_element_kernel


def all(mesh: Any) -> list[int]:
    """Return all element ids."""
    return [elem.id for elem in mesh.elements]


def by_type(mesh: Any, element_type: str) -> list[int]:
    """Return element ids sharing the requested registered kernel type."""
    requested = _kernel_type_identity(element_type)
    if requested is None:
        return []
    return [
        elem.id
        for elem in mesh.elements
        if _kernel_type_identity(elem.type) == requested
    ]


def _kernel_type_identity(element_type: Any) -> str | None:
    """Return a canonical registered type name without substring matching."""
    try:
        kernel = get_element_kernel(str(element_type))
    except NotImplementedError:
        return None
    return str(kernel.type_names[0]).casefold()


def by_ids(mesh: Any, element_ids: Iterable[int]) -> list[int]:
    """Return existing element ids from a requested id collection."""
    requested = {int(element_id) for element_id in element_ids}
    return [elem.id for elem in mesh.elements if elem.id in requested]


def set_all(mesh: Any, name: str) -> ElementSet:
    """Return a named element set containing all elements."""
    return ElementSet(name, all(mesh))


def set_by_type(mesh: Any, name: str, element_type: str) -> ElementSet:
    """Return a named element set selected by element type."""
    return ElementSet(name, by_type(mesh, element_type))


def set_by_ids(mesh: Any, name: str, element_ids: Iterable[int]) -> ElementSet:
    """Return a named element set selected by ids."""
    return ElementSet(name, by_ids(mesh, element_ids))
