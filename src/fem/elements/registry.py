from __future__ import annotations

from .base import ElementKernel
from .hexahedron import Hex8Kernel, Hex20Kernel
from .line import Beam2Kernel, Truss2Kernel
from .quadrilateral import Quad4Kernel, Quad8Kernel
from .tetrahedron import Tet4Kernel, Tet10Kernel
from .triangle import Tri3Kernel, Tri6Kernel


_KERNELS: dict[str, ElementKernel] = {}
_UNSUPPORTED_REDUCED_INTEGRATION_TYPES = frozenset(
    {
        "c3d8r",
        "cps4r",
        "cpe4r",
        "cps8r",
        "cpe8r",
        "c3d20r",
    }
)
_UNSUPPORTED_COUPLED_ELEMENT_TYPES = frozenset({"c3d4t", "c3d10t"})


def register_element_kernel(kernel: ElementKernel) -> None:
    """Register a kernel's canonical type and aliases case-insensitively."""
    names = (kernel.canonical_type, *kernel.aliases)
    keys: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = str(name).casefold()
        if not key:
            raise ValueError("element kernel type names must be nonempty")
        if key in seen:
            raise ValueError(f"element type {name!r} is declared more than once")
        seen.add(key)
        existing = _KERNELS.get(key)
        if existing is not None and existing is not kernel:
            raise ValueError(
                f"element type {name!r} is already registered to "
                f"{existing.canonical_type}"
            )
        keys.append(key)
    for key in keys:
        _KERNELS[key] = kernel


def get_element_kernel(element_type: str) -> ElementKernel:
    """Return the registered element kernel for an element type."""
    key = str(element_type).casefold()
    if key in _UNSUPPORTED_REDUCED_INTEGRATION_TYPES:
        raise NotImplementedError(
            f"Unsupported element type: {element_type}; "
            "reduced integration is not implemented"
        )
    if key in _UNSUPPORTED_COUPLED_ELEMENT_TYPES:
        raise NotImplementedError(
            f"Unsupported element type: {element_type}; "
            "coupled temperature-displacement elements are not implemented"
        )
    if key in _KERNELS:
        return _KERNELS[key]

    raise NotImplementedError(f"Unsupported element type: {element_type}")


def canonical_element_type(element_type: str) -> str:
    """Return the canonical registered name for an element type or alias."""
    return str(get_element_kernel(element_type).canonical_type)


register_element_kernel(Quad4Kernel())
register_element_kernel(Quad8Kernel())
register_element_kernel(Tri6Kernel())
register_element_kernel(Tri3Kernel())
register_element_kernel(Hex8Kernel())
register_element_kernel(Hex20Kernel())
register_element_kernel(Tet4Kernel())
register_element_kernel(Tet10Kernel())
register_element_kernel(Truss2Kernel())
register_element_kernel(Beam2Kernel())
