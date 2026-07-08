from __future__ import annotations

from .base import ElementKernel
from .hexahedron import Hex8Kernel
from .line import Beam2DKernel, Truss2DKernel
from .quadrilateral import Quad4PlaneKernel, Quad8PlaneKernel
from .tetrahedron import Tet4Kernel, Tet10Kernel
from .triangle import Tri3PlaneKernel, Tri6PlaneKernel


_KERNELS: dict[str, ElementKernel] = {}


def register_element_kernel(kernel: ElementKernel) -> None:
    """Register an element kernel for all declared type names."""
    for name in kernel.type_names:
        _KERNELS[name.lower()] = kernel


def get_element_kernel(element_type: str) -> ElementKernel:
    """Return the registered element kernel for an element type."""
    key = str(element_type).lower()
    if key in _KERNELS:
        return _KERNELS[key]

    for name, kernel in _KERNELS.items():
        if name in key:
            return kernel

    raise NotImplementedError(f"Unsupported element type: {element_type}")


register_element_kernel(Quad4PlaneKernel())
register_element_kernel(Quad8PlaneKernel())
register_element_kernel(Tri6PlaneKernel())
register_element_kernel(Tri3PlaneKernel())
register_element_kernel(Hex8Kernel())
register_element_kernel(Tet4Kernel())
register_element_kernel(Tet10Kernel())
register_element_kernel(Truss2DKernel())
register_element_kernel(Beam2DKernel())
