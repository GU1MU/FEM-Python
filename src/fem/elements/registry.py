from __future__ import annotations

from .base import ElementKernel
from .capabilities import (
    ElementCapabilityDescriptor,
    ElementCapabilityRequirement,
)
from .hexahedron import Hex8Kernel, Hex20Kernel
from .line import Beam2Kernel, Truss2Kernel
from .quadrilateral import Quad4Kernel, Quad8Kernel
from .tetrahedron import Tet4Kernel, Tet10Kernel
from .triangle import Tri3Kernel, Tri6Kernel


_KERNELS: dict[str, ElementKernel] = {}
_CAPABILITIES: dict[str, ElementCapabilityDescriptor] = {}
_CANONICAL_CAPABILITIES: dict[str, ElementCapabilityDescriptor] = {}
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


def register_element_kernel(
    kernel: ElementKernel,
    capabilities: ElementCapabilityDescriptor | None = None,
) -> None:
    """Atomically register a kernel and its complete capability descriptor."""
    names = _kernel_names(kernel)
    keys: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
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

    if capabilities is None:
        raise ValueError(
            f"element kernel {kernel.canonical_type!r} requires a "
            "capability descriptor"
        )
    if not isinstance(capabilities, ElementCapabilityDescriptor):
        raise TypeError(
            "capabilities must be an ElementCapabilityDescriptor"
        )
    if capabilities.canonical_type != kernel.canonical_type:
        raise ValueError(
            "capability canonical type must exactly match the kernel "
            "canonical type"
        )
    if capabilities.aliases != kernel.aliases:
        raise ValueError(
            "capability aliases must exactly match the kernel aliases"
        )

    for name, key in zip(names, keys, strict=True):
        existing = _CAPABILITIES.get(key)
        if existing is not None and existing is not capabilities:
            raise ValueError(
                f"element capability {name!r} is already registered to "
                f"{existing.canonical_type}"
            )

    canonical_key = capabilities.canonical_type.casefold()
    existing_canonical = _CANONICAL_CAPABILITIES.get(canonical_key)
    if existing_canonical is not None and existing_canonical is not capabilities:
        raise ValueError(
            f"canonical element capability {capabilities.canonical_type!r} "
            "is already registered"
        )

    for key in keys:
        _KERNELS[key] = kernel
        _CAPABILITIES[key] = capabilities
    _CANONICAL_CAPABILITIES[canonical_key] = capabilities


def get_element_kernel(element_type: str) -> ElementKernel:
    """Return the registered element kernel for an element type."""
    key = str(element_type).casefold()
    if key in _KERNELS:
        return _KERNELS[key]

    _raise_unsupported_element_type(element_type, key)


def get_element_capabilities(element_type: str) -> ElementCapabilityDescriptor:
    """Return immutable capabilities for a canonical type or alias."""

    key = str(element_type).casefold()
    if key in _CAPABILITIES:
        return _CAPABILITIES[key]

    _raise_unsupported_element_type(element_type, key)


def registered_element_capabilities() -> tuple[ElementCapabilityDescriptor, ...]:
    """Return one descriptor per registered kernel in registration order."""

    return tuple(_CANONICAL_CAPABILITIES.values())


def canonical_element_type(element_type: str) -> str:
    """Return the canonical registered name for an element type or alias."""
    return str(get_element_kernel(element_type).canonical_type)


def _kernel_names(kernel: ElementKernel) -> tuple[str, ...]:
    try:
        canonical_type = kernel.canonical_type
        aliases = kernel.aliases
    except AttributeError as exc:
        raise ValueError(
            "element kernels must declare canonical_type and aliases"
        ) from exc
    if not isinstance(canonical_type, str) or not canonical_type:
        raise ValueError("element kernel canonical type must be nonempty")
    if canonical_type != canonical_type.strip():
        raise ValueError("element kernel canonical type must be trimmed")
    if not isinstance(aliases, tuple):
        raise TypeError("element kernel aliases must be a tuple")
    if any(
        not isinstance(alias, str) or not alias or alias != alias.strip()
        for alias in aliases
    ):
        raise ValueError("element kernel aliases must be nonempty trimmed strings")
    return (canonical_type, *aliases)


def _raise_unsupported_element_type(element_type: object, key: str) -> None:
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
    raise NotImplementedError(f"Unsupported element type: {element_type}")


_PLANE_DOF_LABELS = ("U1", "U2")
_PLANE_FORCE_LABELS = ("Fx", "Fy")
_SPATIAL_DOF_LABELS = ("U1", "U2", "U3")
_SPATIAL_FORCE_LABELS = ("Fx", "Fy", "Fz")
_BEAM_DOF_LABELS = ("U1", "U2", "U3", "UR1", "UR2", "UR3")
_BEAM_FORCE_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_BEAM_ORIENTATION_VALIDITY_REQUIREMENT = ElementCapabilityRequirement(
    code="beam.orientation.valid",
    operations=("section.rectangle", "load.line.local"),
)
_BEAM_EXPLICIT_ORIENTATION_REQUIREMENT = ElementCapabilityRequirement(
    code="beam.orientation.explicit",
    operations=("load.line.local",),
)


def _plane_capabilities(
    kernel: ElementKernel,
    node_count: int,
) -> ElementCapabilityDescriptor:
    return ElementCapabilityDescriptor(
        canonical_type=kernel.canonical_type,
        aliases=kernel.aliases,
        family="plane_continuum",
        topological_dimension=2,
        spatial_dimension=2,
        node_count=node_count,
        dofs_per_node=2,
        section_families=("solid",),
        load_kinds=("node", "edge", "gravity"),
        dof_labels=_PLANE_DOF_LABELS,
        force_labels=_PLANE_FORCE_LABELS,
    )


def _solid_capabilities(
    kernel: ElementKernel,
    node_count: int,
) -> ElementCapabilityDescriptor:
    return ElementCapabilityDescriptor(
        canonical_type=kernel.canonical_type,
        aliases=kernel.aliases,
        family="solid_continuum",
        topological_dimension=3,
        spatial_dimension=3,
        node_count=node_count,
        dofs_per_node=3,
        section_families=("solid",),
        load_kinds=("node", "surface", "gravity"),
        dof_labels=_SPATIAL_DOF_LABELS,
        force_labels=_SPATIAL_FORCE_LABELS,
    )


def _register_builtin_capabilities() -> None:
    quad4 = Quad4Kernel()
    register_element_kernel(quad4, _plane_capabilities(quad4, 4))
    quad8 = Quad8Kernel()
    register_element_kernel(quad8, _plane_capabilities(quad8, 8))
    tri6 = Tri6Kernel()
    register_element_kernel(tri6, _plane_capabilities(tri6, 6))
    tri3 = Tri3Kernel()
    register_element_kernel(tri3, _plane_capabilities(tri3, 3))

    hex8 = Hex8Kernel()
    register_element_kernel(hex8, _solid_capabilities(hex8, 8))
    hex20 = Hex20Kernel()
    register_element_kernel(hex20, _solid_capabilities(hex20, 20))
    tet4 = Tet4Kernel()
    register_element_kernel(tet4, _solid_capabilities(tet4, 4))
    tet10 = Tet10Kernel()
    register_element_kernel(tet10, _solid_capabilities(tet10, 10))

    truss2 = Truss2Kernel()
    register_element_kernel(
        truss2,
        ElementCapabilityDescriptor(
            canonical_type=truss2.canonical_type,
            aliases=truss2.aliases,
            family="truss",
            topological_dimension=1,
            spatial_dimension=3,
            node_count=2,
            dofs_per_node=3,
            section_families=("truss",),
            load_kinds=("node", "gravity"),
            dof_labels=_SPATIAL_DOF_LABELS,
            force_labels=_SPATIAL_FORCE_LABELS,
        ),
    )

    beam2 = Beam2Kernel()
    register_element_kernel(
        beam2,
        ElementCapabilityDescriptor(
            canonical_type=beam2.canonical_type,
            aliases=beam2.aliases,
            family="beam",
            topological_dimension=1,
            spatial_dimension=3,
            node_count=2,
            dofs_per_node=6,
            section_families=("beam",),
            load_kinds=("node", "line", "gravity"),
            dof_labels=_BEAM_DOF_LABELS,
            force_labels=_BEAM_FORCE_LABELS,
            requirements=(
                _BEAM_ORIENTATION_VALIDITY_REQUIREMENT,
                _BEAM_EXPLICIT_ORIENTATION_REQUIREMENT,
            ),
        ),
    )


_register_builtin_capabilities()
