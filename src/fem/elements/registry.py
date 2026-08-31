"""Registry of reference-element identities and intrinsic capabilities.

This module deliberately contains no stiffness kernels, material models, or
physics operators. It answers only questions intrinsic to a cell type.
"""

from __future__ import annotations

from collections.abc import Callable

from .capabilities import (
    ElementCapabilityDescriptor,
    ElementCapabilityRequirement,
)
from .beam2 import Beam2Definition
from .hex8 import Hex8Definition
from .hex20 import Hex20Definition
from .quad4 import Quad4Definition
from .quad8 import Quad8Definition
from .tet4 import Tet4Definition
from .tet10 import Tet10Definition
from .tri3 import Tri3Definition
from .tri6 import Tri6Definition
from .truss2 import Truss2Definition
from .contracts import ElementDefinition

_PLANE_DOF_LABELS = ("U1", "U2")
_PLANE_FORCE_LABELS = ("Fx", "Fy")
_SPATIAL_DOF_LABELS = ("U1", "U2", "U3")
_SPATIAL_FORCE_LABELS = ("Fx", "Fy", "Fz")
_BEAM_DOF_LABELS = ("U1", "U2", "U3", "UR1", "UR2", "UR3")
_BEAM_FORCE_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

_BEAM_REQUIREMENTS = (
    ElementCapabilityRequirement(
        code="beam.orientation.valid",
        operations=("section.rectangle", "load.line.local"),
    ),
    ElementCapabilityRequirement(
        code="beam.orientation.explicit",
        operations=("load.line.local",),
    ),
)


def _plane(
    canonical_type: str,
    aliases: tuple[str, ...],
    node_count: int,
) -> ElementCapabilityDescriptor:
    return ElementCapabilityDescriptor(
        canonical_type=canonical_type,
        aliases=aliases,
        family="plane_continuum",
        topological_dimension=2,
        spatial_dimension=2,
        node_count=node_count,
        dofs_per_node=2,
        section_families=("solid",),
        load_kinds=("node", "edge", "body", "gravity"),
        dof_labels=_PLANE_DOF_LABELS,
        force_labels=_PLANE_FORCE_LABELS,
    )


def _solid(
    canonical_type: str,
    aliases: tuple[str, ...],
    node_count: int,
) -> ElementCapabilityDescriptor:
    return ElementCapabilityDescriptor(
        canonical_type=canonical_type,
        aliases=aliases,
        family="solid_continuum",
        topological_dimension=3,
        spatial_dimension=3,
        node_count=node_count,
        dofs_per_node=3,
        section_families=("solid",),
        load_kinds=("node", "surface", "body", "gravity"),
        dof_labels=_SPATIAL_DOF_LABELS,
        force_labels=_SPATIAL_FORCE_LABELS,
    )


_DESCRIPTORS = (
    _plane("Quad4", ("CPS4", "CPE4"), 4),
    _plane("Quad8", ("CPS8", "CPE8"), 8),
    _plane("Tri3", ("CPS3", "CPE3"), 3),
    _plane("Tri6", ("CPS6", "CPE6"), 6),
    _solid("Hex8", ("C3D8",), 8),
    _solid("Hex20", ("C3D20",), 20),
    _solid("Tet4", ("C3D4",), 4),
    _solid("Tet10", ("C3D10",), 10),
    ElementCapabilityDescriptor(
        canonical_type="Truss2",
        aliases=(),
        family="truss",
        topological_dimension=1,
        spatial_dimension=3,
        node_count=2,
        dofs_per_node=3,
        section_families=("truss",),
        load_kinds=("node", "body", "gravity"),
        dof_labels=_SPATIAL_DOF_LABELS,
        force_labels=_SPATIAL_FORCE_LABELS,
    ),
    ElementCapabilityDescriptor(
        canonical_type="Beam2",
        aliases=(),
        family="beam",
        topological_dimension=1,
        spatial_dimension=3,
        node_count=2,
        dofs_per_node=6,
        section_families=("beam",),
        load_kinds=("node", "line", "body", "gravity"),
        dof_labels=_BEAM_DOF_LABELS,
        force_labels=_BEAM_FORCE_LABELS,
        requirements=_BEAM_REQUIREMENTS,
    ),
)

_BY_NAME: dict[str, ElementCapabilityDescriptor] = {}
for _descriptor in _DESCRIPTORS:
    for _name in (_descriptor.canonical_type, *_descriptor.aliases):
        _key = _name.casefold()
        if _key in _BY_NAME:
            raise RuntimeError(f"duplicate element identity {_name!r}")
        _BY_NAME[_key] = _descriptor


_DEFINITION_FACTORIES: dict[str, Callable[[], ElementDefinition]] = {
    "quad4": Quad4Definition,
    "quad8": Quad8Definition,
    "tri3": Tri3Definition,
    "tri6": Tri6Definition,
    "hex8": Hex8Definition,
    "hex20": Hex20Definition,
    "tet4": Tet4Definition,
    "tet10": Tet10Definition,
    "truss2": Truss2Definition,
    "beam2": Beam2Definition,
}

_UNSUPPORTED_REDUCED_INTEGRATION_TYPES = frozenset(
    {"c3d8r", "cps4r", "cpe4r", "cps8r", "cpe8r", "c3d20r"}
)
_UNSUPPORTED_COUPLED_ELEMENT_TYPES = frozenset({"c3d4t", "c3d10t"})


def get_element_capabilities(element_type: str) -> ElementCapabilityDescriptor:
    """Return immutable intrinsic capabilities for a canonical name or alias."""

    key = str(element_type).strip().casefold()
    try:
        return _BY_NAME[key]
    except KeyError:
        _raise_unsupported_element_type(element_type, key)


def registered_element_capabilities() -> tuple[ElementCapabilityDescriptor, ...]:
    """Return one descriptor per canonical element type."""

    return _DESCRIPTORS


def canonical_element_type(element_type: str) -> str:
    """Return the canonical reference-element name for a name or alias."""

    return get_element_capabilities(element_type).canonical_type


def get_element_definition(element_type: str) -> ElementDefinition:
    """Build the canonical geometric definition for one registered element."""

    canonical = canonical_element_type(element_type)
    try:
        return _DEFINITION_FACTORIES[canonical.casefold()]()
    except KeyError as exc:
        raise NotImplementedError(
            f"element definition is not registered for {element_type!r}"
        ) from exc


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


__all__ = [
    "canonical_element_type",
    "get_element_definition",
    "get_element_capabilities",
    "registered_element_capabilities",
]
