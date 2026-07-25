"""Immutable, GUI-independent element capability descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


_ELEMENT_FAMILIES = frozenset(
    {"plane_continuum", "solid_continuum", "truss", "beam"}
)
_SECTION_FAMILIES = frozenset({"solid", "truss", "beam"})
_LOAD_KINDS = frozenset({"node", "edge", "surface", "line", "gravity"})


class ElementCapabilityStatus(str, Enum):
    """Availability of a domain operation exposed by an element kernel."""

    SUPPORTED = "supported"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ElementCapabilityLimitation:
    """Stable limitation attached to one or more domain operations."""

    code: str
    operations: tuple[str, ...]
    message: str
    status: ElementCapabilityStatus = ElementCapabilityStatus.LIMITED

    def __post_init__(self) -> None:
        _require_nonempty_text(self.code, "limitation code")
        _require_text_tuple(self.operations, "limitation operations")
        _require_nonempty_text(self.message, "limitation message")
        if self.status is ElementCapabilityStatus.SUPPORTED:
            raise ValueError("a limitation status cannot be supported")


@dataclass(frozen=True, slots=True)
class ElementCapabilityRequirement:
    """One model-state requirement for a group of domain operations."""

    code: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.code, "requirement code")
        _require_text_tuple(self.operations, "requirement operations")


@dataclass(frozen=True, slots=True)
class ElementCapabilityDescriptor:
    """Intrinsic numerical and modeling capabilities of one element kernel."""

    canonical_type: str
    aliases: tuple[str, ...]
    family: str
    topological_dimension: int
    spatial_dimension: int
    node_count: int
    dofs_per_node: int
    section_families: tuple[str, ...]
    load_kinds: tuple[str, ...]
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    limitations: tuple[ElementCapabilityLimitation, ...] = ()
    requirements: tuple[ElementCapabilityRequirement, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_text(self.canonical_type, "canonical element type")
        _require_text_tuple(self.aliases, "element aliases", allow_empty=True)
        _require_unique_names(
            (self.canonical_type, *self.aliases),
            "canonical element type and aliases",
        )

        if self.family not in _ELEMENT_FAMILIES:
            raise ValueError(f"unsupported element family {self.family!r}")
        _require_positive_int(
            self.topological_dimension,
            "topological dimension",
        )
        _require_positive_int(self.spatial_dimension, "spatial dimension")
        if self.topological_dimension > self.spatial_dimension:
            raise ValueError(
                "topological dimension cannot exceed spatial dimension"
            )
        _require_positive_int(self.node_count, "node count")
        _require_positive_int(self.dofs_per_node, "DOFs per node")

        _require_text_tuple(self.section_families, "section families")
        unsupported_sections = set(self.section_families) - _SECTION_FAMILIES
        if unsupported_sections:
            raise ValueError(
                "unsupported section families: "
                + ", ".join(sorted(unsupported_sections))
            )

        _require_text_tuple(self.load_kinds, "load kinds")
        unsupported_loads = set(self.load_kinds) - _LOAD_KINDS
        if unsupported_loads:
            raise ValueError(
                "unsupported load kinds: " + ", ".join(sorted(unsupported_loads))
            )

        _require_text_tuple(self.dof_labels, "DOF labels")
        _require_text_tuple(self.force_labels, "force labels")
        if len(self.dof_labels) != self.dofs_per_node:
            raise ValueError("DOF label count must equal DOFs per node")
        if len(self.force_labels) != self.dofs_per_node:
            raise ValueError("force label count must equal DOFs per node")

        if not isinstance(self.limitations, tuple):
            raise TypeError("element limitations must be a tuple")
        if any(
            not isinstance(item, ElementCapabilityLimitation)
            for item in self.limitations
        ):
            raise TypeError(
                "element limitations must contain "
                "ElementCapabilityLimitation values"
            )
        _require_unique_names(
            tuple(item.code for item in self.limitations),
            "limitation codes",
        )
        if not isinstance(self.requirements, tuple):
            raise TypeError("element requirements must be a tuple")
        if any(
            not isinstance(item, ElementCapabilityRequirement)
            for item in self.requirements
        ):
            raise TypeError(
                "element requirements must contain "
                "ElementCapabilityRequirement values"
            )
        _require_unique_names(
            tuple(item.code for item in self.requirements),
            "requirement codes",
        )

    @property
    def status(self) -> ElementCapabilityStatus:
        """Return the most restrictive status declared by this descriptor."""

        statuses = {item.status for item in self.limitations}
        if ElementCapabilityStatus.UNAVAILABLE in statuses:
            return ElementCapabilityStatus.UNAVAILABLE
        if ElementCapabilityStatus.LIMITED in statuses:
            return ElementCapabilityStatus.LIMITED
        return ElementCapabilityStatus.SUPPORTED


def _require_nonempty_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")


def _require_text_tuple(
    values: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_nonempty_text(value, label)
    _require_unique_names(values, label)


def _require_unique_names(values: tuple[str, ...], label: str) -> None:
    normalized = [value.casefold() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique case-insensitively")


def _require_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
