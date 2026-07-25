"""Application-level model and authoring capability reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from fem.elements import (
    ElementCapabilityDescriptor,
    get_element_capabilities,
)

from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)


_REGION_KINDS = frozenset(
    {"node_set", "element_set", "edge", "surface"}
)
_DISTRIBUTED_LOAD_KINDS = frozenset({"edge", "surface", "line"})


class AuthoringStatus(str, Enum):
    """Product-level availability derived from domain capabilities."""

    ENABLED = "enabled"
    LIMITED = "limited"
    READ_ONLY = "read_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RegionRef:
    """Typed reference that preserves region namespace identity."""

    kind: str
    name: str

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold()
        name = str(self.name).strip()
        if kind not in _REGION_KINDS:
            raise ValueError(f"unsupported region kind: {self.kind!r}")
        if not name:
            raise ValueError("region name must not be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class AuthoringCapability:
    """Availability and reason for one application authoring operation."""

    operation: str
    status: AuthoringStatus
    diagnostics: tuple[PreflightDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionCapability:
    """Safe capability intersection for one typed model region."""

    region: RegionRef
    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    homogeneous: bool
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    distributed_load_kinds: tuple[str, ...]
    diagnostics: tuple[PreflightDiagnostic, ...] = ()

    @property
    def status(self) -> AuthoringStatus:
        if any(item.blocking for item in self.diagnostics):
            return AuthoringStatus.UNAVAILABLE
        if self.diagnostics:
            return AuthoringStatus.LIMITED
        return AuthoringStatus.ENABLED

    def supports_section(self, section_type: str) -> bool:
        return (
            self.compatible
            and _supports_section_preset(
                self.section_presets,
                section_type,
            )
        )

    def supports_distributed_load(self, load_kind: str) -> bool:
        return (
            self.compatible
            and str(load_kind).strip().casefold()
            in self.distributed_load_kinds
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilityReport:
    """Aggregated model facts and product authoring policy."""

    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    diagnostics: tuple[PreflightDiagnostic, ...]
    regions: tuple[RegionCapability, ...]
    authoring: tuple[AuthoringCapability, ...]

    @property
    def status(self) -> AuthoringStatus:
        if any(item.blocking for item in self.diagnostics):
            return AuthoringStatus.UNAVAILABLE
        if self.diagnostics:
            return AuthoringStatus.LIMITED
        return AuthoringStatus.ENABLED

    def region(self, region: RegionRef) -> RegionCapability:
        """Return a region report without collapsing its namespace."""

        for capability in self.regions:
            if capability.region == region:
                return capability
        return _missing_region_capability(region)

    def operation(self, name: str) -> AuthoringCapability:
        for capability in self.authoring:
            if capability.operation == str(name):
                return capability
        return AuthoringCapability(
            str(name),
            AuthoringStatus.UNAVAILABLE,
            (),
        )

    def supports_section(self, section_type: str) -> bool:
        """Return whether the model contract can author this section type."""

        return self.compatible and _supports_section_preset(
            self.section_presets,
            section_type,
        )


def require_region_kind(region: RegionRef, expected_kind: str) -> str:
    """Validate an application command target before writing a string DTO."""

    if not isinstance(region, RegionRef):
        raise TypeError("authoring target must be RegionRef")
    normalized = str(expected_kind).strip().casefold()
    if region.kind != normalized:
        raise ValueError(
            f"operation requires region kind {normalized!r}, "
            f"got {region.kind!r}"
        )
    return region.name


def describe_model_capabilities(model: Any) -> ModelCapabilityReport:
    """Describe intrinsic model facts and Phase 3 authoring policy."""

    elements = tuple(getattr(getattr(model, "mesh", None), "elements", ()))
    aggregate = _aggregate_capabilities(
        (getattr(element, "type", "") for element in elements),
        subject="model",
    )
    regions = tuple(
        describe_region_capabilities(model, reference)
        for reference in _model_region_refs(model)
    )
    output_diagnostic = PreflightDiagnostic(
        code="output.request.not_executed",
        severity=PreflightSeverity.WARNING,
        stage=PreflightStage.OUTPUT,
        message=(
            "Output requests are preserved but are not executed by the "
            "current solver."
        ),
        subject="output_request",
        path=("analysis", "outputs"),
        remediation="可查看或删除既有输出请求；当前版本不能新建。",
    )
    section_status = (
        AuthoringStatus.UNAVAILABLE
        if not aggregate.compatible or not aggregate.section_families
        else (
            AuthoringStatus.LIMITED
            if aggregate.diagnostics
            else AuthoringStatus.ENABLED
        )
    )
    line_regions = tuple(
        item
        for item in regions
        if item.region.kind == "element_set"
        and item.compatible
        and item.distributed_load_kinds == ("line",)
        and item.families == ("beam",)
    )
    line_status = (
        AuthoringStatus.LIMITED
        if line_regions
        and any(item.diagnostics for item in line_regions)
        else (
            AuthoringStatus.ENABLED
            if line_regions
            else AuthoringStatus.UNAVAILABLE
        )
    )
    authoring = (
        AuthoringCapability("section.create", section_status),
        AuthoringCapability("line_load.create", line_status),
        AuthoringCapability(
            "output_request.create",
            AuthoringStatus.UNAVAILABLE,
            (output_diagnostic,),
        ),
        AuthoringCapability(
            "output_request.existing",
            AuthoringStatus.READ_ONLY,
            (output_diagnostic,),
        ),
    )
    return ModelCapabilityReport(
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        diagnostics=aggregate.diagnostics,
        regions=regions,
        authoring=authoring,
    )


def describe_region_capabilities(
    model: Any,
    region: RegionRef,
) -> RegionCapability:
    """Describe a typed region using a safe element capability intersection."""

    if not isinstance(region, RegionRef):
        raise TypeError("region must be RegionRef")
    try:
        element_lookup = {
            int(element.id): element
            for element in getattr(
                getattr(model, "mesh", None),
                "elements",
                (),
            )
        }
        element_ids = _region_element_ids(model, region, element_lookup)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _missing_region_capability(region, str(error))
    element_types = tuple(
        element_lookup[element_id].type for element_id in element_ids
    )
    aggregate = _aggregate_capabilities(element_types, subject=region)
    return RegionCapability(
        region=region,
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        homogeneous=len(aggregate.canonical_element_types) <= 1,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        distributed_load_kinds=aggregate.distributed_load_kinds,
        diagnostics=aggregate.diagnostics,
    )


def describe_native_authoring_capabilities(
    recipe: Any,
    mesh_settings: Any,
) -> ModelCapabilityReport:
    """Describe a native recipe before a mesh artifact exists."""

    shape = str(
        getattr(mesh_settings, "cell_shape", "")
    ).strip().casefold()
    order = int(getattr(mesh_settings, "order", 1))
    if not shape:
        from fem.geometry.recipes import geometry_dimension

        shape = (
            "tetrahedron"
            if geometry_dimension(recipe) == 3
            else "triangle"
        )
    canonical = {
        ("triangle", 1): "Tri3",
        ("triangle", 2): "Tri6",
        ("quadrilateral", 1): "Quad4",
        ("quadrilateral", 2): "Quad8",
        ("tetrahedron", 1): "Tet4",
        ("tetrahedron", 2): "Tet10",
        ("hexahedron", 1): "Hex8",
        ("hexahedron", 2): "Hex20",
    }.get((shape, order))
    if canonical is None:
        aggregate = _aggregate_capabilities(("",), subject="native_mesh")
    else:
        aggregate = _aggregate_capabilities(
            (canonical,),
            subject="native_mesh",
        )
    return ModelCapabilityReport(
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        diagnostics=aggregate.diagnostics,
        regions=(),
        authoring=(
            AuthoringCapability(
                "section.create",
                (
                    AuthoringStatus.ENABLED
                    if aggregate.compatible
                    else AuthoringStatus.UNAVAILABLE
                ),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _CapabilityAggregate:
    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    distributed_load_kinds: tuple[str, ...]
    diagnostics: tuple[PreflightDiagnostic, ...]


def _aggregate_capabilities(
    element_types: Iterable[Any],
    *,
    subject: Any,
) -> _CapabilityAggregate:
    descriptors: list[ElementCapabilityDescriptor] = []
    diagnostics: list[PreflightDiagnostic] = []
    seen: set[str] = set()
    for element_type in element_types:
        try:
            descriptor = get_element_capabilities(str(element_type))
        except (NotImplementedError, TypeError, ValueError) as error:
            diagnostics.append(
                _unsupported_mix_diagnostic(subject, str(error))
            )
            continue
        if descriptor.canonical_type.casefold() not in seen:
            descriptors.append(descriptor)
            seen.add(descriptor.canonical_type.casefold())
    if not descriptors:
        if not diagnostics:
            diagnostics.append(
                _unsupported_mix_diagnostic(
                    subject,
                    "region contains no elements",
                )
            )
        return _CapabilityAggregate(
            (),
            (),
            False,
            None,
            None,
            None,
            (),
            (),
            (),
            (),
            (),
            (),
            tuple(diagnostics),
        )

    families = _ordered_unique(item.family for item in descriptors)
    topologies = _ordered_unique(
        item.topological_dimension for item in descriptors
    )
    spatial_dimensions = _ordered_unique(
        item.spatial_dimension for item in descriptors
    )
    dof_profiles = _ordered_unique(
        (item.dofs_per_node, item.dof_labels, item.force_labels)
        for item in descriptors
    )
    section_families = _intersection(
        item.section_families for item in descriptors
    )
    load_kinds = _intersection(item.load_kinds for item in descriptors)
    compatible = (
        len(families) == 1
        and len(topologies) == 1
        and len(spatial_dimensions) == 1
        and len(dof_profiles) == 1
        and bool(section_families)
        and bool(load_kinds)
        and not diagnostics
    )
    if not compatible:
        diagnostics.append(
            _unsupported_mix_diagnostic(
                subject,
                "element families or DOF profiles have no safe common contract",
            )
        )
    for descriptor in descriptors:
        for limitation in descriptor.limitations:
            if any(
                existing.code == limitation.code
                for existing in diagnostics
            ):
                continue
            diagnostics.append(
                PreflightDiagnostic(
                    code=limitation.code,
                    severity=PreflightSeverity.WARNING,
                    stage=PreflightStage.CAPABILITY,
                    message=limitation.message,
                    subject=subject,
                    path=("capabilities", descriptor.canonical_type),
                    remediation=(
                        "当前局部轴由单元几何自动确定；请核对方向假设。"
                    ),
                    details={"operations": limitation.operations},
                )
            )
    family = families[0] if len(families) == 1 else None
    profile = dof_profiles[0] if len(dof_profiles) == 1 else None
    return _CapabilityAggregate(
        canonical_element_types=tuple(
            item.canonical_type for item in descriptors
        ),
        families=families,
        compatible=compatible,
        topological_dimension=(
            topologies[0] if len(topologies) == 1 else None
        ),
        spatial_dimension=(
            spatial_dimensions[0]
            if len(spatial_dimensions) == 1
            else None
        ),
        dofs_per_node=None if profile is None else profile[0],
        dof_labels=() if profile is None else profile[1],
        force_labels=() if profile is None else profile[2],
        section_families=section_families,
        section_presets=(
            () if family is None else _section_presets(family)
        ),
        load_kinds=load_kinds,
        distributed_load_kinds=tuple(
            kind for kind in load_kinds
            if kind in _DISTRIBUTED_LOAD_KINDS
        ),
        diagnostics=tuple(diagnostics),
    )


def _model_region_refs(model: Any) -> tuple[RegionRef, ...]:
    refs: list[RegionRef] = []
    for kind, attribute in (
        ("node_set", "node_sets"),
        ("element_set", "element_sets"),
        ("edge", "edges"),
        ("surface", "surfaces"),
    ):
        refs.extend(
            RegionRef(kind, str(name))
            for name in getattr(model, attribute, {})
        )
    internal = getattr(model, "metadata", {}).get(
        "_abaqus_internal_element_sets",
        {},
    )
    public_names = {
        item.name
        for item in refs
        if item.kind == "element_set"
    }
    refs.extend(
        RegionRef("element_set", str(name))
        for name in internal
        if str(name) not in public_names
    )
    return tuple(refs)


def _region_element_ids(
    model: Any,
    region: RegionRef,
    element_lookup: dict[int, Any],
) -> tuple[int, ...]:
    if region.kind == "element_set":
        public = getattr(model, "element_sets", {})
        internal = getattr(model, "metadata", {}).get(
            "_abaqus_internal_element_sets",
            {},
        )
        collection = (
            public[region.name]
            if region.name in public
            else internal[region.name]
        )
        ids = tuple(int(value) for value in collection.element_ids)
    elif region.kind == "edge":
        collection = getattr(model, "edges", {})[region.name]
        ids = tuple(int(entry.elem_id) for entry in collection.edges)
    elif region.kind == "surface":
        collection = getattr(model, "surfaces", {})[region.name]
        ids = tuple(int(entry.elem_id) for entry in collection.faces)
    else:
        node_set = getattr(model, "node_sets", {})[region.name]
        node_ids = {int(value) for value in node_set.node_ids}
        ids = tuple(
            element_id
            for element_id, element in element_lookup.items()
            if any(int(node_id) in node_ids for node_id in element.node_ids)
        )
    unique = _ordered_unique(ids)
    missing = tuple(
        element_id
        for element_id in unique
        if element_id not in element_lookup
    )
    if missing:
        raise KeyError(f"region references missing element {missing[0]}")
    return unique


def _intersection(
    values: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    collections = tuple(values)
    if not collections:
        return ()
    common = set(collections[0])
    for collection in collections[1:]:
        common.intersection_update(collection)
    return tuple(value for value in collections[0] if value in common)


def _supports_section_preset(
    presets: tuple[str, ...],
    section_type: str,
) -> bool:
    normalized = str(section_type).strip().casefold()
    if normalized == "solid":
        return any(
            preset == "solid" or preset.startswith("solid_plane_")
            for preset in presets
        )
    return normalized in presets


def _section_presets(element_family: str) -> tuple[str, ...]:
    from fem.materials import section_presets_for_element_family

    return section_presets_for_element_family(element_family)


def _ordered_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _unsupported_mix_diagnostic(
    subject: Any,
    message: str,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code="model.capability.unsupported_mix",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=subject,
        path=("capabilities",),
        remediation="请选择具有共同单元族与自由度契约的区域。",
    )


def _missing_region_capability(
    region: RegionRef,
    message: str | None = None,
) -> RegionCapability:
    diagnostic = PreflightDiagnostic(
        code="step.reference.invalid",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message or f"region {region.name!r} is not defined",
        subject=region,
        path=("regions", region.kind, region.name),
        remediation="请选择当前模型中存在的同类命名区域。",
    )
    return RegionCapability(
        region=region,
        canonical_element_types=(),
        families=(),
        homogeneous=False,
        compatible=False,
        topological_dimension=None,
        spatial_dimension=None,
        dofs_per_node=None,
        dof_labels=(),
        force_labels=(),
        section_families=(),
        section_presets=(),
        load_kinds=(),
        distributed_load_kinds=(),
        diagnostics=(diagnostic,),
    )


__all__ = [
    "AuthoringCapability",
    "AuthoringStatus",
    "ModelCapabilityReport",
    "RegionCapability",
    "RegionRef",
    "describe_model_capabilities",
    "describe_native_authoring_capabilities",
    "describe_region_capabilities",
    "require_region_kind",
]
