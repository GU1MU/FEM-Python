"""Contextual field registry for supported finite-element result families."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fem.elements import get_element_capabilities
from fem.post.averaging import NodalAveragingPolicy

from .data import FieldDescriptor, ResultDiagnostic
from .fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    PhysicalQuantity,
    ResultFieldId,
    ResultVariable,
)


RECOVERY_CONTRACT = 1

_CONTINUUM_COMPONENTS = {
    "plane_continuum": ("S11", "S22", "S33", "S12"),
    "solid_continuum": ("S11", "S22", "S33", "S12", "S23", "S13"),
}
_FINITE_STRAIN_COMPONENTS = {
    "plane_continuum": ("E11", "E22", "E33", "E12"),
    "solid_continuum": ("E11", "E22", "E33", "E12", "E23", "E13"),
}
_CONTINUUM_DERIVED_COMPONENTS = (
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)


class ResultModelFamily(str, Enum):
    """Stress-recovery family resolved from exact element capabilities."""

    PLANE_CONTINUUM = "plane_continuum"
    SOLID_CONTINUUM = "solid_continuum"
    TRUSS = "truss"
    BEAM = "beam"
    MIXED_UNSUPPORTED = "mixed_unsupported"


class FieldRecoveryKind(str, Enum):
    """Pure materializer selected by one contextual registry entry."""

    PRIMARY = "primary"
    CONTINUUM_STRESS = "continuum_stress"
    FINITE_STRAIN_TENSOR = "finite_strain_tensor"
    FINITE_STRAIN_SCALAR = "finite_strain_scalar"
    MATERIAL_STATE_SCALAR = "material_state_scalar"
    TRUSS_STRAIN = "truss_strain"
    TRUSS_STRESS = "truss_stress"
    BEAM_SECTION_END = "beam_section_end"
    BEAM_SECTION_POINT = "beam_section_point"
    BEAM_INTEGRATION_POINT_SF = "beam_integration_point_sf"
    BEAM_INTEGRATION_POINT_SM = "beam_integration_point_sm"
    BEAM_INTEGRATION_POINT_S11 = "beam_integration_point_s11"
    BEAM_NODE_ENVELOPE = "beam_node_envelope"


@dataclass(frozen=True, slots=True)
class ElementResultProfile:
    """Detached result-relevant projection of element capability facts."""

    family: ResultModelFamily
    canonical_element_types: tuple[str, ...]
    element_families: tuple[str, ...]
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    primary_compatible: bool
    stress_compatible: bool

    def __post_init__(self) -> None:
        if type(self.family) is not ResultModelFamily:
            raise TypeError("family must be ResultModelFamily")
        for name in (
            "canonical_element_types",
            "element_families",
            "dof_labels",
            "force_labels",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value
                for value in values
            ):
                raise TypeError(f"{name} must be a tuple of nonempty strings")
        if self.dofs_per_node is not None and (
            type(self.dofs_per_node) is not int or self.dofs_per_node <= 0
        ):
            raise TypeError("dofs_per_node must be a positive integer or None")
        if type(self.primary_compatible) is not bool:
            raise TypeError("primary_compatible must be bool")
        if type(self.stress_compatible) is not bool:
            raise TypeError("stress_compatible must be bool")
        if self.primary_compatible:
            if self.dofs_per_node is None:
                raise ValueError(
                    "primary-compatible profiles require dofs_per_node"
                )
            if len(self.dof_labels) != self.dofs_per_node:
                raise ValueError(
                    "primary-compatible profile DOF label count must equal "
                    "dofs_per_node"
                )
            if len(self.force_labels) != self.dofs_per_node:
                raise ValueError(
                    "primary-compatible profile force label count must equal "
                    "dofs_per_node"
                )
        elif (
            self.dofs_per_node is not None
            or self.dof_labels
            or self.force_labels
        ):
            raise ValueError(
                "primary-incompatible profiles cannot expose a partial DOF "
                "contract"
            )
        if self.stress_compatible and (
            self.family is ResultModelFamily.MIXED_UNSUPPORTED
            or not self.primary_compatible
        ):
            raise ValueError(
                "stress-compatible profiles require a supported family and "
                "a compatible primary DOF profile"
            )


@dataclass(frozen=True, slots=True)
class FieldRegistryEntry:
    """One descriptor plus the numerical recovery contract that produces it."""

    descriptor: FieldDescriptor
    recovery_kind: FieldRecoveryKind
    recovery_contract: int = RECOVERY_CONTRACT
    default_averaging_policy: NodalAveragingPolicy | None = None

    def __post_init__(self) -> None:
        if type(self.descriptor) is not FieldDescriptor:
            raise TypeError("descriptor must be FieldDescriptor")
        if type(self.recovery_kind) is not FieldRecoveryKind:
            raise TypeError("recovery_kind must be FieldRecoveryKind")
        if type(self.recovery_contract) is not int:
            raise TypeError("recovery_contract must be an integer")
        if self.recovery_contract <= 0:
            raise ValueError("recovery_contract must be positive")
        if (
            self.default_averaging_policy is not None
            and type(self.default_averaging_policy)
            is not NodalAveragingPolicy
        ):
            raise TypeError(
                "default_averaging_policy must be NodalAveragingPolicy or None"
            )
        resolved = (
            self.descriptor.field_id.position
            is FieldPosition.RESOLVED_NODAL
        )
        if resolved != (self.default_averaging_policy is not None):
            raise ValueError(
                "only resolved_nodal entries require a default averaging policy"
            )

    def default_request(self) -> FieldRequest:
        """Return this entry's canonical interactive request."""

        return FieldRequest(
            self.descriptor.field_id,
            averaging_policy=self.default_averaging_policy,
        )

    def default_key(self) -> FieldMaterializationKey:
        """Return the complete default numerical identity."""

        return FieldMaterializationKey(
            self.default_request(),
            self.recovery_contract,
        )


def classify_result_model(model: Any) -> ElementResultProfile:
    """Classify one model without inferring field support from GUI names."""

    try:
        mesh = model.mesh
        mesh_dofs_per_node = mesh.dofs_per_node
    except AttributeError as error:
        raise TypeError(
            "model must expose a mesh with elements and dofs_per_node"
        ) from error
    element_types: list[Any] = []
    for element in mesh.elements:
        try:
            element_type = element.type
        except AttributeError as error:
            raise TypeError("mesh elements must expose type") from error
        if element_type not in element_types:
            element_types.append(element_type)
    return classify_result_element_types(
        tuple(element_types),
        dofs_per_node=mesh_dofs_per_node,
    )


def classify_result_element_types(
    element_types: Iterable[object],
    *,
    dofs_per_node: int,
) -> ElementResultProfile:
    """Classify expected or realized elements through one capability path."""

    if type(dofs_per_node) is not int or dofs_per_node <= 0:
        raise TypeError("dofs_per_node must be a positive integer")
    try:
        requested_types = tuple(element_types)
    except TypeError as error:
        raise TypeError("element_types must be iterable") from error
    descriptors = []
    unsupported_types: list[str] = []
    for element_type in requested_types:
        try:
            descriptors.append(get_element_capabilities(element_type))
        except (NotImplementedError, TypeError, ValueError):
            unsupported_types.append(str(element_type))

    canonical_types = _ordered_unique(
        descriptor.canonical_type for descriptor in descriptors
    )
    element_families = _ordered_unique(
        descriptor.family for descriptor in descriptors
    )
    profiles = _ordered_unique(
        (
            descriptor.dofs_per_node,
            descriptor.dof_labels,
            descriptor.force_labels,
        )
        for descriptor in descriptors
    )
    primary_compatible = (
        bool(requested_types)
        and not unsupported_types
        and len(descriptors) == len(requested_types)
        and len(profiles) == 1
        and profiles[0][0] == dofs_per_node
    )
    if primary_compatible:
        dofs_per_node = profiles[0][0]
        dof_labels = profiles[0][1]
        force_labels = profiles[0][2]
    else:
        dofs_per_node = None
        dof_labels = ()
        force_labels = ()

    family = _stress_family(
        descriptors=tuple(descriptors),
        element_count=len(requested_types),
        unsupported=bool(unsupported_types),
        primary_compatible=primary_compatible,
    )
    return ElementResultProfile(
        family=family,
        canonical_element_types=canonical_types,
        element_families=element_families,
        dofs_per_node=dofs_per_node,
        dof_labels=dof_labels,
        force_labels=force_labels,
        primary_compatible=primary_compatible,
        stress_compatible=family is not ResultModelFamily.MIXED_UNSUPPORTED,
    )


def catalog_diagnostics(
    profile: ElementResultProfile,
) -> tuple[ResultDiagnostic, ...]:
    """Explain profile-level omissions without inventing a field descriptor."""

    if type(profile) is not ElementResultProfile:
        raise TypeError("profile must be ElementResultProfile")
    if profile.family is not ResultModelFamily.MIXED_UNSUPPORTED:
        return ()
    return (
        ResultDiagnostic(
            code="result.catalog.stress_family_unsupported",
            severity="warning",
            message=(
                "The element collection has no single canonical stress "
                "field contract."
            ),
            path=("results", "catalog", "variables", ResultVariable.S.value),
            remediation=(
                "Use one compatible continuum family, homogeneous Truss2, "
                "or homogeneous Beam2 for canonical stress recovery."
            ),
            details={
                "canonical_variable": ResultVariable.S.value,
                "model_family": profile.family.value,
                "canonical_element_types": list(
                    profile.canonical_element_types
                ),
                "element_families": list(profile.element_families),
            },
        ),
    )


def catalog_entries(
    profile: ElementResultProfile,
    *,
    finite_strain: bool = False,
    material_state: bool = False,
) -> tuple[FieldRegistryEntry, ...]:
    """Return the ordered registry for one exact profile and result contract.

    ``finite_strain`` is deliberately result-scoped rather than an element
    capability. A continuum model can produce either the linear result
    contract or captured finite-strain state fields, so the latter must be
    enabled only by a result provider that has those outputs.
    """

    if type(profile) is not ElementResultProfile:
        raise TypeError("profile must be ElementResultProfile")
    if type(finite_strain) is not bool:
        raise TypeError("finite_strain must be bool")
    if type(material_state) is not bool:
        raise TypeError("material_state must be bool")
    entries: list[FieldRegistryEntry] = []
    if profile.primary_compatible:
        entries.extend(_primary_entries(profile.dof_labels))
    if profile.stress_compatible:
        entries.extend(
            _derived_entries(
                profile.family,
                finite_strain=(
                    finite_strain
                    and profile.family
                    in {
                        ResultModelFamily.PLANE_CONTINUUM,
                        ResultModelFamily.SOLID_CONTINUUM,
                    }
                ),
                material_state=(
                    material_state
                    and profile.family
                    in {
                        ResultModelFamily.PLANE_CONTINUUM,
                        ResultModelFamily.SOLID_CONTINUUM,
                    }
                ),
            )
        )
    entries.sort(key=lambda entry: entry.descriptor.order)
    orders = tuple(entry.descriptor.order for entry in entries)
    if len(orders) != len(set(orders)):
        raise RuntimeError("contextual result descriptor orders must be unique")
    field_ids = tuple(entry.descriptor.field_id for entry in entries)
    if len(field_ids) != len(set(field_ids)):
        raise RuntimeError("contextual result field IDs must be unique")
    return tuple(entries)


def descriptor_for(
    profile: ElementResultProfile,
    field_id: ResultFieldId,
    *,
    finite_strain: bool = False,
    material_state: bool = False,
) -> FieldDescriptor:
    """Resolve one descriptor in its model-family context."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    for entry in catalog_entries(
        profile,
        finite_strain=finite_strain,
        material_state=material_state,
    ):
        if entry.descriptor.field_id == field_id:
            return entry.descriptor
    raise KeyError(field_id)


def registry_entry_for(
    profile: ElementResultProfile,
    field_id: ResultFieldId,
    *,
    finite_strain: bool = False,
    material_state: bool = False,
) -> FieldRegistryEntry:
    """Resolve one complete recovery entry in model-family context."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    for entry in catalog_entries(
        profile,
        finite_strain=finite_strain,
        material_state=material_state,
    ):
        if entry.descriptor.field_id == field_id:
            return entry
    raise KeyError(field_id)


def _stress_family(
    *,
    descriptors: tuple[Any, ...],
    element_count: int,
    unsupported: bool,
    primary_compatible: bool,
) -> ResultModelFamily:
    if (
        element_count == 0
        or unsupported
        or len(descriptors) != element_count
        or not primary_compatible
    ):
        return ResultModelFamily.MIXED_UNSUPPORTED
    families = {descriptor.family for descriptor in descriptors}
    if families == {"plane_continuum"}:
        return ResultModelFamily.PLANE_CONTINUUM
    if families == {"solid_continuum"}:
        return ResultModelFamily.SOLID_CONTINUUM
    canonical_types = {descriptor.canonical_type for descriptor in descriptors}
    if families == {"truss"} and canonical_types == {"Truss2"}:
        return ResultModelFamily.TRUSS
    if families == {"beam"} and canonical_types == {"Beam2"}:
        return ResultModelFamily.BEAM
    return ResultModelFamily.MIXED_UNSUPPORTED


def _primary_entries(
    dof_labels: tuple[str, ...],
) -> tuple[FieldRegistryEntry, ...]:
    translations = tuple(
        label for label in dof_labels if _component_number(label, "U")
    )
    rotations = tuple(
        label for label in dof_labels if _component_number(label, "UR")
    )
    entries: list[FieldRegistryEntry] = []
    if translations:
        entries.append(
            _entry(
                ResultVariable.U,
                FieldPosition.NODE,
                FieldAssociation.NODE,
                PhysicalQuantity.DISPLACEMENT,
                translations,
                ("Magnitude",),
                "result.field.u.node",
                "Magnitude",
                0,
                FieldRecoveryKind.PRIMARY,
            )
        )
    if rotations:
        entries.append(
            _entry(
                ResultVariable.UR,
                FieldPosition.NODE,
                FieldAssociation.NODE,
                PhysicalQuantity.ROTATION,
                rotations,
                (),
                "result.field.ur.node",
                rotations[0],
                1,
                FieldRecoveryKind.PRIMARY,
            )
        )
    if translations:
        reactions = tuple(
            f"RF{_component_number(label, 'U')}"
            for label in translations
        )
        entries.append(
            _entry(
                ResultVariable.RF,
                FieldPosition.NODE,
                FieldAssociation.NODE,
                PhysicalQuantity.FORCE,
                reactions,
                ("Magnitude",),
                "result.field.rf.node",
                "Magnitude",
                2,
                FieldRecoveryKind.PRIMARY,
            )
        )
    if rotations:
        moments = tuple(
            f"RM{_component_number(label, 'UR')}"
            for label in rotations
        )
        entries.append(
            _entry(
                ResultVariable.RM,
                FieldPosition.NODE,
                FieldAssociation.NODE,
                PhysicalQuantity.MOMENT,
                moments,
                (),
                "result.field.rm.node",
                moments[0],
                3,
                FieldRecoveryKind.PRIMARY,
            )
        )
    return tuple(entries)


def _derived_entries(
    family: ResultModelFamily,
    *,
    finite_strain: bool = False,
    material_state: bool = False,
) -> tuple[FieldRegistryEntry, ...]:
    if family in {
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.SOLID_CONTINUUM,
    }:
        entries = list(_continuum_entries(family))
        if finite_strain:
            entries.extend(_finite_strain_entries(family))
        elif material_state:
            entries.extend(_material_state_entries(family))
        return tuple(entries)
    if family is ResultModelFamily.TRUSS:
        return (
            _entry(
                ResultVariable.LE,
                FieldPosition.CENTROID,
                FieldAssociation.ELEMENT,
                PhysicalQuantity.STRAIN,
                ("LE11",),
                (),
                "result.field.le.centroid",
                "LE11",
                10,
                FieldRecoveryKind.TRUSS_STRAIN,
            ),
            _entry(
                ResultVariable.S,
                FieldPosition.CENTROID,
                FieldAssociation.ELEMENT,
                PhysicalQuantity.STRESS,
                ("S11",),
                ("Mises",),
                "result.field.s.centroid",
                "Mises",
                20,
                FieldRecoveryKind.TRUSS_STRESS,
            ),
        )
    if family is ResultModelFamily.BEAM:
        section_forces = (
            _entry(
                ResultVariable.SF,
                FieldPosition.INTEGRATION_POINT,
                FieldAssociation.INTEGRATION_POINT,
                PhysicalQuantity.FORCE,
                ("N", "Vy", "Vz"),
                (),
                "result.field.sf.integration_point",
                "N",
                18,
                FieldRecoveryKind.BEAM_INTEGRATION_POINT_SF,
                recovery_contract=4,
            ),
            _entry(
                ResultVariable.SM,
                FieldPosition.INTEGRATION_POINT,
                FieldAssociation.INTEGRATION_POINT,
                PhysicalQuantity.MOMENT,
                ("T", "My", "Mz"),
                (),
                "result.field.sm.integration_point",
                "T",
                19,
                FieldRecoveryKind.BEAM_INTEGRATION_POINT_SM,
                recovery_contract=4,
            ),
        )
        section_points = tuple(
            _entry(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
                FieldAssociation.INTEGRATION_POINT,
                PhysicalQuantity.STRESS,
                ("S11", "S22", "S12"),
                (
                    "Mises",
                    "MaxPrincipal",
                    "MidPrincipal",
                    "MinPrincipal",
                ),
                f"result.field.s.integration_point.section_point.{number}",
                "Mises",
                19 + number,
                FieldRecoveryKind.BEAM_INTEGRATION_POINT_S11,
                recovery_contract=4,
                section_point_number=number,
            )
            for number in range(1, 5)
        )
        return section_forces + section_points
    return ()


def _continuum_entries(
    family: ResultModelFamily,
) -> tuple[FieldRegistryEntry, ...]:
    components = _CONTINUUM_COMPONENTS[family.value]
    definitions = (
        (
            FieldPosition.INTEGRATION_POINT,
            FieldAssociation.INTEGRATION_POINT,
            20,
        ),
        (FieldPosition.CENTROID, FieldAssociation.ELEMENT, 21),
        (FieldPosition.ELEMENT_NODAL, FieldAssociation.ELEMENT_NODE, 22),
        (FieldPosition.NODE_REGION, FieldAssociation.NODE_REGION, 23),
        (
            FieldPosition.RESOLVED_NODAL,
            FieldAssociation.RESOLVED_NODAL,
            24,
        ),
    )
    entries = []
    for position, association, order in definitions:
        entries.append(
            _entry(
                ResultVariable.S,
                position,
                association,
                PhysicalQuantity.STRESS,
                components,
                _CONTINUUM_DERIVED_COMPONENTS,
                f"result.field.s.{position.value}",
                "Mises",
                order,
                FieldRecoveryKind.CONTINUUM_STRESS,
                averaging_policy=(
                    NodalAveragingPolicy()
                    if position is FieldPosition.RESOLVED_NODAL
                    else None
                ),
            )
        )
    return tuple(entries)


def _finite_strain_entries(
    family: ResultModelFamily,
) -> tuple[FieldRegistryEntry, ...]:
    components = _FINITE_STRAIN_COMPONENTS[family.value]
    definitions = (
        (
            FieldPosition.INTEGRATION_POINT,
            FieldAssociation.INTEGRATION_POINT,
            30,
        ),
        (FieldPosition.CENTROID, FieldAssociation.ELEMENT, 31),
        (FieldPosition.ELEMENT_NODAL, FieldAssociation.ELEMENT_NODE, 32),
        (FieldPosition.NODE_REGION, FieldAssociation.NODE_REGION, 33),
        (
            FieldPosition.RESOLVED_NODAL,
            FieldAssociation.RESOLVED_NODAL,
            34,
        ),
    )
    entries: list[FieldRegistryEntry] = []
    for position, association, order in definitions:
        averaging_policy = (
            NodalAveragingPolicy()
            if position is FieldPosition.RESOLVED_NODAL
            else None
        )
        entries.append(
            _entry(
                ResultVariable.E,
                position,
                association,
                PhysicalQuantity.STRAIN,
                components,
                (),
                f"result.field.e.{position.value}",
                components[0],
                order,
                FieldRecoveryKind.FINITE_STRAIN_TENSOR,
                averaging_policy=averaging_policy,
            )
        )
        entries.append(
            _entry(
                ResultVariable.PEEQ,
                position,
                association,
                PhysicalQuantity.STRAIN,
                ("PEEQ",),
                (),
                f"result.field.peeq.{position.value}",
                "PEEQ",
                order + 10,
                FieldRecoveryKind.FINITE_STRAIN_SCALAR,
                averaging_policy=averaging_policy,
            )
        )
    return tuple(entries)


def _material_state_entries(
    family: ResultModelFamily,
) -> tuple[FieldRegistryEntry, ...]:
    """Return constitutive state fields independent of geometry measure."""

    definitions = (
        (FieldPosition.INTEGRATION_POINT, FieldAssociation.INTEGRATION_POINT, 40),
        (FieldPosition.CENTROID, FieldAssociation.ELEMENT, 41),
        (FieldPosition.ELEMENT_NODAL, FieldAssociation.ELEMENT_NODE, 42),
        (FieldPosition.NODE_REGION, FieldAssociation.NODE_REGION, 43),
        (FieldPosition.RESOLVED_NODAL, FieldAssociation.RESOLVED_NODAL, 44),
    )
    return tuple(
        _entry(
            ResultVariable.PEEQ,
            position,
            association,
            PhysicalQuantity.STRAIN,
            ("PEEQ",),
            (),
            f"result.field.peeq.{position.value}",
            "PEEQ",
            order,
            FieldRecoveryKind.MATERIAL_STATE_SCALAR,
            averaging_policy=(
                NodalAveragingPolicy()
                if position is FieldPosition.RESOLVED_NODAL
                else None
            ),
        )
        for position, association, order in definitions
    )


def _entry(
    variable: ResultVariable,
    position: FieldPosition,
    association: FieldAssociation,
    quantity: PhysicalQuantity,
    components: tuple[str, ...],
    derived_components: tuple[str, ...],
    label_key: str,
    default_component: str,
    order: int,
    recovery_kind: FieldRecoveryKind,
    *,
    averaging_policy: NodalAveragingPolicy | None = None,
    recovery_contract: int = RECOVERY_CONTRACT,
    section_point_number: int | None = None,
) -> FieldRegistryEntry:
    return FieldRegistryEntry(
        descriptor=FieldDescriptor(
            field_id=ResultFieldId(
                variable,
                position,
                section_point_number=section_point_number,
            ),
            association=association,
            quantity=quantity,
            components=components,
            derived_components=derived_components,
            label_key=label_key,
            unit_label=None,
            default_component=default_component,
            order=order,
        ),
        recovery_kind=recovery_kind,
        recovery_contract=recovery_contract,
        default_averaging_policy=averaging_policy,
    )


def _component_number(label: str, prefix: str) -> int | None:
    if not label.startswith(prefix):
        return None
    suffix = label[len(prefix):]
    if suffix not in {"1", "2", "3"}:
        return None
    return int(suffix)


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


__all__ = [
    "ElementResultProfile",
    "FieldRecoveryKind",
    "FieldRegistryEntry",
    "RECOVERY_CONTRACT",
    "ResultModelFamily",
    "catalog_diagnostics",
    "catalog_entries",
    "classify_result_element_types",
    "classify_result_model",
    "descriptor_for",
    "registry_entry_for",
]
