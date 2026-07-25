from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..core.model import ElementSet, MaterialDefinition, SectionAssignment
from ..elements.beam_section import parse_beam2_section
from .sections import (
    MaterialPropertyError,
    SectionCompatibilityError,
    SectionPropertyError,
    resolve_section_properties,
)


_SECTION_KEYS_METADATA = "_section_property_keys_by_element"
_SECTION_ORIGINALS_METADATA = "_section_original_properties_by_element"
_SECTION_IDENTITIES_METADATA = "_section_property_element_identity_by_element"
_SECTION_METADATA_KEYS = (
    _SECTION_KEYS_METADATA,
    _SECTION_ORIGINALS_METADATA,
    _SECTION_IDENTITIES_METADATA,
)


@dataclass(frozen=True, slots=True)
class SectionResolutionIssue:
    """One deterministic problem found by pure section resolution."""

    code: str
    message: str
    assignment_index: int | None = None
    element_set: str | None = None
    material: str | None = None
    section_type: str | None = None
    element_id: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedSectionAssignment:
    """One source assignment retained in caller order."""

    assignment_index: int
    element_set: str
    material: str
    section_type: str
    element_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EffectiveSectionAssignment:
    """The final last-match assignment and owned properties for one element."""

    element_id: int
    assignment_index: int
    element_set: str
    material: str
    section_type: str
    applied_properties: dict[str, Any]
    effective_properties: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_properties",
            deepcopy(dict(self.applied_properties)),
        )
        object.__setattr__(
            self,
            "effective_properties",
            deepcopy(dict(self.effective_properties)),
        )


@dataclass(frozen=True, slots=True)
class SectionResolution:
    """Pure model-wide section resolution with explicit failure facts."""

    assignments: tuple[ResolvedSectionAssignment, ...]
    effective_assignments: tuple[EffectiveSectionAssignment, ...]
    uncovered_element_ids: tuple[int, ...]
    missing_materials: tuple[str, ...]
    missing_element_sets: tuple[str, ...]
    missing_element_ids: tuple[int, ...]
    incompatible_element_ids: tuple[int, ...]
    issues: tuple[SectionResolutionIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether every declared assignment resolved successfully."""

        return not self.issues

    @property
    def fully_covered(self) -> bool:
        """Return whether every model element has a declared assignment."""

        return not self.uncovered_element_ids

    @property
    def assignment_order(self) -> tuple[int, ...]:
        """Return source assignment indices in their preserved order."""

        return tuple(item.assignment_index for item in self.assignments)

    def for_element(
        self,
        element_id: int,
    ) -> EffectiveSectionAssignment | None:
        """Return the final assignment for an element, if it is covered."""

        target = int(element_id)
        return next(
            (
                item
                for item in self.effective_assignments
                if item.element_id == target
            ),
            None,
        )

    def require_valid(self) -> SectionResolution:
        """Return this resolution or raise its first canonical error."""

        if not self.issues:
            return self
        issue = self.issues[0]
        if issue.code in {
            "definition.material.missing",
            "definition.section.reference_missing",
        }:
            raise KeyError(issue.message)
        raise ValueError(issue.message)


def add(model: Any, material: MaterialDefinition) -> MaterialDefinition:
    """Add a material definition to a model."""
    model.materials[material.name] = material
    return material


def assign(
    model: Any,
    material: str | MaterialDefinition,
    element_set: str | ElementSet,
    section_type: str = "solid",
    **properties: Any,
) -> SectionAssignment:
    """Assign material and section properties to an element set."""
    material_name = material.name if isinstance(material, MaterialDefinition) else str(material)
    element_set_name = element_set.name if isinstance(element_set, ElementSet) else str(element_set)
    section = SectionAssignment(
        element_set_name,
        material_name,
        section_type,
        dict(properties),
    )
    model.sections.append(section)
    return section


def apply_sections(model: Any) -> None:
    """Apply the same pure last-match resolution used by validation."""

    element_lookup = _element_lookup(model)
    resolution = resolve_sections(model, element_lookup=element_lookup)
    resolution.require_valid()
    _validate_unassigned_beam_sections(model, element_lookup, resolution)
    props_snapshot = {
        element_id: (elem.props, deepcopy(elem.props))
        for element_id, elem in element_lookup.items()
    }
    metadata_snapshot = {
        key: (
            key in model.metadata,
            model.metadata.get(key),
            deepcopy(model.metadata.get(key)),
        )
        for key in _SECTION_METADATA_KEYS
    }

    try:
        section_keys = model.metadata.setdefault(_SECTION_KEYS_METADATA, {})
        original_values = model.metadata.setdefault(_SECTION_ORIGINALS_METADATA, {})
        element_identities = model.metadata.setdefault(_SECTION_IDENTITIES_METADATA, {})
        _restore_section_properties(
            element_lookup,
            section_keys,
            original_values,
            element_identities,
        )

        for effective in resolution.effective_assignments:
            element_id = effective.element_id
            elem = element_lookup[element_id]
            props = effective.applied_properties
            baseline = original_values.setdefault(element_id, {})
            _restore_tracked_keys(
                elem,
                section_keys.get(element_id, ()),
                baseline,
            )
            for key in props:
                if key not in baseline:
                    baseline[key] = (
                        key in elem.props,
                        deepcopy(elem.props.get(key)),
                    )
            elem.props.update(props)
            section_keys[element_id] = tuple(props)
            element_identities[element_id] = id(elem)
    except Exception:
        _restore_apply_sections_state(model, props_snapshot, metadata_snapshot)
        raise


def resolve_sections(
    model: Any,
    *,
    element_lookup: dict[int, Any] | None = None,
) -> SectionResolution:
    """Resolve every assignment without mutating model or element props."""

    lookup = (
        _element_lookup(model)
        if element_lookup is None
        else dict(element_lookup)
    )
    assignments: list[ResolvedSectionAssignment] = []
    effective_by_element: dict[int, EffectiveSectionAssignment] = {}
    targeted_element_ids: set[int] = set()
    issues: list[SectionResolutionIssue] = []
    missing_materials: list[str] = []
    missing_element_sets: list[str] = []
    missing_element_ids: list[int] = []
    incompatible_element_ids: list[int] = []

    for assignment_index, section in enumerate(
        getattr(model, "sections", ()),
    ):
        element_set_name = str(section.element_set)
        material_name = str(section.material)
        declared_type = str(section.section_type)
        material = getattr(model, "materials", {}).get(material_name)
        try:
            element_set = _section_element_set(model, element_set_name)
        except KeyError:
            element_set = None

        element_ids = (
            ()
            if element_set is None
            else tuple(int(value) for value in element_set.element_ids)
        )
        assignments.append(
            ResolvedSectionAssignment(
                assignment_index=assignment_index,
                element_set=element_set_name,
                material=material_name,
                section_type=declared_type,
                element_ids=element_ids,
            )
        )

        if material is None:
            _append_unique(missing_materials, material_name)
            issues.append(
                SectionResolutionIssue(
                    code="definition.material.missing",
                    message=f"material {material_name} is not defined",
                    assignment_index=assignment_index,
                    element_set=element_set_name,
                    material=material_name,
                    section_type=declared_type,
                )
            )
        if element_set is None:
            _append_unique(missing_element_sets, element_set_name)
            issues.append(
                SectionResolutionIssue(
                    code="definition.section.reference_missing",
                    message=(
                        f"element set {element_set_name} is not defined"
                    ),
                    assignment_index=assignment_index,
                    element_set=element_set_name,
                    material=material_name,
                    section_type=declared_type,
                )
            )
        if element_set is None:
            continue

        for element_id in element_ids:
            elem = lookup.get(element_id)
            if elem is None:
                _append_unique(missing_element_ids, element_id)
                issues.append(
                    SectionResolutionIssue(
                        code="definition.section.reference_missing",
                        message=f"element {element_id} is not defined",
                        assignment_index=assignment_index,
                        element_set=element_set_name,
                        material=material_name,
                        section_type=declared_type,
                        element_id=element_id,
                    )
                )
                continue
            targeted_element_ids.add(element_id)
            # Last assignment wins as a declaration too: a later invalid
            # assignment cannot leave an earlier valid assignment effective.
            effective_by_element.pop(element_id, None)
            if material is None:
                continue

            try:
                resolved = resolve_section_properties(
                    elem.type,
                    material.properties,
                    declared_type,
                    section.properties,
                    baseline_properties=_restored_properties(
                        model,
                        element_id,
                        elem,
                    ),
                )
            except SectionCompatibilityError as exc:
                _append_unique(incompatible_element_ids, element_id)
                issues.append(
                    _element_issue(
                        "definition.section.incompatible",
                        exc,
                        assignment_index,
                        element_set_name,
                        material_name,
                        declared_type,
                        element_id,
                    )
                )
                continue
            except MaterialPropertyError as exc:
                issues.append(
                    _element_issue(
                        "definition.material.invalid",
                        exc,
                        assignment_index,
                        element_set_name,
                        material_name,
                        declared_type,
                        element_id,
                    )
                )
                continue
            except (SectionPropertyError, NotImplementedError) as exc:
                if isinstance(exc, NotImplementedError):
                    _append_unique(incompatible_element_ids, element_id)
                    code = "definition.section.incompatible"
                else:
                    code = "definition.section.invalid"
                issues.append(
                    _element_issue(
                        code,
                        exc,
                        assignment_index,
                        element_set_name,
                        material_name,
                        declared_type,
                        element_id,
                    )
                )
                continue

            applied = dict(resolved.applied_properties)
            applied["material"] = material_name
            applied["section_type"] = resolved.section_type
            applied["_stress_material_signature"] = (
                "material",
                material_name,
                _freeze_signature(material.properties),
            )
            applied["_stress_section_signature"] = (
                "section",
                resolved.section_type,
                _freeze_signature(section.properties),
            )
            effective = dict(resolved.effective_properties)
            effective.update(applied)
            effective_by_element[element_id] = EffectiveSectionAssignment(
                element_id=element_id,
                assignment_index=assignment_index,
                element_set=element_set_name,
                material=material_name,
                section_type=resolved.section_type,
                applied_properties=applied,
                effective_properties=effective,
            )

    effective_assignments = tuple(
        effective_by_element[element_id]
        for element_id in lookup
        if element_id in effective_by_element
    )
    uncovered = tuple(
        element_id
        for element_id in lookup
        if element_id not in targeted_element_ids
    )
    return SectionResolution(
        assignments=tuple(assignments),
        effective_assignments=effective_assignments,
        uncovered_element_ids=uncovered,
        missing_materials=tuple(missing_materials),
        missing_element_sets=tuple(missing_element_sets),
        missing_element_ids=tuple(missing_element_ids),
        incompatible_element_ids=tuple(incompatible_element_ids),
        issues=tuple(issues),
    )


def _validate_unassigned_beam_sections(
    model: Any,
    element_lookup: dict[int, Any],
    resolution: SectionResolution,
) -> None:
    """Preserve validation for directly-authored uncovered beam elements."""

    covered = {
        item.element_id for item in resolution.effective_assignments
    }
    for element_id, elem in element_lookup.items():
        if (
            element_id in covered
            or _element_capabilities(elem.type).family != "beam"
        ):
            continue
        try:
            parse_beam2_section(
                _restored_properties(model, element_id, elem)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Element {element_id} has invalid Beam2 section: {exc}"
            ) from exc


def _element_capabilities(element_type: str) -> Any:
    """Query lazily to preserve elements → materials import compatibility."""

    from ..elements import get_element_capabilities

    return get_element_capabilities(element_type)


def _element_issue(
    code: str,
    error: Exception,
    assignment_index: int,
    element_set: str,
    material: str,
    section_type: str,
    element_id: int,
) -> SectionResolutionIssue:
    """Return a stable issue with element context."""

    if code == "definition.material.invalid":
        message = (
            f"Element {element_id} has invalid material {material}: {error}"
        )
    elif code == "definition.section.incompatible":
        message = f"Element {element_id} has incompatible section: {error}"
    else:
        message = f"Element {element_id} has invalid section: {error}"
    return SectionResolutionIssue(
        code=code,
        message=message,
        assignment_index=assignment_index,
        element_set=element_set,
        material=material,
        section_type=section_type,
        element_id=element_id,
    )


def _append_unique(values: list[Any], value: Any) -> None:
    """Append a fact once while preserving discovery order."""

    if value not in values:
        values.append(value)


def _restored_properties(model: Any, element_id: int, elem: Any) -> dict[str, Any]:
    """Return element properties with prior assignment-derived keys restored."""
    props = deepcopy(elem.props)
    section_keys = model.metadata.get(_SECTION_KEYS_METADATA, {})
    original_values = model.metadata.get(_SECTION_ORIGINALS_METADATA, {})
    element_identities = model.metadata.get(_SECTION_IDENTITIES_METADATA, {})
    keys = section_keys.get(element_id, ())
    expected_identity = element_identities.get(element_id)
    if expected_identity is not None and expected_identity != id(elem):
        return props

    baseline = original_values.get(element_id, {})
    for key in keys:
        existed, value = baseline.get(key, (False, None))
        if existed:
            props[key] = deepcopy(value)
        else:
            props.pop(key, None)
    return props


def _restore_apply_sections_state(
    model: Any,
    props_snapshot: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    metadata_snapshot: dict[str, tuple[bool, Any, Any]],
) -> None:
    """Roll back element props and section tracking metadata after a failure."""
    for original_props, values in props_snapshot.values():
        original_props.clear()
        original_props.update(deepcopy(values))

    for key, (existed, original_value, values) in metadata_snapshot.items():
        if not existed:
            model.metadata.pop(key, None)
            continue
        if isinstance(original_value, dict) and isinstance(values, dict):
            original_value.clear()
            original_value.update(deepcopy(values))
            model.metadata[key] = original_value
        else:
            model.metadata[key] = original_value


def _element_lookup(model: Any) -> dict[int, Any]:
    """Return elements by unique id for section application."""
    lookup: dict[int, Any] = {}
    for elem in model.mesh.elements:
        element_id = int(elem.id)
        if element_id in lookup:
            raise ValueError("element ids must be unique")
        lookup[element_id] = elem
    return lookup


def _restore_section_properties(
    element_lookup: dict[int, Any],
    section_keys: dict[int, tuple[str, ...]],
    original_values: dict[int, dict[str, tuple[bool, Any]]],
    element_identities: dict[int, int],
) -> None:
    """Restore element properties that a previous section application replaced."""
    for raw_element_id, keys in tuple(section_keys.items()):
        element_id = int(raw_element_id)
        elem = element_lookup.get(element_id)
        if elem is None:
            continue
        expected_identity = element_identities.get(
            raw_element_id,
            element_identities.get(element_id),
        )
        if expected_identity is not None and expected_identity != id(elem):
            continue
        baseline = original_values.get(raw_element_id, original_values.get(element_id, {}))
        _restore_tracked_keys(elem, keys, baseline)
    section_keys.clear()
    original_values.clear()
    element_identities.clear()


def _restore_tracked_keys(
    elem: Any,
    keys: tuple[str, ...],
    baseline: dict[str, tuple[bool, Any]],
) -> None:
    """Restore one element's currently derived section keys."""
    for key in keys:
        existed, value = baseline.get(key, (False, None))
        if existed:
            elem.props[key] = deepcopy(value)
        else:
            elem.props.pop(key, None)


def _section_element_set(model: Any, name: str) -> ElementSet:
    """Return a public or importer-internal element set used by a section."""
    if name in model.element_sets:
        return model.element_sets[name]
    internal_sets = model.metadata.get("_abaqus_internal_element_sets", {})
    if name in internal_sets:
        return internal_sets[name]
    raise KeyError(f"element set {name} is not defined")


def _freeze_signature(value: Any) -> Any:
    """Recursively freeze material and section data for region comparisons."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_signature(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_signature(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
