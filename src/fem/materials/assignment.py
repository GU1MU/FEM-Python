from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from ..core.model import ElementSet, MaterialDefinition, SectionAssignment
from ..elements.beam_section import parse_beam2_section


_SECTION_KEYS_METADATA = "_section_property_keys_by_element"
_SECTION_ORIGINALS_METADATA = "_section_original_properties_by_element"
_SECTION_IDENTITIES_METADATA = "_section_property_element_identity_by_element"
_SECTION_METADATA_KEYS = (
    _SECTION_KEYS_METADATA,
    _SECTION_ORIGINALS_METADATA,
    _SECTION_IDENTITIES_METADATA,
)


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
    """Copy assigned material and section data onto element props."""
    element_lookup = _element_lookup(model)
    resolved_sections = _resolve_sections(model, element_lookup)
    _validate_effective_beam2_sections(model, element_lookup, resolved_sections)
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

        for element_set, props in resolved_sections:
            for element_id in element_set.element_ids:
                elem = element_lookup[element_id]
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


def _resolve_sections(
    model: Any,
    element_lookup: dict[int, Any],
) -> list[tuple[ElementSet, dict[str, Any]]]:
    """Resolve and validate every section without mutating model state."""
    resolved_sections: list[tuple[ElementSet, dict[str, Any]]] = []
    for section in model.sections:
        if section.material not in model.materials:
            raise KeyError(f"material {section.material} is not defined")
        element_set = _section_element_set(model, section.element_set)

        props = dict(model.materials[section.material].properties)
        props.update(section.properties)
        props["material"] = section.material
        props["section_type"] = section.section_type
        props["_stress_material_signature"] = (
            "material",
            section.material,
            _freeze_signature(model.materials[section.material].properties),
        )
        props["_stress_section_signature"] = (
            "section",
            section.section_type,
            _freeze_signature(section.properties),
        )
        for element_id in element_set.element_ids:
            if element_id not in element_lookup:
                raise KeyError(f"element {element_id} is not defined")
        resolved_sections.append((element_set, props))
    return resolved_sections


def _validate_effective_beam2_sections(
    model: Any,
    element_lookup: dict[int, Any],
    resolved_sections: list[tuple[ElementSet, dict[str, Any]]],
) -> None:
    """Validate the final Beam2 properties without mutating model state."""
    effective = {
        element_id: _restored_properties(model, element_id, elem)
        for element_id, elem in element_lookup.items()
    }
    last_assignment: dict[int, dict[str, Any]] = {}
    for element_set, props in resolved_sections:
        for element_id in element_set.element_ids:
            last_assignment[element_id] = props
    for element_id, props in last_assignment.items():
        effective[element_id].update(props)

    for element_id, elem in element_lookup.items():
        if str(elem.type).casefold() != "beam2":
            continue
        try:
            parse_beam2_section(effective[element_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Element {element_id} has invalid Beam2 section: {exc}"
            ) from exc


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
