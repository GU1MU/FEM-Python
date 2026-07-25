"""Pure section-schema validation driven by element capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ..elements.beam_section import parse_beam2_section


BEAM_SECTION_TYPES = ("rectangle", "solid_circle", "hollow_circle")
SECTION_TYPES = ("solid", "truss", *BEAM_SECTION_TYPES)
SECTION_PRESETS = (
    "solid_plane_stress",
    "solid_plane_strain",
    "solid",
    "truss",
    *BEAM_SECTION_TYPES,
)
_SECTION_PRESET_ELEMENT_TYPES = {
    "solid_plane_stress": "CPS3",
    "solid_plane_strain": "CPE3",
    "solid": "Tet4",
    "truss": "Truss2",
    "rectangle": "Beam2",
    "solid_circle": "Beam2",
    "hollow_circle": "Beam2",
}
_SECTION_PRESETS_BY_ELEMENT_FAMILY = {
    "plane_continuum": (
        "solid_plane_stress",
        "solid_plane_strain",
    ),
    "solid_continuum": ("solid",),
    "truss": ("truss",),
    "beam": BEAM_SECTION_TYPES,
}
_BEAM_DIMENSION_FIELDS = (
    "radius",
    "outer_radius",
    "inner_radius",
    "height",
    "width",
)


class SectionSchemaError(ValueError):
    """Base class for invalid known section semantics."""


class MaterialPropertyError(SectionSchemaError):
    """A material does not satisfy its element-family requirements."""


class SectionPropertyError(SectionSchemaError):
    """A section has missing or invalid effective properties."""


class SectionCompatibilityError(SectionSchemaError):
    """A known section type is incompatible with an element family."""


@dataclass(frozen=True, slots=True)
class ResolvedSectionProperties:
    """Owned effective properties for one element/section combination."""

    element_type: str
    element_family: str
    section_family: str
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


def element_section_family(element_type: str) -> str:
    """Return the catalog section family accepted by one element type."""

    capabilities = _element_capabilities(element_type)
    if len(capabilities.section_families) != 1:
        raise SectionCompatibilityError(
            f"element type {element_type!r} has no unique section family"
        )
    return capabilities.section_families[0]


def section_family(section_type: str) -> str:
    """Return the catalog family for one concrete section type."""

    normalized = _normalized_section_type(section_type)
    if normalized == "solid":
        return "solid"
    if normalized == "truss":
        return "truss"
    if normalized in BEAM_SECTION_TYPES:
        return "beam"
    raise SectionPropertyError(
        f"unsupported section type {section_type!r}; expected one of "
        + ", ".join(SECTION_TYPES)
    )


def section_type_for_preset(preset: str) -> str:
    """Return the existing DTO section type represented by one preset."""

    normalized = str(preset).strip().casefold()
    if normalized not in _SECTION_PRESET_ELEMENT_TYPES:
        raise SectionPropertyError(
            f"unsupported section preset {preset!r}; expected one of "
            + ", ".join(SECTION_PRESETS)
        )
    if normalized.startswith("solid_plane_") or normalized == "solid":
        return "solid"
    return normalized


def section_presets_for_element_family(
    element_family: str,
) -> tuple[str, ...]:
    """Return authorable presets defined by the single section schema."""

    try:
        return _SECTION_PRESETS_BY_ELEMENT_FAMILY[
            str(element_family).strip().casefold()
        ]
    except KeyError as exc:
        raise SectionCompatibilityError(
            f"unsupported element family {element_family!r}"
        ) from exc


def resolve_section_preset_properties(
    preset: str,
    material_properties: Mapping[str, Any],
    section_properties: Mapping[str, Any],
) -> ResolvedSectionProperties:
    """Validate GUI/application authoring data through the domain schema."""

    normalized = str(preset).strip().casefold()
    try:
        element_type = _SECTION_PRESET_ELEMENT_TYPES[normalized]
    except KeyError as exc:
        raise SectionPropertyError(
            f"unsupported section preset {preset!r}; expected one of "
            + ", ".join(SECTION_PRESETS)
        ) from exc
    properties = deepcopy(dict(section_properties))
    if normalized == "solid_plane_stress":
        properties["plane_type"] = "stress"
    elif normalized == "solid_plane_strain":
        properties["plane_type"] = "strain"
    return resolve_section_properties(
        element_type,
        material_properties,
        section_type_for_preset(normalized),
        properties,
    )


def validate_material_properties(
    element_family: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    """Return owned material data valid for one element family."""

    if not isinstance(properties, Mapping):
        raise MaterialPropertyError("material properties must be a mapping")
    if element_family not in {
        "plane_continuum",
        "solid_continuum",
        "truss",
        "beam",
    }:
        raise MaterialPropertyError(
            f"unsupported element family {element_family!r}"
        )

    validated = deepcopy(dict(properties))
    validated["E"] = _required_positive(
        validated,
        "E",
        MaterialPropertyError,
        "material",
    )
    if element_family != "truss":
        validated["nu"] = _required_poisson_ratio(validated)
    if "rho" in validated:
        validated["rho"] = _nonnegative_finite(
            validated["rho"],
            "rho",
            MaterialPropertyError,
            "material",
        )
    return validated


def resolve_section_properties(
    element_type: str,
    material_properties: Mapping[str, Any],
    section_type: str,
    section_properties: Mapping[str, Any],
    *,
    baseline_properties: Mapping[str, Any] | None = None,
) -> ResolvedSectionProperties:
    """Resolve and validate one assignment without mutating its inputs."""

    capabilities = _element_capabilities(element_type)
    element_family = capabilities.family
    baseline = deepcopy(dict(baseline_properties or {}))
    if not isinstance(section_properties, Mapping):
        raise SectionPropertyError("section properties must be a mapping")
    section_data = deepcopy(dict(section_properties))
    material_data = validate_material_properties(
        element_family,
        material_properties,
    )

    combined = dict(baseline)
    combined.update(material_data)
    combined.update(section_data)
    concrete_type = _effective_section_type(
        section_type,
        element_family,
        combined,
    )
    concrete_family = section_family(concrete_type)
    if concrete_family not in capabilities.section_families:
        raise SectionCompatibilityError(
            f"section type {concrete_type!r} is incompatible with "
            f"element type {capabilities.canonical_type!r} "
            f"({element_family})"
        )

    applied = dict(material_data)
    applied.update(section_data)
    applied["section_type"] = concrete_type
    effective = dict(baseline)
    effective.update(applied)

    if element_family == "plane_continuum":
        _reject_known_geometry_fields(
            effective,
            allowed={"thickness", "plane_type"},
            label="two-dimensional solid",
        )
        thickness = _positive_finite(
            effective.get("thickness", 1.0),
            "thickness",
            SectionPropertyError,
            "section",
        )
        plane_type = _plane_type(element_type, effective)
        applied["thickness"] = thickness
        applied["plane_type"] = plane_type
        effective["thickness"] = thickness
        effective["plane_type"] = plane_type
    elif element_family == "solid_continuum":
        _reject_known_geometry_fields(
            effective,
            allowed=set(),
            label="three-dimensional solid",
        )
        if "plane_type" in effective:
            raise SectionPropertyError(
                "three-dimensional solid sections do not use plane_type"
            )
    elif element_family == "truss":
        _reject_known_geometry_fields(
            effective,
            allowed={"area"},
            label="truss",
        )
        area = _required_positive(
            effective,
            "area",
            SectionPropertyError,
            "section",
        )
        applied["area"] = area
        effective["area"] = area
    elif element_family == "beam":
        effective["section_type"] = concrete_type
        try:
            parsed = parse_beam2_section(effective)
        except (KeyError, TypeError, ValueError) as exc:
            raise SectionPropertyError(str(exc)) from exc
        for name in _beam_dimensions(parsed.section_type):
            value = getattr(parsed, name)
            applied[name] = value
            effective[name] = value

    return ResolvedSectionProperties(
        element_type=capabilities.canonical_type,
        element_family=element_family,
        section_family=concrete_family,
        section_type=concrete_type,
        applied_properties=applied,
        effective_properties=effective,
    )


def _effective_section_type(
    declared_type: Any,
    element_family: str,
    combined: Mapping[str, Any],
) -> str:
    normalized = _normalized_section_type(declared_type)
    embedded = combined.get("section_type")
    embedded_type = (
        _normalized_section_type(embedded)
        if isinstance(embedded, str) and embedded.strip()
        else None
    )

    if normalized == "beam":
        if embedded_type in BEAM_SECTION_TYPES:
            return embedded_type
        raise SectionPropertyError(
            "beam section requires a concrete rectangle, solid_circle, "
            "or hollow_circle section_type"
        )
    if normalized == "plane" and element_family == "plane_continuum":
        return "solid"

    # Older programmatic assignments used SectionAssignment's historical
    # ``solid`` default for trusses.  An effective area makes that intent
    # unambiguous while newly authored definitions use ``truss`` explicitly.
    if (
        normalized == "solid"
        and element_family == "truss"
        and "area" in combined
    ):
        return "truss"
    if (
        normalized == "solid"
        and element_family == "beam"
        and embedded_type in BEAM_SECTION_TYPES
    ):
        return embedded_type
    return normalized


def _normalized_section_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionPropertyError(
            "section type must be a nonempty string"
        )
    return value.strip().casefold()


def _plane_type(
    element_type: str,
    properties: Mapping[str, Any],
) -> str:
    raw_value = properties.get("plane_type")
    if raw_value is not None:
        normalized = str(raw_value).strip().casefold()
        if normalized.startswith("stress"):
            return "stress"
        if normalized.startswith("strain"):
            return "strain"
        raise SectionPropertyError(
            f"plane_type {raw_value!r} is invalid; expected 'stress' or 'strain'"
        )

    for candidate in (element_type, properties.get("abaqus_type")):
        normalized = str(candidate or "").strip().casefold()
        if normalized.startswith("cpe"):
            return "strain"
        if normalized.startswith("cps"):
            return "stress"
    return "stress"


def _required_positive(
    properties: Mapping[str, Any],
    name: str,
    error_type: type[SectionSchemaError],
    owner: str,
) -> float:
    if name not in properties or properties[name] in (None, ""):
        raise error_type(f"{owner} is missing required property {name}")
    return _positive_finite(
        properties[name],
        name,
        error_type,
        owner,
    )


def _positive_finite(
    value: Any,
    name: str,
    error_type: type[SectionSchemaError],
    owner: str,
) -> float:
    converted = _finite_number(value, name, error_type, owner)
    if converted <= 0.0:
        raise error_type(
            f"{owner} property {name} must be finite and > 0, got {value!r}"
        )
    return converted


def _nonnegative_finite(
    value: Any,
    name: str,
    error_type: type[SectionSchemaError],
    owner: str,
) -> float:
    converted = _finite_number(value, name, error_type, owner)
    if converted < 0.0:
        raise error_type(
            f"{owner} property {name} must be finite and >= 0, got {value!r}"
        )
    return converted


def _finite_number(
    value: Any,
    name: str,
    error_type: type[SectionSchemaError],
    owner: str,
) -> float:
    if isinstance(value, bool):
        raise error_type(
            f"{owner} property {name} must be numeric, got {value!r}"
        )
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{owner} property {name} must be numeric, got {value!r}"
        ) from exc
    if not isfinite(converted):
        raise error_type(
            f"{owner} property {name} must be finite, got {value!r}"
        )
    return converted


def _required_poisson_ratio(properties: Mapping[str, Any]) -> float:
    if "nu" not in properties or properties["nu"] in (None, ""):
        raise MaterialPropertyError(
            "material is missing required property nu"
        )
    value = _finite_number(
        properties["nu"],
        "nu",
        MaterialPropertyError,
        "material",
    )
    if not -1.0 < value < 0.5:
        raise MaterialPropertyError(
            "material property nu must be finite and satisfy "
            f"-1 < nu < 0.5, got {properties['nu']!r}"
        )
    return value


def _reject_known_geometry_fields(
    properties: Mapping[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    known = {"area", "thickness", *_BEAM_DIMENSION_FIELDS}
    unsupported = sorted(
        name
        for name in known.difference(allowed)
        if name in properties and properties[name] not in (None, "")
    )
    if unsupported:
        raise SectionPropertyError(
            f"{label} section does not use "
            + ", ".join(unsupported)
        )


def _beam_dimensions(section_type: str) -> tuple[str, ...]:
    if section_type == "rectangle":
        return ("height", "width")
    if section_type == "solid_circle":
        return ("radius",)
    return ("outer_radius", "inner_radius")


def _element_capabilities(element_type: str) -> Any:
    """Query the catalog lazily to keep kernel/material imports acyclic."""

    from ..elements import get_element_capabilities

    return get_element_capabilities(element_type)


__all__ = [
    "BEAM_SECTION_TYPES",
    "MaterialPropertyError",
    "ResolvedSectionProperties",
    "SECTION_PRESETS",
    "SECTION_TYPES",
    "SectionCompatibilityError",
    "SectionPropertyError",
    "SectionSchemaError",
    "element_section_family",
    "resolve_section_properties",
    "resolve_section_preset_properties",
    "section_family",
    "section_presets_for_element_family",
    "section_type_for_preset",
    "validate_material_properties",
]
