"""Shared GUI policy for published result fields and localized positions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Protocol

from fem.results import (
    FieldAvailability,
    FieldLocation,
    FieldPosition,
    FieldState,
    ResultFieldId,
    ResultDisplayComputation,
    ResultProvider,
    ResultRegionKey,
    ResultVariable,
    result_region_sort_key,
)


class _SectionPointView(Protocol):
    """Application-facing section-point values needed for presentation."""

    number: int
    local_y: float
    local_z: float


_POSITION_LABELS = {
    FieldPosition.NODE: "节点",
    FieldPosition.INTEGRATION_POINT: "积分点",
    FieldPosition.CENTROID: "质心",
    FieldPosition.ELEMENT_NODAL: "单元节点（未平均）",
    FieldPosition.NODE_REGION: "节点区域值（未跨区域平均）",
    FieldPosition.RESOLVED_NODAL: "区域内节点平均",
    FieldPosition.SECTION_END: "截面",
}
_VARIABLE_LABELS = {
    ResultVariable.U: "位移 U",
    ResultVariable.UR: "转角 UR",
    ResultVariable.RF: "反力 RF",
    ResultVariable.RM: "反力矩 RM",
    ResultVariable.SF: "截面力 SF",
    ResultVariable.SM: "截面矩 SM",
    ResultVariable.LE: "对数应变 LE",
    ResultVariable.E: "应变 E",
    ResultVariable.PEEQ: "塑性应变 PEEQ",
    ResultVariable.S: "应力 S",
}
_RECTANGLE_SECTION_POINT_LABELS = {
    1: "右上",
    2: "左上",
    3: "左下",
    4: "右下",
}
_DISPLAY_COMPUTATION_LABELS = {
    ResultDisplayComputation.INTEGRATION_POINT_MARKERS: "积分点标记",
    ResultDisplayComputation.CENTROID_CELLS: "单元质心值",
    ResultDisplayComputation.ELEMENT_NODAL_QUILT: "单元节点（未平均）",
    ResultDisplayComputation.NODAL_AVERAGED: "区域内节点平均",
}
_TREE_POSITION_PRIORITY = {
    # The result tree represents the physical variable.  When several
    # storage/recovery positions exist, use the same default users see in
    # the contour display: values averaged within each material region.
    FieldPosition.RESOLVED_NODAL: 0,
    FieldPosition.NODE: 0,
    FieldPosition.CENTROID: 1,
    FieldPosition.ELEMENT_NODAL: 2,
    FieldPosition.INTEGRATION_POINT: 3,
    FieldPosition.SECTION_END: 4,
    FieldPosition.SECTION_POINT: 4,
    FieldPosition.NODE_REGION: 9,
}


def result_region_display_labels(
    region_keys: Iterable[ResultRegionKey],
) -> dict[ResultRegionKey, str]:
    """Return stable, human-readable labels for result-region identities.

    The canonical JSON remains the machine identity and is intentionally not
    used as the primary GUI label.  Ordering is deterministic so labels stay
    stable across frames and across query/probe tables.
    """

    keys = tuple(region_keys)
    if any(type(key) is not ResultRegionKey for key in keys):
        raise TypeError("region_keys must contain ResultRegionKey values")
    unique = sorted(set(keys), key=result_region_sort_key)
    return {
        key: (
            f"区域 {index} · "
            f"{_result_region_signature_caption(key, role='material')} · "
            f"{_result_region_signature_caption(key, role='section')}"
        )
        for index, key in enumerate(unique, start=1)
    }


def _result_region_signature_caption(
    region_key: ResultRegionKey,
    *,
    role: str,
) -> str:
    signature = (
        region_key.material_signature
        if role == "material"
        else region_key.section_signature
    )
    fallback = "材料签名" if role == "material" else "截面签名"
    try:
        payload = json.loads(signature.canonical_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, list) or not payload:
        return fallback
    kind = payload[0]
    value = payload[1] if len(payload) > 1 else None
    if role == "material":
        if kind in {"material", "material_id"} and isinstance(
            value,
            (str, int, float),
        ):
            return f"材料 {value}"
        if kind == "effective":
            return "有效材料参数"
        return fallback
    if kind == "section":
        if value is None:
            return "截面未指定"
        return f"截面 {value}"
    return fallback


def result_field_is_visible(availability: FieldAvailability) -> bool:
    """Expose supported GUI fields without collapsing beam field identities."""

    if type(availability) is not FieldAvailability:
        raise TypeError("availability must be FieldAvailability")
    field_id = availability.descriptor.field_id
    if field_id.variable not in {
        ResultVariable.S,
        ResultVariable.E,
        ResultVariable.PEEQ,
    }:
        return True
    if field_id.section_point_number is not None:
        return field_id.position in {
            FieldPosition.SECTION_POINT,
            FieldPosition.INTEGRATION_POINT,
        }
    if field_id.position is FieldPosition.SECTION_END:
        return True
    return field_id.position in {
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
        FieldPosition.RESOLVED_NODAL,
    }


def visible_result_fields(
    fields: Iterable[FieldAvailability],
) -> tuple[FieldAvailability, ...]:
    """Return GUI-visible fields without changing their catalog order."""

    return tuple(
        availability
        for availability in fields
        if result_field_is_visible(availability)
    )


def result_position_label(position: FieldPosition) -> str:
    """Return the user-facing location name for one result position."""

    if type(position) is not FieldPosition:
        raise TypeError("position must be FieldPosition")
    return _POSITION_LABELS.get(position, position.value)


def result_field_position_label(
    field_id: ResultFieldId,
    *,
    section_point_labels: Mapping[int, str] | None = None,
) -> str:
    """Return the exact position label for one complete field identity."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    if field_id.section_point_number is not None:
        if section_point_labels is not None:
            label = section_point_labels.get(field_id.section_point_number)
            if label:
                return str(label)
        return f"截面点 {field_id.section_point_number}"
    return result_position_label(field_id.position)


def result_display_computation_label(
    computation: ResultDisplayComputation,
) -> str | None:
    """Return a concise user label for an explicit render computation."""

    if type(computation) is not ResultDisplayComputation:
        raise TypeError(
            "computation must be ResultDisplayComputation"
        )
    return _DISPLAY_COMPUTATION_LABELS.get(computation)


def section_point_relative_position_label(point: _SectionPointView) -> str:
    """Return a rectangular corner name, preserving IDs for other shapes."""

    if point.local_y != 0.0 and point.local_z != 0.0:
        horizontal = "右" if point.local_y > 0.0 else "左"
        vertical = "上" if point.local_z > 0.0 else "下"
        return f"{horizontal}{vertical}"
    return f"截面点 {point.number}"


def section_point_labels_from_locations(
    locations: Iterable[FieldLocation | None],
) -> dict[int, str]:
    """Infer rectangular corner labels from materialized field locations."""

    points = tuple(
        location.section_point
        for location in locations
        if location is not None and location.section_point is not None
    )
    if not points or any(
        point.local_y == 0.0 or point.local_z == 0.0 for point in points
    ):
        return {}
    return dict(_RECTANGLE_SECTION_POINT_LABELS)


def result_provider_section_point_labels(
    provider: ResultProvider,
) -> dict[int, str]:
    """Return corner labels when a provider contains only rectangular beams."""

    if type(provider) is not ResultProvider:
        raise TypeError("provider must be a ResultProvider")
    model_result = provider.model_result
    if model_result is not None:
        section_types = {
            str(element.props.get("section_type", "")).strip().casefold()
            for element in model_result.model.mesh.elements
            if str(element.type).strip().casefold() == "beam2"
        }
        if section_types == {"rectangle"}:
            return dict(_RECTANGLE_SECTION_POINT_LABELS)
        if section_types:
            return {}

    locations = tuple(
        location
        for field in provider.snapshot.fields
        if field.key.request.field_id.section_point_number is not None
        for location in field.locations
    )
    labels = section_point_labels_from_locations(locations)
    if labels:
        return labels

    projection = provider.model_projection
    summaries = {} if projection is None else projection.summaries
    section_types = {
        str(
            section.get("properties", {}).get(
                "section_type",
                section.get("section_type", ""),
            )
        ).strip().casefold()
        for section in summaries.get("sections", ())
        if isinstance(section, Mapping)
    }
    if section_types == {"rectangle"}:
        return dict(_RECTANGLE_SECTION_POINT_LABELS)
    return {}


def result_field_is_beam_section(field_id: ResultFieldId) -> bool:
    """Return whether a field is one of the GUI beam section results."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    return (
        field_id.section_point_number is not None
        or field_id.position is FieldPosition.SECTION_END
    )


def result_tree_fields(
    fields: Iterable[FieldAvailability],
) -> tuple[FieldAvailability, ...]:
    """Return one concise result-tree item per user-facing variable.

    A catalog retains every exact position because the result display,
    Probe, XY data, and export paths need those identities.  The result tree
    is a variable chooser, however, so exposing ``S`` or ``E`` once for every
    position makes it look as though several different physical variables
    exist.  This projection keeps the preferred nodal-average field when it
    is available and leaves exact position selection to the display/query
    controls.

    Beam section-point fields are intentionally retained as a group: their
    section-point number is a physical location within the beam section, not
    another duplicate result-variable choice.
    """

    visible = visible_result_fields(fields)
    selected: list[tuple[int, FieldAvailability]] = []
    grouped: dict[ResultVariable, list[tuple[int, FieldAvailability]]] = {}
    beam_variables: set[ResultVariable] = set()

    for index, availability in enumerate(visible):
        field_id = availability.descriptor.field_id
        if result_field_is_beam_section(field_id):
            selected.append((index, availability))
            beam_variables.add(field_id.variable)
            continue
        grouped.setdefault(field_id.variable, []).append(
            (index, availability)
        )

    for variable, candidates in grouped.items():
        # If a beam exposes section-point stress, that grouped representation
        # is the useful tree entry; do not add a second generic S entry.
        if variable in beam_variables:
            continue
        selected.append(
            min(
                candidates,
                key=lambda item: (
                    item[1].state is FieldState.UNAVAILABLE,
                    _TREE_POSITION_PRIORITY.get(
                        item[1].descriptor.field_id.position,
                        99,
                    ),
                    item[0],
                ),
            )
        )

    selected.sort(key=lambda item: item[0])
    return tuple(availability for _index, availability in selected)


def result_field_has_section_points(field_id: ResultFieldId) -> bool:
    """Return whether field rows carry explicit section-point locations."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    return field_id.section_point_number is not None


def result_variable_label(variable: ResultVariable) -> str:
    """Return the user-facing name for one result variable."""

    if type(variable) is not ResultVariable:
        raise TypeError("variable must be ResultVariable")
    return _VARIABLE_LABELS.get(variable, variable.value)
