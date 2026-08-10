"""Shared GUI policy for published result fields and localized positions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fem.application.results import (
    FieldAvailability,
    FieldLocation,
    FieldPosition,
    ResultFieldId,
    ResultProvider,
    ResultVariable,
)
from fem.elements.beam_section import BeamSectionPoint


_POSITION_LABELS = {
    FieldPosition.NODE: "节点",
    FieldPosition.ELEMENT_NODAL: "节点",
    FieldPosition.SECTION_END: "截面",
}
_VARIABLE_LABELS = {
    ResultVariable.U: "位移 U",
    ResultVariable.UR: "转角 UR",
    ResultVariable.RF: "反力 RF",
    ResultVariable.RM: "反力矩 RM",
    ResultVariable.LE: "对数应变 LE",
    ResultVariable.S: "应力 S",
}
_RECTANGLE_SECTION_POINT_LABELS = {
    1: "右上",
    2: "左上",
    3: "左下",
    4: "右下",
}


def result_field_is_visible(availability: FieldAvailability) -> bool:
    """Expose supported GUI fields without collapsing beam field identities."""

    if type(availability) is not FieldAvailability:
        raise TypeError("availability must be FieldAvailability")
    field_id = availability.descriptor.field_id
    if field_id.variable is not ResultVariable.S:
        return True
    if field_id.position in {
        FieldPosition.SECTION_POINT,
        FieldPosition.SECTION_END,
    }:
        return True
    return field_id.position is FieldPosition.ELEMENT_NODAL


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
    if field_id.position is FieldPosition.SECTION_POINT:
        if section_point_labels is not None:
            label = section_point_labels.get(field_id.section_point_number)
            if label:
                return str(label)
        return f"截面点 {field_id.section_point_number}"
    return result_position_label(field_id.position)


def section_point_relative_position_label(point: BeamSectionPoint) -> str:
    """Return a rectangular corner name, preserving IDs for other shapes."""

    if type(point) is not BeamSectionPoint:
        raise TypeError("point must be a BeamSectionPoint")
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
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
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
    return field_id.position in (
        FieldPosition.SECTION_POINT,
        FieldPosition.SECTION_END,
    )


def result_field_has_section_points(field_id: ResultFieldId) -> bool:
    """Return whether field rows carry explicit section-point locations."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    return field_id.position is FieldPosition.SECTION_POINT


def result_variable_label(variable: ResultVariable) -> str:
    """Return the user-facing name for one result variable."""

    if type(variable) is not ResultVariable:
        raise TypeError("variable must be ResultVariable")
    return _VARIABLE_LABELS.get(variable, variable.value)
