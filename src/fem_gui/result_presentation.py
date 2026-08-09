"""Shared GUI policy for published result fields and localized positions."""

from __future__ import annotations

from collections.abc import Iterable

from fem.application.results import (
    FieldAvailability,
    FieldPosition,
    ResultFieldId,
    ResultVariable,
)


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


def result_field_position_label(field_id: ResultFieldId) -> str:
    """Return the exact position label for one complete field identity."""

    if type(field_id) is not ResultFieldId:
        raise TypeError("field_id must be ResultFieldId")
    if field_id.position is FieldPosition.SECTION_POINT:
        return f"截面点 {field_id.section_point_number}"
    return result_position_label(field_id.position)


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
