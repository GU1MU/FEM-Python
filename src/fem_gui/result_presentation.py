"""Shared GUI policy for published result fields and localized positions."""

from __future__ import annotations

from collections.abc import Iterable

from fem.application.results import (
    FieldAvailability,
    FieldPosition,
    ResultVariable,
)


_POSITION_LABELS = {
    FieldPosition.NODE: "节点",
    FieldPosition.ELEMENT_NODAL: "节点",
}


def result_field_is_visible(availability: FieldAvailability) -> bool:
    """Keep only the supported nodal representation for stress in the GUI."""

    if type(availability) is not FieldAvailability:
        raise TypeError("availability must be FieldAvailability")
    field_id = availability.descriptor.field_id
    return (
        field_id.variable is not ResultVariable.S
        or field_id.position is FieldPosition.ELEMENT_NODAL
    )


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
