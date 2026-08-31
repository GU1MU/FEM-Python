"""Quad4 ownership must remain split by responsibility."""

from fem.elements.quad4 import Quad4Definition
from fem.physics.mechanics import (
    Quad4LinearOperator,
    get_mechanical_operator,
    get_recovery_service,
)


def test_quad4_reference_definition_has_no_mechanics_api():
    definition = Quad4Definition()

    assert definition.canonical_type == "Quad4"
    assert not hasattr(definition, "stiffness")
    assert not hasattr(definition, "material")


def test_quad4_mechanical_operator_is_owned_by_physics():
    operator = get_mechanical_operator(None, "CPS4")

    assert isinstance(operator, Quad4LinearOperator)
    assert operator is get_recovery_service("Quad4")


def test_non_quad4_legacy_mechanics_remains_available_outside_elements():
    assert get_recovery_service("Quad8").canonical_type == "Quad8"
