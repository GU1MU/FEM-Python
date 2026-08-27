"""Registry of concrete mechanical operators.

Reference-element identity lives in :mod:`fem.elements.registry`. This
registry owns stiffness, loading, and recovery implementations.
"""

from __future__ import annotations

from fem.elements import canonical_element_type
from fem.physics.contracts import PhysicsOperator

from .operators.adapter import MechanicalOperatorAdapter
from .operators.common import LinearElementService
from .operators.continuum_hexahedron import Hex8LinearOperator, Hex20LinearOperator
from .operators.line import Beam2LinearOperator, Truss2LinearOperator
from .operators.continuum_quadrilateral import Quad8LinearOperator
from .operators.continuum_tetrahedron import Tet4LinearOperator, Tet10LinearOperator
from .operators.continuum_triangle import Tri3LinearOperator, Tri6LinearOperator
from .operators.continuum_quad4 import Quad4LinearOperator
from .operators.continuum import ContinuumMechanicsOperator

_LINEAR_SERVICES: tuple[LinearElementService, ...] = (
    Quad4LinearOperator(),
    Quad8LinearOperator(),
    Tri3LinearOperator(),
    Tri6LinearOperator(),
    Hex8LinearOperator(),
    Hex20LinearOperator(),
    Tet4LinearOperator(),
    Tet10LinearOperator(),
    Truss2LinearOperator(),
    Beam2LinearOperator(),
)

_BY_NAME: dict[str, LinearElementService] = {}
for _operator in _LINEAR_SERVICES:
    _canonical = canonical_element_type(_operator.canonical_type)
    if _canonical != _operator.canonical_type:
        raise RuntimeError(
            f"mechanical operator {_operator.canonical_type!r} does not match "
            f"reference-element identity {_canonical!r}"
        )
    for _name in (_operator.canonical_type, *_operator.aliases):
        _key = _name.casefold()
        if _key in _BY_NAME:
            raise RuntimeError(f"duplicate mechanical operator identity {_name!r}")
        _BY_NAME[_key] = _operator


def get_recovery_service(element_type: str) -> LinearElementService:
    """Return the small-strain result/load service for one element family."""

    canonical = canonical_element_type(element_type)
    return _BY_NAME[canonical.casefold()]


def get_mechanical_operator(mesh: object, element_type: str) -> PhysicsOperator:
    """Return the canonical physics-evaluation view of one element service."""

    operator = get_recovery_service(element_type)
    if isinstance(operator, PhysicsOperator):
        return operator
    return MechanicalOperatorAdapter(mesh, operator)


__all__ = [
    "ContinuumMechanicsOperator",
    "get_mechanical_operator",
    "get_recovery_service",
]
