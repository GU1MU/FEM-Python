"""Local mechanics operators grouped by physical formulation.

The package intentionally has no ``linear`` or ``nonlinear`` subpackage.
Linearity is a composition of an analysis procedure, kinematics, and
material model; it is not an element-family namespace.
"""

from .adapter import MechanicalOperatorAdapter
from .continuum_hexahedron import Hex20LinearOperator, Hex8LinearOperator
from .continuum_quad4 import Quad4LinearOperator
from .continuum_quadrilateral import Quad8LinearOperator
from .continuum_tetrahedron import Tet10LinearOperator, Tet4LinearOperator
from .continuum_triangle import Tri3LinearOperator, Tri6LinearOperator
from .line import Beam2LinearOperator, Truss2LinearOperator
from .plane import PlaneProperties, plane_thickness

__all__ = [
    "Beam2LinearOperator",
    "Hex20LinearOperator",
    "Hex8LinearOperator",
    "MechanicalOperatorAdapter",
    "PlaneProperties",
    "Quad4LinearOperator",
    "Quad8LinearOperator",
    "Tet10LinearOperator",
    "Tet4LinearOperator",
    "Tri3LinearOperator",
    "Tri6LinearOperator",
    "Truss2LinearOperator",
    "plane_thickness",
]
