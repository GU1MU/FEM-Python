"""Reference elements: topology, interpolation, quadrature, and DOF shape."""

from __future__ import annotations

from .capabilities import (
    ElementCapabilityDescriptor,
    ElementCapabilityLimitation,
    ElementCapabilityRequirement,
    ElementCapabilityStatus,
)
from .beam2 import Beam2Definition
from .contracts import ElementDefinition
from .hex20 import Hex20Definition
from .hex8 import Hex8Definition
from .quad4 import Quad4Definition
from .quad8 import Quad8Definition
from .tet10 import Tet10Definition
from .tet4 import Tet4Definition
from .tri3 import Tri3Definition
from .tri6 import Tri6Definition
from .truss2 import Truss2Definition
from .registry import (
    canonical_element_type,
    get_element_definition,
    get_element_capabilities,
    registered_element_capabilities,
)

__all__ = [
    "ElementCapabilityDescriptor",
    "ElementCapabilityLimitation",
    "ElementCapabilityRequirement",
    "ElementCapabilityStatus",
    "ElementDefinition",
    "Beam2Definition",
    "Hex20Definition",
    "Hex8Definition",
    "Quad4Definition",
    "Quad8Definition",
    "Tet10Definition",
    "Tet4Definition",
    "Tri3Definition",
    "Tri6Definition",
    "canonical_element_type",
    "get_element_definition",
    "get_element_capabilities",
    "registered_element_capabilities",
]
