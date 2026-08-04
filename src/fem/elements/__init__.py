from __future__ import annotations

from .base import ElementKernel
from .beam_frame import (
    BEAM_DEFAULT_LOCAL_Y_REFERENCE,
    BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY,
    BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY,
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BEAM_ORIENTATION_PARALLEL_TOLERANCE,
    BeamFrame,
    BeamOrientation,
    BeamOrientationError,
    BeamOrientationInvalidError,
    BeamOrientationParallelError,
    BeamOrientationUnsupportedTargetError,
    parse_beam_orientation,
    resolve_beam_frame,
)
from .capabilities import (
    ElementCapabilityDescriptor,
    ElementCapabilityLimitation,
    ElementCapabilityRequirement,
    ElementCapabilityStatus,
)
from .registry import (
    canonical_element_type,
    get_element_capabilities,
    get_element_kernel,
    register_element_kernel,
    registered_element_capabilities,
)

__all__ = [
    "BEAM_DEFAULT_LOCAL_Y_REFERENCE",
    "BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY",
    "BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY",
    "BEAM_LOCAL_Y_REFERENCE_KEY",
    "BEAM_ORIENTATION_PARALLEL_TOLERANCE",
    "BeamFrame",
    "BeamOrientation",
    "BeamOrientationError",
    "BeamOrientationInvalidError",
    "BeamOrientationParallelError",
    "BeamOrientationUnsupportedTargetError",
    "ElementCapabilityDescriptor",
    "ElementCapabilityLimitation",
    "ElementCapabilityRequirement",
    "ElementCapabilityStatus",
    "ElementKernel",
    "canonical_element_type",
    "get_element_capabilities",
    "get_element_kernel",
    "parse_beam_orientation",
    "register_element_kernel",
    "registered_element_capabilities",
    "resolve_beam_frame",
]
