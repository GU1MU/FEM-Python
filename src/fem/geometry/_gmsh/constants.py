"""Numerical tolerances shared by Gmsh geometry operations."""

_PLANAR_TOLERANCE = 1.0e-10
_LOOP_WINDING_REFINEMENTS = tuple(2**power for power in range(3, 14))
# OpenCASCADE expands Gmsh bounding boxes by this numerical safety gap.
_OCC_BOUNDING_BOX_PADDING = 1.0e-7
