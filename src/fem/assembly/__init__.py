"""Canonical global sparse assembly."""

from .contracts import Assembly, AssemblyResult
from .bindings import CompiledElement, LocalFieldBinding, OperatorBinding
from .sparse import SparseAssembler
from .mass import assemble_mass_matrix
from .stiffness import (
    assemble_global_stiffness,
    assemble_global_stiffness_sparse,
)

__all__ = [
    "Assembly",
    "AssemblyResult",
    "CompiledElement",
    "LocalFieldBinding",
    "OperatorBinding",
    "SparseAssembler",
    "assemble_mass_matrix",
    "assemble_global_stiffness",
    "assemble_global_stiffness_sparse",
]
