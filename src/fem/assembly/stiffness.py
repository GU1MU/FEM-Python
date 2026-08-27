"""Linear-stiffness convenience API backed by the canonical assembler."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.physics.contracts import PhysicsOperator
from fem.physics.mechanics import (
    get_mechanical_operator,
)
from fem.state import EvaluationContext, SolutionState

from .sparse import SparseAssembler


def assemble_global_stiffness(
    mesh: Any,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Assemble a dense stiffness view through the single sparse scatter."""

    return assemble_global_stiffness_sparse(mesh, strict=strict).toarray()


def assemble_global_stiffness_sparse(
    mesh: Any,
    *,
    strict: bool = True,
) -> csr_matrix:
    """Assemble all existing linear element families through SparseAssembler."""

    if type(strict) is not bool:
        raise TypeError("strict must be bool")
    operators: dict[str, PhysicsOperator] = {}
    for element in mesh.elements:
        key = str(element.type).casefold()
        if key in operators:
            continue
        operators[key] = get_mechanical_operator(mesh, element.type)
    assembler = SparseAssembler.from_displacement_mesh(
        mesh,
        operator=operators,
        require_symmetric_tangent=strict,
    )
    solution = SolutionState.zeros(assembler.dof_space)
    return assembler.assemble(
        solution,
        context=EvaluationContext(load_factor=1.0),
    ).tangent


__all__ = ["assemble_global_stiffness", "assemble_global_stiffness_sparse"]
