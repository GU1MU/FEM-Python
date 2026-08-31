"""Compiled equation constraints and enforcement algorithms."""

from .contracts import ConstraintSet
from .dirichlet import apply_dirichlet_equations

__all__ = ["ConstraintSet", "apply_dirichlet_equations"]
