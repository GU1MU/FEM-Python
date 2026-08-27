from __future__ import annotations

import importlib.util

from fem.assembly import Assembly, SparseAssembler
from fem.results import ModelResult as LegacyModelResult
from fem.physics.mechanics import KinematicsModel, TotalLagrangianKinematics
from fem.materials import J2PlasticityMaterial, MaterialModel
from fem.problem import StaticEquilibriumProblem
from fem.results import ModelResult
from fem.state import StateManager, TransactionalStateManager


def test_arch01_v2_paths_are_canonical_and_old_combinations_are_gone() -> None:
    assert isinstance(TotalLagrangianKinematics(), KinematicsModel)
    assert isinstance(TransactionalStateManager(), StateManager)
    assert isinstance(J2PlasticityMaterial(210.0, 0.3, 1.0), MaterialModel)
    assert SparseAssembler is not None
    assert StaticEquilibriumProblem is not None
    assert Assembly is not None
    assert LegacyModelResult is ModelResult

    for retired in (
        "fem.physics.mechanics.linear",
        "fem.physics.mechanics.operators.plane_properties",
        "fem.problem.adapters.finite_strain",
        "fem.problem.adapters.total_lagrangian",
        "fem.materials.material_point",
        "fem.elements.total_lagrangian",
        "fem.elements.quad4.kinematics",
    ):
        assert importlib.util.find_spec(retired) is None
