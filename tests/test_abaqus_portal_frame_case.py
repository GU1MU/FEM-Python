from __future__ import annotations

from pathlib import Path

import numpy as np

from fem import abaqus
from fem.solvers.static_linear import solve


CASE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
    / "portal_frame_b31_wind_snow.inp"
)


def test_connected_portal_frame_imports_and_solves_with_snow_load() -> None:
    imported = abaqus.read_with_report(CASE)
    model = imported.model

    assert model.mesh.num_nodes == 117
    assert len(model.mesh.elements) == 276
    assert {element.type for element in model.mesh.elements} == {"Beam2"}
    assert len(model.node_sets["BASE_SUPPORTS"].node_ids) == 18
    assert len(model.element_sets["ROOF_SNOW"].element_ids) == 144

    snow_step = next(step for step in model.steps if step.name == "SnowLoad")
    assert len(snow_step.line_loads) == 1
    assert snow_step.line_loads[0].target == "ROOF_SNOW"
    assert snow_step.line_loads[0].vector == (0.0, 0.0, -1800.0)
    assert snow_step.line_loads[0].coordinate_system == "global"
    assert tuple(notice.code for notice in imported.notices) == (
        "abaqus.b31.euler_bernoulli_approximation",
        "abaqus.b31.shared_node_frame_approximation",
    )

    result = solve(model, "SnowLoad")

    assert np.isfinite(result.U).all()
    assert np.max(np.abs(result.U)) > 0.0
