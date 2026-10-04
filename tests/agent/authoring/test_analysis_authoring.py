from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.analysis_authoring import (
    AnalysisAuthoringError,
    ConfirmedDisplacement,
    ConfirmedLoad,
)


def test_missing_displacement_confirmation_fails_closed() -> None:
    displacement = ConfirmedDisplacement(
        "Displacement-固定端", "Step-静力", "Edge-固定端", "edge",
        1, 2, 0.0, "mm", "uniform", True,
    )
    with pytest.raises(AnalysisAuthoringError, match="not confirmed"):
        replace(displacement, confirmed=False)


def test_pressure_direction_requires_signed_magnitude() -> None:
    load = ConfirmedLoad(
        "Load-拉伸", "Step-静力", "Edge-加载端", "edge",
        "edge_pressure", None, (), -12.0, "outward_normal",
        "N/mm", "uniform", True,
    )
    assert load.magnitude == -12.0
    with pytest.raises(AnalysisAuthoringError, match="positive inward"):
        replace(load, magnitude=12.0)
