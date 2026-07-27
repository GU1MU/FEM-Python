from __future__ import annotations

import builtins

import pytest

from fem.application.preprocessing import generate_fem_model
from fem.geometry import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings


def test_native_1d_preprocessing_fails_closed_before_gmsh_import(monkeypatch) -> None:
    recipe = WireGeometry(
        "Wire",
        (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0)),
        (WireMember("M1", "P1", "P2"),),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gmsh":
            raise AssertionError("Phase 1 must reject before importing gmsh")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(
        NotImplementedError,
        match="native 1D preprocessing is not enabled",
    ):
        generate_fem_model(
            recipe,
            MeshSettings(
                0.25,
                cell_shape="line",
                line_element_type="Truss2",
            ),
        )
