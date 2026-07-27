from __future__ import annotations

import pytest

from fem.geometry import LogicalEntityRef
from fem.mesh.settings import LocalMeshControl, MeshSettings


@pytest.mark.parametrize("element_type", ["Truss2", "Beam2"])
def test_line_mesh_settings_require_one_explicit_formulation(element_type) -> None:
    settings = MeshSettings(
        0.25,
        cell_shape="line",
        line_element_type=element_type,
    )

    assert settings.order == 1
    assert settings.cell_shape == "line"
    assert settings.line_element_type == element_type


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cell_shape": "line"},
        {"cell_shape": "line", "line_element_type": "truss2"},
        {"cell_shape": "line", "line_element_type": "B31"},
        {"cell_shape": "line", "line_element_type": "T3D2"},
        {"cell_shape": "line", "order": 2, "line_element_type": "Beam2"},
        {"cell_shape": "triangle", "line_element_type": "Truss2"},
    ),
)
def test_line_formulation_cross_field_invariants_are_strict(kwargs) -> None:
    with pytest.raises(ValueError):
        MeshSettings(0.25, **kwargs)


def test_line_settings_keep_local_control_validation_and_canonical_order() -> None:
    controls = (
        LocalMeshControl(LogicalEntityRef("edge:M2"), 0.1),
        LocalMeshControl(LogicalEntityRef("point:P2"), 0.1),
    )

    settings = MeshSettings(
        0.25,
        cell_shape="line",
        local_controls=controls,
        line_element_type="Truss2",
    )

    assert tuple(control.target.logical_id for control in settings.local_controls) == (
        "point:P2",
        "edge:M2",
    )


def test_existing_positional_mesh_settings_field_order_is_unchanged() -> None:
    settings = MeshSettings(0.5, 2, "quadrilateral", ())

    assert settings.size == 0.5
    assert settings.order == 2
    assert settings.cell_shape == "quadrilateral"
    assert settings.local_controls == ()
    assert settings.line_element_type is None
