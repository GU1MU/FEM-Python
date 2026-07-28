from __future__ import annotations

import pytest

from fem.application.definitions import NamedRegion, NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.core.model import AnalysisStep, DisplacementConstraint
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    RectangleGeometry,
)
from fem.io.project_v3 import ProjectV3EncodeError, encode_project_v3
from fem.io.project_v4 import decode_project_v4, encode_project_v4


def _surface_boundary_snapshot() -> ProjectSnapshot:
    recipe = ExtrudedGeometry(
        RectangleGeometry("BlockBase", 2.0, 1.0),
        1.0,
    )
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
        named_regions=(
            NamedRegion(
                "FixedSurface",
                (LogicalEntityRef("face:bottom"),),
            ),
        ),
        analysis_definitions=(
            AnalysisStep(
                "Load",
                boundaries=(
                    DisplacementConstraint(
                        "FixedSurface",
                        1,
                        3,
                        target_kind="surface",
                    ),
                ),
            ),
        ),
    )


def test_v4_roundtrips_typed_displacement_region_target() -> None:
    original = _surface_boundary_snapshot()

    payload = encode_project_v4(original)
    reopened = decode_project_v4(payload)

    encoded = payload["project"]["authoring"]["definitions"]["steps"][0][
        "boundaries"
    ][0]
    assert set(encoded) == {
        "target",
        "target_kind",
        "first_component",
        "last_component",
        "value",
    }
    assert encoded["target_kind"] == "surface"
    assert "node_ids" not in encoded
    assert reopened.model is None
    assert reopened.named_regions == original.named_regions
    assert (
        reopened.analysis_definitions[0].boundaries
        == original.analysis_definitions[0].boundaries
    )


def test_v3_rejects_typed_displacement_region_target_without_data_loss() -> None:
    with pytest.raises(ProjectV3EncodeError, match="target_kind"):
        encode_project_v3(_surface_boundary_snapshot())
