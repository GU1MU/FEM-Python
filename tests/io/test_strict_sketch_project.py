from __future__ import annotations

from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchRectangle,
)
from fem.io.project_v3 import decode_project_v3, encode_project_v3
from fem.io.project_v1 import decode_project_v1


def _sketch() -> SketchGeometry:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 2.0, 1.0),
        SketchPoint("P4", 0.0, 1.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
    )
    return SketchGeometry("Strict", SketchPlane.xy(), points, curves)


def test_v3_round_trip_preserves_strict_curve_graph() -> None:
    recipe = _sketch()
    snapshot = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("Part", "Body"),),
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )

    payload = encode_project_v3(snapshot)
    reopened = decode_project_v3(payload)

    assert (
        payload["project"]["authoring"]["geometry"]["curves"][0]["type"]
        == "line"
    )
    assert reopened.geometry_recipe == recipe
    assert reopened.feature_history == snapshot.feature_history


def test_v3_encoding_upgrades_legacy_contours_to_curve_graph() -> None:
    recipe = SketchGeometry(
        "Legacy",
        (
            SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
            SketchCircle("cut", 1.0, 0.5, 0.25),
        ),
    )
    snapshot = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("Part", "Body"),),
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )

    payload = encode_project_v3(snapshot)
    geometry = payload["project"]["authoring"]["geometry"]

    assert "contours" not in geometry
    assert geometry["points"]
    assert geometry["curves"]
    assert decode_project_v3(payload).geometry_recipe.is_strict


def test_v1_sketch_migration_returns_a_strict_curve_graph() -> None:
    payload = {
        "schema": 1,
        "logical_topology_version": 1,
        "source": "native",
        "parts": [{"name": "Part", "body_name": "Body"}],
        "geometry": {
            "type": "SketchGeometry",
            "name": "Migrated",
            "contours": [
                {
                    "type": "rectangle",
                    "operation": "material",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 2.0,
                    "height": 1.0,
                }
            ],
        },
        "mesh_settings": None,
        "feature_history": [],
        "named_regions": [],
        "materials": [],
        "sections": [],
        "assignments": [],
        "steps": [],
    }

    migrated = decode_project_v1(payload)

    assert migrated.geometry_recipe.is_strict
    assert migrated.geometry_recipe.points
    assert migrated.geometry_recipe.curves
    assert migrated.feature_history == derive_feature_history(
        migrated.geometry_recipe
    )
