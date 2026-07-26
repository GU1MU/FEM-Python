from __future__ import annotations

from fem.application.feature_history import (
    derive_feature_history,
    derive_geometry_feature_rows,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    RectangleGeometry,
    RotatedGeometry,
)


def test_feature_history_matches_current_shallow_summary_projection() -> None:
    recipe = ExtrudedGeometry(
        RotatedGeometry(
            MovedGeometry(RectangleGeometry("Plate", 4.0, 2.0), 1.0, -2.0),
            "z",
            30.0,
        ),
        3.0,
    )

    history = derive_feature_history(recipe)

    assert tuple((record.name, record.kind) for record in history) == (
        ("Base-1", "base"),
        ("Move-1", "move"),
        ("Rotate-1", "rotate"),
        ("Extrude-1", "extrude"),
    )
    assert tuple(record.payload["summary"] for record in history) == (
        "基础体  矩形  4 × 2",
        "移动  X=1，Y=-2，Z=0",
        "旋转  Z 轴，30°",
        "拉伸  高度=3",
    )


def test_boolean_history_projects_object_chain_and_current_tool_summary() -> None:
    recipe = BooleanGeometry(
        "Cut",
        "cut",
        RectangleGeometry("Plate", 4.0, 2.0),
        MovedGeometry(DiskGeometry("Hole", 0.25), 2.0, 1.0),
    )

    history = derive_feature_history(recipe)

    assert tuple(record.name for record in history) == ("Base-1", "Cut-1")
    assert history[-1].payload == {"summary": "切除  工具体=Hole"}
    assert derive_geometry_feature_rows(recipe) == (
        "基础体  矩形  4 × 2",
        "切除  工具体=Hole",
    )


def test_feature_history_module_has_no_gui_dependency() -> None:
    import fem.application.feature_history as feature_history

    assert "fem_gui" not in feature_history.__file__
    assert (
        "fem_gui"
        not in open(
            feature_history.__file__,
            encoding="utf-8",
        ).read()
    )
