from __future__ import annotations
from tests.helpers.profile_sketches import (
    two_profile_sketch,
    profile_face_id,
)
import pytest
from fem.geometry import (
    ExtrudedGeometry,
)
from fem_gui.geometry_preview import build_geometry_preview


def test_selected_only_preview_omits_unselected_profile() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")

    preview = build_geometry_preview(
        ExtrudedGeometry(sketch, 2.0, (first,))
    )

    assert max(point[0] for point in preview.points) == pytest.approx(2.0)
    assert min(point[2] for point in preview.points) == pytest.approx(0.0)
    assert max(point[2] for point in preview.points) == pytest.approx(2.0)
    assert "face:side/L1" in preview.face_logical_ids
    assert "face:side/L5" not in preview.face_logical_ids
    assert preview.body_logical_id == "body:domain"
