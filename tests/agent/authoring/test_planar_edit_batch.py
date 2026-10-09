from __future__ import annotations

from copy import deepcopy

import pytest

from fem.geometry import SketchArc, SketchGeometry, SketchLine, SketchPlane, SketchPoint
from fem_agent import geometry_authoring


def _sketch_with_arc_profile() -> SketchGeometry:
    return SketchGeometry(
        "Plate with arc cutout",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 10.0, 0.0),
            SketchPoint("P3", 10.0, 10.0),
            SketchPoint("P4", 0.0, 10.0),
            SketchPoint("P5", 2.0, 2.0),
            SketchPoint("P6", 3.0, 2.0),
            SketchPoint("P7", 4.0, 2.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchArc("A1", "P5", "P6", "P7", "ccw"),
            SketchLine("L5", "P7", "P5"),
        ),
    )


def test_batch_updates_curves_and_constraints_before_one_final_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sketch = _sketch_with_arc_profile()
    edits = [
        {"operation": "delete_curves", "curve_ids": ["L4"]},
        {
            "operation": "update_line",
            "line_id": "L1",
            "start": {"point_id": "P2"},
            "end": {"point_id": "P1"},
        },
        {"operation": "update_arc", "arc_id": "A1", "orientation": "cw"},
        {
            "operation": "add_constraint",
            "constraint": {"kind": "horizontal", "line_id": "L1", "enabled": False},
        },
        {
            "operation": "replace_constraint",
            "constraint_id": "K1",
            "constraint": {"kind": "vertical", "line_id": "L2", "enabled": False},
        },
        {"operation": "delete_constraints", "constraint_ids": ["K1"]},
        {
            "operation": "add_line",
            "start": {"point_id": "P4"},
            "end": {"point_id": "P1"},
        },
    ]
    submitted_edits = deepcopy(edits)
    validated_sketches = []
    build_draft = geometry_authoring._planar_edit_draft

    def record_validation(candidate: SketchGeometry):
        validated_sketches.append(candidate)
        return build_draft(candidate)

    monkeypatch.setattr(geometry_authoring, "_planar_edit_draft", record_validation)

    draft = geometry_authoring.apply_planar_edit_batch(sketch, edits=edits)

    assert validated_sketches == [draft.recipe]
    assert draft.proof.exact is True
    assert draft.recipe.curve("L1").start_point_id == "P2"
    assert draft.recipe.curve("L1").end_point_id == "P1"
    assert draft.recipe.curve("A1").orientation == "cw"
    assert draft.recipe.constraints == ()
    assert {curve.id for curve in draft.recipe.curves} == {
        "L1", "L2", "L3", "L4", "L5", "A1"
    }
    assert draft.recipe.curve("L4").start_point_id == "P4"
    assert draft.recipe.curve("L4").end_point_id == "P1"
    assert sketch.curve("L1").start_point_id == "P1"
    assert sketch.curve("A1").orientation == "ccw"
    assert edits == submitted_edits


@pytest.mark.parametrize(
    ("edit", "error_type", "message"),
    (
        ({"operation": "add_line"}, KeyError, "'start'"),
        ({"operation": "add_line", "start": {}}, KeyError, "'end'"),
        ({"operation": "add_arc", "start": {}}, KeyError, "'center'"),
        (
            {"operation": "add_arc", "start": {}, "center": {}},
            KeyError,
            "'end'",
        ),
        (
            {"operation": "add_arc", "start": {}, "center": {}, "end": {}},
            KeyError,
            "'orientation'",
        ),
        (
            {
                "operation": "add_line",
                "start": {"point_id": "P1"},
                "end": {"point_id": "P2"},
                "unexpected": True,
            },
            ValueError,
            "edits[1] has unsupported fields",
        ),
        ({"operation": "batch", "edits": []}, ValueError, "edits[1] uses an unsupported operation"),
        ({"operation": "add_path_slot"}, ValueError, "edits[1] uses an unsupported operation"),
        ({"operation": "translate"}, ValueError, "edits[1] uses an unsupported operation"),
    ),
)
def test_batch_reports_original_positional_and_unsupported_operation_errors(
    edit: dict[str, object], error_type: type[Exception], message: str
) -> None:
    sketch = _sketch_with_arc_profile()
    edits = [{"operation": "update_point", "point_id": "P3", "x": 11.0}, edit]
    submitted_edits = deepcopy(edits)

    with pytest.raises(error_type) as raised:
        geometry_authoring.apply_planar_edit_batch(sketch, edits=edits)

    assert str(raised.value) == message
    assert sketch.point("P3").u == 10.0
    assert edits == submitted_edits


@pytest.mark.parametrize(
    ("operation", "handler_name"),
    (
        ("update_line", "_update_planar_line_sketch"),
        ("update_arc", "_update_planar_arc_sketch"),
        ("delete_curves", "_delete_planar_curves_sketch"),
        ("add_constraint", "_add_planar_constraint_sketch"),
        ("replace_constraint", "_replace_planar_constraint_sketch"),
        ("delete_constraints", "_delete_planar_constraints_sketch"),
    ),
)
def test_batch_keyword_errors_keep_the_original_handler_name(
    operation: str, handler_name: str
) -> None:
    with pytest.raises(TypeError) as raised:
        geometry_authoring.apply_planar_edit_batch(
            _sketch_with_arc_profile(),
            edits=[{"operation": operation, "unexpected": True}],
        )

    assert str(raised.value) == (
        f"{handler_name}() got an unexpected keyword argument 'unexpected'"
    )
