from __future__ import annotations

from dataclasses import replace

import pytest

from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    SketchAngleDimension,
    BoxGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    FaceSketchBooleanStepProof,
    FaceSketchWorkplaneStrategy,
    SketchCoincidentConstraint,
    SketchConcentricConstraint,
    SketchCircle,
    SketchDistanceDimension,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchPerpendicularConstraint,
    SketchRadiusDimension,
    SketchTangentConstraint,
    SketchVerticalConstraint,
)
from fem.io.project import decode_project
from fem.io import project_v7 as _v7
from fem.io.project_v12 import ProjectV12EncodeError, encode_project_v12
from fem.io.project_v13 import (
    ProjectV13DecodeError,
    decode_project_v13,
    encode_project_v13,
)
from fem.mesh.settings import MeshSettings


def _constrained_snapshot():
    sketch = SketchGeometry(
        "约束矩形",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 0.0),
            SketchPoint("P3", 2.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
            SketchPoint("P5", 1.0, 0.5),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchCircle("C1", "P5", 0.25),
        ),
        (
            SketchCoincidentConstraint("G1", "P1", "P2", "inferred", False),
            SketchPointOnCurveConstraint("G2", "P3", "L1"),
            SketchHorizontalConstraint("G3", "L1"),
            SketchVerticalConstraint("G4", "L2"),
            SketchFixedConstraint("G5", "P1", 0.0, 0.0),
            SketchDistanceDimension("D1", "P1", "P2", 2.0, False),
            SketchRadiusDimension("D2", "C1", 0.25),
        ),
    )
    settings = MeshSettings(size=0.5)
    part = NativePart(geometry_recipe=sketch, mesh_settings=settings)
    return ProjectSnapshot(
        source_kind="native",
        source_path=None,
        parts=(part,),
        geometry_recipe=None,
        mesh_settings=None,
        feature_history=derive_feature_history(sketch),
        active_part_id=part.id,
    )


def _unconstrained_snapshot():
    source = _constrained_snapshot()
    sketch = source.parts[0].geometry_recipe
    unconstrained = SketchGeometry(
        sketch.name, sketch.plane, sketch.points, sketch.curves
    )
    part = replace(source.parts[0], geometry_recipe=unconstrained)
    return replace(source, parts=(part,), feature_history=derive_feature_history(unconstrained))


def _geometry_payload(payload: dict) -> dict:
    return payload["project"]["authoring"]["parts"][0]["geometry"]


def test_v13_round_trips_every_constraint_field() -> None:
    source = _constrained_snapshot()

    payload = encode_project_v13(source)
    reopened = decode_project_v13(payload)

    assert payload["schema"] == 13
    assert reopened.parts[0].geometry_recipe.constraints == (
        source.parts[0].geometry_recipe.constraints
    )
    assert _geometry_payload(payload)["constraints"][0] == {
        "type": "coincident",
        "id": "G1",
        "first_point_id": "P1",
        "second_point_id": "P2",
        "source": "inferred",
        "enabled": False,
    }


def test_v13_round_trips_every_advanced_constraint_wire_type() -> None:
    source = _constrained_snapshot()
    old_sketch = source.parts[0].geometry_recipe
    points = (*old_sketch.points, SketchPoint("P6", 2.0, 0.5))
    curves = (
        *old_sketch.curves,
        SketchCircle("C2", "P6", 0.25),
    )
    advanced = (
        SketchParallelConstraint("A1", "L1", "L3", enabled=False),
        SketchPerpendicularConstraint("A2", "L1", "L2", source="inferred"),
        SketchTangentConstraint("A3", "L1", "C1", 7),
        SketchEqualLengthConstraint("A4", "L1", "L3"),
        SketchEqualRadiusConstraint("A5", "C1", "C2"),
        SketchConcentricConstraint("A6", "C1", "C2"),
        SketchAngleDimension("A7", "L1", "L2", 1.25, False),
    )
    sketch = SketchGeometry(
        old_sketch.name,
        old_sketch.plane,
        points,
        curves,
        (*old_sketch.constraints, *advanced),
    )
    part = replace(source.parts[0], geometry_recipe=sketch)
    snapshot = replace(
        source,
        parts=(part,),
        feature_history=derive_feature_history(sketch),
    )

    payload = encode_project_v13(snapshot)
    reopened = decode_project_v13(payload)
    encoded = _geometry_payload(payload)["constraints"][-7:]

    assert reopened.parts[0].geometry_recipe.constraints[-7:] == advanced
    assert [item["type"] for item in encoded] == [
        "parallel", "perpendicular", "tangent", "equal_length",
        "equal_radius", "concentric", "angle",
    ]
    assert encoded[2]["branch_hint"] == 7
    assert encoded[-1]["driving"] is False


@pytest.mark.parametrize("mutation", ["unknown", "dangling", "illegal"])
def test_v13_rejects_unknown_dangling_and_illegal_constraints(mutation: str) -> None:
    payload = encode_project_v13(_constrained_snapshot())
    constraints = _geometry_payload(payload)["constraints"]
    if mutation == "unknown":
        constraints[0]["type"] = "parallel"
    elif mutation == "dangling":
        constraints[0]["first_point_id"] = "missing"
    else:
        constraints[-1]["value"] = 0.0

    with pytest.raises(ProjectV13DecodeError):
        decode_project_v13(payload)


def test_v12_reads_with_empty_constraints_and_refuses_to_drop_new_constraints() -> None:
    old_payload = encode_project_v12(_unconstrained_snapshot())
    reopened = decode_project(old_payload).snapshot

    assert reopened.parts[0].geometry_recipe.constraints == ()
    with pytest.raises(ProjectV12EncodeError, match="constraints"):
        encode_project_v12(_constrained_snapshot())


def test_v13_round_trips_constraints_in_nested_face_sketch_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _constrained_snapshot()
    sketch = source.parts[0].geometry_recipe
    face_feature = FaceSketchBooleanGeometry(
        BoxGeometry("Base", 4.0, 4.0, 2.0),
        "FSB1",
        "面草图特征",
        "face:top",
        FaceSketchWorkplaneStrategy("x"),
        sketch,
        FaceSketchBooleanOperation.CUT,
        FaceSketchBooleanDirection.INWARD,
        1.0,
        ("profile:material",),
        step_proofs=(
            FaceSketchBooleanStepProof(
                "profile:material",
                    (
                        BooleanLineageEntity("body", "body:domain", "test body"),
                        BooleanLineageEntity("face", "face:top", "test face"),
                    ),
                    (
                        BooleanLineageMapping(
                            "target", "body:domain", "body:domain", "preserved"
                        ),
                        BooleanLineageMapping(
                            "tool", "face:top", "face:top", "preserved"
                        ),
                ),
            ),
        ),
    )
    part = replace(source.parts[0], geometry_recipe=face_feature)
    nested = replace(source, parts=(part,), feature_history=derive_feature_history(face_feature))
    monkeypatch.setattr(_v7, "_authenticate_part", lambda part, *, encode: None)

    payload = encode_project_v13(nested)
    reopened = decode_project_v13(payload)

    encoded_sketch = _geometry_payload(payload)["sketch"]
    restored = reopened.parts[0].geometry_recipe
    assert encoded_sketch["constraints"]
    assert isinstance(restored, FaceSketchBooleanGeometry)
    assert restored.sketch.constraints == sketch.constraints
