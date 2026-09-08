from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application.face_sketch_boolean import prepare_face_sketch_boolean
from fem.application.recipe_compiler import compile_recipe
from fem.application.session import ModelSession
from fem.application.units import UnitContext
from fem.geometry import (
    BoxGeometry,
    FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    SketchGeometry,
    SketchLine,
    SketchPoint,
    model,
)
from fem.geometry.recipe_analysis import analyze_sketch_profiles
from fem.geometry.sketch_support import resolve_face_workplane
from fem.io.project_v11 import (
    ProjectV11DecodeError,
    decode_project_v11,
    encode_project_v11,
)


def _committed_face_sketch_project() -> tuple[dict[str, object], BoxGeometry]:
    base = BoxGeometry("基础实体", 4.0, 4.0, 2.0)
    with model("project-v11-face-sketch-prepare", dimension=3) as cad:
        compiled = compile_recipe(cad, base)
        workplane = resolve_face_workplane(
            cad,
            compiled.logical_entities,
            "face:top",
        )
        sketch = SketchGeometry(
            "面草图",
            plane=workplane.plane,
            points=(
                SketchPoint("p1", -0.5, -0.5),
                SketchPoint("p2", 0.5, -0.5),
                SketchPoint("p3", 0.5, 0.5),
                SketchPoint("p4", -0.5, 0.5),
            ),
            curves=(
                SketchLine("c1", "p1", "p2"),
                SketchLine("c2", "p2", "p3"),
                SketchLine("c3", "p3", "p4"),
                SketchLine("c4", "p4", "p1"),
            ),
        )
        profile_id = analyze_sketch_profiles(sketch).profiles[0].id
        prepared = prepare_face_sketch_boolean(
            cad,
            FaceSketchBooleanGeometry(
                base,
                "FSB1",
                "拉伸合并-1",
                "face:top",
                workplane.strategy,
                sketch,
                FaceSketchBooleanOperation.FUSE,
                FaceSketchBooleanDirection.OUTWARD,
                1.0,
                (profile_id,),
            ),
        ).geometry

    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型",
        UnitContext("mm", "N", "MPa"),
        base,
        part_name="部件",
    )
    before = session.snapshot()
    session.commit_face_sketch_boolean(
        "P1",
        "body:domain",
        prepared,
        expected_session_id=before.session_id,
        expected_part_revision=before.part_revisions["P1"],
        expected_session_revision=before.session_revision,
        expected_body_recipe=base,
        expected_support_face_id="face:top",
        expected_workplane_strategy=workplane.strategy,
        expected_sketch_revision=1,
        sketch_revision=1,
        expected_preview_generation=1,
        preview_generation=1,
    )
    return encode_project_v11(session.prepare_project_save()), base


@pytest.mark.gmsh
def test_real_v11_face_sketch_round_trip_restores_undo_and_redo(
    real_gmsh: object,
) -> None:
    payload, base = _committed_face_sketch_project()
    loaded = decode_project_v11(deepcopy(payload))

    assert payload["schema"] == 11
    assert isinstance(loaded.parts[0].geometry_recipe, FaceSketchBooleanGeometry)
    assert len(loaded.face_sketch_boolean_undo_records) == 1

    reopened = ModelSession()
    reopened.replace_from_snapshot(loaded)
    committed_part = reopened.snapshot().parts[0]
    assert reopened.can_undo_face_sketch_boolean("P1")

    reopened.undo_face_sketch_boolean("P1")
    undone_part = reopened.snapshot().parts[0]
    assert undone_part.geometry_recipe == base
    assert undone_part.mesh_settings is None
    assert reopened.can_redo_face_sketch_boolean("P1")

    reopened.redo_face_sketch_boolean("P1")
    assert reopened.snapshot().parts[0] == committed_part


@pytest.mark.gmsh
@pytest.mark.parametrize("tamper", ["workplane", "proof"])
def test_real_v11_rejects_tampered_face_sketch_replay(
    tamper: str,
    real_gmsh: object,
) -> None:
    payload, _base = _committed_face_sketch_project()
    geometry = payload["project"]["authoring"]["parts"][0]["geometry"]
    if tamper == "workplane":
        geometry["workplane_strategy"]["seed_axis"] = "z"
    else:
        geometry["step_proofs"][0]["topology_mappings"] = []

    with pytest.raises(ProjectV11DecodeError, match="工作面|坐标|证明|proof"):
        decode_project_v11(payload)
