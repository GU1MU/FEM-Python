from __future__ import annotations

from copy import deepcopy

import pytest

from fem import geometry
from fem.application import (
    ModelSession,
    NamedRegion,
    NativePart,
    prepare_solid_body_boolean,
    prepare_strict_body_recipe_preview,
)
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    BooleanBodyContext,
    BooleanGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    MultiBodyGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SolidBody,
    BooleanLineageResolutionError,
    next_boolean_feature_id,
    install_proven_body_boolean,
    undo_solid_body_feature,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.io.project_v5 import (
    ProjectV5DecodeError,
    decode_project_v5,
    encode_project_v5,
)
from fem.mesh.settings import MeshSettings
from fem.mesh.settings import LocalMeshControl
from fem_gui.geometry_preview import build_strict_body_boolean_preview


def _cut_geometry() -> MultiBodyGeometry:
    return MultiBodyGeometry(
        "Boolean Geometry",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Box", 2.0, 2.0, 2.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    CylinderGeometry("Cylinder", 0.4, 2.0),
                    1.0,
                    1.0,
                    0.0,
                ),
            ),
        ),
    )


def test_boolean_undo_restores_lossy_reference_transitions_exactly() -> None:
    source = MultiBodyGeometry(
        "Lossy Mapping",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 1.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                BoxGeometry("Tool", 1.0, 1.0, 1.0),
            ),
        ),
    )
    combined_face = "face:boolean/BF1/combined/top/top"
    context = BooleanBodyContext(
        "BF1",
        "B1",
        "B2",
        "Tool",
        (
            BooleanLineageEntity(
                "face",
                combined_face,
                "boolean.combined",
            ),
            BooleanLineageEntity(
                "body",
                "body:domain",
                "boolean.result",
            ),
        ),
        (
            BooleanLineageMapping(
                "target",
                "body:domain",
                "body:domain",
                "preserved",
            ),
            BooleanLineageMapping(
                "target",
                "face:top",
                combined_face,
                "derived",
            ),
            BooleanLineageMapping(
                "tool",
                "body:domain",
                "body:domain",
                "derived",
            ),
            BooleanLineageMapping(
                "tool",
                "face:top",
                combined_face,
                "derived",
            ),
        ),
    )
    boolean = BooleanGeometry(
        "Target",
        "fuse",
        source.body("B1").recipe,
        source.body("B2").recipe,
        context,
    )
    committed = install_proven_body_boolean(source, boolean)
    original_regions = (
        NamedRegion(
            "TargetTop",
            (LogicalEntityRef("face:B1/top"),),
        ),
        NamedRegion(
            "ToolTop",
            (LogicalEntityRef("face:B2/top"),),
        ),
        NamedRegion(
            "ToolBottom",
            (LogicalEntityRef("face:B2/bottom"),),
        ),
        NamedRegion(
            "ToolBody",
            (LogicalEntityRef("body:B2"),),
        ),
    )
    original_settings = MeshSettings(
        size=0.5,
        cell_shape="tetrahedron",
        local_controls=(
            LocalMeshControl(LogicalEntityRef("face:B1/top"), 0.2),
            LocalMeshControl(LogicalEntityRef("face:B2/bottom"), 0.15),
        ),
    )
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=original_settings,
    )
    session.replace_named_regions(original_regions)

    session.replace_native_geometry_inputs((NativePart(),), committed)
    transitioned = session.snapshot()
    assert set(transitioned.named_regions) == {"TargetTop", "ToolTop"}
    assert transitioned.mesh_settings.local_controls == (
        LocalMeshControl(
            LogicalEntityRef(f"face:B1/{combined_face.split(':', 1)[1]}"),
            0.2,
        ),
    )

    restored = undo_solid_body_feature(committed, "B1")
    session.replace_native_geometry_inputs((NativePart(),), restored)
    undone = session.snapshot()

    assert tuple(undone.named_regions.values()) == original_regions
    assert undone.mesh_settings == original_settings


@pytest.mark.gmsh
def test_strict_body_cut_proves_and_recompiles_complete_lineage(
    real_gmsh,
) -> None:
    del real_gmsh
    source = _cut_geometry()

    with geometry.model("strict-body-cut-proof", dimension=3) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "cut",
        )

    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("face:B1/left"),
                    0.2,
                ),
                LocalMeshControl(
                    LogicalEntityRef("face:B2/outer"),
                    0.15,
                ),
            ),
        ),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "TargetLeft",
                (LogicalEntityRef("face:B1/left"),),
            ),
            NamedRegion(
                "ToolOuter",
                (LogicalEntityRef("face:B2/outer"),),
            ),
            NamedRegion(
                "ToolBody",
                (LogicalEntityRef("body:B2"),),
            ),
        )
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        prepared.geometry,
    )
    committed = session.snapshot()

    assert tuple(body.id for body in prepared.geometry.bodies) == ("B1",)
    assert prepared.recipe.body_context is not None
    assert prepared.recipe.body_context.proven
    assert prepared.proof.generated_intersections, tuple(
        logical_id
        for logical_id in prepared.proof.logical_entities
        if logical_id.startswith("edge:")
    )
    assert all(
        entities
        for entities in prepared.proof.logical_entities.values()
    )
    assert "edge:top-left" in prepared.proof.logical_entities
    assert "point:top-front-left" in prepared.proof.logical_entities
    for mapping in prepared.proof.topology_mappings:
        source_kind = mapping.source_logical_id.split(":", 1)[0]
        target_kind = mapping.target_logical_id.split(":", 1)[0]
        if source_kind != target_kind:
            assert "/intersection/" in mapping.target_logical_id
    preview = build_strict_body_boolean_preview(
        prepared.geometry,
        prepared.preview,
    )
    assert preview.faces
    assert all(len(face) in {3, 4} for face in preview.faces)
    assert set(preview.face_logical_ids) == {
        f"face:B1/{logical_id.split(':', 1)[1]}"
        for logical_id in prepared.proof.logical_entities
        if logical_id.startswith("face:")
    }
    assert set(preview.face_body_logical_ids) == {"body:B1"}
    assert tuple(committed.named_regions) == ("TargetLeft", "ToolOuter")
    tool_result_ref = committed.named_regions["ToolOuter"].references[0]
    assert tool_result_ref.logical_id.startswith(
        "face:B1/boolean/BF1/tool/outer"
    )
    assert set(committed.mesh_settings.local_controls) == {
        LocalMeshControl(LogicalEntityRef("face:B1/left"), 0.2),
        LocalMeshControl(tool_result_ref, 0.15),
    }
    saved_snapshot = session.prepare_project_save().snapshot
    payload = encode_project_v5(saved_snapshot)
    reopened_snapshot = decode_project_v5(payload)
    assert reopened_snapshot == saved_snapshot
    reopened_session = ModelSession()
    reopened_session.replace_from_snapshot(reopened_snapshot)

    restored = undo_solid_body_feature(
        reopened_snapshot.geometry_recipe,
        "B1",
    )
    reopened_session.replace_native_geometry_inputs(
        (NativePart(),),
        restored,
    )
    undone = reopened_session.snapshot()
    assert undone.named_regions["ToolOuter"].references == (
        LogicalEntityRef("face:B2/outer"),
    )
    assert undone.named_regions["ToolBody"].references == (
        LogicalEntityRef("body:B2"),
    )
    assert set(undone.mesh_settings.local_controls) == {
        LocalMeshControl(LogicalEntityRef("face:B1/left"), 0.2),
        LocalMeshControl(LogicalEntityRef("face:B2/outer"), 0.15),
    }

    with geometry.model("strict-body-cut-recompile", dimension=3) as cad:
        compiled = compile_recipe(cad, prepared.geometry)

    with geometry.model("strict-body-cut-reopen-preview", dimension=3) as cad:
        reopened_occ_preview = prepare_strict_body_recipe_preview(
            cad,
            "B1",
            prepared.recipe,
        )

    assert len(compiled.domain) == 1
    assert set(compiled.logical_entities) == {
        entity.logical_id
        for entity in describe_recipe_topology(prepared.geometry).entities
        if entity.selectable
    }
    reopened_preview = build_strict_body_boolean_preview(
        prepared.geometry,
        reopened_occ_preview,
    )
    assert set(reopened_preview.face_logical_ids) == set(
        preview.face_logical_ids
    )

    broken = deepcopy(payload)
    context = broken["project"]["authoring"]["geometry"]["bodies"][0][
        "recipe"
    ]["body_context"]
    context["target_body_id"] = "B9"
    with pytest.raises(
        ProjectV5DecodeError,
        match=r"bodies\[0\].*body_context\.target_body_id",
    ):
        decode_project_v5(broken)


@pytest.mark.gmsh
def test_strict_body_fuse_consumes_tool_and_preserves_target_id(
    real_gmsh,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Fuse Geometry",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 2.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("Tool", 2.0, 1.0, 1.0),
                    1.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )

    with geometry.model("strict-body-fuse-proof", dimension=3) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "fuse",
        )

    assert tuple(body.id for body in prepared.geometry.bodies) == ("B1",)
    assert prepared.geometry.body("B1").name == "Target"
    context = prepared.recipe.body_context
    assert context is not None
    assert context.target_body_id == "B1"
    assert context.tool_body_id == "B2"
    assert context.tool_body_name == "Tool"

    restored = undo_solid_body_feature(prepared.geometry, "B1")
    assert tuple(body.id for body in restored.bodies) == ("B1", "B2")
    assert restored.retired_boolean_feature_ids == ("BF1",)
    assert next_boolean_feature_id(restored) == "BF2"


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("target", "tool", "operation"),
    (
        (
            BoxGeometry("Target", 2.0, 2.0, 2.0),
            MovedGeometry(
                RotatedGeometry(
                    BoxGeometry("Tool", 1.0, 1.0, 1.0),
                    "z",
                    30.0,
                ),
                1.8,
                0.5,
                0.5,
            ),
            "fuse",
        ),
        (
            ExtrudedGeometry(
                RectangleGeometry("Target", 2.0, 2.0),
                2.0,
            ),
            MovedGeometry(
                ExtrudedGeometry(
                    DiskGeometry("Tool", 0.4),
                    2.0,
                ),
                1.0,
                1.0,
                0.0,
            ),
            "cut",
        ),
        (
            BoxGeometry("Target", 1.0, 1.0, 1.0),
            MovedGeometry(
                CylinderGeometry("Tool", 0.35, 1.0),
                1.0,
                0.5,
                0.0,
            ),
            "fuse",
        ),
    ),
)
def test_strict_body_boolean_matrix_replays_exactly(
    real_gmsh,
    target,
    tool,
    operation,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Matrix",
        (
            SolidBody("B1", "Target", target),
            SolidBody("B2", "Tool", tool),
        ),
    )
    with geometry.model(
        f"strict-body-{operation}-matrix",
        dimension=3,
    ) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            operation,
        )
    with geometry.model(
        f"strict-body-{operation}-matrix-replay",
        dimension=3,
    ) as cad:
        compiled = compile_recipe(cad, prepared.geometry)
    assert len(compiled.domain) == 1


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("operation", "tool", "diagnostic"),
    (
        (
            "cut",
            BoxGeometry("Tool", 3.0, 3.0, 3.0),
            "boolean.result.empty",
        ),
        (
            "fuse",
            MovedGeometry(
                BoxGeometry("Tool", 1.0, 1.0, 1.0),
                5.0,
                0.0,
                0.0,
            ),
            "boolean.result.volume-count",
        ),
    ),
)
def test_strict_body_boolean_rejects_empty_or_split_result(
    real_gmsh,
    operation,
    tool,
    diagnostic,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Rejected",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 1.0, 1.0, 1.0),
            ),
            SolidBody("B2", "Tool", tool),
        ),
    )
    with geometry.model(
        f"strict-body-{operation}-rejected",
        dimension=3,
    ) as cad:
        with pytest.raises(
            BooleanLineageResolutionError,
            match=diagnostic,
        ):
            prepare_solid_body_boolean(
                cad,
                source,
                "B1",
                "B2",
                operation,
            )


@pytest.mark.gmsh
def test_strict_body_boolean_rejects_result_touching_unaffected_body(
    real_gmsh,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Three Bodies",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 2.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("Tool", 1.0, 1.0, 1.0),
                    1.5,
                    0.0,
                    0.0,
                ),
            ),
            SolidBody(
                "B3",
                "Unaffected",
                MovedGeometry(
                    BoxGeometry("Unaffected", 1.0, 1.0, 1.0),
                    2.5,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry.model("strict-body-unaffected", dimension=3) as cad:
        with pytest.raises(
            BooleanLineageResolutionError,
            match="boolean.unaffected.overlap",
        ):
            prepare_solid_body_boolean(
                cad,
                source,
                "B1",
                "B2",
                "fuse",
            )


@pytest.mark.gmsh
def test_strict_body_boolean_preserves_disjoint_unaffected_body(
    real_gmsh,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Three Bodies",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 2.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("Tool", 1.0, 1.0, 1.0),
                    1.5,
                    0.0,
                    0.0,
                ),
            ),
            SolidBody(
                "B3",
                "Unaffected",
                MovedGeometry(
                    BoxGeometry("Unaffected", 1.0, 1.0, 1.0),
                    5.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry.model(
        "strict-body-unaffected-preserved",
        dimension=3,
    ) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "fuse",
        )

    assert tuple(body.id for body in prepared.geometry.bodies) == ("B1", "B3")
    assert prepared.geometry.body("B3") == source.body("B3")


@pytest.mark.gmsh
def test_strict_body_boolean_occ_failure_leaves_source_unchanged(
    real_gmsh,
    monkeypatch,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "Failure",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 1.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("Tool", 1.0, 1.0, 1.0),
                    0.5,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )

    def fail_occ(*_args, **_kwargs):
        raise RuntimeError("simulated OCC failure")

    with geometry.model("strict-body-occ-failure", dimension=3) as cad:
        monkeypatch.setattr(cad, "fuse", fail_occ)
        with pytest.raises(RuntimeError, match="simulated OCC failure"):
            prepare_solid_body_boolean(
                cad,
                source,
                "B1",
                "B2",
                "fuse",
            )

    assert tuple(body.id for body in source.bodies) == ("B1", "B2")
