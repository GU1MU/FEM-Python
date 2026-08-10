from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from fem import geometry
from fem.application import (
    ModelSession,
    NamedRegion,
    NativePart,
    TransitionEffect,
    next_planar_boolean_feature_id,
    prepare_planar_boolean,
    prepare_strict_body_recipe_preview,
    prepare_strict_planar_recipe_preview,
)
from fem.application.native_regions import RecipeRegionSelector
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    PlanarBooleanLineageResolutionError,
    RectangleGeometry,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.io.project_v5 import ProjectV5DecodeError, decode_project_v5
from fem.io.project_v6 import (
    ProjectV6DecodeError,
    decode_project_v6,
    encode_project_v6,
)
from fem.mesh.settings import MeshSettings
from fem_gui.geometry_preview import build_strict_planar_boolean_preview
from tests.geometry.test_profile_extrusion import (
    profile_face_id,
    two_profile_sketch,
)


def _rectangle_tool(
    name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[SketchGeometry, tuple[str, ...]]:
    sketch = SketchGeometry(
        name,
        SketchPlane.xy(),
        (
            SketchPoint("P1", x0, y0),
            SketchPoint("P2", x1, y0),
            SketchPoint("P3", x1, y1),
            SketchPoint("P4", x0, y1),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )
    return sketch, _profile_face_ids(sketch)


def _circle_tool(
    name: str,
    circles: tuple[tuple[float, float, float], ...],
) -> tuple[SketchGeometry, tuple[str, ...]]:
    points = tuple(
        SketchPoint(f"P{index}", x, y)
        for index, (x, y, _radius) in enumerate(circles, start=1)
    )
    curves = tuple(
        SketchCircle(f"C{index}", f"P{index}", radius)
        for index, (_x, _y, radius) in enumerate(circles, start=1)
    )
    sketch = SketchGeometry(name, SketchPlane.xy(), points, curves)
    return sketch, _profile_face_ids(sketch)


def _profile_face_ids(sketch: SketchGeometry) -> tuple[str, ...]:
    from fem.geometry import analyze_sketch_profiles

    return tuple(
        f"face:{profile.id}" for profile in analyze_sketch_profiles(sketch).profiles
    )


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("case_name", "tool_factory"),
    (
        (
            "contained-hole",
            lambda: _circle_tool("Hole", ((2.0, 1.5, 0.5),)),
        ),
        (
            "edge-notch",
            lambda: _circle_tool("Notch", ((4.0, 1.5, 0.7),)),
        ),
        (
            "two-holes",
            lambda: _circle_tool(
                "Holes",
                ((1.25, 1.5, 0.35), (2.75, 1.5, 0.35)),
            ),
        ),
    ),
)
def test_planar_cut_matrix_has_complete_replayable_lineage(
    real_gmsh,
    case_name,
    tool_factory,
) -> None:
    del real_gmsh
    target = RectangleGeometry("Target", 4.0, 3.0)
    tool, tool_face_ids = tool_factory()

    with geometry.model(f"planar-{case_name}", dimension=2) as cad:
        prepared = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            tool,
            tool_face_ids,
            "cut",
        )

    context = prepared.geometry.planar_context
    assert context is not None and context.proven
    assert context.target_face_id == "face:domain"
    assert context.tool_face_ids == tool_face_ids
    assert next_planar_boolean_feature_id(prepared.geometry) == "PB2"
    assert prepared.proof.generated_intersections
    assert all(prepared.proof.logical_entities.values())

    preview = build_strict_planar_boolean_preview(
        prepared.geometry,
        prepared.preview,
    )
    expected_faces = {
        entity.logical_id for entity in context.result_entities if entity.kind == "face"
    }
    assert set(preview.face_logical_ids) == expected_faces
    assert all(logical_id is not None for logical_id in preview.edge_logical_ids)
    expected_points = {
        entity.logical_id
        for entity in context.result_entities
        if entity.kind == "point"
    }
    assert {
        logical_id for logical_id in preview.point_logical_ids if logical_id is not None
    } == expected_points

    with geometry.model(f"planar-{case_name}-replay", dimension=2) as cad:
        compiled = compile_recipe(cad, prepared.geometry)
    selectable = {
        entity.logical_id
        for entity in describe_recipe_topology(prepared.geometry).entities
        if entity.selectable
    }
    assert set(compiled.logical_entities) == selectable

    with geometry.model(
        f"planar-{case_name}-preview-replay",
        dimension=2,
    ) as cad:
        replay = prepare_strict_planar_recipe_preview(
            cad,
            prepared.geometry,
        )
    assert Counter(replay.face_logical_ids) == Counter(
        prepared.preview.face_logical_ids
    )
    assert Counter(replay.edge_logical_ids) == Counter(
        prepared.preview.edge_logical_ids
    )


@pytest.mark.gmsh
def test_split_cut_expands_face_reference_and_v6_undo_restores_it(
    real_gmsh,
) -> None:
    del real_gmsh
    target = RectangleGeometry("Target", 4.0, 2.0)
    tool, tool_face_ids = _rectangle_tool("Strip", 1.5, -1.0, 2.5, 3.0)
    with geometry.model("planar-split-cut", dimension=2) as cad:
        prepared = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            tool,
            tool_face_ids,
            "cut",
        )

    result_faces = tuple(
        entity.logical_id
        for entity in prepared.proof.result_entities
        if entity.kind == "face"
    )
    assert len(result_faces) == 2
    assert all("/result/" in logical_id for logical_id in result_faces)

    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        target,
        mesh_settings=MeshSettings(0.4),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "MaterialFace",
                (LogicalEntityRef("face:domain"),),
            ),
            NamedRegion(
                "SplitBoundary",
                (LogicalEntityRef("edge:top"),),
            ),
        )
    )
    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        prepared.geometry,
    )
    committed = session.snapshot()
    assert {
        reference.logical_id
        for reference in committed.named_regions["MaterialFace"].references
    } == set(result_faces)
    assert "SplitBoundary" not in committed.named_regions
    assert TransitionEffect.NAMED_REGIONS_CLEARED in delta.effects
    saved = session.prepare_project_save().snapshot
    assert len(saved.boolean_reference_undo_records) == 1
    payload = encode_project_v6(saved)
    assert payload["schema"] == 6
    reopened = decode_project_v6(payload)
    assert reopened == saved
    falsely_labeled_v5 = dict(payload)
    falsely_labeled_v5["schema"] = 5
    with pytest.raises(ProjectV5DecodeError, match="planar_context"):
        decode_project_v5(falsely_labeled_v5)
    forged = deepcopy(payload)
    forged_context = forged["project"]["authoring"]["geometry"][
        "planar_context"
    ]
    forged_context["result_entities"][0]["semantic_role"] += ".forged"
    with pytest.raises(
        ProjectV6DecodeError,
        match="topology fingerprint|proof",
    ):
        decode_project_v6(forged)
    with geometry.model("planar-split-reopened-preview", dimension=2) as cad:
        reopened_preview = prepare_strict_planar_recipe_preview(
            cad,
            reopened.geometry_recipe,
        )
    assert set(reopened_preview.face_logical_ids) == set(result_faces)
    reopened_model = generate_fem_model(
        reopened.geometry_recipe,
        reopened.mesh_settings,
    )
    assert reopened_model.mesh.elements

    reopened_session = ModelSession()
    reopened_session.replace_from_snapshot(reopened)
    reopened_session.replace_native_geometry_inputs((NativePart(),), target)
    restored = reopened_session.snapshot()
    assert restored.named_regions["MaterialFace"].references == (
        LogicalEntityRef("face:domain"),
    )
    assert restored.named_regions["SplitBoundary"].references == (
        LogicalEntityRef("edge:top"),
    )


@pytest.mark.gmsh
def test_planar_cut_preserves_unaffected_object_faces(real_gmsh) -> None:
    del real_gmsh
    target = two_profile_sketch()
    selected = profile_face_id(target, "L1")
    unaffected = profile_face_id(target, "L5")
    tool, tool_face_ids = _circle_tool("Hole", ((1.0, 0.5, 0.2),))

    with geometry.model("planar-unaffected-face", dimension=2) as cad:
        prepared = prepare_planar_boolean(
            cad,
            target,
            selected,
            tool,
            tool_face_ids,
            "cut",
        )
    with geometry.model("planar-unaffected-face-replay", dimension=2) as cad:
        compiled = compile_recipe(cad, prepared.geometry)

    assert unaffected in compiled.logical_entities
    assert len(compiled.domain) == 2
    assert any(
        mapping.source == "target"
        and mapping.source_logical_id == unaffected
        and mapping.target_logical_id == unaffected
        and mapping.relation == "preserved"
        for mapping in prepared.proof.topology_mappings
    )


@pytest.mark.gmsh
@pytest.mark.parametrize("bridge_end_x", (5.0, 6.0))
def test_planar_fuse_rejects_contact_with_unaffected_object_face(
    real_gmsh,
    bridge_end_x,
) -> None:
    del real_gmsh
    target = two_profile_sketch()
    selected = profile_face_id(target, "L1")
    tool, tool_face_ids = _rectangle_tool(
        "Bridge",
        1.0,
        0.2,
        bridge_end_x,
        0.8,
    )

    with geometry.model("planar-unaffected-fuse-contact", dimension=2) as cad:
        with pytest.raises(
            PlanarBooleanLineageResolutionError,
            match="planar-boolean.fuse.unaffected-contact",
        ):
            prepare_planar_boolean(
                cad,
                target,
                selected,
                tool,
                tool_face_ids,
                "fuse",
            )


@pytest.mark.gmsh
def test_fuse_accepts_circle_crossing_target_boundary(real_gmsh) -> None:
    del real_gmsh
    target = RectangleGeometry("Fuse Target", 3.0, 2.0)
    tool, tool_face_ids = _circle_tool(
        "Fuse Circle",
        ((3.0, 1.0, 0.6),),
    )

    with geometry.model("planar-circle-boundary-fuse", dimension=2) as cad:
        fused = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            tool,
            tool_face_ids,
            "fuse",
        )

    result_faces = tuple(
        entity for entity in fused.proof.result_entities if entity.kind == "face"
    )
    intersection_points = tuple(
        logical_id
        for logical_id in fused.proof.generated_intersections
        if logical_id.startswith("point:")
    )
    assert len(result_faces) == 1
    assert len(intersection_points) == 2
    assert fused.preview.faces

    with geometry.model("planar-circle-boundary-fuse-replay", dimension=2) as cad:
        compiled = compile_recipe(cad, fused.geometry)
    with geometry.model(
        "planar-circle-boundary-fuse-preview-replay",
        dimension=2,
    ) as cad:
        replay = prepare_strict_planar_recipe_preview(cad, fused.geometry)
    assert len(compiled.domain) == 1
    assert replay.faces


@pytest.mark.gmsh
def test_overlapping_fuse_and_extruded_cut_remain_exact_and_meshable(
    real_gmsh,
) -> None:
    del real_gmsh
    target = RectangleGeometry("Fuse Target", 3.0, 2.0)
    fuse_tool, fuse_face_ids = _rectangle_tool(
        "Fuse Tool",
        2.0,
        0.5,
        4.0,
        1.5,
    )
    with geometry.model("planar-fuse", dimension=2) as cad:
        fused = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            fuse_tool,
            fuse_face_ids,
            "fuse",
        )
    assert (
        len([entity for entity in fused.proof.result_entities if entity.kind == "face"])
        == 1
    )

    cut_tool, cut_face_ids = _circle_tool(
        "Extruded Hole",
        ((1.5, 1.0, 0.4),),
    )
    with geometry.model("planar-cut-for-extrusion", dimension=2) as cad:
        cut = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            cut_tool,
            cut_face_ids,
            "cut",
        )
    extrusion = ExtrudedGeometry(cut.geometry, 1.0)
    with geometry.model("planar-cut-extrusion", dimension=3) as cad:
        compiled = compile_recipe(cad, extrusion)
    with geometry.model("planar-cut-extrusion-preview", dimension=3) as cad:
        extrusion_preview = prepare_strict_body_recipe_preview(
            cad,
            "B1",
            extrusion,
        )
    assert len(compiled.domain) == 1
    assert compiled.region_bindings[RecipeRegionSelector.HOLE]
    assert extrusion_preview.faces
    assert all(extrusion_preview.face_logical_ids)

    planar_model = generate_fem_model(cut.geometry, MeshSettings(0.5))
    assert planar_model.mesh.nodes
    assert planar_model.mesh.elements
    model = generate_fem_model(extrusion, MeshSettings(0.5))
    assert model.mesh.nodes
    assert model.mesh.elements


@pytest.mark.gmsh
def test_planar_fuse_accepts_boundary_arc_and_point_touching_rectangle(
    real_gmsh,
) -> None:
    del real_gmsh
    target = RectangleGeometry("Target", 3.0, 2.0)
    tool = SketchGeometry(
        "Fuse Tool",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 3.0, 0.0),
            SketchPoint("P2", 3.0, 1.0),
            SketchPoint("P3", 3.0, 2.0),
            SketchPoint("P4", 1.0, -1.0),
            SketchPoint("P5", 3.0, -1.0),
            SketchPoint("P6", 3.0, 0.0),
            SketchPoint("P7", 1.0, 0.0),
        ),
        (
            SketchArc("A1", "P1", "P2", "P3", "ccw"),
            SketchLine("L1", "P3", "P1"),
            SketchLine("L2", "P4", "P5"),
            SketchLine("L3", "P5", "P6"),
            SketchLine("L4", "P6", "P7"),
            SketchLine("L5", "P7", "P4"),
        ),
    )

    with geometry.model("planar-boundary-arc-fuse", dimension=2) as cad:
        prepared = prepare_planar_boolean(
            cad,
            target,
            "face:domain",
            tool,
            _profile_face_ids(tool),
            "fuse",
        )

    assert len(
        [entity for entity in prepared.proof.result_entities if entity.kind == "face"]
    ) == 1


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("operation", "tool_factory", "diagnostic"),
    (
        (
            "cut",
            lambda: _rectangle_tool(
                "Outside",
                10.0,
                0.0,
                11.0,
                1.0,
            ),
            "planar-boolean.cut.no-op",
        ),
        (
            "cut",
            lambda: _rectangle_tool("Cover", -2.0, -2.0, 6.0, 6.0),
            "planar-boolean.result.empty",
        ),
        (
            "fuse",
            lambda: _rectangle_tool(
                "Disjoint",
                10.0,
                0.0,
                11.0,
                1.0,
            ),
            "planar-boolean.fuse.disjoint",
        ),
    ),
)
def test_planar_boolean_rejects_no_op_empty_and_disjoint_results(
    real_gmsh,
    operation,
    tool_factory,
    diagnostic,
) -> None:
    del real_gmsh
    target = RectangleGeometry("Target", 4.0, 3.0)
    tool, tool_face_ids = tool_factory()
    with geometry.model(
        f"planar-reject-{operation}-{diagnostic}",
        dimension=2,
    ) as cad:
        with pytest.raises(
            PlanarBooleanLineageResolutionError,
            match=diagnostic,
        ):
            prepare_planar_boolean(
                cad,
                target,
                "face:domain",
                tool,
                tool_face_ids,
                operation,
            )
