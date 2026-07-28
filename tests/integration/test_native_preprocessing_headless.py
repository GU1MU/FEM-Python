from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fem import geometry
from fem.application import MeshEntityRef, ModelSession, NamedRegion, NativePart
from fem.application.native_regions import (
    NativeRegionValidationError,
    RecipeRegionSelector,
)
from fem.application.preprocessing import (
    LogicalRecipeTopologyResolver,
    TopologyResolutionError,
    generate_fem_model,
)
from fem.application.recipe_compiler import compile_recipe
from fem.core.model import FEMModel
from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem.mesh.settings import MeshSizeFalloff
from fem.selection import edges as mesh_edges


@pytest.mark.parametrize(
    ("recipe", "settings", "expected_type"),
    (
        (
            RectangleGeometry("headless-triangle", 2.0, 1.0),
            MeshSettings(0.4, cell_shape="triangle"),
            "Tri3",
        ),
        (
            RectangleGeometry("headless-quadrilateral", 2.0, 1.0),
            MeshSettings(0.4, cell_shape="quadrilateral"),
            "Quad4",
        ),
        (
            BoxGeometry("headless-tetrahedron", 1.0, 0.8, 0.6),
            MeshSettings(0.35, cell_shape="tetrahedron"),
            "Tet4",
        ),
        (
            BoxGeometry("headless-hexahedron", 1.0, 0.8, 0.6),
            MeshSettings(0.35, cell_shape="hexahedron"),
            "Hex8",
        ),
    ),
)
def test_explicit_native_inputs_generate_canonical_models(
    real_gmsh,
    recipe,
    settings,
    expected_type,
) -> None:
    del real_gmsh
    model = generate_fem_model(recipe, settings)

    assert isinstance(model, FEMModel)
    assert model.name == recipe.name
    assert {element.type for element in model.mesh.elements} == {expected_type}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


def test_mesh_task_snapshot_generates_before_installing_a_refined_mesh_scope(
    real_gmsh,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project("Headless")
    session.replace_geometry(
        (NativePart(),),
        RectangleGeometry("headless-snapshot", 2.0, 1.0),
    )
    session.replace_mesh_settings(
        MeshSettings(
            0.5,
            local_controls=(LocalMeshControl(LogicalEntityRef("edge:right"), 0.1),),
        )
    )
    task = session.prepare_mesh_generation()

    model = generate_fem_model(task)
    delta = session.accept_generated_model(task.token, model)
    assert delta.accepted
    nodes_by_id = {
        int(node.id): node
        for node in model.mesh.nodes
    }
    selected = tuple(
        row
        for row in mesh_edges.boundary(model.mesh)
        if all(
            abs(float(nodes_by_id[int(node_id)].x) - 2.0) <= 1.0e-8
            for node_id in row[2]
        )
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "RefinedEdge",
                tuple(MeshEntityRef.edge(*row) for row in selected),
            ),
        )
    )

    assert selected
    assert session.snapshot().artifact is not None
    assert session.snapshot().model.edges["RefinedEdge"].edges
    assert "RefinedEdge" not in session.snapshot().model.node_sets


def test_mesh_task_snapshot_rejects_named_region_override() -> None:
    session = ModelSession()
    session.new_native_project("Headless")
    session.replace_geometry(
        (NativePart(),),
        RectangleGeometry("headless-snapshot-override", 2.0, 1.0),
    )
    session.replace_mesh_settings(MeshSettings(0.5))
    task = session.prepare_mesh_generation()

    with pytest.raises(TypeError, match="不能重复传入 named_regions"):
        generate_fem_model(task, named_regions=())


def test_headless_preprocessing_module_has_no_gui_imports() -> None:
    module_path = (
        Path(__file__).parents[2] / "src" / "fem" / "application" / "preprocessing.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        name == "fem_gui"
        or name.startswith("fem_gui.")
        or name == "PySide6"
        or name.startswith("PySide6.")
        or name == "pyvista"
        or name.startswith("pyvista.")
        for name in imports
    )


def test_box_logical_edges_ignore_unrelated_backend_tag_order(real_gmsh) -> None:
    del real_gmsh
    recipe = BoxGeometry("mapped-box", 2.0, 1.5, 0.75)
    with geometry.model("mapped-box-session", dimension=3) as cad:
        cad.point(20.0, 20.0, 20.0)
        cad.box(10.0, 10.0, 10.0, 0.2, 0.3, 0.4)

        compiled = compile_recipe(cad, recipe)

        edges = tuple(
            compiled.resolve(LogicalEntityRef(entity.logical_id))
            for entity in compiled.catalog.entities_of("edge")
        )
        assert all(len(group) == 1 for group in edges)
        assert len({group[0] for group in edges}) == 12
        points = tuple(
            compiled.resolve(LogicalEntityRef(entity.logical_id))[0]
            for entity in compiled.catalog.entities_of("point")
        )
        faces = tuple(
            compiled.resolve(LogicalEntityRef(entity.logical_id))[0]
            for entity in compiled.catalog.entities_of("face")
        )
        assert len(set(points)) == 8
        assert len(set(faces)) == 6
        first_bounds = cad.bounding_box(edges[0][0])
        assert first_bounds[1] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[2] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[4] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[5] == pytest.approx(0.0, abs=1.0e-7)
        for kind, dimension in (
            ("point", 0),
            ("edge", 1),
            ("face", 2),
            ("body", 3),
        ):
            for logical in compiled.catalog.entities_of(kind):
                entities = compiled.resolve(LogicalEntityRef(logical.logical_id))
                assert entities
                assert {entity.dimension for entity in entities} == {dimension}


def test_strict_sketch_compiles_profiles_and_stable_curve_groups(real_gmsh) -> None:
    del real_gmsh
    recipe = SketchGeometry(
        "strict-plate",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 4.0, 0.0),
            SketchPoint("P3", 4.0, 3.0),
            SketchPoint("P4", 0.0, 3.0),
            SketchPoint("P5", 2.0, 1.5),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchCircle("C1", "P5", 0.5),
        ),
    )

    with geometry.model("strict-sketch-session", dimension=2) as cad:
        compiled = compile_recipe(cad, recipe)

        assert compiled.catalog.exact
        assert len(compiled.domain) == 1
        assert len(compiled.resolve(LogicalEntityRef("edge:C1"))) == 4
        assert len(compiled.resolve(LogicalEntityRef("edge:outer-loop"))) == 4
        assert len(compiled.resolve(LogicalEntityRef("edge:hole-loop"))) == 4
        profile_faces = tuple(
            entity
            for entity in compiled.catalog.entities_of(
                "face",
                selectable_only=True,
            )
            if entity.semantic_role == "sketch.profile"
        )
        assert len(profile_faces) == 1
        assert len(
            compiled.resolve(LogicalEntityRef(profile_faces[0].logical_id))
        ) == 1
        assert compiled.resolve(LogicalEntityRef("body:domain")) == compiled.domain
        for point in recipe.points:
            assert len(
                compiled.resolve(LogicalEntityRef(f"point:{point.id}"))
            ) == 1


@pytest.mark.parametrize("invalid_reference", (True, 1.9, "edge:right"))
def test_logical_resolver_accepts_only_typed_references(
    real_gmsh,
    invalid_reference,
) -> None:
    del real_gmsh
    with geometry.model("mapped-strict-id-session", dimension=2) as cad:
        compiled = compile_recipe(cad, RectangleGeometry("strict-id", 2.0, 1.0))

        with pytest.raises(TypeError, match="LogicalEntityRef"):
            compiled.resolve(invalid_reference)

        with pytest.raises(TopologyResolutionError, match="已失效"):
            compiled.resolve(LogicalEntityRef("edge:missing"))


def test_named_region_and_local_control_reject_untyped_references(real_gmsh) -> None:
    del real_gmsh
    with pytest.raises(TypeError, match="LogicalEntityRef"):
        NamedRegion("Unsafe", (True,))
    with pytest.raises(TypeError, match="LogicalEntityRef"):
        LocalMeshControl(True, 0.2)


@pytest.mark.parametrize("former_builtin_name", ("DOMAIN", "LEFT"))
def test_former_builtin_names_are_available_for_mesh_scopes(
    real_gmsh,
    former_builtin_name,
) -> None:
    del real_gmsh
    recipe = RectangleGeometry("former-builtin-region", 2.0, 1.0)
    settings = MeshSettings(0.4)
    model = generate_fem_model(recipe, settings)
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), recipe)
    session.replace_mesh_settings(settings)
    task = session.prepare_mesh_generation()
    session.accept_generated_model(task.token, model)
    session.replace_named_regions(
        (
            NamedRegion(
                former_builtin_name,
                (MeshEntityRef.node(model.mesh.nodes[0].id),),
            ),
        )
    )

    assert former_builtin_name in session.snapshot().model.node_sets


def test_mesh_scope_does_not_reuse_the_local_control_geometry_resolver(
    real_gmsh,
) -> None:
    del real_gmsh

    class RecordingResolver(LogicalRecipeTopologyResolver):
        def __init__(self) -> None:
            self.calls = []

        def resolve(self, cad, recipe, topology, reference):
            entities = super().resolve(cad, recipe, topology, reference)
            self.calls.append((reference, entities))
            return entities

    resolver = RecordingResolver()
    model = generate_fem_model(
        RectangleGeometry("shared-resolver", 2.0, 1.0),
        MeshSettings(
            0.4,
            local_controls=(LocalMeshControl(LogicalEntityRef("edge:right"), 0.15),),
        ),
        resolver=resolver,
    )

    selected = [
        entities
        for reference, entities in resolver.calls
        if reference == LogicalEntityRef("edge:right")
    ]
    assert len(selected) == 1
    assert not model.edges


def test_global_and_target_radius_falloff_use_their_frozen_scales(
    real_gmsh,
    monkeypatch,
) -> None:
    del real_gmsh
    from fem.mesh import gmsh as gmsh_meshing

    calls = []
    original = gmsh_meshing.Mesher.threshold_field

    def recording_threshold(self, distance, **kwargs):
        calls.append(dict(kwargs))
        return original(self, distance, **kwargs)

    monkeypatch.setattr(
        gmsh_meshing.Mesher,
        "threshold_field",
        recording_threshold,
    )
    target = LogicalEntityRef("edge:hole-loop")
    generate_fem_model(
        PlateWithHoleGeometry("falloff", 2.0, 1.0, 1.0, 0.5, 0.2),
        MeshSettings(
            0.4,
            local_controls=(
                LocalMeshControl(
                    target,
                    0.1,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
                LocalMeshControl(
                    target,
                    0.12,
                    MeshSizeFalloff("global_size", 0.0, 2.0),
                ),
            ),
        ),
    )

    assert calls == [
        {
            "size_min": 0.12,
            "size_max": 0.4,
            "dist_min": 0.0,
            "dist_max": 0.8,
        },
        {
            "size_min": 0.1,
            "size_max": 0.4,
            "dist_min": 0.05,
            "dist_max": 0.4,
        },
    ]


def test_rigid_transform_keeps_lower_logical_entities_live(real_gmsh) -> None:
    del real_gmsh
    recipe = MovedGeometry(
        RotatedGeometry(
            RectangleGeometry("transformed", 2.0, 1.0),
            "z",
            90.0,
        ),
        3.0,
        -2.0,
    )
    with geometry.model("mapped-transform-session", dimension=2) as cad:
        compiled = compile_recipe(cad, recipe)

        entities = tuple(
            compiled.resolve(LogicalEntityRef(logical.logical_id))
            for kind in ("point", "edge", "face")
            for logical in compiled.catalog.entities_of(kind)
        )

        assert all(group for group in entities)
        for group in entities:
            for entity in group:
                cad.bounding_box(entity)
                cad.center_of_mass(entity)
        bottom = cad.bounding_box(compiled.resolve(LogicalEntityRef("edge:bottom"))[0])
        assert bottom[0] == pytest.approx(3.0, abs=1.0e-7)
        assert bottom[3] == pytest.approx(3.0, abs=1.0e-7)
        assert bottom[1] == pytest.approx(-2.0, abs=1.0e-7)
        assert bottom[4] == pytest.approx(0.0, abs=1.0e-7)


def test_extrusion_recovers_caps_sides_top_edges_and_verticals(real_gmsh) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(
        RectangleGeometry("extruded-map", 2.0, 1.0),
        0.5,
    )
    with geometry.model("mapped-extrusion-session", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)

        assert (
            len(
                {
                    compiled.resolve(LogicalEntityRef(entity.logical_id))[0]
                    for entity in compiled.catalog.entities_of("edge")
                }
            )
            == 12
        )
        assert (
            len(
                {
                    compiled.resolve(LogicalEntityRef(entity.logical_id))[0]
                    for entity in compiled.catalog.entities_of("face")
                }
            )
            == 6
        )
        for entity in compiled.catalog.entities_of("point"):
            point = compiled.resolve(LogicalEntityRef(entity.logical_id))
            assert len(point) == 1
            assert point[0].dimension == 0
        bottom_edge = cad.bounding_box(
            compiled.resolve(LogicalEntityRef("edge:bottom/bottom"))[0]
        )
        top_edge = cad.bounding_box(
            compiled.resolve(LogicalEntityRef("edge:top/bottom"))[0]
        )
        vertical = cad.bounding_box(
            compiled.resolve(LogicalEntityRef("edge:vertical/bottom-left"))[0]
        )
        bottom_face = cad.bounding_box(
            compiled.resolve(LogicalEntityRef("face:bottom"))[0]
        )
        top_face = cad.bounding_box(compiled.resolve(LogicalEntityRef("face:top"))[0])
        assert bottom_edge[2] == pytest.approx(0.0, abs=1.0e-7)
        assert bottom_edge[5] == pytest.approx(0.0, abs=1.0e-7)
        assert top_edge[2] == pytest.approx(0.5, abs=1.0e-6)
        assert top_edge[5] == pytest.approx(0.5, abs=1.0e-6)
        assert vertical[2] == pytest.approx(0.0, abs=1.0e-6)
        assert vertical[5] == pytest.approx(0.5, abs=1.0e-6)
        assert bottom_face[2] == pytest.approx(0.0, abs=1.0e-6)
        assert top_face[5] == pytest.approx(0.5, abs=1.0e-6)


def test_perforated_extrusion_keeps_inner_and_outer_side_groups_distinct(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(
        PlateWithHoleGeometry("perforated", 2.0, 1.0, 1.0, 0.5, 0.2),
        0.4,
    )
    with geometry.model("mapped-perforated-session", dimension=3) as cad:
        compiled = compile_recipe(cad, recipe)

        hole_side = set(compiled.resolve(LogicalEntityRef("face:side/hole-loop")))
        outer_sides = set(compiled.resolve(LogicalEntityRef("face:side/outer-loop")))

        assert len(hole_side) == 1
        assert len(outer_sides) == 4
        assert hole_side.isdisjoint(outer_sides)
        assert len(compiled.resolve(LogicalEntityRef("edge:bottom/hole-loop"))) == 1
        assert len(compiled.resolve(LogicalEntityRef("edge:bottom/outer-loop"))) == 4
        assert set(compiled.region_bindings[RecipeRegionSelector.HOLE]) == hole_side
        assert set(compiled.region_bindings[RecipeRegionSelector.OUTER]) == outer_sides
        assert not hasattr(compiled, "groups")
        assert not hasattr(compiled, "hole_boundary")

    model = generate_fem_model(recipe, MeshSettings(0.3))
    assert model.mesh.elements
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


def test_unproven_subentities_fail_closed_but_domain_meshing_remains_available(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = BooleanGeometry(
        "unproven-fuse",
        "fuse",
        RectangleGeometry("left", 2.0, 1.0),
        MovedGeometry(RectangleGeometry("right", 2.0, 1.0), 1.0, 0.0),
    )
    with geometry.model("mapped-unproven-session", dimension=2) as cad:
        compiled = compile_recipe(cad, recipe)
        with pytest.raises(TopologyResolutionError, match="不可用于建模"):
            compiled.resolve(LogicalEntityRef("edge:missing"))

    model = generate_fem_model(recipe, MeshSettings(0.4))
    assert model.mesh.elements
    assert not model.element_sets
    with pytest.raises(NativeRegionValidationError):
        generate_fem_model(
            recipe,
            MeshSettings(0.4),
            named_regions=(
                NamedRegion(
                    "Unsafe",
                    (LogicalEntityRef("body:result"),),
                ),
            ),
        )
    with pytest.raises(NativeRegionValidationError):
        generate_fem_model(
            recipe,
            MeshSettings(
                0.4,
                local_controls=(
                    LocalMeshControl(
                        LogicalEntityRef("edge:missing"),
                        0.2,
                    ),
                ),
            ),
        )


def test_periodic_display_points_are_absent_from_cad_point_contract(
    real_gmsh,
) -> None:
    del real_gmsh
    with geometry.model("mapped-cylinder-session", dimension=3) as cad:
        compiled = compile_recipe(
            cad,
            CylinderGeometry("mapped-cylinder", 0.5, 1.0),
        )

        with pytest.raises(TopologyResolutionError, match="已失效"):
            compiled.resolve(LogicalEntityRef("point:seam"))
        assert len(compiled.resolve(LogicalEntityRef("edge:bottom-rim"))) == 1
        assert len(compiled.resolve(LogicalEntityRef("edge:top-rim"))) == 1
