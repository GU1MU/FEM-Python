from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fem import geometry
from fem.application import ModelSession, NamedRegion, NativePart
from fem.application.preprocessing import (
    LogicalRecipeTopologyResolver,
    TopologyResolutionError,
    generate_fem_model,
)
from fem.application.recipe_compiler import compile_recipe
from fem.core.model import FEMModel
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings


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
    assert set(model.element_sets["DOMAIN"].element_ids) == {
        element.id for element in model.mesh.elements
    }


def test_mesh_task_snapshot_generates_and_installs_named_refined_region(
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
            local_controls=(LocalMeshControl("edge", 2, 0.1),),
        )
    )
    session.replace_named_regions((NamedRegion("RefinedEdge", "edge", (2,)),))
    task = session.prepare_mesh_generation()

    model = generate_fem_model(task)

    assert model.node_sets["RefinedEdge"].node_ids
    assert model.edges["RefinedEdge"].edges
    assert len(model.node_sets["RefinedEdge"].node_ids) > len(
        model.node_sets["LEFT"].node_ids
    )
    delta = session.accept_generated_model(task.token, model)
    assert delta.accepted
    assert session.snapshot().artifact is not None


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

        edges = tuple(compiled.resolve("edge", index) for index in range(1, 13))
        assert all(len(group) == 1 for group in edges)
        assert len({group[0] for group in edges}) == 12
        points = tuple(compiled.resolve("point", index)[0] for index in range(1, 9))
        faces = tuple(compiled.resolve("face", index)[0] for index in range(1, 7))
        assert len(set(points)) == 8
        assert len(set(faces)) == 6
        first_bounds = cad.bounding_box(edges[0][0])
        assert first_bounds[1] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[2] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[4] == pytest.approx(0.0, abs=1.0e-7)
        assert first_bounds[5] == pytest.approx(0.0, abs=1.0e-7)
        for kind, count, dimension in (
            ("point", 8, 0),
            ("edge", 12, 1),
            ("face", 6, 2),
            ("body", 1, 3),
        ):
            for entity_id in range(1, count + 1):
                entities = compiled.resolve(kind, entity_id)
                assert entities
                assert {entity.dimension for entity in entities} == {dimension}


@pytest.mark.parametrize("invalid_id", (True, 1.9, "1"))
def test_logical_resolver_rejects_coercible_entity_ids(real_gmsh, invalid_id) -> None:
    del real_gmsh
    with geometry.model("mapped-strict-id-session", dimension=2) as cad:
        compiled = compile_recipe(cad, RectangleGeometry("strict-id", 2.0, 1.0))

        with pytest.raises(TopologyResolutionError, match="已失效"):
            compiled.resolve("edge", invalid_id)

        with pytest.raises(TopologyResolutionError, match="已失效"):
            compiled.resolve("curve", 1)


def test_named_region_rejects_coercible_entity_id(real_gmsh) -> None:
    del real_gmsh
    with pytest.raises(TopologyResolutionError, match="编号 True"):
        generate_fem_model(
            RectangleGeometry("strict-region", 2.0, 1.0),
            MeshSettings(0.4),
            named_regions=(NamedRegion("Unsafe", "edge", (True,)),),
        )
    with pytest.raises(ValueError, match="大于零的整数"):
        LocalMeshControl("edge", True, 0.2)


@pytest.mark.parametrize("reserved_name", ("DOMAIN", "LEFT"))
def test_named_regions_cannot_override_builtin_groups(
    real_gmsh,
    reserved_name,
) -> None:
    del real_gmsh
    with pytest.raises(ValueError, match="与内建区域冲突"):
        generate_fem_model(
            RectangleGeometry("reserved-region", 2.0, 1.0),
            MeshSettings(0.4),
            named_regions=(NamedRegion(reserved_name, "edge", (1,)),),
        )


def test_named_region_and_local_control_share_one_resolver_mapping(real_gmsh) -> None:
    del real_gmsh

    class RecordingResolver(LogicalRecipeTopologyResolver):
        def __init__(self) -> None:
            self.calls = []

        def resolve(self, cad, recipe, topology, entity_kind, entity_id):
            entities = super().resolve(
                cad,
                recipe,
                topology,
                entity_kind,
                entity_id,
            )
            self.calls.append((entity_kind, entity_id, entities))
            return entities

    resolver = RecordingResolver()
    model = generate_fem_model(
        RectangleGeometry("shared-resolver", 2.0, 1.0),
        MeshSettings(
            0.4,
            local_controls=(LocalMeshControl("edge", 2, 0.15),),
        ),
        named_regions=(NamedRegion("Selected", "edge", (2,)),),
        resolver=resolver,
    )

    selected = [
        entities
        for kind, entity_id, entities in resolver.calls
        if kind == "edge" and entity_id == 2
    ]
    assert len(selected) == 2
    assert selected[0] == selected[1]
    assert model.edges["Selected"].edges


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
            compiled.resolve(kind, entity_id)
            for kind, count in (("point", 4), ("edge", 4), ("face", 1))
            for entity_id in range(1, count + 1)
        )

        assert all(group for group in entities)
        for group in entities:
            for entity in group:
                cad.bounding_box(entity)
                cad.center_of_mass(entity)
        bottom = cad.bounding_box(compiled.resolve("edge", 1)[0])
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
            len({compiled.resolve("edge", entity_id)[0] for entity_id in range(1, 13)})
            == 12
        )
        assert (
            len({compiled.resolve("face", entity_id)[0] for entity_id in range(1, 7)})
            == 6
        )
        for entity_id in range(1, 9):
            point = compiled.resolve("point", entity_id)
            assert len(point) == 1
            assert point[0].dimension == 0
        bottom_edge = cad.bounding_box(compiled.resolve("edge", 1)[0])
        top_edge = cad.bounding_box(compiled.resolve("edge", 5)[0])
        vertical = cad.bounding_box(compiled.resolve("edge", 9)[0])
        bottom_face = cad.bounding_box(compiled.resolve("face", 1)[0])
        top_face = cad.bounding_box(compiled.resolve("face", 2)[0])
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

        hole_side = set(compiled.resolve("face", 3))
        outer_sides = set(compiled.resolve("face", 4))

        assert len(hole_side) == 1
        assert len(outer_sides) == 4
        assert hole_side.isdisjoint(outer_sides)
        assert len(compiled.resolve("edge", 1)) == 1
        assert len(compiled.resolve("edge", 2)) == 4
        assert set(compiled.groups["HOLE"]) == hole_side
        assert set(compiled.groups["OUTER"]) == outer_sides
        assert set(compiled.hole_boundary) == hole_side

    model = generate_fem_model(recipe, MeshSettings(0.3))
    hole_faces = {
        (face.elem_id, face.local_index)
        for face in model.surfaces["HOLE"].faces
    }
    outer_faces = {
        (face.elem_id, face.local_index)
        for face in model.surfaces["OUTER"].faces
    }
    assert hole_faces
    assert outer_faces
    assert hole_faces.isdisjoint(outer_faces)


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
            compiled.resolve("edge", 1)

    model = generate_fem_model(recipe, MeshSettings(0.4))
    assert model.element_sets["DOMAIN"].element_ids
    with pytest.raises(TopologyResolutionError):
        generate_fem_model(
            recipe,
            MeshSettings(0.4),
            named_regions=(NamedRegion("Unsafe", "edge", (1,)),),
        )
    with pytest.raises(TopologyResolutionError):
        generate_fem_model(
            recipe,
            MeshSettings(
                0.4,
                local_controls=(LocalMeshControl("edge", 1, 0.2),),
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

        with pytest.raises(TopologyResolutionError, match="编号 1"):
            compiled.resolve("point", 1)
        assert len(compiled.resolve("edge", 1)) == 1
        assert len(compiled.resolve("edge", 2)) == 1
