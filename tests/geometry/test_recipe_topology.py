from __future__ import annotations

import pytest

from fem.geometry.recipe_topology import (
    can_preserve_logical_references,
    describe_recipe_topology,
    topology_fingerprint_for_recipe,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
)


def _wire(points=None, members=None) -> WireGeometry:
    return WireGeometry(
        "Wire",
        (
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0, 0.0),
            WirePoint("P3", 1.0, 2.0, 0.0),
        )
        if points is None
        else points,
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3"),
        )
        if members is None
        else members,
    )


def _counts(topology) -> tuple[int, int, int, int]:
    return tuple(
        len(topology.entities_of(kind)) for kind in ("point", "edge", "face", "body")
    )


def test_rectangle_has_stable_four_point_four_edge_topology() -> None:
    first = describe_recipe_topology(RectangleGeometry("A", 2.0, 1.0))
    resized = describe_recipe_topology(RectangleGeometry("B", 12.0, 0.5))

    assert first.exact is True
    assert _counts(first) == (4, 4, 1, 1)
    assert first.signature == resized.signature
    assert first.entity("edge:bottom").semantic_role == "boundary.bottom"
    assert set(first.signature.logical_ids) == {
        "point:bottom-left",
        "point:bottom-right",
        "point:top-right",
        "point:top-left",
        "edge:bottom",
        "edge:right",
        "edge:top",
        "edge:left",
        "face:domain",
        "body:domain",
    }
    assert first.selectable_entities() == first.entities
    assert first.entity("point:bottom-left").dimension == 0
    assert first.entity("edge:bottom").dimension == 1
    assert first.entity("face:domain").dimension == 2
    assert first.entity("body:domain").dimension == 2
    assert tuple(entity.logical_id for entity in first.entities_of("edge")) == (
        "edge:bottom",
        "edge:right",
        "edge:top",
        "edge:left",
    )


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            RectangleGeometry("A", 2.0, 1.0),
            RectangleGeometry("B", 12.0, 0.5),
        ),
        (
            MovedGeometry(RectangleGeometry("A", 2.0, 1.0), 1.0, 2.0),
            MovedGeometry(RectangleGeometry("B", 3.0, 4.0), -2.0, 5.0),
        ),
        (
            RotatedGeometry(RectangleGeometry("A", 2.0, 1.0), "z", 15.0),
            RotatedGeometry(RectangleGeometry("B", 3.0, 4.0), "z", 70.0),
        ),
    ),
    ids=("rectangle-parameters", "move-parameters", "rotate-parameters"),
)
def test_exact_matching_signatures_preserve_logical_references(
    before,
    after,
) -> None:
    assert can_preserve_logical_references(before, after)


def test_changed_or_unproven_topology_cannot_preserve_logical_references() -> None:
    rectangle = RectangleGeometry("Plate", 2.0, 1.0)
    unproven = BooleanGeometry(
        "Union",
        "fuse",
        rectangle,
        MovedGeometry(RectangleGeometry("Tool", 1.0, 0.5), 0.5, 0.25),
    )

    assert not can_preserve_logical_references(
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        rectangle,
    )
    assert not can_preserve_logical_references(rectangle, unproven)
    assert not can_preserve_logical_references(object(), rectangle)


def test_box_exposes_all_eight_points_twelve_edges_and_six_faces() -> None:
    topology = describe_recipe_topology(BoxGeometry("Box", 2.0, 3.0, 4.0))

    assert topology.exact is True
    assert _counts(topology) == (8, 12, 6, 1)
    assert len(topology.signature.logical_ids) == len(
        set(topology.signature.logical_ids)
    )
    assert topology.entity("edge:vertical-back-left").semantic_role == (
        "boundary.vertical-back-left"
    )
    assert topology.entity("face:front").semantic_role == "boundary.front"
    assert topology.entity("body:domain").selectable is True
    assert topology.entity("body:domain").dimension == 3
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:bottom-front",
        "edge:bottom-right",
        "edge:bottom-back",
        "edge:bottom-left",
        "edge:top-front",
        "edge:top-right",
        "edge:top-back",
        "edge:top-left",
        "edge:vertical-front-left",
        "edge:vertical-front-right",
        "edge:vertical-back-right",
        "edge:vertical-back-left",
    )
    assert tuple(entity.logical_id for entity in topology.entities_of("face")) == (
        "face:bottom",
        "face:top",
        "face:front",
        "face:right",
        "face:back",
        "face:left",
    )


def test_periodic_primitives_omit_backend_seams() -> None:
    disk = describe_recipe_topology(DiskGeometry("Disk", 1.0))
    cylinder = describe_recipe_topology(CylinderGeometry("Cylinder", 1.0, 2.0))

    assert _counts(disk) == (0, 1, 1, 1)
    assert disk.signature.logical_ids == (
        "edge:outer",
        "face:domain",
        "body:domain",
    )
    assert _counts(cylinder) == (0, 2, 3, 1)
    assert {entity.logical_id for entity in cylinder.entities_of("face")} == {
        "face:bottom",
        "face:outer",
        "face:top",
    }
    assert not any("tag" in logical_id for logical_id in cylinder.signature.logical_ids)


def test_plate_with_hole_uses_inner_and_outer_compatibility_groups() -> None:
    topology = describe_recipe_topology(
        PlateWithHoleGeometry("Plate", 10.0, 8.0, 4.0, 3.0, 1.0)
    )

    assert _counts(topology) == (4, 2, 1, 1)
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:hole-loop",
        "edge:outer-loop",
    )
    assert topology.entity("edge:hole-loop").semantic_role == "boundary.hole-loop"
    assert topology.signature.exact is True


@pytest.mark.parametrize(
    ("contour", "expected"),
    [
        (SketchRectangle("material", 2.0, 3.0, 4.0, 5.0), (4, 4, 1, 1)),
        (SketchCircle("material", 2.0, 3.0, 4.0), (0, 1, 1, 1)),
    ],
)
def test_single_material_sketch_has_proven_topology(contour, expected) -> None:
    topology = describe_recipe_topology(SketchGeometry("Sketch", (contour,)))

    assert topology.exact is True
    assert _counts(topology) == expected
    assert topology.diagnostics == ()


@pytest.mark.parametrize(
    ("cut", "expected_point_count", "operation"),
    [
        (
            SketchCircle("cut", 14.0, 23.0, 1.0),
            4,
            "sketch.cut-contained-circle",
        ),
        (
            SketchRectangle("cut", 12.0, 22.0, 2.0, 1.0),
            8,
            "sketch.cut-contained-rectangle",
        ),
    ],
)
def test_strictly_contained_sketch_cut_uses_grouped_hole_topology(
    cut,
    expected_point_count,
    operation,
) -> None:
    recipe = SketchGeometry(
        "Opening",
        (
            SketchRectangle("material", 10.0, 20.0, 8.0, 6.0),
            cut,
        ),
    )

    topology = describe_recipe_topology(recipe)

    assert topology.exact is True
    assert _counts(topology) == (expected_point_count, 2, 1, 1)
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:hole-loop",
        "edge:outer-loop",
    )
    assert topology.transition.operation == operation
    assert topology.transition.proven is True


def test_proven_sketch_hole_topology_propagates_through_extrusion() -> None:
    sketch = SketchGeometry(
        "Opening",
        (
            SketchRectangle("material", 0.0, 0.0, 8.0, 6.0),
            SketchCircle("cut", 4.0, 3.0, 1.0),
        ),
    )

    topology = describe_recipe_topology(ExtrudedGeometry(sketch, 2.0))

    assert _counts(topology) == (8, 8, 4, 1)
    assert tuple(entity.logical_id for entity in topology.entities_of("face")) == (
        "face:bottom",
        "face:top",
        "face:side/hole-loop",
        "face:side/outer-loop",
    )


def test_composite_sketch_fails_closed_with_an_unselectable_diagnostic() -> None:
    recipe = SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 0.5, 1.5, 0.5),
        ),
    )

    topology = describe_recipe_topology(recipe)

    assert topology.exact is False
    assert topology.selectable_entities() == ()
    assert topology.entities[0].logical_id == "face:result"
    assert topology.entities[0].diagnostic_code == "sketch.topology-unproven"
    assert topology.diagnostics[0].affected_logical_ids == (
        "face:result",
        "body:result",
    )
    assert topology.transition.proven is False


def test_move_and_rotate_preserve_every_logical_id_and_signature() -> None:
    base = RectangleGeometry("Plate", 2.0, 1.0)
    moved = MovedGeometry(base, 3.0, -2.0)
    rotated = RotatedGeometry(moved, "z", 37.0)

    base_topology = describe_recipe_topology(base)
    moved_topology = describe_recipe_topology(moved)
    rotated_topology = describe_recipe_topology(rotated)

    assert moved_topology.entities == base_topology.entities
    assert moved_topology.signature == base_topology.signature
    assert rotated_topology.entities == base_topology.entities
    assert rotated_topology.signature == base_topology.signature
    assert moved_topology.transition.preserved_logical_ids == (
        moved_topology.signature.logical_ids
    )
    assert rotated_topology.transition.preserved_logical_ids == (
        rotated_topology.signature.logical_ids
    )
    assert not moved_topology.transition.derived_logical_ids


def test_rigid_transform_preserves_an_unknown_catalog_without_making_it_selectable() -> (
    None
):
    sketch = SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 0.5, 1.5, 0.5),
        ),
    )
    moved = describe_recipe_topology(MovedGeometry(sketch, 1.0, 2.0))

    assert moved.exact is False
    assert moved.selectable_entities() == ()
    assert moved.transition.preserved_logical_ids == (
        "face:result",
        "body:result",
    )
    assert moved.diagnostics[0].code == "sketch.topology-unproven"


def test_rectangle_extrusion_derives_box_topology_without_backend_tags() -> None:
    first = describe_recipe_topology(
        ExtrudedGeometry(RectangleGeometry("Plate", 2.0, 1.0), 3.0)
    )
    resized = describe_recipe_topology(
        ExtrudedGeometry(RectangleGeometry("Other", 4.0, 7.0), 9.0)
    )

    assert first.exact is True
    assert _counts(first) == (8, 12, 6, 1)
    assert first.signature == resized.signature
    assert first.entity("face:side/bottom").semantic_role == (
        "sweep.boundary.outer"
    )
    assert first.entity("edge:vertical/bottom-left").semantic_role == (
        "sweep.corner.bottom-left"
    )
    assert set(first.transition.derived_logical_ids) == set(first.signature.logical_ids)
    assert first.transition.preserved_logical_ids == ()


def test_disk_and_plate_with_hole_extrusions_keep_only_proven_semantics() -> None:
    cylinder = describe_recipe_topology(
        ExtrudedGeometry(DiskGeometry("Disk", 2.0), 3.0)
    )
    perforated = describe_recipe_topology(
        ExtrudedGeometry(
            PlateWithHoleGeometry("Plate", 10.0, 8.0, 4.0, 3.0, 1.0),
            2.0,
        )
    )

    assert _counts(cylinder) == (0, 2, 3, 1)
    assert {entity.logical_id for entity in cylinder.entities_of("face")} == {
        "face:bottom",
        "face:side/outer",
        "face:top",
    }
    assert _counts(perforated) == (8, 8, 4, 1)
    assert tuple(entity.logical_id for entity in perforated.entities_of("edge")) == (
        "edge:bottom/hole-loop",
        "edge:bottom/outer-loop",
        "edge:top/hole-loop",
        "edge:top/outer-loop",
        "edge:vertical/bottom-left",
        "edge:vertical/bottom-right",
        "edge:vertical/top-right",
        "edge:vertical/top-left",
    )
    assert tuple(entity.logical_id for entity in perforated.entities_of("face")) == (
        "face:bottom",
        "face:top",
        "face:side/hole-loop",
        "face:side/outer-loop",
    )
    assert perforated.entity("face:side/hole-loop").semantic_role == (
        "sweep.boundary.hole"
    )
    assert perforated.entity("face:side/outer-loop").semantic_role == (
        "sweep.boundary.outer"
    )


def test_extruding_unproven_sketch_topology_remains_unselectable() -> None:
    sketch = SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 0.5, 1.5, 0.5),
        ),
    )

    topology = describe_recipe_topology(ExtrudedGeometry(sketch, 2.0))

    assert topology.exact is False
    assert topology.selectable_entities() == ()
    assert topology.entities[0].logical_id == "body:result"
    assert (
        topology.diagnostics[0].code
        == "extrude.source-face.topology-unproven"
    )


def test_contained_circle_cut_has_a_proven_hole_transition() -> None:
    object_geometry = RectangleGeometry("Plate", 10.0, 8.0)
    tool = MovedGeometry(DiskGeometry("Hole", 1.0), 4.0, 3.0)
    recipe = BooleanGeometry("Cut", "cut", object_geometry, tool)

    topology = describe_recipe_topology(recipe)

    assert topology.exact is True
    assert _counts(topology) == (4, 2, 1, 1)
    assert topology.entity("edge:hole-loop").selectable is True
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:hole-loop",
        "edge:outer-loop",
    )
    assert set(topology.transition.preserved_logical_ids) == {
        "point:bottom-left",
        "point:bottom-right",
        "point:top-right",
        "point:top-left",
        "face:domain",
        "body:domain",
    }
    assert set(topology.transition.derived_logical_ids) == {
        "edge:hole-loop",
        "edge:outer-loop",
    }


def test_contained_rectangle_cut_has_proven_inner_points_and_edges() -> None:
    object_geometry = MovedGeometry(
        RectangleGeometry("Plate", 10.0, 8.0),
        -5.0,
        -4.0,
    )
    tool = MovedGeometry(
        RectangleGeometry("Opening", 2.0, 1.0),
        -1.0,
        -0.5,
    )

    topology = describe_recipe_topology(
        BooleanGeometry("Cut", "cut", object_geometry, tool)
    )

    assert topology.exact is True
    assert _counts(topology) == (8, 2, 1, 1)
    assert topology.entity("point:hole-bottom-left").selectable is True
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:hole-loop",
        "edge:outer-loop",
    )


@pytest.mark.parametrize("operation", ["fuse", "fragment"])
def test_intersection_dependent_boolean_topology_fails_closed(operation) -> None:
    recipe = BooleanGeometry(
        "Boolean",
        operation,
        RectangleGeometry("Object", 2.0, 1.0),
        MovedGeometry(RectangleGeometry("Tool", 1.0, 1.0), 1.0, 0.0),
    )

    topology = describe_recipe_topology(recipe)

    assert topology.exact is False
    assert topology.selectable_entities() == ()
    assert topology.entity("face:result").selectable is False
    assert topology.diagnostics[0].code == "boolean.topology-unproven"
    assert topology.transition.proven is False
    assert len(topology.transition.source_signatures) == 2


def test_touching_cut_is_not_claimed_as_a_proven_hole() -> None:
    recipe = BooleanGeometry(
        "Touching",
        "cut",
        RectangleGeometry("Object", 4.0, 4.0),
        MovedGeometry(DiskGeometry("Tool", 1.0), 1.0, 2.0),
    )

    topology = describe_recipe_topology(recipe)

    assert topology.exact is False
    assert topology.selectable_entities() == ()


def test_signature_is_hashable_and_queries_reject_unknown_kinds() -> None:
    topology = describe_recipe_topology(RectangleGeometry("Plate", 2.0, 1.0))

    assert {topology.signature} == {topology.signature}
    with pytest.raises(ValueError, match="Unsupported logical entity kind"):
        topology.entities_of("curve")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        topology.entity("edge:missing")
    assert not hasattr(topology, "logical_entity")
    assert not hasattr(topology, "logical_index")


def test_unsupported_objects_are_rejected_without_cad_execution() -> None:
    with pytest.raises(TypeError, match="Unsupported native geometry recipe"):
        describe_recipe_topology(object())  # type: ignore[arg-type]


def test_wire_exposes_named_points_members_and_one_domain_body() -> None:
    topology = describe_recipe_topology(_wire())

    assert topology.dimension == 1
    assert topology.exact is True
    assert _counts(topology) == (3, 2, 0, 1)
    assert tuple(entity.logical_id for entity in topology.entities_of("point")) == (
        "point:P1",
        "point:P2",
        "point:P3",
    )
    assert tuple(entity.logical_id for entity in topology.entities_of("edge")) == (
        "edge:M1",
        "edge:M2",
    )
    assert topology.entity("point:P1").semantic_role == "wire.point"
    assert topology.entity("edge:M1").semantic_role == "wire.member"
    assert topology.entity("body:domain").dimension == 1
    assert topology.entity("body:domain").semantic_role == "domain"
    assert topology.transition.operation == "primitive.wire"


def test_wire_fingerprint_is_order_independent_but_identity_sensitive() -> None:
    original = _wire()
    reordered = _wire(
        points=(
            WirePoint("P3", 1.0, 2.0, 0.0),
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0, 0.0),
        ),
        members=(
            WireMember("M2", "P2", "P3"),
            WireMember("M1", "P1", "P2"),
        ),
    )
    moved = WireGeometry(
        "Other",
        (
            WirePoint("P1", 10.0, 10.0, 0.0),
            WirePoint("P2", 11.0, 10.0, 0.0),
            WirePoint("P3", 11.0, 12.0, 0.0),
        ),
        original.members,
    )
    renamed = WireGeometry(
        "Other",
        (
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0, 0.0),
            WirePoint("P3-renamed", 1.0, 2.0, 0.0),
        ),
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3-renamed"),
        ),
    )
    reversed_member = _wire(
        members=(
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P3", "P2"),
        )
    )
    reconnected = _wire(
        members=(
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P1", "P3"),
        )
    )

    assert topology_fingerprint_for_recipe(original) == (
        topology_fingerprint_for_recipe(reordered)
    )
    assert topology_fingerprint_for_recipe(original) == (
        topology_fingerprint_for_recipe(reversed_member)
    )
    assert can_preserve_logical_references(original, reordered)
    assert can_preserve_logical_references(original, moved)
    assert not can_preserve_logical_references(original, renamed)
    assert not can_preserve_logical_references(original, reconnected)


def test_wire_rigid_transforms_preserve_logical_references() -> None:
    original = _wire()
    transformed = RotatedGeometry(MovedGeometry(original, 3.0, 4.0, 5.0), "y", 35.0)

    assert describe_recipe_topology(transformed).entities == (
        describe_recipe_topology(original).entities
    )
    assert can_preserve_logical_references(original, transformed)
