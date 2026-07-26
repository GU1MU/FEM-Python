from __future__ import annotations

from dataclasses import replace

import pytest

from fem.geometry import (
    LogicalEntityRef,
    TargetRadiusResolutionError,
    logical_ref_sort_key,
    resolve_legacy_hole_target,
    resolve_target_radius,
)
from fem.geometry.recipe_topology import (
    TOPOLOGY_REFERENCE_CONTRACT,
    describe_recipe_topology,
    topology_fingerprint_for_recipe,
    topology_fingerprint_from_topology,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)


@pytest.mark.parametrize(
    "value,error_type",
    [
        (1, TypeError),
        (True, TypeError),
        ("", ValueError),
        (" edge:left", ValueError),
        ("edge:left ", ValueError),
        ("edge", ValueError),
        ("edge:", ValueError),
        ("curve:left", ValueError),
    ],
)
def test_logical_entity_ref_is_strict(value, error_type) -> None:
    with pytest.raises(error_type):
        LogicalEntityRef(value)


def test_logical_reference_kind_and_canonical_order_are_explicit() -> None:
    references = (
        LogicalEntityRef("body:domain"),
        LogicalEntityRef("edge:right"),
        LogicalEntityRef("point:origin"),
        LogicalEntityRef("edge:left"),
        LogicalEntityRef("face:domain"),
    )

    assert LogicalEntityRef("edge:left").kind == "edge"
    assert tuple(
        reference.logical_id
        for reference in sorted(references, key=logical_ref_sort_key)
    ) == (
        "point:origin",
        "edge:left",
        "edge:right",
        "face:domain",
        "body:domain",
    )
    with pytest.raises(TypeError):
        sorted(references)


def test_topology_fingerprint_is_complete_and_catalog_order_independent() -> None:
    recipe = RectangleGeometry("Plate", 4.0, 2.0)
    topology = describe_recipe_topology(recipe)
    reordered = replace(topology, entities=tuple(reversed(topology.entities)))

    fingerprint = topology_fingerprint_for_recipe(recipe)

    assert fingerprint.contract == TOPOLOGY_REFERENCE_CONTRACT == 2
    assert fingerprint.dimension == 2
    assert fingerprint.exact is True
    assert len(fingerprint.entities) == len(topology.entities)
    assert topology_fingerprint_from_topology(reordered) == fingerprint
    assert tuple(record.kind for record in fingerprint.entities) == (
        "point",
        "point",
        "point",
        "point",
        "edge",
        "edge",
        "edge",
        "edge",
        "face",
        "body",
    )


def test_topology_fingerprint_detects_semantic_contract_changes() -> None:
    topology = describe_recipe_topology(RectangleGeometry("Plate", 4.0, 2.0))
    changed_entity = replace(
        topology.entities[0],
        semantic_role="corner.changed",
    )
    changed = replace(
        topology,
        entities=(changed_entity, *topology.entities[1:]),
    )

    assert topology_fingerprint_from_topology(changed) != (
        topology_fingerprint_from_topology(topology)
    )


def _plate(radius: float = 0.5) -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry("Plate", 6.0, 4.0, 3.0, 2.0, radius)


def _circle_cut(radius: float = 0.5) -> SketchGeometry:
    return SketchGeometry(
        "Sketch",
        (
            SketchRectangle("material", 0.0, 0.0, 6.0, 4.0),
            SketchCircle("cut", 3.0, 2.0, radius),
        ),
    )


def _boolean_cut(radius: float = 0.5) -> BooleanGeometry:
    return BooleanGeometry(
        "Cut",
        "cut",
        RectangleGeometry("Plate", 6.0, 4.0),
        MovedGeometry(DiskGeometry("Tool", radius), 3.0, 2.0),
    )


@pytest.mark.parametrize(
    "recipe",
    [
        _plate(),
        _circle_cut(),
        _boolean_cut(),
        MovedGeometry(_plate(), 1.0, -2.0),
        RotatedGeometry(_circle_cut(), "z", 32.0),
    ],
)
def test_legacy_hole_target_and_radius_follow_proven_lineage(recipe) -> None:
    target = resolve_legacy_hole_target(recipe)

    assert target == LogicalEntityRef("edge:hole-loop")
    assert resolve_target_radius(recipe, target) == pytest.approx(0.5)


def test_extruded_hole_uses_the_cylindrical_side_lineage() -> None:
    recipe = ExtrudedGeometry(_boolean_cut(0.75), 2.0)

    target = resolve_legacy_hole_target(recipe)

    assert target == LogicalEntityRef("face:side/hole-loop")
    assert resolve_target_radius(recipe, target) == pytest.approx(0.75)
    assert resolve_target_radius(
        recipe,
        LogicalEntityRef("edge:bottom/hole-loop"),
    ) == pytest.approx(0.75)


def test_target_radius_uses_current_recipe_parameters() -> None:
    before = _plate(0.4)
    after = _plate(0.9)
    target = LogicalEntityRef("edge:hole-loop")

    assert resolve_target_radius(before, target) == pytest.approx(0.4)
    assert resolve_target_radius(after, target) == pytest.approx(0.9)


def test_disk_outer_and_non_circular_holes_are_not_legacy_holes() -> None:
    disk = DiskGeometry("Disk", 2.0)
    rectangular_cut = SketchGeometry(
        "Sketch",
        (
            SketchRectangle("material", 0.0, 0.0, 6.0, 4.0),
            SketchRectangle("cut", 2.0, 1.0, 1.0, 1.0),
        ),
    )

    with pytest.raises(TargetRadiusResolutionError):
        resolve_legacy_hole_target(disk)
    with pytest.raises(TargetRadiusResolutionError):
        resolve_target_radius(disk, LogicalEntityRef("edge:outer"))
    with pytest.raises(TargetRadiusResolutionError):
        resolve_legacy_hole_target(rectangular_cut)
