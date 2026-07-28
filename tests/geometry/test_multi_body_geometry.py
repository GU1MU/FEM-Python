from __future__ import annotations

from dataclasses import replace

import pytest

from fem.geometry import (
    BoxGeometry,
    BooleanResult,
    CylinderGeometry,
    EntityRef,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    RectangleGeometry,
    SolidBody,
    add_solid_body,
    analyze_body_relations,
    delete_solid_body,
    materialize_multi_body,
    next_body_id,
    rename_solid_body,
    transform_solid_body,
    require_meshable_body_relations,
)
from fem.geometry.recipe_topology import (
    canonicalize_multi_body_logical_id,
    describe_recipe_topology,
    surviving_logical_reference_ids,
    topology_fingerprint_for_recipe,
)
from fem.geometry.solid_boolean_lineage import (
    BooleanLineageResolutionError,
    BooleanOperandEvidence,
    EntityEvidence,
    _unambiguous_support_ids,
    validate_solid_boolean_input_map,
)


def _two_bodies() -> MultiBodyGeometry:
    return MultiBodyGeometry(
        "Part-1 Geometry",
        (
            SolidBody("B1", "Body-1", BoxGeometry("Box", 2.0, 3.0, 4.0)),
            SolidBody(
                "B2",
                "Body-2",
                CylinderGeometry("Cylinder", 0.5, 2.0),
            ),
        ),
    )


def test_multi_body_canonicalizes_order_and_separates_name_from_identity() -> None:
    geometry = MultiBodyGeometry(
        "Geometry",
        tuple(reversed(_two_bodies().bodies)),
    )

    assert tuple(body.id for body in geometry.bodies) == ("B1", "B2")
    renamed = rename_solid_body(geometry, "B1", "Target")
    assert renamed.body("B1").name == "Target"
    assert topology_fingerprint_for_recipe(renamed) == (
        topology_fingerprint_for_recipe(geometry)
    )


@pytest.mark.parametrize("body_id", ["", "B0", "b1", "B01", " B1"])
def test_solid_body_rejects_invalid_stable_id(body_id: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        SolidBody(body_id, "Body", BoxGeometry("Box", 1.0, 1.0, 1.0))


def test_multi_body_rejects_duplicate_ids_and_names() -> None:
    body = SolidBody("B1", "Body-1", BoxGeometry("Box", 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="IDs"):
        MultiBodyGeometry("Geometry", (body, replace(body, name="Other")))
    with pytest.raises(ValueError, match="names"):
        MultiBodyGeometry(
            "Geometry",
            (
                body,
                SolidBody(
                    "B2",
                    "Body-1",
                    CylinderGeometry("Cylinder", 0.5, 1.0),
                ),
            ),
        )


def test_solid_body_requires_single_solid_recipe() -> None:
    with pytest.raises(ValueError, match="single-solid"):
        SolidBody("B1", "Body-1", RectangleGeometry("Rectangle", 1.0, 1.0))


def test_multi_profile_extrusion_materializes_canonical_singleton_bodies() -> None:
    first = RectangleGeometry("Rectangle", 1.0, 1.0)
    extrusion = ExtrudedGeometry(first, 2.0, ("face:domain",))
    geometry = materialize_multi_body(extrusion)

    assert tuple(body.id for body in geometry.bodies) == ("B1",)
    assert geometry.body("B1").recipe.source_face_ids == ("face:domain",)


def test_add_delete_transform_and_allocator_preserve_other_body_identity() -> None:
    before = materialize_multi_body(BoxGeometry("Box", 2.0, 2.0, 2.0))
    added = add_solid_body(
        before,
        CylinderGeometry("Cylinder", 0.5, 2.0),
    )
    moved = transform_solid_body(
        added,
        "B2",
        move=(0.5, 0.0, 0.0),
    )

    assert next_body_id(moved) == "B3"
    survivors = surviving_logical_reference_ids(before, moved)
    assert "body:B1" in survivors
    assert "face:B1/top" in survivors
    assert moved.body("B1") == before.body("B1")
    deleted = delete_solid_body(moved, "B2")
    assert deleted is not None
    assert deleted.bodies == before.bodies
    assert deleted.retired_body_ids == ("B2",)
    assert delete_solid_body(before, "B1") is None


def test_deleted_body_id_is_retired_and_never_reused() -> None:
    geometry = _two_bodies()
    after_delete = delete_solid_body(geometry, "B2")
    assert after_delete is not None

    after_add = add_solid_body(
        after_delete,
        CylinderGeometry("Replacement", 0.25, 1.0),
    )

    assert after_delete.retired_body_ids == ("B2",)
    assert tuple(body.id for body in after_add.bodies) == ("B1", "B3")


def test_multi_body_topology_uses_body_namespace_and_internal_aggregate() -> None:
    topology = describe_recipe_topology(_two_bodies())

    assert topology.exact
    assert topology.entity("body:B1").selectable
    assert topology.entity("body:B2").selectable
    assert not topology.entity("body:domain").selectable
    assert topology.entity("face:B1/top").semantic_role == "boundary.top"
    assert topology.entity("face:B2/top").semantic_role == "boundary.top"


def test_single_body_legacy_aliases_are_unambiguous_only() -> None:
    single = materialize_multi_body(BoxGeometry("Box", 1.0, 1.0, 1.0))
    assert canonicalize_multi_body_logical_id(single, "body:domain") == "body:B1"
    assert canonicalize_multi_body_logical_id(single, "face:top") == "face:B1/top"
    with pytest.raises(KeyError):
        canonicalize_multi_body_logical_id(_two_bodies(), "face:top")


def test_boolean_lineage_rejects_ambiguous_same_dimension_support() -> None:
    bounds = (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    evidence = BooleanOperandEvidence(
        None,
        (
            EntityEvidence("face:first", 2, "Plane", bounds),
            EntityEvidence("face:second", 2, "Plane", bounds),
        ),
        None,
        1.0,
    )

    with pytest.raises(
        BooleanLineageResolutionError,
        match="output-ambiguous",
    ):
        _unambiguous_support_ids(
            EntityEvidence("", 2, "Plane", bounds),
            evidence,
            operand_label="target",
        )


def test_boolean_input_map_must_prove_target_result_ownership() -> None:
    owner = object()
    result = EntityRef(3, 1, owner, object())
    foreign = EntityRef(3, 2, owner, object())

    with pytest.raises(
        BooleanLineageResolutionError,
        match="input-map.invalid",
    ):
        validate_solid_boolean_input_map(
            BooleanResult((result,), ((foreign,), ()))
        )


def test_disjoint_cylinders_are_not_blocked_by_overlapping_aabbs() -> None:
    geometry = MultiBodyGeometry(
        "Conservative",
        (
            SolidBody(
                "B1",
                "First",
                CylinderGeometry("First", 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Second",
                MovedGeometry(
                    CylinderGeometry("Second", 1.0, 1.0),
                    1.5,
                    1.5,
                    0.0,
                ),
            ),
        ),
    )

    assert analyze_body_relations(geometry)[0].relation == "disjoint"
    require_meshable_body_relations(geometry)
