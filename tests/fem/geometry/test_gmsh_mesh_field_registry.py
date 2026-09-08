from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fem import geometry
from fem.mesh import gmsh as meshing
from fem.mesh.gmsh._field_registry import (
    _MeshFieldRegistry,
    _normalize_active_tags,
)


def _construct(
    registry: _MeshFieldRegistry,
    tag: int,
    field_type: str = "Distance",
) -> meshing.MeshFieldRef:
    def allocate(record):
        record(tag)

    return registry.construct(field_type, allocate, lambda unused: None)


def test_mesh_field_reference_has_the_current_immutable_contract() -> None:
    owner = object()
    reference = meshing.MeshFieldRef(5, "Distance", owner, object())

    assert (reference.tag, reference.field_type) == (5, "Distance")
    assert not hasattr(reference, "__dict__")
    assert "object" not in repr(reference)
    with pytest.raises(FrozenInstanceError):
        reference.tag = 6  # type: ignore[misc]
    with pytest.raises(ValueError, match="mesh field type"):
        meshing.MeshFieldRef(1, "Box", owner, object())  # type: ignore[arg-type]


def test_field_owner_identity_is_runtime_local_and_tag_reuse_is_fresh() -> None:
    first_registry = _MeshFieldRegistry("first")
    second_registry = _MeshFieldRegistry("second")
    original = _construct(first_registry, 3)
    replacement = _construct(first_registry, 3)
    foreign = _construct(second_registry, 3)

    with pytest.raises(meshing.StaleMeshFieldError, match="stale"):
        first_registry.normalize((original,), operation="threshold_field")
    assert first_registry.normalize(
        (replacement,), operation="threshold_field"
    ) == (replacement,)
    with pytest.raises(meshing.MeshFieldOwnershipError, match="another"):
        first_registry.normalize((foreign,), operation="threshold_field")
    assert first_registry.owner_token is not second_registry.owner_token


def test_native_field_disappearance_invalidates_only_matching_identity() -> None:
    registry = _MeshFieldRegistry("live-fields")
    first = _construct(registry, 1)
    second = _construct(registry, 2)

    with pytest.raises(meshing.StaleMeshFieldError, match="no longer exists"):
        registry.assert_liveness(
            (first,),
            frozenset({2}),
            operation="threshold_field",
        )

    assert registry.normalize((second,), operation="background_field") == (second,)
    with pytest.raises(meshing.StaleMeshFieldError, match="stale"):
        registry.normalize((first,), operation="threshold_field")


def test_field_inputs_reject_duplicates_before_native_liveness() -> None:
    registry = _MeshFieldRegistry("duplicates")
    field = _construct(registry, 1, "Threshold")

    with pytest.raises(ValueError, match="duplicate-free"):
        registry.normalize((field, field), operation="min_field")


def test_post_allocation_failure_rolls_back_and_preserves_primary_exception() -> None:
    registry = _MeshFieldRegistry("rollback")
    removed: list[int] = []
    primary = RuntimeError("configuration failed")

    def allocate(record):
        record(7)
        raise primary

    with pytest.raises(RuntimeError, match="configuration failed") as caught:
        registry.construct("Distance", allocate, removed.append)

    assert caught.value is primary
    assert removed == [7]


def test_rollback_failure_is_a_note_on_the_primary_exception() -> None:
    registry = _MeshFieldRegistry("rollback-note")

    def allocate(record):
        record(8)
        raise RuntimeError("configuration failed")

    def rollback(tag: int) -> None:
        raise RuntimeError(f"remove {tag} failed")

    with pytest.raises(RuntimeError, match="configuration failed") as caught:
        registry.construct("Distance", allocate, rollback)

    assert caught.value.__notes__ == [
        "geometry model 'rollback-note': mesh-field rollback also failed while "
        "removing field 8: remove 8 failed"
    ]


@pytest.mark.parametrize("raw", [[1, 0], [1, True], ["1"], None])
def test_active_native_field_tags_reject_malformed_values(raw: object) -> None:
    with pytest.raises(
        geometry.GeometryError,
        match="Gmsh returned an invalid mesh field list",
    ):
        _normalize_active_tags(raw, "malformed")
