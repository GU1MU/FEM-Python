from __future__ import annotations

import gc
import tracemalloc

import pytest

from fem.application import (
    CompressedMeshEntityRefs,
    MeshEntityRef,
    NamedRegion,
)
from fem.application.native_scope_materialization import materialize_native_scopes
from fem.application.preprocessing import _active_part_regions
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel


@pytest.mark.parametrize(
    "references",
    (
        tuple(MeshEntityRef.node(value, part_id="P1") for value in (7, 2, 3, 4)),
        tuple(MeshEntityRef.element(value, part_id="P1") for value in (9, 1, 2)),
        (
            MeshEntityRef.edge(8, 2, (4, 5), part_id="P2"),
            MeshEntityRef.edge(3, 0, (1, 2), part_id="P1"),
        ),
        (
            MeshEntityRef.face(8, 1, (4, 5, 6), part_id="P2"),
            MeshEntityRef.face(3, 0, (1, 2, 3), part_id="P1"),
        ),
    ),
)
def test_all_mesh_reference_kinds_round_trip_with_stable_lazy_order(
    references,
) -> None:
    expected = tuple(
        sorted(
            references,
            key=lambda item: (
                item.part_id or "",
                item.identity,
                item.node_ids,
            ),
        )
    )
    region = NamedRegion("Scope", references)

    assert isinstance(region.references, CompressedMeshEntityRefs)
    assert len(region.references) == len(expected)
    assert tuple(region.references) == expected
    assert all(reference in region.references for reference in expected)
    assert region.references[0] == expected[0]


def _peak_bytes(factory) -> int:
    gc.collect()
    tracemalloc.start()
    value = factory()
    gc.collect()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert value
    return peak


def test_contiguous_100k_storage_reduces_retained_memory_by_at_least_80_percent() -> None:
    old_bytes = _peak_bytes(
        lambda: tuple(MeshEntityRef.node(value) for value in range(1, 100_001))
    )
    compact_bytes = _peak_bytes(
        lambda: CompressedMeshEntityRefs(
            MeshEntityRef.node(value) for value in range(1, 100_001)
        )
    )

    assert compact_bytes <= old_bytes * 0.20


def test_sparse_storage_does_not_exceed_legacy_reference_storage() -> None:
    ids = range(1, 200_000, 2)
    old_bytes = _peak_bytes(
        lambda: tuple(MeshEntityRef.element(value) for value in ids)
    )
    compact_bytes = _peak_bytes(
        lambda: CompressedMeshEntityRefs(
            MeshEntityRef.element(value) for value in ids
        )
    )

    assert compact_bytes <= old_bytes


def test_mesh_revision_mismatch_fails_before_scope_materialization() -> None:
    model = FEMModel(
        Mesh2D(
            [Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
            [Element2D(1, [1, 2], "Truss2")],
        )
    )
    region = NamedRegion("Nodes", (MeshEntityRef.node(1),)).bind_mesh_revision(4)

    with pytest.raises(ValueError, match="stale"):
        materialize_native_scopes(
            model,
            previous_names=(),
            regions=(region,),
            mesh_revision=5,
        )


def test_bound_compact_references_cannot_be_rebound_to_a_new_mesh() -> None:
    references = CompressedMeshEntityRefs(
        (MeshEntityRef.node(1),),
        mesh_revision=4,
    )

    with pytest.raises(ValueError, match="stale"):
        references.bind_mesh_revision(5)


def test_active_part_filter_does_not_expand_compact_references(
    monkeypatch,
) -> None:
    region = NamedRegion(
        "Nodes",
        (
            MeshEntityRef.node(1, part_id="P1"),
            MeshEntityRef.node(2, part_id="P2"),
        ),
    ).bind_mesh_revision(3)

    def unexpected_iteration(self):
        del self
        raise AssertionError("Part filtering must consume compact groups")

    monkeypatch.setattr(
        CompressedMeshEntityRefs,
        "__iter__",
        unexpected_iteration,
    )

    projected = _active_part_regions((region,), frozenset({"P2"}))

    assert len(projected) == 1
    assert projected[0].references.compact_groups() == (
        ("P2", (2, 2)),
    )


def test_node_materialization_consumes_compact_sequence_without_tuple_expansion() -> None:
    model = FEMModel(
        Mesh2D(
            [Node2D(value, float(value), 0.0) for value in range(1, 5)],
            [],
        )
    )
    region = NamedRegion(
        "Nodes",
        tuple(MeshEntityRef.node(value) for value in range(1, 5)),
    ).bind_mesh_revision(7)

    updated = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(region,),
        mesh_revision=7,
    )

    assert updated.node_sets["Nodes"].node_ids == (1, 2, 3, 4)
