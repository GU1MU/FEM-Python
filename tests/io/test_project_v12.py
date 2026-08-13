from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from fem.application import CompressedMeshEntityRefs, MeshEntityRef, NamedRegion
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    materialize_native_scopes,
)
from fem.application.session import ModelSession, ProjectSnapshot
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel
from fem.geometry import MovedGeometry, RectangleGeometry
from fem.io.project import decode_project, dumps_project, encode_project, loads_project
from fem.io.project_v11 import encode_project_v11
from fem.io.project_v12 import decode_project_v12, encode_project_v12
from fem.mesh.settings import MeshSettings


def _mesh_scope_snapshot() -> ProjectSnapshot:
    session = ModelSession()
    session.new_native_project("compact-scopes")
    session.add_native_part(
        RectangleGeometry("R1", 2.0, 1.0),
        name="Part-1",
        mesh_settings=MeshSettings(0.5),
    )
    session.add_native_part(
        MovedGeometry(RectangleGeometry("R2", 1.0, 1.0), 3.0, 0.0),
        name="Part-2",
        mesh_settings=MeshSettings(0.5),
    )
    base = session.prepare_project_save().snapshot
    return replace(
        base,
        named_regions=(
            NamedRegion(
                "Nodes",
                tuple(MeshEntityRef.node(value, part_id="P1") for value in (1, 2, 3, 8)),
            ),
            NamedRegion(
                "Elements",
                tuple(MeshEntityRef.element(value, part_id="P2") for value in (10, 11, 14)),
            ),
            NamedRegion(
                "Edges-A",
                (MeshEntityRef.edge(10, 0, (1, 2), part_id="P1"),),
            ),
            NamedRegion(
                "Edges-B",
                (MeshEntityRef.edge(10, 0, (1, 2), part_id="P1"),),
            ),
            NamedRegion(
                "Faces",
                (MeshEntityRef.face(20, 1, (3, 4, 5), part_id="P2"),),
            ),
        ),
    )


def test_v12_round_trip_uses_ranges_grouped_part_ids_and_shared_topology() -> None:
    source = _mesh_scope_snapshot()
    payload = encode_project_v12(source)
    reopened = decode_project_v12(payload)
    authoring = payload["project"]["authoring"]
    encoded_regions = authoring["named_regions"]

    assert "references" not in encoded_regions[0]
    assert encoded_regions[0]["compact_references"]["groups"] == [
        {"part_id": "P1", "ranges": [[1, 3], [8, 8]]}
    ]
    assert len(authoring["mesh_scope_topology"]["rows"]) == 2
    assert reopened.named_regions == source.named_regions
    edge_a = reopened.named_regions[2].references
    edge_b = reopened.named_regions[3].references
    assert isinstance(edge_a, CompressedMeshEntityRefs)
    assert edge_a.topology is edge_b.topology


def test_schema_v11_project_is_read_edited_and_first_save_migrates_to_current() -> None:
    source = _mesh_scope_snapshot()
    legacy = encode_project_v11(source)
    loaded = decode_project(legacy)
    edited = replace(
        loaded.snapshot,
        named_regions=(
            *loaded.snapshot.named_regions,
            NamedRegion("More", (MeshEntityRef.node(99, part_id="P1"),)),
        ),
    )

    migrated = encode_project(edited)
    reopened = loads_project(dumps_project(edited))

    assert loaded.source_schema == 11
    assert migrated["schema"] == 14
    assert "compact_references" in migrated["project"]["authoring"]["named_regions"][0]
    assert reopened.snapshot.named_regions == edited.named_regions


def test_old_and_new_wire_formats_decode_to_identical_scope_inputs() -> None:
    source = _mesh_scope_snapshot()
    old = decode_project(encode_project_v11(source)).snapshot
    new = decode_project(encode_project_v12(source)).snapshot

    assert tuple(
        (region.name, tuple(region.references)) for region in old.named_regions
    ) == tuple(
        (region.name, tuple(region.references)) for region in new.named_regions
    )
    base_model = FEMModel(
        Mesh2D(
            [Node2D(value, float(value), 0.0) for value in range(1, 9)],
            [
                Element2D(value, [1, 2], "Truss2")
                for value in (10, 11, 14)
            ],
        ),
        metadata={
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {"node_ids": tuple(range(1, 9)), "element_ids": ()},
                "P2": {"node_ids": (), "element_ids": (10, 11, 14)},
            }
        },
    )
    old_model = materialize_native_scopes(
        deepcopy(base_model),
        previous_names=(),
        regions=old.named_regions[:2],
    )
    new_model = materialize_native_scopes(
        deepcopy(base_model),
        previous_names=(),
        regions=new.named_regions[:2],
    )

    assert old_model.node_sets == new_model.node_sets
    assert old_model.element_sets == new_model.element_sets


def test_v12_writer_consumes_compact_arrays_without_reference_expansion(
    monkeypatch,
) -> None:
    source = _mesh_scope_snapshot()

    def unexpected_iteration(self):
        del self
        raise AssertionError("v12 writer must consume compact arrays directly")

    monkeypatch.setattr(
        CompressedMeshEntityRefs,
        "__iter__",
        unexpected_iteration,
    )

    payload = encode_project_v12(source)

    assert payload["schema"] == 12
    assert payload["project"]["authoring"]["named_regions"][0][
        "compact_references"
    ]["kind"] == "node"
