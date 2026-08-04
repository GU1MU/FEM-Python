from __future__ import annotations

from dataclasses import replace

import pytest

from fem import geometry
from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    NativePart,
    prepare_part_boolean,
    prepare_solid_body_boolean,
)
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    LogicalEntityRef,
    MovedGeometry,
    BoxGeometry,
    CylinderGeometry,
    RectangleGeometry,
    MultiBodyGeometry,
    SolidBody,
    transform_solid_body,
)
from fem.io import (
    CURRENT_PROJECT_SCHEMA,
    dumps_project,
    load_project,
    loads_project,
    save_project,
)
from fem.io.project_v6 import dumps_project_v6
from fem.io.project_v7 import (
    ProjectV7DecodeError,
    ProjectV7EncodeError,
    decode_project_v7,
    encode_project_v7,
)
from fem.io.project_migration import migrate_project_snapshot_to_v7
from fem.mesh.settings import MeshSettings


def _two_part_snapshot() -> ProjectSnapshot:
    session = ModelSession()
    session.new_native_project("多部件模型")
    session.add_native_part(
        RectangleGeometry("草图-1", 4.0, 2.0),
        name="部件-1",
        mesh_settings=MeshSettings(0.5),
    )
    session.add_native_part(
        MovedGeometry(
            RectangleGeometry("草图-2", 2.0, 1.0),
            6.0,
            0.0,
        ),
        name="部件-2",
        mesh_settings=MeshSettings(0.25),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "两个面",
                (
                    LogicalEntityRef("face:P1/domain"),
                    LogicalEntityRef("face:P2/domain"),
                ),
            ),
        )
    )
    return session.prepare_project_save().snapshot


def test_v7_round_trip_preserves_part_ids_and_namespaces() -> None:
    source = _two_part_snapshot()
    payload = encode_project_v7(source)
    reopened = loads_project(dumps_project(source)).snapshot

    assert CURRENT_PROJECT_SCHEMA == 12
    assert payload["schema"] == 7
    assert reopened == source
    assert tuple(part.id for part in reopened.parts) == ("P1", "P2")
    assert tuple(
        reference.logical_id
        for reference in reopened.named_regions[0].references
    ) == ("face:P1/domain", "face:P2/domain")


def test_v7_round_trip_preserves_an_empty_native_model(tmp_path) -> None:
    session = ModelSession()
    session.new_native_project("模型-1")

    target = save_project(
        tmp_path / "empty.femproj",
        session.prepare_project_save(),
    )
    reopened = load_project(target).snapshot
    installed = ModelSession()
    installed.replace_from_snapshot(reopened)

    assert reopened.parts == ()
    assert reopened.active_part_id is None
    assert installed.snapshot().parts == ()
    assert installed.snapshot().active_part_id is None
    assert installed.can_save


def test_v7_mesh_entity_reference_round_trip_keeps_part_owner() -> None:
    source = replace(
        _two_part_snapshot(),
        named_regions=(
            NamedRegion(
                "网格节点",
                (MeshEntityRef.node(1, part_id="P1"),),
            ),
        ),
    )

    payload = encode_project_v7(source)
    reference_payload = payload["project"]["authoring"]["named_regions"][0][
        "references"
    ][0]
    reopened = decode_project_v7(payload)

    assert reference_payload["part_id"] == "P1"
    assert reopened.named_regions[0].references[0].part_id == "P1"


def test_v7_round_trip_preserves_no_active_part_when_all_are_suppressed() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 1.0, 1.0),
        name="部件-1",
    )
    session.suppress_native_part("P1")
    source = session.prepare_project_save().snapshot

    reopened = decode_project_v7(encode_project_v7(source))
    installed = ModelSession()
    installed.replace_from_snapshot(reopened)

    assert reopened.active_part_id is None
    assert installed.snapshot().active_part_id is None
    assert installed.snapshot().geometry_recipe is None


def test_v6_single_geometry_migrates_to_owned_p1() -> None:
    recipe = RectangleGeometry("旧草图", 4.0, 2.0)
    legacy = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("旧部件", "旧实体"),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(0.5),
        feature_history=derive_feature_history(recipe),
        named_regions=(
            NamedRegion(
                "旧面",
                (LogicalEntityRef("face:domain"),),
            ),
        ),
    )

    loaded = loads_project(dumps_project_v6(legacy))

    assert loaded.source_schema == 6
    assert loaded.notices
    assert loaded.snapshot.parts[0].id == "P1"
    assert loaded.snapshot.parts[0].name == "旧部件"
    assert (
        loaded.snapshot.named_regions[0].references[0].logical_id
        == "face:P1/domain"
    )


def test_legacy_multi_body_splits_into_deterministic_parts() -> None:
    legacy_geometry = MultiBodyGeometry(
        "旧多实体",
        (
            SolidBody(
                "B1",
                "旧部件-1",
                BoxGeometry("实体-1", 1.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "旧部件-2",
                MovedGeometry(
                    BoxGeometry("实体-2", 1.0, 1.0, 1.0),
                    3.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
        retired_body_ids=("B3",),
    )
    legacy = ProjectSnapshot(
        geometry_recipe=legacy_geometry,
        mesh_settings=MeshSettings(0.5, cell_shape="tetrahedron"),
        named_regions=(
            NamedRegion(
                "两个顶面",
                (
                    LogicalEntityRef("face:B1/top"),
                    LogicalEntityRef("face:B2/top"),
                ),
            ),
        ),
    )

    migrated, notices = migrate_project_snapshot_to_v7(legacy)

    assert notices
    assert tuple(part.id for part in migrated.parts) == ("P1", "P2")
    assert tuple(part.name for part in migrated.parts) == (
        "旧部件-1",
        "旧部件-2",
    )
    assert migrated.retired_part_ids == ("P3",)
    assert tuple(
        reference.logical_id
        for reference in migrated.named_regions[0].references
    ) == ("face:P1/top", "face:P2/top")


def test_v7_rejects_unknown_part_reference() -> None:
    payload = encode_project_v7(_two_part_snapshot())
    payload["project"]["authoring"]["named_regions"][0]["references"][0] = (
        "face:P9/domain"
    )

    with pytest.raises(ProjectV7DecodeError, match="P9"):
        decode_project_v7(payload)


def test_v7_part_boolean_round_trip_authenticates_proof(real_gmsh) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("目标", 2.0, 1.0, 1.0),
        name="目标部件",
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("工具", 1.0, 1.0, 1.0),
            1.5,
            0.0,
            0.0,
        ),
        name="工具部件",
    )
    before = session.snapshot()
    with geometry.model("v7 部件布尔", dimension=3) as cad:
        prepared = prepare_part_boolean(
            cad,
            before.part("P1"),
            before.part("P2"),
            "fuse",
            result_part_id="P3",
            feature_id="PBF1",
            result_name="合并结果-1",
        )
    session.apply_part_boolean(
        "P1",
        "P2",
        "fuse",
        "合并结果-1",
        result=prepared,
    )
    source = session.prepare_project_save().snapshot
    payload = encode_project_v7(source)

    assert decode_project_v7(payload) == source

    context = payload["project"]["authoring"]["parts"][2]["geometry"][
        "part_context"
    ]
    context["topology_mappings"] = context["topology_mappings"][:-1]
    with pytest.raises(ProjectV7DecodeError):
        decode_project_v7(payload)

    forged_source = replace(
        source.parts[0],
        geometry_recipe=BoxGeometry("伪造源", 3.0, 1.0, 1.0),
    )
    forged = replace(
        source,
        parts=(forged_source, *source.parts[1:]),
    )
    with pytest.raises(ProjectV7EncodeError, match="live source"):
        encode_project_v7(forged)


def test_active_v6_body_boolean_migrates_to_suppressed_sources(
    real_gmsh,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "旧多实体几何",
        (
            SolidBody(
                "B1",
                "目标实体",
                BoxGeometry("目标", 2.0, 2.0, 2.0),
            ),
            SolidBody(
                "B2",
                "工具实体",
                MovedGeometry(
                    CylinderGeometry("工具", 0.4, 2.0),
                    1.0,
                    1.0,
                    0.0,
                ),
            ),
        ),
        retired_body_ids=("B3",),
    )
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=MeshSettings(0.5, cell_shape="tetrahedron"),
    )
    with geometry.model("旧实体布尔", dimension=3) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "cut",
        )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        prepared.geometry,
    )

    migrated, notices = migrate_project_snapshot_to_v7(
        session.prepare_project_save().snapshot
    )

    assert notices
    assert tuple(part.id for part in migrated.parts) == ("P1", "P2", "P4")
    assert migrated.retired_part_ids == ("P3",)
    assert migrated.part_boolean_undo_records[0].feature_id == "PBF1"
    assert migrated.active_part_id == "P4"
    assert migrated.parts[0].suppressed
    assert migrated.parts[1].suppressed
    assert not migrated.parts[2].suppressed
    assert migrated.parts[2].geometry_recipe.part_context.proven
    assert decode_project_v7(encode_project_v7(migrated)) == migrated

    reopened = ModelSession()
    reopened.replace_from_snapshot(migrated)
    reopened.undo_part_boolean("P4")
    restored = reopened.snapshot()
    assert not restored.part("P1").suppressed
    assert not restored.part("P2").suppressed


def test_nested_v6_body_booleans_migrate_to_reversible_part_dag(
    real_gmsh,
) -> None:
    del real_gmsh
    source = MultiBodyGeometry(
        "旧嵌套布尔",
        (
            SolidBody(
                "B1",
                "外层目标",
                BoxGeometry("外层目标", 2.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "内层目标",
                MovedGeometry(
                    BoxGeometry("内层目标", 2.0, 1.0, 1.0),
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
            SolidBody(
                "B3",
                "内层工具",
                MovedGeometry(
                    BoxGeometry("内层工具", 2.0, 1.0, 1.0),
                    11.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry.model("旧嵌套布尔-内层", dimension=3) as cad:
        inner = prepare_solid_body_boolean(
            cad,
            source,
            "B2",
            "B3",
            "fuse",
        )
    placed_inner = transform_solid_body(
        inner.geometry,
        "B2",
        move=(-9.0, 0.0, 0.0),
    )
    with geometry.model("旧嵌套布尔-外层", dimension=3) as cad:
        outer = prepare_solid_body_boolean(
            cad,
            placed_inner,
            "B1",
            "B2",
            "fuse",
        )
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=MeshSettings(0.5, cell_shape="tetrahedron"),
    )
    session.replace_native_geometry_inputs((NativePart(),), inner.geometry)
    session.replace_native_geometry_inputs((NativePart(),), placed_inner)
    session.replace_native_geometry_inputs((NativePart(),), outer.geometry)

    migrated, _notices = migrate_project_snapshot_to_v7(
        session.prepare_project_save().snapshot
    )

    assert tuple(part.id for part in migrated.parts) == (
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    )
    assert tuple(
        record.feature_id for record in migrated.part_boolean_undo_records
    ) == ("PBF1", "PBF2")
    assert migrated.active_part_id == "P5"
    for record in migrated.part_boolean_undo_records:
        live = next(
            part
            for part in migrated.parts
            if part.id == record.result_part_id
        )
        assert live.provenance == record.result_part.provenance
        assert (
            live.geometry_recipe == record.result_part.geometry_recipe
            or getattr(live.geometry_recipe, "base", None)
            == record.result_part.geometry_recipe
        )
    assert decode_project_v7(encode_project_v7(migrated)) == migrated

    reopened = ModelSession()
    reopened.replace_from_snapshot(migrated)
    reopened.undo_part_boolean("P5")
    assert reopened.snapshot().active_part_id == "P1"
    assert not reopened.snapshot().part("P4").suppressed
    reopened.undo_part_boolean("P4")
    restored = reopened.snapshot()
    assert restored.active_part_id == "P2"
    assert not restored.part("P2").suppressed
    assert not restored.part("P3").suppressed
