from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import pytest

from fem import geometry as geometry_runtime
from fem.application import (
    ModelSession,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
    SessionStateError,
    prepare_solid_body_boolean,
)
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    BooleanBodyContext,
    BoxGeometry,
    BooleanGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    LogicalEntityRef,
    MovedGeometry,
    MultiBodyGeometry,
    SolidBody,
    add_solid_body,
    delete_solid_body,
    transform_solid_body,
    logical_ref_sort_key,
    undo_solid_body_feature,
)
from fem.core.model import MaterialDefinition
from fem.io.project import decode_project
from fem.io.project_v4 import encode_project_v4
from fem.io.project_v5 import (
    ProjectV5DecodeError,
    ProjectV5EncodeError,
    decode_project_v5,
    encode_project_v5,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings


def _multi_body_snapshot(
    *,
    named_regions=(),
    mesh_settings: MeshSettings | None = None,
    body_count: int = 2,
) -> ProjectSnapshot:
    bodies = (
            SolidBody(
                "B1",
                "Body-1",
                BoxGeometry("Box-1", 1.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Body-2",
                MovedGeometry(
                    BoxGeometry("Box-2", 1.0, 1.0, 1.0),
                    2.0,
                    0.0,
                    0.0,
                ),
            ),
        )
    geometry = MultiBodyGeometry(
        "Geometry",
        bodies[:body_count],
    )
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=geometry,
        mesh_settings=mesh_settings or MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
        ),
        feature_history=derive_feature_history(geometry),
        named_regions=tuple(named_regions),
    )


def _forged_boolean_snapshot() -> ProjectSnapshot:
    result_face = "face:boolean/BF1/combined/top/top"
    context = BooleanBodyContext(
        "BF1",
        "B1",
        "B2",
        "Tool",
        (
            BooleanLineageEntity(
                "face",
                result_face,
                "boolean.combined",
            ),
            BooleanLineageEntity(
                "body",
                "body:domain",
                "boolean.result",
            ),
        ),
        (
            BooleanLineageMapping(
                "target",
                "body:domain",
                "body:domain",
                "preserved",
            ),
            BooleanLineageMapping(
                "target",
                "face:top",
                result_face,
                "derived",
            ),
            BooleanLineageMapping(
                "tool",
                "face:top",
                result_face,
                "derived",
            ),
        ),
    )
    recipe = BooleanGeometry(
        "Target",
        "fuse",
        BoxGeometry("Target", 1.0, 1.0, 1.0),
        MovedGeometry(
            BoxGeometry("Tool", 1.0, 1.0, 1.0),
            0.5,
            0.0,
            0.0,
        ),
        context,
    )
    source = MultiBodyGeometry(
        "Boolean Source",
        (
            SolidBody("B1", "Target", recipe.object_geometry),
            SolidBody("B2", "Tool", recipe.tool_geometry),
        ),
    )
    geometry = MultiBodyGeometry(
        "Boolean Geometry",
        (SolidBody("B1", "Target", recipe),),
        ("B2",),
    )
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
        ),
    )
    session.replace_native_geometry_inputs((NativePart(),), geometry)
    return session.prepare_project_save().snapshot


@lru_cache(maxsize=1)
def _proven_boolean_snapshot() -> ProjectSnapshot:
    source = MultiBodyGeometry(
        "Boolean Geometry",
        (
            SolidBody(
                "B1",
                "Target",
                BoxGeometry("Target", 1.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("Tool", 1.0, 1.0, 1.0),
                    0.5,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry_runtime.model(
        "project-v5-proven-boolean-fixture",
        dimension=3,
    ) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "fuse",
        )
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        source,
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
        ),
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        prepared.geometry,
    )
    return session.prepare_project_save().snapshot


def test_v5_multi_body_roundtrip_uses_canonical_refs_and_part_wire() -> None:
    snapshot = _multi_body_snapshot(
        named_regions=(
                NamedRegion(
                "SurfaceTop",
                (LogicalEntityRef("face:B1/top"),),
            ),
        ),
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("face:B2/top"),
                    0.2,
                ),
            ),
        ),
    )

    payload = encode_project_v5(snapshot)
    reopened = decode_project_v5(payload)

    assert payload["schema"] == 5
    assert payload["project"]["authoring"]["part"] == {"name": "Part-1"}
    assert reopened == snapshot


def test_v5_writer_rejects_legacy_alias_for_canonical_multi_body() -> None:
    snapshot = _multi_body_snapshot(
        body_count=1,
        named_regions=(
            NamedRegion(
                "Legacy",
                (LogicalEntityRef("face:top"),),
            ),
        )
    )

    with pytest.raises(
        ProjectV5EncodeError,
        match="canonical Body namespace",
    ):
        encode_project_v5(snapshot)


def test_v5_round_trip_preserves_model_name_and_accepts_older_payloads() -> None:
    snapshot = replace(
        _multi_body_snapshot(body_count=1),
        model_name="Bracket",
    )
    payload = encode_project_v5(snapshot)

    assert payload["project"]["authoring"]["model_name"] == "Bracket"
    assert decode_project_v5(payload).model_name == "Bracket"

    legacy_payload = deepcopy(payload)
    del legacy_payload["project"]["authoring"]["model_name"]
    assert decode_project_v5(legacy_payload).model_name == "Model-1"


def test_v5_reader_rejects_legacy_alias_with_reference_path() -> None:
    snapshot = _multi_body_snapshot(
        body_count=1,
        named_regions=(
            NamedRegion(
                "SurfaceTop",
                (LogicalEntityRef("face:B1/top"),),
            ),
        )
    )
    payload = encode_project_v5(snapshot)
    payload["project"]["authoring"]["named_regions"][0]["references"][0] = (
        "face:top"
    )

    with pytest.raises(
        ProjectV5DecodeError,
        match=r"named_regions\[0\].*canonical Body namespace",
    ):
        decode_project_v5(payload)


def test_v4_single_body_refs_migrate_atomically_to_v5_namespace() -> None:
    geometry = BoxGeometry("Box", 1.0, 1.0, 1.0)
    legacy = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(name="Part", body_name="Legacy Body"),),
        geometry_recipe=geometry,
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
            local_controls=(
                LocalMeshControl(LogicalEntityRef("face:top"), 0.2),
            ),
        ),
        feature_history=derive_feature_history(geometry),
        named_regions=(
            NamedRegion(
                "SolidRegion",
                (LogicalEntityRef("body:domain"),),
            ),
        ),
    )

    loaded = decode_project(encode_project_v4(legacy))

    migrated = loaded.snapshot
    assert loaded.source_schema == 4
    assert migrated.active_part_id == "P1"
    assert migrated.parts[0].name == "Legacy Body"
    assert migrated.parts[0].geometry_recipe == geometry
    assert migrated.named_regions[0].references == (
        LogicalEntityRef("body:P1/domain"),
    )
    assert migrated.mesh_settings.local_controls[0].target == (
        LogicalEntityRef("face:P1/top")
    )


def test_v5_reader_rejects_non_multi_body_3d_geometry() -> None:
    payload = encode_project_v5(_multi_body_snapshot())
    legacy = BoxGeometry("Box", 1.0, 1.0, 1.0)
    v4_payload = encode_project_v4(
        ProjectSnapshot(
            source_kind="native",
            parts=(NativePart(),),
            geometry_recipe=legacy,
            mesh_settings=MeshSettings(
                size=0.5,
                cell_shape="tetrahedron",
            ),
            feature_history=derive_feature_history(legacy),
        )
    )
    payload["project"]["authoring"]["geometry"] = (
        v4_payload["project"]["authoring"]["geometry"]
    )

    with pytest.raises(ProjectV5DecodeError, match="MultiBodyGeometry"):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_forged_incomplete_boolean_proof() -> None:
    payload = encode_project_v5(_proven_boolean_snapshot())
    context = payload["project"]["authoring"]["geometry"]["bodies"][0][
        "recipe"
    ]["body_context"]
    context["topology_mappings"] = context["topology_mappings"][:1]

    with pytest.raises(
        ProjectV5DecodeError,
        match="map every result entity",
    ):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_unknown_boolean_mapping_source() -> None:
    payload = encode_project_v5(_proven_boolean_snapshot())
    mappings = payload["project"]["authoring"]["geometry"]["bodies"][0][
        "recipe"
    ]["body_context"]["topology_mappings"]
    mapping = next(
        item
        for item in mappings
        if item["source"] == "tool"
        and item["source_logical_id"].startswith("face:")
    )
    mapping["source_logical_id"] = "face:missing"
    mappings.sort(
        key=lambda item: (
            item["source"],
            logical_ref_sort_key(
                LogicalEntityRef(item["source_logical_id"])
            ),
            logical_ref_sort_key(
                LogicalEntityRef(item["target_logical_id"])
            ),
            item["relation"],
        )
    )

    with pytest.raises(
        ProjectV5DecodeError,
        match="unknown tool source",
    ):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_duplicate_active_boolean_feature_ids() -> None:
    payload = encode_project_v5(_proven_boolean_snapshot())
    geometry = payload["project"]["authoring"]["geometry"]
    duplicate = {
        **geometry["bodies"][0],
        "id": "B3",
        "name": "Second Target",
        "recipe": {
            **geometry["bodies"][0]["recipe"],
            "body_context": {
                **geometry["bodies"][0]["recipe"]["body_context"],
                "target_body_id": "B3",
                "tool_body_id": "B4",
                "tool_body_name": "Second Tool",
            },
        },
    }
    geometry["bodies"].append(duplicate)
    geometry["retired_body_ids"].append("B4")

    with pytest.raises(
        ProjectV5DecodeError,
        match="duplicates active feature",
    ):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_active_retired_boolean_feature_conflict() -> None:
    payload = encode_project_v5(_proven_boolean_snapshot())
    payload["project"]["authoring"]["geometry"][
        "retired_boolean_feature_ids"
    ] = ["BF1"]

    with pytest.raises(
        ProjectV5DecodeError,
        match="conflicts with active feature",
    ):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_noncanonical_boolean_proof_order() -> None:
    payload = encode_project_v5(_proven_boolean_snapshot())
    entities = payload["project"]["authoring"]["geometry"]["bodies"][0][
        "recipe"
    ]["body_context"]["result_entities"]
    entities.reverse()

    with pytest.raises(
        ProjectV5DecodeError,
        match="canonical logical order",
    ):
        decode_project_v5(payload)


@pytest.mark.gmsh
def test_v5_rejects_structurally_consistent_forged_boolean_proof() -> None:
    with pytest.raises(
        ProjectV5EncodeError,
        match="OCC proof authentication failed",
    ):
        encode_project_v5(_forged_boolean_snapshot())


@pytest.mark.gmsh
def test_v5_roundtrip_accepts_nested_boolean_tool_history() -> None:
    source = MultiBodyGeometry(
        "Nested Boolean Geometry",
        (
            SolidBody(
                "B1",
                "Outer Target",
                BoxGeometry("Outer Target", 2.0, 1.0, 1.0),
            ),
            SolidBody(
                "B2",
                "Inner Target",
                MovedGeometry(
                    BoxGeometry("Inner Target", 2.0, 1.0, 1.0),
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
            SolidBody(
                "B3",
                "Inner Tool",
                MovedGeometry(
                    BoxGeometry("Inner Tool", 2.0, 1.0, 1.0),
                    11.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry_runtime.model(
        "project-v5-nested-boolean-inner",
        dimension=3,
    ) as cad:
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
    with geometry_runtime.model(
        "project-v5-nested-boolean-outer",
        dimension=3,
    ) as cad:
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
        mesh_settings=MeshSettings(
            size=0.5,
            cell_shape="tetrahedron",
        ),
    )
    session.replace_native_geometry_inputs((NativePart(),), inner.geometry)
    session.replace_native_geometry_inputs((NativePart(),), placed_inner)
    session.replace_native_geometry_inputs((NativePart(),), outer.geometry)
    snapshot = session.prepare_project_save().snapshot

    assert decode_project_v5(encode_project_v5(snapshot)) == snapshot


@pytest.mark.gmsh
def test_v5_save_prunes_undo_records_for_deleted_boolean_history() -> None:
    session = ModelSession()
    session.replace_from_snapshot(_proven_boolean_snapshot())
    with_extra_body = add_solid_body(
        session.snapshot().geometry_recipe,
        MovedGeometry(
            BoxGeometry("Survivor", 1.0, 1.0, 1.0),
            10.0,
            0.0,
            0.0,
        ),
        name="Survivor",
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        with_extra_body,
    )
    without_boolean_body = delete_solid_body(with_extra_body, "B1")
    assert without_boolean_body is not None
    session.replace_native_geometry_inputs(
        (NativePart(),),
        without_boolean_body,
    )

    snapshot = session.prepare_project_save().snapshot

    assert snapshot.boolean_reference_undo_records == ()
    assert decode_project_v5(encode_project_v5(snapshot)) == snapshot


@pytest.mark.gmsh
def test_v5_preserves_historical_definition_context_for_conflicting_undo() -> None:
    fixture = _proven_boolean_snapshot()
    record = fixture.boolean_reference_undo_records[0]
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        record.before_geometry,
        mesh_settings=fixture.mesh_settings,
    )
    session.replace_named_regions(
        (NamedRegion("ToolBody", (LogicalEntityRef("body:B2"),)),)
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "ToolBody"),),
        (),
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        record.after_geometry,
    )
    assert session.snapshot().assignments == ()

    session.replace_model_definitions((), (), (), ())
    snapshot = session.prepare_project_save().snapshot
    reopened = decode_project_v5(encode_project_v5(snapshot))
    assert reopened == snapshot

    reopened_session = ModelSession()
    reopened_session.replace_from_snapshot(reopened)
    restored_geometry = undo_solid_body_feature(
        reopened.geometry_recipe,
        "B1",
    )
    with pytest.raises(
        SessionStateError,
        match="undo-reference-conflict",
    ):
        reopened_session.replace_native_geometry_inputs(
            (NativePart(),),
            restored_geometry,
        )
