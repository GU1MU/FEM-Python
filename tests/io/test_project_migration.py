from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from fem.application.definitions import (
    NamedRegion,
    NativePart,
    RegionAssignment,
)
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    NodalLoad,
    SurfaceLoad,
)
from fem.elements import BeamOrientation
from fem.geometry.recipes import (
    PlateWithHoleGeometry,
)
from fem.geometry.references import LogicalEntityRef
from fem.io.project_migration import ProjectMigrationNotice
from fem.io.project_v1 import (
    ProjectV1DecodeError,
    ProjectV1EncodeError,
    _decode_project_v1_loaded,
    decode_project_v1,
    encode_project_v1,
    load_project_v1,
    save_project_v1,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff


FIXTURES = Path(__file__).parents[1] / "helpers" / "fixtures" / "femproj" / "v1"
GLOBAL_FALLOFF = MeshSizeFalloff("global_size", 0.0, 2.0)
RADIUS_FALLOFF = MeshSizeFalloff("target_radius", 0.25, 2.0)


def _rectangle_payload() -> dict[str, object]:
    return {
        "schema": 1,
        "logical_topology_version": 1,
        "source": "native",
        "parts": [{"name": "Part-1", "body_name": "Body-1"}],
        "geometry": {
            "type": "RectangleGeometry",
            "name": "Rectangle",
            "width": 4.0,
            "height": 2.0,
        },
        "mesh_settings": None,
        "feature_history": [],
        "named_regions": [],
        "materials": [],
        "sections": [],
        "assignments": [],
        "steps": [],
    }


def _plate_payload() -> dict[str, object]:
    payload = _rectangle_payload()
    payload["geometry"] = {
        "type": "PlateWithHoleGeometry",
        "name": "Plate",
        "width": 10.0,
        "height": 6.0,
        "hole_x": 5.0,
        "hole_y": 3.0,
        "hole_radius": 1.0,
    }
    return payload


def _snapshot(
    *,
    recipe: object | None = None,
    controls: tuple[LocalMeshControl, ...] = (),
    steps: tuple[AnalysisStep, ...] = (),
) -> ProjectSnapshot:
    geometry = (
        PlateWithHoleGeometry("Plate", 10.0, 6.0, 5.0, 3.0, 1.0)
        if recipe is None
        else recipe
    )
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=geometry,
        mesh_settings=MeshSettings(1.0, local_controls=controls),
        feature_history=derive_feature_history(geometry),
        analysis_definitions=steps,
    )


def _notices(payload: dict[str, object]) -> tuple[
    ProjectSnapshot,
    tuple[ProjectMigrationNotice, ...],
]:
    return _decode_project_v1_loaded(payload)


def test_real_minimal_and_canonical_full_fixtures_migrate_without_rewrite() -> None:
    for name in (
        "minimal_rectangle.femproj",
        "full_rectangle_canonical.femproj",
    ):
        path = FIXTURES / name
        before = path.read_bytes()

        snapshot = load_project_v1(path)

        assert snapshot.source_path == path
        assert snapshot.feature_history == derive_feature_history(
            snapshot.geometry_recipe
        )
        assert path.read_bytes() == before


def test_real_line_load_fixture_preserves_bytes_and_fails_native_capability() -> None:
    path = FIXTURES / "line_load_unsupported.femproj"
    before = path.read_bytes()

    with pytest.raises(ProjectV1DecodeError, match="beam_element_set"):
        load_project_v1(path)

    assert path.read_bytes() == before


def test_mapping_level_entry_returns_typed_notices_and_canonicalizes_cache() -> None:
    payload = _rectangle_payload()
    payload["feature_history"] = [
        {
            "name": "Legacy-1",
            "kind": "legacy",
            "payload": {"summary": "stale"},
        }
    ]

    snapshot, notices = _notices(payload)

    assert snapshot.feature_history == derive_feature_history(
        snapshot.geometry_recipe
    )
    assert all(type(notice) is ProjectMigrationNotice for notice in notices)
    assert [notice.code for notice in notices] == [
        "project.schema.v1",
        "project.v1.feature_history_rederived",
    ]
    assert notices[-1].path == "$.feature_history"


def test_missing_parts_defaults_but_explicit_zero_or_multiple_rejects() -> None:
    missing = _rectangle_payload()
    missing.pop("parts")

    assert decode_project_v1(missing).parts == (NativePart(),)

    for parts in ([], [{}, {}]):
        payload = _rectangle_payload()
        payload["parts"] = parts
        with pytest.raises(ProjectV1DecodeError, match=r"\$\.parts.*只包含一个"):
            decode_project_v1(payload)


@pytest.mark.parametrize(
    ("ordinal", "logical_id"),
    [
        (1, "edge:bottom"),
        (2, "edge:right"),
        (4, "edge:left"),
    ],
)
def test_frozen_v1_ordinal_first_middle_last_migrates_to_logical_ref(
    ordinal: int,
    logical_id: str,
) -> None:
    payload = _rectangle_payload()
    payload["named_regions"] = [
        {
            "name": "USER_EDGE",
            "entity_kind": "edge",
            "entity_ids": [ordinal],
        }
    ]

    snapshot = decode_project_v1(payload)

    assert snapshot.named_regions == (
        NamedRegion("USER_EDGE", (LogicalEntityRef(logical_id),)),
    )


@pytest.mark.parametrize(
    ("geometry", "entity_kind", "ordinal", "logical_id"),
    [
        (
            {"type": "DiskGeometry", "name": "D", "radius": 2.0},
            "edge",
            1,
            "edge:outer",
        ),
        (
            {
                "type": "PlateWithHoleGeometry",
                "name": "P",
                "width": 10.0,
                "height": 6.0,
                "hole_x": 5.0,
                "hole_y": 3.0,
                "hole_radius": 1.0,
            },
            "edge",
            2,
            "edge:outer-loop",
        ),
        (
            {
                "type": "BoxGeometry",
                "name": "B",
                "width": 2.0,
                "depth": 3.0,
                "height": 4.0,
            },
            "face",
            6,
            "face:left",
        ),
        (
            {
                "type": "CylinderGeometry",
                "name": "C",
                "radius": 2.0,
                "height": 4.0,
            },
            "face",
            2,
            "face:top",
        ),
        (
            {
                "type": "SketchGeometry",
                "name": "S",
                "contours": [
                    {
                        "type": "rectangle",
                        "operation": "material",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 4.0,
                        "height": 2.0,
                    }
                ],
            },
            "edge",
            3,
            "edge:top",
        ),
        (
            {
                "type": "MovedGeometry",
                "base": {
                    "type": "RectangleGeometry",
                    "name": "R",
                    "width": 4.0,
                    "height": 2.0,
                },
                "dx": 1.0,
                "dy": 2.0,
                "dz": 0.0,
            },
            "edge",
            4,
            "edge:left",
        ),
        (
            {
                "type": "RotatedGeometry",
                "base": {
                    "type": "RectangleGeometry",
                    "name": "R",
                    "width": 4.0,
                    "height": 2.0,
                },
                "axis": "z",
                "angle_degrees": 30.0,
            },
            "point",
            2,
            "point:bottom-right",
        ),
        (
            {
                "type": "ExtrudedGeometry",
                "base": {
                    "type": "RectangleGeometry",
                    "name": "R",
                    "width": 4.0,
                    "height": 2.0,
                },
                "height": 3.0,
            },
            "face",
            6,
            "face:side/left",
        ),
        (
            {
                "type": "BooleanGeometry",
                "name": "Cut",
                "operation": "cut",
                "object": {
                    "type": "RectangleGeometry",
                    "name": "R",
                    "width": 4.0,
                    "height": 2.0,
                },
                "tool": {
                    "type": "MovedGeometry",
                    "base": {
                        "type": "DiskGeometry",
                        "name": "D",
                        "radius": 0.5,
                    },
                    "dx": 2.0,
                    "dy": 1.0,
                    "dz": 0.0,
                },
            },
            "edge",
            1,
            "edge:hole-loop",
        ),
    ],
)
def test_v1_ordinal_migration_covers_supported_recipe_families(
    geometry: dict[str, object],
    entity_kind: str,
    ordinal: int,
    logical_id: str,
) -> None:
    payload = _rectangle_payload()
    payload["geometry"] = geometry
    payload["named_regions"] = [
        {
            "name": "USER_REF",
            "entity_kind": entity_kind,
            "entity_ids": [ordinal],
        }
    ]

    snapshot = decode_project_v1(payload)

    assert snapshot.named_regions[0].references == (
        LogicalEntityRef(logical_id),
    )


@pytest.mark.parametrize("ordinal", [0, -1, True, 5])
def test_invalid_v1_ordinal_fails_closed(ordinal: object) -> None:
    payload = _rectangle_payload()
    payload["named_regions"] = [
        {
            "name": "USER_EDGE",
            "entity_kind": "edge",
            "entity_ids": [ordinal],
        }
    ]

    with pytest.raises(ProjectV1DecodeError, match="entity_ids|ordinal"):
        decode_project_v1(payload)


@pytest.mark.parametrize("reference_owner", ["region", "control"])
def test_missing_topology_version_with_integer_reference_rejects(
    reference_owner: str,
) -> None:
    payload = _rectangle_payload()
    payload.pop("logical_topology_version")
    if reference_owner == "region":
        payload["named_regions"] = [
            {
                "name": "USER_EDGE",
                "entity_kind": "edge",
                "entity_ids": [1],
            }
        ]
    else:
        payload["mesh_settings"] = {
            "size": 1.0,
            "local_controls": [
                {
                    "entity_kind": "edge",
                    "entity_id": 1,
                    "size": 0.5,
                }
            ],
        }

    with pytest.raises(ProjectV1DecodeError, match="逻辑拓扑契约版本"):
        decode_project_v1(payload)


def test_unknown_topology_version_rejects() -> None:
    payload = _rectangle_payload()
    payload["logical_topology_version"] = 99

    with pytest.raises(ProjectV1DecodeError, match="不支持的逻辑拓扑契约版本"):
        decode_project_v1(payload)


def test_empty_named_region_reports_name_and_json_path() -> None:
    payload = _rectangle_payload()
    payload["named_regions"] = [
        {
            "name": "EMPTY_REGION",
            "entity_kind": "edge",
            "entity_ids": [],
        }
    ]

    with pytest.raises(
        ProjectV1DecodeError,
        match=r"\$\.named_regions\[0\].*EMPTY_REGION",
    ):
        decode_project_v1(payload)


def test_local_size_without_topology_version_uses_unique_hole_radius_proof() -> None:
    payload = _plate_payload()
    payload.pop("logical_topology_version")
    payload["mesh_settings"] = {
        "size": 1.0,
        "local_size": 0.2,
        "local_controls": [],
    }

    snapshot, notices = _notices(payload)

    assert snapshot.mesh_settings.local_controls == (
        LocalMeshControl(
            LogicalEntityRef("edge:hole-loop"),
            0.2,
            RADIUS_FALLOFF,
        ),
    )
    assert "project.v1.local_size_migrated" in {
        notice.code for notice in notices
    }


def test_local_size_rejects_disk_outer_as_unproven_hole() -> None:
    payload = _rectangle_payload()
    payload["geometry"] = {
        "type": "DiskGeometry",
        "name": "Disk",
        "radius": 2.0,
    }
    payload["mesh_settings"] = {
        "size": 1.0,
        "local_size": 0.2,
        "local_controls": [],
    }

    with pytest.raises(ProjectV1DecodeError, match="唯一圆孔 target"):
        decode_project_v1(payload)


def test_local_size_and_ordinary_hole_control_coexist_as_distinct_profiles() -> None:
    payload = _plate_payload()
    payload["mesh_settings"] = {
        "size": 1.0,
        "local_size": 0.2,
        "local_controls": [
            {
                "entity_kind": "edge",
                "entity_id": 1,
                "size": 0.4,
            }
        ],
    }

    snapshot = decode_project_v1(payload)

    assert snapshot.mesh_settings.local_controls == (
        LocalMeshControl(
            LogicalEntityRef("edge:hole-loop"),
            0.4,
            GLOBAL_FALLOFF,
        ),
        LocalMeshControl(
            LogicalEntityRef("edge:hole-loop"),
            0.2,
            RADIUS_FALLOFF,
        ),
    )


def test_duplicate_legacy_control_same_size_dedupes_and_conflict_rejects() -> None:
    payload = _plate_payload()
    payload["mesh_settings"] = {
        "size": 1.0,
        "local_controls": [
            {"entity_kind": "edge", "entity_id": 1, "size": 0.4},
            {"entity_kind": "edge", "entity_id": 1, "size": 0.4},
        ],
    }

    snapshot = decode_project_v1(payload)

    assert len(snapshot.mesh_settings.local_controls) == 1

    conflict = deepcopy(payload)
    conflict["mesh_settings"]["local_controls"][1]["size"] = 0.3
    with pytest.raises(ProjectV1DecodeError, match="冲突 size"):
        decode_project_v1(conflict)


@pytest.mark.parametrize(
    ("collection", "entry"),
    [
        (
            "boundaries",
            {
                "target": 1,
                "first_component": 1,
                "last_component": 2,
                "value": 0.0,
            },
        ),
        ("cloads", {"target": 2, "component": 1, "value": 10.0}),
        (
            "line_loads",
            {
                "target": 3,
                "vector": [1.0, 0.0],
                "coordinate_system": "global",
            },
        ),
        (
            "gravity_loads",
            {"acceleration": [0.0, -9.81], "target": 4},
        ),
    ],
)
def test_integer_analysis_targets_decode_wire_then_migration_rejects(
    collection: str,
    entry: dict[str, object],
) -> None:
    payload = _rectangle_payload()
    payload["steps"] = [{"name": "Load", collection: [entry]}]

    with pytest.raises(ProjectV1DecodeError, match="mesh integer target"):
        decode_project_v1(payload)


def test_stable_target_missing_or_capability_mismatch_rejects() -> None:
    missing = _rectangle_payload()
    missing["steps"] = [
        {
            "name": "Load",
            "boundaries": [
                {
                    "target": "MISSING",
                    "first_component": 1,
                    "last_component": 2,
                    "value": 0.0,
                }
            ],
        }
    ]
    mismatch = deepcopy(missing)
    mismatch["steps"][0]["boundaries"][0]["target"] = "DOMAIN"

    with pytest.raises(ProjectV1DecodeError, match="unknown native region"):
        decode_project_v1(missing)
    with pytest.raises(ProjectV1DecodeError, match="cannot produce 'node_set'"):
        decode_project_v1(mismatch)


def test_v1_edge_and_surface_loads_obey_recipe_dimension_capabilities() -> None:
    rectangle = _rectangle_payload()
    rectangle["steps"] = [
        {
            "name": "Load",
            "edge_loads": [
                {
                    "edge": "LEFT",
                    "vector": [0.0, -1.0],
                    "magnitude": None,
                    "load_type": "traction",
                }
            ],
        }
    ]
    box = _rectangle_payload()
    box["geometry"] = {
        "type": "BoxGeometry",
        "name": "Box",
        "width": 2.0,
        "depth": 3.0,
        "height": 4.0,
    }
    box["steps"] = [
        {
            "name": "Load",
            "surface_loads": [
                {
                    "surface": "FRONT",
                    "vector": [0.0, 0.0, -1.0],
                    "magnitude": None,
                    "load_type": "traction",
                }
            ],
        }
    ]

    rectangle_snapshot = decode_project_v1(rectangle)
    box_snapshot = decode_project_v1(box)

    assert rectangle_snapshot.analysis_definitions[0].edge_loads == (
        EdgeLoad("LEFT", (0.0, -1.0)),
    )
    assert box_snapshot.analysis_definitions[0].surface_loads == (
        SurfaceLoad("FRONT", (0.0, 0.0, -1.0)),
    )

    wrong_2d = deepcopy(rectangle)
    wrong_2d["steps"][0] = {
        "name": "Load",
        "surface_loads": [
            {
                "surface": "LEFT",
                "vector": [0.0, -1.0],
                "magnitude": None,
                "load_type": "traction",
            }
        ],
    }
    wrong_3d = deepcopy(box)
    wrong_3d["steps"][0] = {
        "name": "Load",
        "edge_loads": [
            {
                "edge": "FRONT",
                "vector": [0.0, 0.0, -1.0],
                "magnitude": None,
                "load_type": "traction",
            }
        ],
    }

    with pytest.raises(ProjectV1DecodeError, match="cannot produce 'surface'"):
        decode_project_v1(wrong_2d)
    with pytest.raises(ProjectV1DecodeError, match="cannot produce 'edge'"):
        decode_project_v1(wrong_3d)


def test_current_writer_reverse_matrix_round_trips_both_profiles() -> None:
    target = LogicalEntityRef("edge:hole-loop")
    snapshot = _snapshot(
        controls=(
            LocalMeshControl(target, 0.4, GLOBAL_FALLOFF),
            LocalMeshControl(target, 0.2, RADIUS_FALLOFF),
        )
    )

    payload = encode_project_v1(snapshot)
    reopened = decode_project_v1(payload)

    assert payload["mesh_settings"]["local_size"] == 0.2
    assert payload["mesh_settings"]["local_controls"] == [
        {"entity_kind": "edge", "entity_id": 1, "size": 0.4}
    ]
    assert reopened == snapshot


def test_current_writer_rejects_unsupported_falloff_and_unproven_radius_target() -> None:
    hole = LogicalEntityRef("edge:hole-loop")
    unsupported = _snapshot(
        controls=(
            LocalMeshControl(
                hole,
                0.2,
                MeshSizeFalloff("global_size", 0.1, 2.0),
            ),
        )
    )
    wrong_target = _snapshot(
        controls=(
            LocalMeshControl(
                LogicalEntityRef("edge:outer-loop"),
                0.2,
                RADIUS_FALLOFF,
            ),
        )
    )

    with pytest.raises(ProjectV1EncodeError, match="falloff.*无法由 v1"):
        encode_project_v1(unsupported)
    with pytest.raises(ProjectV1EncodeError, match="legacy hole target"):
        encode_project_v1(wrong_target)


def test_current_writer_rejects_multiple_target_radius_controls() -> None:
    snapshot = _snapshot(
        controls=(
            LocalMeshControl(
                LogicalEntityRef("edge:hole-loop"),
                0.2,
                RADIUS_FALLOFF,
            ),
            LocalMeshControl(
                LogicalEntityRef("edge:outer-loop"),
                0.3,
                RADIUS_FALLOFF,
            ),
        )
    )

    with pytest.raises(ProjectV1EncodeError, match="多个.*target_radius"):
        encode_project_v1(snapshot)


def test_current_writer_dedupes_same_control_and_rejects_conflicting_size() -> None:
    target = LogicalEntityRef("edge:hole-loop")
    first = LocalMeshControl(target, 0.4, GLOBAL_FALLOFF)
    same = LocalMeshControl(target, 0.4, GLOBAL_FALLOFF)
    conflict = LocalMeshControl(target, 0.3, GLOBAL_FALLOFF)

    duplicate_settings = MeshSettings(1.0, local_controls=(first,))
    object.__setattr__(
        duplicate_settings,
        "local_controls",
        (first, same),
    )
    duplicate = replace(_snapshot(), mesh_settings=duplicate_settings)

    assert len(
        encode_project_v1(duplicate)["mesh_settings"]["local_controls"]
    ) == 1

    conflict_settings = MeshSettings(1.0, local_controls=(first,))
    object.__setattr__(
        conflict_settings,
        "local_controls",
        (first, conflict),
    )
    conflicting = replace(_snapshot(), mesh_settings=conflict_settings)
    with pytest.raises(ProjectV1EncodeError, match="冲突 size"):
        encode_project_v1(conflicting)


@pytest.mark.parametrize(
    "steps",
    [
        (
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint(1, 1, 2),),
            ),
        ),
        (
            AnalysisStep(
                "Load",
                cloads=(NodalLoad(2, 1, 10.0),),
            ),
        ),
        (
            AnalysisStep(
                "Load",
                line_loads=(LineLoad(3, (1.0, 0.0)),),
            ),
        ),
        (
            AnalysisStep(
                "Load",
                gravity_loads=(GravityLoad((0.0, -9.81), 4),),
            ),
        ),
    ],
)
def test_current_v1_writer_rejects_integer_analysis_target(
    steps: tuple[AnalysisStep, ...],
) -> None:
    with pytest.raises(ProjectV1EncodeError, match="mesh integer target"):
        encode_project_v1(_snapshot(steps=steps))


def test_current_v1_writer_rejects_parts_runtime_model_and_stale_history() -> None:
    base = _snapshot()

    with pytest.raises(ProjectV1EncodeError, match="恰好包含一个"):
        encode_project_v1(replace(base, parts=()))
    with pytest.raises(ProjectV1EncodeError, match="恰好包含一个"):
        encode_project_v1(replace(base, parts=(NativePart(), NativePart("P2"))))
    with pytest.raises(ProjectV1EncodeError, match="模型制品"):
        encode_project_v1(replace(base, model=object()))
    with pytest.raises(ProjectV1EncodeError, match="feature_history"):
        encode_project_v1(replace(base, feature_history=()))


def test_orientation_guard_precedes_native_beam_capability_rejection() -> None:
    base = _snapshot()
    explicit = replace(
        base,
        region_assignments=(
            RegionAssignment(
                "Beam",
                "MISSING_BEAM_REGION",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
    )

    with pytest.raises(
        ProjectV1EncodeError,
        match="v1 不支持 Beam orientation",
    ):
        encode_project_v1(explicit)


def test_v1_atomic_writer_keeps_target_on_reverse_projection_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "legacy.femproj"
    destination.write_text("previous", encoding="utf-8")
    unsupported = _snapshot(
        controls=(
            LocalMeshControl(
                LogicalEntityRef("edge:hole-loop"),
                0.2,
                MeshSizeFalloff("target_radius", 0.5, 2.0),
            ),
        )
    )

    with pytest.raises(ProjectV1EncodeError):
        save_project_v1(destination, unsupported)

    assert destination.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []
