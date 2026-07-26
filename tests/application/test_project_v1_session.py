from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fem.application import (
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
    TokenStatus,
)
from fem.application.feature_history import derive_feature_history
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
)
from fem.elements import BeamOrientation
from fem.geometry.recipes import PlateWithHoleGeometry
from fem.geometry.references import LogicalEntityRef
from fem.io import project_v1
from fem.io.project_v1 import (
    ProjectV1DecodeError,
    ProjectV1EncodeError,
    decode_project_v1,
    dumps_project_v1,
    encode_project_v1,
    load_project_v1,
    loads_project_v1,
    save_project_v1,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff


def _project_snapshot() -> ProjectSnapshot:
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("LEFT", 1, 2, 0.0),),
        cloads=(NodalLoad("RIGHT", 1, 100.0),),
        edge_loads=(EdgeLoad("LOADED_EDGE", (0.0, -2.0, 0.0)),),
        gravity_loads=(GravityLoad((0.0, -9.81)),),
        outputs=(OutputRequest("field", "node", ("U",), {"frequency": 1}),),
        metadata={"increments": [1, 2]},
    )
    geometry = PlateWithHoleGeometry(
        "Plate",
        10.0,
        5.0,
        5.0,
        2.5,
        1.0,
    )
    hole = LogicalEntityRef("edge:hole-loop")
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("Plate", "Body"),),
        geometry_recipe=geometry,
        mesh_settings=MeshSettings(
            1.0,
            order=2,
            cell_shape="quadrilateral",
            local_controls=(
                LocalMeshControl(hole, 0.5),
                LocalMeshControl(
                    hole,
                    0.25,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
            ),
        ),
        feature_history=derive_feature_history(geometry),
        named_regions=(
            NamedRegion(
                "LOADED_EDGE",
                (LogicalEntityRef("edge:outer-loop"),),
            ),
        ),
        material_definitions=(
            MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),
        ),
        section_definitions=(
            SectionDefinition(
                "Section-1",
                "Steel",
                properties={"thickness": 2.0},
            ),
        ),
        region_assignments=(RegionAssignment("Section-1", "DOMAIN"),),
        analysis_definitions=(step,),
    )


def test_v1_round_trip_preserves_legacy_local_size_and_editable_inputs():
    original = _project_snapshot()

    payload = encode_project_v1(original)
    reopened = loads_project_v1(dumps_project_v1(original))

    assert payload["mesh_settings"]["local_size"] == 0.25
    assert payload["mesh_settings"]["local_controls"] == [
        {
            "entity_kind": "edge",
            "entity_id": 1,
            "size": 0.5,
        }
    ]
    assert payload["assignments"] == [
        {
            "section_name": "Section-1",
            "region_name": "DOMAIN",
        }
    ]
    assert reopened.region_assignments[0].beam_orientation is None
    assert reopened == original


def test_old_v1_missing_new_keys_uses_compatible_defaults():
    payload = encode_project_v1(_project_snapshot())
    payload["mesh_settings"].pop("local_size")

    reopened = decode_project_v1(payload)

    assert reopened.analysis_definitions[0].line_loads == ()
    assert reopened.region_assignments[0].beam_orientation is None
    assert reopened.mesh_settings.local_controls == (
        LocalMeshControl(LogicalEntityRef("edge:hole-loop"), 0.5),
    )


@pytest.mark.parametrize(
    ("keep_named_regions", "keep_local_controls"),
    ((True, False), (False, True)),
)
def test_legacy_v1_topology_references_fail_closed_instead_of_remapping(
    keep_named_regions,
    keep_local_controls,
):
    payload = encode_project_v1(_project_snapshot())
    payload.pop("logical_topology_version")
    if not keep_named_regions:
        payload["named_regions"] = []
    if not keep_local_controls:
        payload["mesh_settings"]["local_controls"] = []

    with pytest.raises(ProjectV1DecodeError, match="逻辑拓扑契约版本.*重新选择"):
        decode_project_v1(payload)


def test_legacy_v1_without_topology_references_remains_loadable():
    payload = encode_project_v1(_project_snapshot())
    payload.pop("logical_topology_version")
    payload["named_regions"] = []
    payload["mesh_settings"]["local_controls"] = []
    payload["steps"][0]["edge_loads"] = []

    reopened = decode_project_v1(payload)

    assert reopened.named_regions == ()
    assert reopened.mesh_settings.local_controls == (
        LocalMeshControl(
            LogicalEntityRef("edge:hole-loop"),
            0.25,
            MeshSizeFalloff("target_radius", 0.25, 2.0),
        ),
    )


def test_unknown_logical_topology_version_is_rejected():
    payload = encode_project_v1(_project_snapshot())
    payload["logical_topology_version"] = 999

    with pytest.raises(ProjectV1DecodeError, match="不支持的逻辑拓扑契约版本"):
        decode_project_v1(payload)


@pytest.mark.parametrize("invalid_version", (True, 1.5, "1"))
def test_logical_topology_version_requires_a_strict_integer(invalid_version):
    payload = encode_project_v1(_project_snapshot())
    payload["logical_topology_version"] = invalid_version

    with pytest.raises(
        ProjectV1DecodeError,
        match=r"\$\.logical_topology_version 必须是整数",
    ):
        decode_project_v1(payload)


def test_invalid_detached_decode_leaves_current_session_field_equal():
    session = ModelSession()
    session.replace_from_snapshot(_project_snapshot())
    before = session.snapshot()
    payload = encode_project_v1(_project_snapshot())
    payload["steps"][0]["edge_loads"][0]["vector"] = [1.0, float("nan")]

    with pytest.raises(ProjectV1DecodeError, match="有限数值"):
        decode_project_v1(payload)

    assert session.snapshot() == before


def test_successful_decode_can_be_installed_with_one_cas_replacement():
    session = ModelSession()
    initial = session.snapshot()
    decoded = loads_project_v1(dumps_project_v1(_project_snapshot()))

    delta = session.replace_from_snapshot(
        decoded,
        expected_session_revision=initial.session_revision,
    )

    current = session.snapshot()
    assert delta.accepted
    assert current.session_id != initial.session_id
    assert current.geometry_recipe == decoded.geometry_recipe
    assert current.steps[0].edge_loads == decoded.analysis_definitions[0].edge_loads
    assert not current.dirty


def test_atomic_save_validates_temp_then_replaces_target(tmp_path):
    target = tmp_path / "plate.femproj"
    target.write_text("old contents", encoding="utf-8")

    returned = save_project_v1(target, _project_snapshot())

    assert returned == target
    reopened = load_project_v1(target)
    assert reopened.source_path == target
    assert reopened.mesh_settings.local_controls == (
        LocalMeshControl(LogicalEntityRef("edge:hole-loop"), 0.5),
        LocalMeshControl(
            LogicalEntityRef("edge:hole-loop"),
            0.25,
            MeshSizeFalloff("target_radius", 0.25, 2.0),
        ),
    )
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_failed_atomic_replace_keeps_previous_target_and_removes_temp(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "plate.femproj"
    target.write_text("previous", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(project_v1.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_project_v1(target, _project_snapshot())

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_save_snapshot_does_not_mark_changed_session_clean(tmp_path):
    session = ModelSession()
    session.replace_from_snapshot(_project_snapshot())
    save_snapshot = session.prepare_project_save()

    session.replace_mesh_settings(MeshSettings(2.0))
    target = save_project_v1(tmp_path / "plate.femproj", save_snapshot)
    delta = session.accept_project_saved(save_snapshot.token, target)

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_REVISION
    assert session.snapshot().dirty
    saved = load_project_v1(target)
    assert any(
        control.falloff.reference == "target_radius"
        for control in saved.mesh_settings.local_controls
    )
    assert session.snapshot().mesh_settings == MeshSettings(2.0)


def test_save_snapshot_exposes_only_detached_copies(tmp_path):
    session = ModelSession()
    session.replace_from_snapshot(_project_snapshot())
    save_snapshot = session.prepare_project_save()
    exposed = save_snapshot.snapshot
    exposed.material_definitions[0].properties["E"] = 999.0
    exposed.section_definitions[0].properties["thickness"] = 999.0
    exposed.feature_history[0].payload["summary"] = "tampered"
    exposed.analysis_definitions[0].metadata["increments"].append(999)

    target = save_project_v1(tmp_path / "plate.femproj", save_snapshot)
    delta = session.accept_project_saved(save_snapshot.token, target)
    saved = load_project_v1(target)

    assert delta.accepted
    assert saved.material_definitions[0].properties["E"] == 210000.0
    assert saved.section_definitions[0].properties["thickness"] == 2.0
    assert saved.feature_history == derive_feature_history(
        saved.geometry_recipe
    )
    assert saved.analysis_definitions[0].metadata["increments"] == [1, 2]
    assert not session.snapshot().dirty


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["named_regions"].append(
            dict(payload["named_regions"][0])
        ),
        lambda payload: payload["steps"][0]["edge_loads"][0].update(
            {"unexpected": "field"}
        ),
    ],
)
def test_decode_rejects_unknown_duplicate_or_invalid_state(mutate):
    payload = encode_project_v1(_project_snapshot())
    mutate(payload)

    with pytest.raises(ProjectV1DecodeError):
        decode_project_v1(payload)


def test_json_duplicate_keys_and_non_finite_numbers_are_rejected():
    duplicate = (
        '{"schema":1,"schema":1,"source":"native",'
        '"geometry":{"type":"RectangleGeometry","name":"R",'
        '"width":1,"height":1}}'
    )
    non_finite = json.dumps(
        {
            "schema": 1,
            "source": "native",
            "geometry": {
                "type": "RectangleGeometry",
                "name": "R",
                "width": float("nan"),
                "height": 1,
            },
        }
    )

    with pytest.raises(ProjectV1DecodeError, match="重复键"):
        loads_project_v1(duplicate)
    with pytest.raises(ProjectV1DecodeError, match="非有限"):
        loads_project_v1(non_finite)


def test_encoder_blocks_state_v1_cannot_preserve(tmp_path):
    target = tmp_path / "plate.femproj"
    target.write_text("previous", encoding="utf-8")
    base = _project_snapshot()
    unsupported_metadata = ProjectSnapshot(
        source_kind=base.source_kind,
        source_path=base.source_path,
        parts=base.parts,
        geometry_recipe=base.geometry_recipe,
        mesh_settings=base.mesh_settings,
        feature_history=base.feature_history,
        named_regions=base.named_regions,
        material_definitions=(
            MaterialDefinition("Steel", {"table": (1.0, 2.0)}),
        ),
        section_definitions=base.section_definitions,
        region_assignments=base.region_assignments,
        analysis_definitions=base.analysis_definitions,
    )

    with pytest.raises(ProjectV1EncodeError, match="无法由 JSON 无损表示"):
        save_project_v1(target, unsupported_metadata)

    assert target.read_text(encoding="utf-8") == "previous"


def test_explicit_beam_orientation_fails_closed_before_atomic_replace(
    tmp_path,
):
    target = tmp_path / "plate.femproj"
    target.write_text("previous", encoding="utf-8")
    base = _project_snapshot()
    explicit = replace(
        base,
        region_assignments=(
            RegionAssignment(
                "Section-1",
                "DOMAIN",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
    )

    with pytest.raises(
        ProjectV1EncodeError,
        match=r"v1 不支持 Beam orientation",
    ):
        save_project_v1(target, explicit)

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_explicit_orientation_save_failure_keeps_session_dirty(
    tmp_path,
):
    session = ModelSession()
    session.replace_from_snapshot(_project_snapshot())
    before_edit = session.snapshot()
    session.replace_model_definitions(
        before_edit.material_definitions,
        before_edit.section_definitions,
        (
            RegionAssignment(
                "Section-1",
                "DOMAIN",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
        before_edit.analysis_definitions,
    )
    before_save = session.snapshot()
    save_snapshot = session.prepare_project_save()
    target = tmp_path / "plate.femproj"
    target.write_text("previous", encoding="utf-8")

    with pytest.raises(ProjectV1EncodeError):
        save_project_v1(target, save_snapshot)

    after = session.snapshot()
    assert after.dirty
    assert after.project_revision == before_save.project_revision
    assert (
        after.saved_project_revision
        == before_save.saved_project_revision
    )
    assert after.project_path == before_save.project_path
    assert target.read_text(encoding="utf-8") == "previous"
