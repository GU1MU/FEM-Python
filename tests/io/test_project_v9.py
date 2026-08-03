from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import LogicalEntityRef, PlateWithHoleGeometry
from fem.io.project import CURRENT_PROJECT_SCHEMA, decode_project, encode_project
from fem.io.project_v7 import ProjectV7DecodeError, decode_project_v7, encode_project_v7
from fem.io.project_v8 import ProjectV8DecodeError, decode_project_v8, encode_project_v8
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff


def _session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        PlateWithHoleGeometry(
            "实体-偏心孔板",
            10.0,
            6.0,
            6.5,
            2.0,
            1.0,
        ),
        part_name="部件-偏心孔板",
    )
    return session


def test_current_schema_preserves_a3_automatic_strict_mesh_intent() -> None:
    session = _session()
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(
            0.5,
            order=2,
            cell_shape="quadrilateral",
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:hole-loop"),
                    0.15,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
            ),
            auto_level=4,
            strict_cell_shape=True,
        ),
    )

    payload = encode_project(session.prepare_project_save())
    loaded = decode_project(payload)
    settings = loaded.snapshot.parts[0].mesh_settings

    assert CURRENT_PROJECT_SCHEMA == 11
    assert payload["schema"] == 11
    assert payload["project"]["authoring"]["parts"][0]["mesh_settings"][
        "intent_mode"
    ] == "automatic"
    assert settings.auto_level == 4
    assert settings.strict_cell_shape is True
    assert settings.local_controls[0].target == LogicalEntityRef(
        "edge:P1/hole-loop"
    )


def test_a3_v7_and_v8_remain_strict_and_reject_v9_mesh_fields() -> None:
    session = _session()
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.5, cell_shape="triangle"),
    )

    v7 = encode_project_v7(session.prepare_project_save())
    v7_extended = deepcopy(v7)
    v7_extended["project"]["authoring"]["parts"][0]["mesh_settings"][
        "auto_level"
    ] = 4
    with pytest.raises(ProjectV7DecodeError, match="未知字段|字段"):
        decode_project_v7(v7_extended)

    v8 = encode_project_v8(session.prepare_project_save())
    v8_extended = deepcopy(v8)
    v8_extended["project"]["authoring"]["parts"][0]["mesh_settings"].update(
        {
            "intent_mode": "automatic",
            "auto_level": 4,
            "strict_cell_shape": True,
        }
    )
    with pytest.raises(ProjectV8DecodeError, match="无效|未知字段|字段"):
        decode_project_v8(v8_extended)


def test_a3_schema_v8_reads_as_explicit_non_strict_mesh_intent() -> None:
    session = _session()
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.5, cell_shape="quadrilateral"),
    )
    payload = encode_project_v8(session.prepare_project_save())

    loaded = decode_project(payload)
    settings = loaded.snapshot.parts[0].mesh_settings

    assert loaded.source_schema == 8
    assert settings.auto_level is None
    assert settings.strict_cell_shape is False
