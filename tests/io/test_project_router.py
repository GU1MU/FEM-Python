from __future__ import annotations

import json
from pathlib import Path

import pytest

import fem.io as fem_io
from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry.recipes import RectangleGeometry
from fem.io._project_errors import (
    ProjectDecodeError,
    UnsupportedProjectSchemaError,
)
from fem.io.project import (
    CURRENT_PROJECT_SCHEMA,
    LoadedProject,
    decode_project,
    dumps_project,
    encode_project,
    load_project,
    loads_project,
    save_project,
)
from fem.io.project_v2 import (
    dumps_project_v2,
    load_project_v2,
)
from fem.io.project_v5 import dumps_project_v5
from fem.io.project_v7 import ProjectV7DecodeError, load_project_v7
from fem.mesh.settings import MeshSettings


FIXTURES = Path(__file__).parents[1] / "fixtures" / "femproj" / "v1"


def _snapshot(*, source_path: Path | None = None) -> ProjectSnapshot:
    recipe = RectangleGeometry(name="Rectangle", width=4.0, height=2.0)
    return ProjectSnapshot(
        source_kind="native",
        source_path=source_path,
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(size=0.5),
        feature_history=derive_feature_history(recipe),
    )


def test_generic_writer_always_emits_current_schema(tmp_path: Path) -> None:
    snapshot = _snapshot()

    payload = encode_project(snapshot)
    dumped = dumps_project(snapshot)
    target = save_project(tmp_path / "current.femproj", snapshot)

    assert CURRENT_PROJECT_SCHEMA == 7
    assert payload["schema"] == CURRENT_PROJECT_SCHEMA
    assert json.loads(dumped)["schema"] == CURRENT_PROJECT_SCHEMA
    assert json.loads(target.read_text(encoding="utf-8"))["schema"] == 7
    assert load_project_v7(target).source_path == target


def test_generic_current_reader_returns_loaded_project_with_path_invariant(
    tmp_path: Path,
) -> None:
    target = tmp_path / "native.femproj"
    target.write_text(dumps_project(_snapshot()), encoding="utf-8")

    loaded = load_project(target)

    assert type(loaded) is LoadedProject
    assert loaded.path == target
    assert loaded.snapshot.source_path == target
    assert loaded.source_schema == 7
    assert loaded.notices == ()


def test_generic_router_still_reads_frozen_v5_projects(tmp_path: Path) -> None:
    target = tmp_path / "v5.femproj"
    target.write_text(
        dumps_project_v5(_snapshot()),
        encoding="utf-8",
    )

    loaded = load_project(target)

    assert loaded.source_schema == 5
    assert loaded.snapshot.source_path == target
    assert tuple(notice.code for notice in loaded.notices) == (
        "project.schema.v7.native_parts",
    )


def test_generic_router_still_reads_frozen_v2_projects(tmp_path: Path) -> None:
    target = tmp_path / "v2.femproj"
    target.write_text(dumps_project_v2(_snapshot()), encoding="utf-8")

    loaded = load_project(target)

    assert loaded.source_schema == 2
    assert loaded.snapshot.source_path == target
    assert load_project_v2(target).source_path == target


def test_generic_v1_reader_preserves_migration_notices_without_writing() -> None:
    source = FIXTURES / "minimal_rectangle.femproj"
    before = source.read_bytes()

    loaded = load_project(source)

    assert loaded.source_schema == 1
    assert loaded.path == source
    assert loaded.snapshot.source_path == source
    assert loaded.notices
    assert loaded.notices[0].code == "project.schema.v1"
    assert source.read_bytes() == before


def test_loads_without_source_path_keeps_both_paths_none() -> None:
    loaded = loads_project(dumps_project(_snapshot()))

    assert loaded.path is None
    assert loaded.snapshot.source_path is None


@pytest.mark.parametrize("schema", [True, 2.0, "2", None])
def test_router_requires_a_strict_integer_schema(schema: object) -> None:
    with pytest.raises(ProjectDecodeError, match=r"\$\.schema.*严格整数"):
        decode_project({"schema": schema})


def test_router_requires_schema_and_rejects_future_schema() -> None:
    with pytest.raises(ProjectDecodeError, match=r"\$\.schema.*缺失"):
        decode_project({})
    with pytest.raises(
        UnsupportedProjectSchemaError,
        match=r"\$\.schema=99.*schema 1、2、3、4、5、6 和 7",
    ):
        decode_project({"schema": 99})


def test_router_parses_json_once_and_rejects_duplicate_keys() -> None:
    with pytest.raises(ProjectDecodeError, match="重复键"):
        loads_project('{"schema": 2, "schema": 1}')


def test_decode_project_rejects_serialized_input() -> None:
    with pytest.raises(TypeError, match="loads_project"):
        decode_project(b'{"schema": 2}')  # type: ignore[arg-type]


def test_v7_format_error_keeps_concrete_version_error() -> None:
    payload = encode_project(_snapshot())
    payload["format"] = "wrong"

    with pytest.raises(ProjectV7DecodeError, match=r"\$\.format"):
        decode_project(payload)


def test_fem_io_exports_generic_and_explicit_versioned_project_apis() -> None:
    expected = {
        "decode_project",
        "dumps_project",
        "encode_project",
        "load_project",
        "loads_project",
        "save_project",
        "decode_project_v1",
        "encode_project_v1",
        "load_project_v1",
        "save_project_v1",
        "decode_project_v2",
        "encode_project_v2",
        "load_project_v2",
        "save_project_v2",
        "decode_project_v3",
        "encode_project_v3",
        "load_project_v3",
        "save_project_v3",
        "decode_project_v4",
        "encode_project_v4",
        "load_project_v4",
        "save_project_v4",
        "decode_project_v5",
        "encode_project_v5",
        "load_project_v5",
        "save_project_v5",
        "decode_project_v6",
        "encode_project_v6",
        "load_project_v6",
        "save_project_v6",
        "decode_project_v7",
        "encode_project_v7",
        "load_project_v7",
        "save_project_v7",
        "ProjectMigrationNotice",
    }

    assert expected.issubset(set(fem_io.__all__))
    assert not hasattr(fem_io, "load_native_project")
    assert not hasattr(fem_io, "save_native_project")
