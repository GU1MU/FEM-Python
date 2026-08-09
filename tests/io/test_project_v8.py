from __future__ import annotations

from fem.application import ModelSession, UnitContext
from fem.geometry import PlateWithHoleGeometry, RectangleGeometry
from fem.io.project import (
    CURRENT_PROJECT_SCHEMA,
    decode_project,
    load_project,
    save_project,
)
from fem.io.project_v7 import encode_project_v7


def _units() -> UnitContext:
    return UnitContext(
        length="mm",
        force="N",
        stress="MPa",
        density=None,
        acceleration=None,
        convention="N-mm-MPa",
    )


def test_a2_project_save_and_reopen_preserves_units_and_accepted_geometry(
    tmp_path,
) -> None:
    session = ModelSession()
    recipe = PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        _units(),
        recipe,
        part_name="部件-偏心孔板",
    )
    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "plate.femproj", prepared)
    session.accept_project_saved(prepared.token, target)

    loaded = load_project(target)
    reopened = ModelSession()
    reopened.replace_from_snapshot(loaded.snapshot)
    snapshot = reopened.snapshot()

    assert CURRENT_PROJECT_SCHEMA == 13
    assert loaded.source_schema == 13
    assert snapshot.unit_context == _units()
    assert snapshot.parts[0].geometry_recipe == recipe
    assert snapshot.parts[0].name == "部件-偏心孔板"


def test_a2_schema_v7_migrates_with_explicit_missing_unit_context() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-旧项目",
        _units(),
        RectangleGeometry("实体-矩形", 2.0, 1.0),
        part_name="部件-矩形",
    )
    payload = encode_project_v7(session.prepare_project_save())

    loaded = decode_project(payload)

    assert loaded.source_schema == 7
    assert loaded.snapshot.unit_context is None
