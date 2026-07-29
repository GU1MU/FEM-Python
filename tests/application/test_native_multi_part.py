from __future__ import annotations

import pytest

from fem.application import (
    ModelSession,
    NamedRegion,
    PartRevisionConflictError,
    RegionAssignment,
    SectionDefinition,
    SessionStateError,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
)
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    RectangleGeometry,
)
from fem.mesh.settings import MeshSettings
from tests.geometry.test_profile_extrusion import (
    profile_face_id,
    two_profile_sketch,
)


def test_empty_model_allocates_stable_parts_without_replacement() -> None:
    session = ModelSession()
    session.new_native_project("模型-1")

    assert session.snapshot().parts == ()
    session.add_native_part(
        RectangleGeometry("草图-1", 4.0, 2.0),
        name="部件-1",
        mesh_settings=MeshSettings(0.5),
    )
    first = session.snapshot().parts[0]
    session.add_native_part(
        BoxGeometry("长方体-1", 1.0, 1.0, 1.0),
        name="部件-2",
        mesh_settings=MeshSettings(0.4),
    )

    snapshot = session.snapshot()
    assert tuple(part.id for part in snapshot.parts) == ("P1", "P2")
    assert snapshot.active_part_id == "P2"
    assert snapshot.parts[0] == first
    assert snapshot.parts[0].geometry_recipe.name == "草图-1"


def test_unproven_three_dimensional_boolean_is_rejected_atomically(
    real_gmsh,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    recipe = BooleanGeometry(
        "未证明合并",
        "fuse",
        BoxGeometry("目标", 1.0, 1.0, 1.0),
        MovedGeometry(
            BoxGeometry("工具", 1.0, 1.0, 1.0),
            3.0,
            0.0,
            0.0,
        ),
    )

    with pytest.raises(ValueError, match="拓扑证明"):
        session.add_native_part(recipe, name="无效部件")

    assert session.snapshot().parts == ()
    assert session.retired_part_ids == ()


def test_aggregate_definition_rejects_surface_as_solid_assignment() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("实体", 1.0, 1.0, 1.0),
        name="部件-1",
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "顶面",
                (LogicalEntityRef("face:P1/top"),),
            ),
        )
    )

    with pytest.raises(ValueError, match="cannot produce 'element_set'"):
        session.replace_model_definitions(
            (MaterialDefinition("钢", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("实体截面", "钢"),),
            (RegionAssignment("实体截面", "顶面"),),
            (),
        )

    assert session.snapshot().assignments == ()


def test_part_edit_preserves_unrelated_namespaces_and_checks_part_revision() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 4.0, 2.0),
        name="部件-1",
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 3.0, 1.0),
        name="部件-2",
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "P1 面",
                (LogicalEntityRef("face:P1/domain"),),
            ),
            NamedRegion(
                "P2 面",
                (LogicalEntityRef("face:P2/domain"),),
            ),
        )
    )
    before = session.snapshot()

    session.replace_part_geometry(
        "P2",
        RectangleGeometry("草图-2", 5.0, 1.0),
        expected_part_revision=before.part_revision("P2"),
        expected_session_revision=before.session_revision,
    )
    after = session.snapshot()

    assert after.part("P1") == before.part("P1")
    assert after.named_regions["P1 面"] == before.named_regions["P1 面"]
    with pytest.raises(PartRevisionConflictError):
        session.replace_part_geometry(
            "P2",
            RectangleGeometry("草图-2", 6.0, 1.0),
            expected_part_revision=before.part_revision("P2"),
        )


def test_deleted_part_ids_are_retired_and_not_reused() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 1.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 1.0, 1.0),
        name="部件-2",
    )
    session.delete_native_part("P2")
    session.add_native_part(
        RectangleGeometry("草图-3", 1.0, 1.0),
        name="部件-3",
    )

    assert tuple(part.id for part in session.snapshot().parts) == (
        "P1",
        "P3",
    )
    assert session.retired_part_ids == ("P2",)


def test_delete_removes_only_the_deleted_part_namespace() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 1.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 1.0, 1.0),
        name="部件-2",
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "共享面",
                (
                    LogicalEntityRef("face:P1/domain"),
                    LogicalEntityRef("face:P2/domain"),
                ),
            ),
            NamedRegion(
                "第二部件面",
                (LogicalEntityRef("face:P2/domain"),),
            ),
        )
    )

    session.delete_native_part("P2")
    snapshot = session.snapshot()

    assert snapshot.named_regions["共享面"].references == (
        LogicalEntityRef("face:P1/domain"),
    )
    assert "第二部件面" not in snapshot.named_regions


def test_rename_settings_and_suppression_are_part_scoped() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 2.0, 1.0),
        name="部件-1",
        mesh_settings=MeshSettings(0.5),
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 3.0, 1.0),
        name="部件-2",
        mesh_settings=MeshSettings(0.4),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "第一部件面",
                (LogicalEntityRef("face:P1/domain"),),
            ),
        )
    )
    before = session.snapshot()

    session.rename_native_part("P2", "工具部件")
    session.replace_part_mesh_settings("P2", MeshSettings(0.2))
    session.suppress_native_part("P2")
    after = session.snapshot()

    assert after.part("P1") == before.part("P1")
    assert after.named_regions["第一部件面"] == before.named_regions["第一部件面"]
    assert after.part("P2").name == "工具部件"
    assert after.part("P2").mesh_settings.size == 0.2
    assert after.part("P2").suppressed
    assert after.active_part_id == "P1"
    with pytest.raises(SessionStateError, match="suppressed"):
        session.set_active_native_part("P2")
    with pytest.raises(SessionStateError, match="suppressed"):
        session.rename_native_part("P2", "被锁定的工具部件")


def test_definition_edit_validates_the_aggregate_part_namespace() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 2.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 2.0, 1.0),
        name="部件-2",
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "跨部件面",
                (
                    LogicalEntityRef("face:P1/domain"),
                    LogicalEntityRef("face:P2/domain"),
                ),
            ),
        )
    )

    session.replace_model_definitions((), (), (), ())

    assert session.snapshot().named_regions["跨部件面"].references == (
        LogicalEntityRef("face:P1/domain"),
        LogicalEntityRef("face:P2/domain"),
    )


def test_suppressed_part_reference_is_preserved_but_blocks_preflight() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("草图-1", 2.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        RectangleGeometry("草图-2", 2.0, 1.0),
        name="部件-2",
    )
    region = NamedRegion(
        "第二部件边",
        (LogicalEntityRef("edge:P2/bottom"),),
    )
    session.replace_named_regions((region,))
    session.replace_model_definitions(
        (),
        (),
        (),
        (
            AnalysisStep(
                "载荷步",
                boundaries=(
                    DisplacementConstraint("第二部件边", 1, 2),
                ),
            ),
        ),
    )
    session.suppress_native_part("P2")

    assert session.snapshot().named_regions["第二部件边"] == region
    with pytest.raises(SessionStateError, match="suppressed Part"):
        session.prepare_mesh_generation()


def test_multi_profile_extrusion_is_atomic_and_has_shared_undo() -> None:
    sketch = two_profile_sketch()
    face_ids = (
        profile_face_id(sketch, "L1"),
        profile_face_id(sketch, "L5"),
    )
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(sketch, name="二维部件")
    before = session.snapshot()
    recipes = tuple(
        ExtrudedGeometry(sketch, 2.0, (face_id,))
        for face_id in reversed(face_ids)
    )

    session.replace_part_with_extruded_siblings(
        "P1",
        recipes,
        expected_part_revision=before.part_revision("P1"),
        expected_session_revision=before.session_revision,
    )
    extruded = session.snapshot()
    assert tuple(part.id for part in extruded.parts) == ("P1", "P2")
    assert all(part.dimension == 3 for part in extruded.parts)
    assert extruded.part("P1").geometry_recipe.source_face_ids == (
        min(face_ids),
    )
    assert extruded.part("P2").geometry_recipe.source_face_ids == (
        max(face_ids),
    )

    session.undo_part_extrusion("P1")
    restored = session.snapshot()
    assert restored.parts == before.parts
    assert session.retired_part_ids == ("P2",)


def test_multi_profile_extrusion_failure_keeps_session_atomic() -> None:
    sketch = two_profile_sketch()
    face_ids = (
        profile_face_id(sketch, "L1"),
        profile_face_id(sketch, "L5"),
    )
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(sketch, name="二维部件")
    before = session.snapshot()
    valid = ExtrudedGeometry(sketch, 2.0, (min(face_ids),))
    multi_solid = ExtrudedGeometry(sketch, 2.0, tuple(sorted(face_ids)))

    with pytest.raises(ValueError, match="single solid"):
        session.replace_part_with_extruded_siblings(
            "P1",
            (valid, multi_solid),
        )

    assert session.snapshot() == before
    assert session.retired_part_ids == ()
