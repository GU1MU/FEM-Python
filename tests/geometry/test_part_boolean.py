from __future__ import annotations

import pytest

from fem import geometry
from fem.application import ModelSession, prepare_part_boolean
from fem.geometry import BoxGeometry, LogicalEntityRef, MovedGeometry
from fem.mesh.settings import LocalMeshControl, MeshSettings


def _overlapping_parts() -> tuple[ModelSession, object, object]:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("目标实体", 2.0, 1.0, 1.0),
        name="目标部件",
        mesh_settings=MeshSettings(
            0.4,
            cell_shape="tetrahedron",
            local_controls=(
                LocalMeshControl(LogicalEntityRef("face:left"), 0.2),
            ),
        ),
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("工具实体", 1.0, 1.0, 1.0),
            1.5,
            0.0,
            0.0,
        ),
        name="工具部件",
    )
    target, tool = session.snapshot().parts
    return session, target, tool


@pytest.mark.parametrize("operation", ["fuse", "cut"])
def test_part_boolean_creates_result_suppresses_sources_and_undoes(
    real_gmsh,
    operation: str,
) -> None:
    del real_gmsh
    session, target, tool = _overlapping_parts()
    result_name = "合并结果-1" if operation == "fuse" else "切除结果-1"
    with geometry.model("实体布尔", dimension=3) as cad:
        result = prepare_part_boolean(
            cad,
            target,
            tool,
            operation,
            result_part_id="P3",
            feature_id="PBF1",
            result_name=result_name,
        )

    before = session.snapshot()
    session.apply_part_boolean(
        "P1",
        "P2",
        operation,
        result_name,
        result=result,
        expected_target_revision=before.part_revision("P1"),
        expected_tool_revision=before.part_revision("P2"),
        expected_session_revision=before.session_revision,
    )
    committed = session.snapshot()

    assert tuple(part.id for part in committed.parts) == ("P1", "P2", "P3")
    assert committed.part("P1").suppressed
    assert committed.part("P2").suppressed
    assert committed.active_part_id == "P3"
    assert committed.part("P3").provenance.feature_id == "PBF1"
    assert committed.part("P3").geometry_recipe.part_context.proven
    assert tuple(
        control.target.logical_id
        for control in committed.part("P3").mesh_settings.local_controls
    ) == ("face:P3/left",)

    session.undo_part_boolean("P3")
    restored = session.snapshot()
    assert restored.parts == before.parts
    assert session.retired_part_ids == ("P3",)
    assert session.retired_part_boolean_feature_ids == ("PBF1",)


def test_disjoint_fuse_fails_without_mutating_session(real_gmsh) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("目标", 1.0, 1.0, 1.0),
        name="目标部件",
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("工具", 1.0, 1.0, 1.0),
            3.0,
            0.0,
            0.0,
        ),
        name="工具部件",
    )
    before = session.snapshot()
    with pytest.raises(ValueError):
        with geometry.model("不相交合并", dimension=3) as cad:
            prepare_part_boolean(
                cad,
                before.part("P1"),
                before.part("P2"),
                "fuse",
                result_part_id="P3",
                feature_id="PBF1",
                result_name="合并结果-1",
            )
    assert session.snapshot() == before


def test_disjoint_cut_is_rejected_as_no_op(real_gmsh) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("目标", 1.0, 1.0, 1.0),
        name="目标部件",
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("工具", 1.0, 1.0, 1.0),
            3.0,
            0.0,
            0.0,
        ),
        name="工具部件",
    )
    before = session.snapshot()

    with pytest.raises(ValueError, match="no-op"):
        with geometry.model("无效切除", dimension=3) as cad:
            prepare_part_boolean(
                cad,
                before.part("P1"),
                before.part("P2"),
                "cut",
                result_part_id="P3",
                feature_id="PBF1",
                result_name="切除结果-1",
            )

    assert session.snapshot() == before


@pytest.mark.parametrize(
    ("target", "tool", "error_pattern"),
    (
        (
            BoxGeometry("目标", 1.0, 1.0, 1.0),
            MovedGeometry(
                BoxGeometry("工具", 2.0, 2.0, 2.0),
                -0.5,
                -0.5,
                -0.5,
            ),
            "empty|volume-count",
        ),
        (
            BoxGeometry("目标", 3.0, 1.0, 1.0),
            MovedGeometry(
                BoxGeometry("工具", 1.0, 1.0, 1.0),
                1.0,
                0.0,
                0.0,
            ),
            "volume-count",
        ),
    ),
)
def test_empty_and_split_cuts_are_rejected(
    real_gmsh,
    target,
    tool,
    error_pattern: str,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(target, name="目标部件")
    session.add_native_part(tool, name="工具部件")
    before = session.snapshot()

    with pytest.raises(ValueError, match=error_pattern):
        with geometry.model("无效切除结果", dimension=3) as cad:
            prepare_part_boolean(
                cad,
                before.part("P1"),
                before.part("P2"),
                "cut",
                result_part_id="P3",
                feature_id="PBF1",
                result_name="切除结果-1",
            )

    assert session.snapshot() == before


def test_chained_boolean_undo_restores_only_direct_sources(real_gmsh) -> None:
    del real_gmsh
    session, target, tool = _overlapping_parts()
    with geometry.model("第一层布尔", dimension=3) as cad:
        first = prepare_part_boolean(
            cad,
            target,
            tool,
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
        result=first,
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("第三实体", 1.0, 1.0, 1.0),
            2.0,
            0.0,
            0.0,
        ),
        name="第三部件",
    )
    before_outer = session.snapshot()
    with geometry.model("第二层布尔", dimension=3) as cad:
        second = prepare_part_boolean(
            cad,
            before_outer.part("P3"),
            before_outer.part("P4"),
            "fuse",
            result_part_id="P5",
            feature_id="PBF2",
            result_name="合并结果-2",
        )
    session.apply_part_boolean(
        "P3",
        "P4",
        "fuse",
        "合并结果-2",
        result=second,
    )

    chained = session.snapshot()
    assert chained.part("P1").suppressed
    assert chained.part("P2").suppressed
    assert chained.part("P3").suppressed
    assert chained.part("P4").suppressed
    assert chained.active_part_id == "P5"

    session.undo_part_boolean("P5")
    restored = session.snapshot()
    assert restored.part("P1").suppressed
    assert restored.part("P2").suppressed
    assert not restored.part("P3").suppressed
    assert not restored.part("P4").suppressed
    assert restored.active_part_id == "P3"


def test_commit_rejects_proof_after_source_geometry_changes(real_gmsh) -> None:
    del real_gmsh
    session, target, tool = _overlapping_parts()
    with geometry.model("过期布尔证明", dimension=3) as cad:
        prepared = prepare_part_boolean(
            cad,
            target,
            tool,
            "fuse",
            result_part_id="P3",
            feature_id="PBF1",
            result_name="合并结果-1",
        )
    session.replace_part_geometry(
        "P1",
        BoxGeometry("已修改目标", 2.5, 1.0, 1.0),
    )
    before = session.snapshot()

    with pytest.raises(RuntimeError, match="no longer matches source"):
        session.apply_part_boolean(
            "P1",
            "P2",
            "fuse",
            "合并结果-1",
            result=prepared,
        )

    assert session.snapshot() == before


def test_result_rename_and_mesh_edit_do_not_block_boolean_undo(
    real_gmsh,
) -> None:
    del real_gmsh
    session, target, tool = _overlapping_parts()
    with geometry.model("可撤销布尔", dimension=3) as cad:
        prepared = prepare_part_boolean(
            cad,
            target,
            tool,
            "fuse",
            result_part_id="P3",
            feature_id="PBF1",
            result_name="合并结果-1",
        )
    before = session.snapshot()
    session.apply_part_boolean(
        "P1",
        "P2",
        "fuse",
        "合并结果-1",
        result=prepared,
    )
    session.rename_native_part("P3", "用户结果")
    session.replace_part_mesh_settings(
        "P3",
        MeshSettings(0.2, cell_shape="tetrahedron"),
    )

    session.undo_part_boolean("P3")

    assert session.snapshot().parts == before.parts
