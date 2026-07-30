from __future__ import annotations

import pytest

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    generate_fem_model,
)
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
)
from fem.geometry import (
    BoxGeometry,
    LogicalEntityRef,
    MovedGeometry,
    RectangleGeometry,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem.mesh.settings import MeshSettings


def test_disjoint_parts_mesh_with_deterministic_global_ids(real_gmsh) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project("多部件模型")
    session.add_native_part(
        RectangleGeometry("矩形-1", 2.0, 1.0),
        name="部件-1",
        mesh_settings=MeshSettings(0.4),
    )
    session.add_native_part(
        MovedGeometry(
            RectangleGeometry("矩形-2", 1.0, 1.0),
            4.0,
            0.0,
        ),
        name="部件-2",
        mesh_settings=MeshSettings(0.3),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "全部材料面",
                (
                    LogicalEntityRef("face:P1/domain"),
                    LogicalEntityRef("face:P2/domain"),
                ),
            ),
        )
    )

    model = generate_fem_model(session.prepare_mesh_generation())
    ownership = model.metadata["_native_part_ownership"]
    catalog = model.metadata["_native_scope_catalog"]

    assert tuple(ownership) == ("P1", "P2")
    assert set(ownership["P1"]["node_ids"]).isdisjoint(
        ownership["P2"]["node_ids"]
    )
    assert set(ownership["P1"]["element_ids"]).isdisjoint(
        ownership["P2"]["element_ids"]
    )
    assert [node.id for node in model.mesh.nodes] == list(
        range(1, len(model.mesh.nodes) + 1)
    )
    assert [element.id for element in model.mesh.elements] == list(
        range(1, len(model.mesh.elements) + 1)
    )
    assert "face:P1/domain" in catalog
    assert "face:P2/domain" in catalog
    assert set(model.element_sets["全部材料面"].element_ids) == {
        element.id for element in model.mesh.elements
    }


def test_sketch_hole_center_nodes_stay_out_of_scope_catalog(
    real_gmsh,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project("两孔平板模型")
    session.add_native_part(
        SketchGeometry(
            "两孔平板",
            SketchPlane.xy(),
            (
                SketchPoint("P1", 0.0, 0.0),
                SketchPoint("P2", 100.0, 0.0),
                SketchPoint("P3", 100.0, 200.0),
                SketchPoint("P4", 0.0, 200.0),
                SketchPoint("P5", 30.0, 60.0),
                SketchPoint("P6", 70.0, 140.0),
            ),
            (
                SketchLine("L1", "P1", "P2"),
                SketchLine("L2", "P2", "P3"),
                SketchLine("L3", "P3", "P4"),
                SketchLine("L4", "P4", "P1"),
                SketchCircle("C5", "P5", 10.0),
                SketchCircle("C6", "P6", 15.0),
            ),
        ),
        name="部件-两孔板",
        mesh_settings=MeshSettings(15.0),
    )

    # 孔心点的 gmsh 节点不属于任何域单元；聚合重编号曾因 catalog
    # 引用这些幽灵节点抛出 KeyError。
    model = generate_fem_model(session.prepare_mesh_generation())

    mesh_node_ids = {int(node.id) for node in model.mesh.nodes}
    catalog = model.metadata["_native_scope_catalog"]
    assert "point:P1/P5" in catalog
    assert "point:P1/P6" in catalog
    assert catalog["point:P1/P5"]["node_ids"] == ()
    assert catalog["point:P1/P6"]["node_ids"] == ()
    for entry in catalog.values():
        assert set(entry["node_ids"]) <= mesh_node_ids


def test_mixed_dimensions_are_rejected_before_analysis() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("二维", 1.0, 1.0),
        name="二维部件",
    )
    session.add_native_part(
        BoxGeometry("三维", 1.0, 1.0, 1.0),
        name="三维部件",
    )

    with pytest.raises(ValueError, match="mixed-dimension"):
        generate_fem_model(session.prepare_mesh_generation())


def test_touching_three_dimensional_parts_require_boolean() -> None:
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("实体-1", 1.0, 1.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("实体-2", 1.0, 1.0, 1.0),
            1.0,
            0.0,
            0.0,
        ),
        name="部件-2",
    )

    with pytest.raises(ValueError, match="part.overlap.mesh-blocked"):
        generate_fem_model(session.prepare_mesh_generation())


def test_suppressed_part_is_excluded_from_relation_checks_and_mesh(
    real_gmsh,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        BoxGeometry("实体-1", 1.0, 1.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        MovedGeometry(
            BoxGeometry("实体-2", 1.0, 1.0, 1.0),
            1.0,
            0.0,
            0.0,
        ),
        name="部件-2",
    )
    session.suppress_native_part("P2")

    model = generate_fem_model(session.prepare_mesh_generation())

    assert tuple(model.metadata["_native_part_ownership"]) == ("P1",)


def test_mesh_scope_owner_is_materialized_and_enforced(real_gmsh) -> None:
    del real_gmsh
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(
        RectangleGeometry("矩形-1", 1.0, 1.0),
        name="部件-1",
    )
    session.add_native_part(
        MovedGeometry(
            RectangleGeometry("矩形-2", 1.0, 1.0),
            3.0,
            0.0,
        ),
        name="部件-2",
    )
    task = session.prepare_mesh_generation()
    model = generate_fem_model(task)
    session.accept_generated_model(task.token, model)

    owned = mesh_references_for_logical_entities(
        model,
        (LogicalEntityRef("face:P2/domain"),),
        mesh_kind="element",
    )
    assert {reference.part_id for reference in owned} == {"P2"}

    p1_element = model.metadata["_native_part_ownership"]["P1"][
        "element_ids"
    ][0]
    with pytest.raises(ValueError, match="does not belong"):
        session.replace_named_regions(
            (
                NamedRegion(
                    "伪造所有者",
                    (
                        MeshEntityRef.element(
                            p1_element,
                            part_id="P2",
                        ),
                    ),
                ),
            )
        )
