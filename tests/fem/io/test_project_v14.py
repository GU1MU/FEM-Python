from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application.session import ProjectSnapshot
from fem.core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
)
from fem.io.project import decode_project, encode_project
from fem.io.project_v13 import encode_project_v13
from fem.io.project_v14 import (
    ProjectV14DecodeError,
    decode_project_v14,
    dumps_project_v14,
    encode_project_v14,
    load_project_v14,
    save_project_v14,
)
import fem.io.project_v14 as project_v14_module


def _model_2d() -> FEMModel:
    return FEMModel(
        mesh=Mesh2D(
            [Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
            [Element2D(7, [1, 2, 3], "Tri3", {"base": 2.5})],
        ),
        name="二维网格",
        node_sets={"FIXED": NodeSet("FIXED", (1, 3))},
        element_sets={"DOMAIN": ElementSet("DOMAIN", (7,))},
        surfaces={"FACE": Surface("FACE", (ElementFace(7, 0, (1, 2, 3)),))},
        edges={"EDGE": Edge("EDGE", (ElementEdge(7, 0, (1, 2)),))},
        metadata={"generator": "native", "options": {"quadratic": False}},
    )


def _model_3d() -> FEMModel:
    return FEMModel(
        mesh=Mesh3D(
            [Node3D(2, 0.0, 0.0, 0.0), Node3D(9, 1.0, 0.0, 0.0)],
            [Element3D(4, [2, 9], "Beam2", {"reference": [0.0, 1.0, 0.0]})],
            dofs_per_node=6,
        ),
        name="梁网格",
        node_sets={"ROOT": NodeSet("ROOT", (2,))},
        element_sets={"MEMBER": ElementSet("MEMBER", (4,))},
    )


@pytest.mark.parametrize("model", (_model_2d(), _model_3d()))
def test_v14_round_trips_mesh_topology_scopes_and_base_properties(model) -> None:
    source = ProjectSnapshot(model=model, model_name="持久化模型")

    payload = encode_project_v14(source)
    reopened = decode_project_v14(payload)
    restored = reopened.model

    assert payload["schema"] == 14
    assert restored is not None
    assert type(restored.mesh) is type(model.mesh)
    assert restored.mesh.dofs_per_node == model.mesh.dofs_per_node
    assert [vars(node) for node in restored.mesh.nodes] == [
        vars(node) for node in model.mesh.nodes
    ]
    assert [
        (element.id, element.node_ids, element.type, element.props)
        for element in restored.mesh.elements
    ] == [
        (element.id, element.node_ids, element.type, element.props)
        for element in model.mesh.elements
    ]
    assert restored.node_sets == model.node_sets
    assert restored.element_sets == model.element_sets
    assert restored.surfaces == model.surfaces
    assert restored.edges == model.edges
    assert restored.metadata == model.metadata


def test_current_router_persists_model_while_v13_remains_meshless() -> None:
    source = ProjectSnapshot(model=_model_2d())

    current = decode_project(encode_project(source)).snapshot
    legacy = decode_project(encode_project_v13(source)).snapshot

    assert current.model is not None
    assert legacy.model is None


def test_v14_rejects_mesh_connectivity_to_unknown_node() -> None:
    payload = deepcopy(encode_project_v14(ProjectSnapshot(model=_model_2d())))
    payload["project"]["model_artifact"]["elements"][0]["node_ids"] = [1, 99, 3]

    with pytest.raises(ProjectV14DecodeError, match="不存在的节点"):
        decode_project_v14(payload)


def test_v14_uses_compact_deterministic_json_for_mesh_payload() -> None:
    source = ProjectSnapshot(model=_model_2d())

    serialized = dumps_project_v14(source)

    assert serialized.endswith("\n")
    assert serialized.count("\n") == 1
    assert dumps_project_v14(decode_project_v14(serialized)) == serialized


def test_v14_atomic_save_does_not_repeat_full_project_decode(
    tmp_path,
    monkeypatch,
) -> None:
    source = ProjectSnapshot(model=_model_2d())
    target = tmp_path / "model.fempy"
    original_loader = load_project_v14
    monkeypatch.setattr(
        project_v14_module,
        "load_project_v14",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("save must not decode the complete temporary project")
        ),
    )

    assert save_project_v14(target, source) == target
    assert original_loader(target).model is not None
