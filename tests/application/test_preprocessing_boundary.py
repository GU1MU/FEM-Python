from __future__ import annotations

from types import SimpleNamespace

import pytest

import fem.application.preprocessing as preprocessing
from fem.application.native_mesh_contract import NativeMeshContract
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import (
    CompiledRecipeTopology,
    TopologyResolutionError,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings


@pytest.mark.gmsh
def test_native_1d_preprocessing_generates_a_truss_model() -> None:
    recipe = WireGeometry(
        "Wire",
        (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0)),
        (WireMember("M1", "P1", "P2"),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.25,
            cell_shape="line",
            line_element_type="Truss2",
        ),
    )

    assert model.mesh.dofs_per_node == 3
    assert {element.type for element in model.mesh.elements} == {"Truss2"}
    assert [node.id for node in model.mesh.nodes] == [1, 2]
    assert [element.id for element in model.mesh.elements] == [1]
    catalog = model.metadata["_native_scope_catalog"]
    assert catalog["edge:M1"]["element_ids"] == (1,)
    assert catalog["body:domain"]["element_ids"] == (1,)
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces
    assert not model.node_sets


def _wire_audit_fixture(
    elements: tuple[tuple[int, tuple[int, ...]], ...],
    member_element_ids: dict[str, tuple[int, ...]],
):
    recipe = WireGeometry(
        "WireAudit",
        (
            WirePoint("P1", 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0),
            WirePoint("P3", 2.0, 0.0),
        ),
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3"),
        ),
    )
    logical_entities = {
        logical_id: (logical_id,)
        for logical_id in (
            "point:P1",
            "point:P2",
            "point:P3",
            "edge:M1",
            "edge:M2",
            "body:domain",
        )
    }
    topology = CompiledRecipeTopology(
        domain=(),
        boundary=(),
        catalog=describe_recipe_topology(recipe),
        logical_entities=logical_entities,
        region_bindings={},
    )
    mesh = SimpleNamespace(
        elements=tuple(
            SimpleNamespace(id=element_id, node_ids=node_ids, type="Truss2")
            for element_id, node_ids in elements
        )
    )
    node_ids_by_entity = {
        "point:P1": (1,),
        "point:P2": (2,),
        "point:P3": (3,),
    }
    element_ids_by_entity = {
        **member_element_ids,
        "body:domain": tuple(element_id for element_id, _ in elements),
    }
    return (
        recipe,
        topology,
        mesh,
        NativeMeshContract(1, "line", 1, "Truss2", "Truss2", True),
        node_ids_by_entity,
        element_ids_by_entity,
    )


def _patch_wire_audit_io(monkeypatch, node_ids_by_entity, element_ids_by_entity):
    monkeypatch.setattr(
        preprocessing.gmsh_io,
        "entity_node_ids",
        lambda _source, entities: node_ids_by_entity[tuple(entities)[0]],
    )
    monkeypatch.setattr(
        preprocessing.gmsh_io,
        "entity_element_ids",
        lambda _source, entities: element_ids_by_entity[tuple(entities)[0]],
    )


def test_native_1d_audit_rejects_undeclared_member_node_sharing(monkeypatch) -> None:
    fixture = _wire_audit_fixture(
        (
            (1, (1, 99)),
            (2, (99, 2)),
            (3, (2, 99)),
            (4, (99, 3)),
        ),
        {"edge:M1": (1, 2), "edge:M2": (3, 4)},
    )
    recipe, topology, mesh, contract, node_ids, element_ids = fixture
    _patch_wire_audit_io(monkeypatch, node_ids, element_ids)

    with pytest.raises(TopologyResolutionError, match="undeclared mesh nodes"):
        preprocessing._audit_native_wire_mesh(
            recipe,
            topology,
            object(),
            mesh,
            contract,
        )


def test_native_1d_audit_rejects_fragmented_member_chain(monkeypatch) -> None:
    fixture = _wire_audit_fixture(
        (
            (1, (1, 99)),
            (2, (100, 2)),
            (3, (2, 3)),
        ),
        {"edge:M1": (1, 2), "edge:M2": (3,)},
    )
    recipe, topology, mesh, contract, node_ids, element_ids = fixture
    _patch_wire_audit_io(monkeypatch, node_ids, element_ids)

    with pytest.raises(
        TopologyResolutionError,
        match="fragmented into disconnected element chains",
    ):
        preprocessing._audit_native_wire_mesh(
            recipe,
            topology,
            object(),
            mesh,
            contract,
        )
