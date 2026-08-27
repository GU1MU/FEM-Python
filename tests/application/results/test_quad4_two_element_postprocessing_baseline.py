from __future__ import annotations

import numpy as np

from fem.results import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
)
from fem.model.mesh import Element2D, Mesh2D, Node2D
from fem.model import FEMModel
from fem.results import ModelResult
from fem.post.averaging import NodalAveragingPolicy


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="quad4-two-element-result",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=1,
        step_name="Step-1",
        run_id="run-1",
    )


def _two_quad4_result() -> ModelResult:
    properties = {
        "E": 100.0,
        "nu": 0.0,
        "plane_type": "stress",
        "thickness": 1.0,
    }
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
            Node2D(5, 2.0, 0.0),
            Node2D(6, 2.0, 1.0),
        ],
        elements=[
            Element2D(10, [1, 2, 3, 4], "Quad4", properties),
            Element2D(20, [2, 5, 6, 3], "Quad4", properties),
        ],
    )
    displacement = np.zeros(mesh.num_dofs, dtype=float)
    # The right element receives a different extension so the shared edge
    # carries a meaningful element-side stress comparison.
    displacement[mesh.global_dof(2, 0)] = 0.02
    displacement[mesh.global_dof(3, 0)] = 0.02
    displacement[mesh.global_dof(5, 0)] = 0.20
    displacement[mesh.global_dof(6, 0)] = 0.20
    return ModelResult(
        model=FEMModel(mesh=mesh),
        step=None,
        U=displacement,
        reactions=np.zeros(mesh.num_dofs, dtype=float),
    )


def _field_request(
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
) -> FieldRequest:
    return FieldRequest(
        ResultFieldId(ResultVariable.S, position),
        averaging_policy=policy,
    )


def test_quad4_two_element_recovery_preserves_sample_counts_and_identity() -> None:
    provider = build_result_provider(_source(), _two_quad4_result())
    keys = tuple(
        provider.resolve_request(_field_request(position))
        for position in (
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
        )
    )
    patch = provider.materialize(keys)
    fields = {
        field.key.request.field_id.position: field for field in patch.fields
    }

    assert len(fields[FieldPosition.INTEGRATION_POINT].locations) == 8
    assert len(fields[FieldPosition.CENTROID].locations) == 2
    assert len(fields[FieldPosition.ELEMENT_NODAL].locations) == 8
    assert {
        (location.element_id, location.integration_point)
        for location in fields[FieldPosition.INTEGRATION_POINT].locations
    } == {
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (20, 1),
        (20, 2),
        (20, 3),
        (20, 4),
    }
    assert {
        (location.element_id, location.local_node, location.node_id)
        for location in fields[FieldPosition.ELEMENT_NODAL].locations
    } == {
        (element_id, local_node, node_id)
        for element_id, node_ids in (
            (10, (1, 2, 3, 4)),
            (20, (2, 5, 6, 3)),
        )
        for local_node, node_id in enumerate(node_ids, start=1)
    }


def test_quad4_two_element_average_merges_shared_nodes_only_in_same_region() -> None:
    provider = build_result_provider(_source(), _two_quad4_result())
    key = provider.resolve_request(
        _field_request(
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(100.0),
        )
    )
    field = provider.apply(provider.materialize((key,))).field(key)

    # Six physical nodes exist.  The two shared-edge nodes each receive one
    # averaged row, while the two elements still contribute eight raw sides.
    assert len(field.locations) == 6
    assert {location.node_id for location in field.locations} == {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert {
        location.node_id
        for location in field.locations
        if location.averaged is True
    } == {2, 3}
    assert {
        location.node_id
        for location in field.locations
        if location.averaged is False
    } == {1, 4, 5, 6}


def test_quad4_two_element_zero_threshold_keeps_element_side_discontinuity() -> None:
    provider = build_result_provider(_source(), _two_quad4_result())
    key = provider.resolve_request(
        _field_request(
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(0.0),
        )
    )
    field = provider.apply(provider.materialize((key,))).field(key)

    assert len(field.locations) == 8
    assert all(location.averaged is False for location in field.locations)
    shared_node_rows = [
        location for location in field.locations if location.node_id in {2, 3}
    ]
    assert {
        (location.element_id, location.local_node, location.node_id)
        for location in shared_node_rows
    } == {
        (10, 2, 2),
        (10, 3, 3),
        (20, 1, 2),
        (20, 4, 3),
    }
