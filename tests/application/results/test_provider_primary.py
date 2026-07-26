from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fem.application.results.data import FieldState
from fem.application.results.fields import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
)
from fem.application.results.provider import build_result_provider
from fem.application.results.registry import ResultModelFamily
from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.core.model import AnalysisStep, FEMModel, LineLoad
from fem.core.result import ModelResult
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import result_region_key_for_element


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=7,
        step_name="Step-1",
        run_id="run-1",
    )


def _request(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
) -> FieldRequest:
    return FieldRequest(
        ResultFieldId(variable, position),
        averaging_policy=policy,
        gauss_order=gauss_order,
    )


def _two_dimensional_result() -> ModelResult:
    mesh = Mesh2D(
        nodes=[
            Node2D(30, 3.0, 3.5),
            Node2D(10, 1.0, 1.5),
            Node2D(50, 5.0, 5.5),
            Node2D(20, 2.0, 2.5),
        ],
        elements=[
            Element2D(
                90,
                [10, 20, 30, 50],
                "Quad4",
                props={
                    "E": 10.0,
                    "nu": 0.25,
                    "tag": {"nested": [1, 2]},
                },
            )
        ],
    )
    model = FEMModel(mesh=mesh)
    return ModelResult(
        model=model,
        step=None,
        U=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        reactions=np.asarray(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        ),
    )


def _three_dimensional_result() -> ModelResult:
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(4, 0.0, 0.0, 1.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 0.0, 1.0, 0.0),
        ],
        elements=[
            Element3D(
                40,
                [1, 2, 3, 4],
                "Tet4",
                props={"E": 20.0, "nu": 0.3},
            )
        ],
    )
    model = FEMModel(mesh=mesh)
    return ModelResult(
        model=model,
        step=None,
        U=np.arange(1.0, 13.0),
        reactions=np.arange(101.0, 113.0),
    )


def _beam_result() -> ModelResult:
    mesh = Mesh3D(
        nodes=[
            Node3D(30, 3.0, 0.0, 0.0),
            Node3D(10, 1.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                70,
                [10, 30],
                "Beam2",
                props={
                    "E": 200.0,
                    "nu": 0.3,
                    "section_type": "solid_rectangle",
                    "width": 2.0,
                    "height": 1.0,
                },
            )
        ],
        dofs_per_node=6,
    )
    step = AnalysisStep(
        name="Step-1",
        line_loads=(LineLoad(70, (0.0, -2.0, 0.0)),),
    )
    model = FEMModel(mesh=mesh, steps=[step])
    return ModelResult(
        model=model,
        step=step,
        U=np.arange(1.0, 13.0),
        reactions=np.arange(101.0, 113.0),
    )


def _field_values(provider, variable: ResultVariable) -> np.ndarray:
    key = provider.resolve_request(
        _request(variable, FieldPosition.NODE)
    )
    return provider.field(key).values


def test_2d_provider_builds_mesh_order_topology_and_eager_primary_fields() -> None:
    result = _two_dimensional_result()
    provider = build_result_provider(_source(), result)

    assert provider.source == _source()
    assert provider.profile.family is ResultModelFamily.PLANE_CONTINUUM
    assert provider.snapshot.generation == 0
    topology = provider.snapshot.topology
    assert topology.node_ids == (10, 20, 30, 50)
    assert topology.node_coordinates == pytest.approx(np.asarray(
        [
            [1.0, 1.5, 0.0],
            [2.0, 2.5, 0.0],
            [3.0, 3.5, 0.0],
            [5.0, 5.5, 0.0],
        ]
    ))
    assert topology.nodal_displacements == pytest.approx(np.asarray(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 8.0, 0.0],
        ]
    ))
    assert topology.element_ids == (90,)
    assert topology.element_types == ("Quad4",)
    assert topology.connectivity == ((10, 20, 30, 50),)
    assert topology.element_region_keys == (
        result_region_key_for_element(result.model.mesh.elements[0]),
    )

    assert _field_values(provider, ResultVariable.U) == pytest.approx(np.asarray(
        [
            [1.0, 2.0, np.sqrt(5.0)],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, np.sqrt(61.0)],
            [7.0, 8.0, np.sqrt(113.0)],
        ]
    ))
    assert _field_values(provider, ResultVariable.RF) == pytest.approx(np.asarray(
        [
            [10.0, 20.0, np.sqrt(500.0)],
            [30.0, 40.0, 50.0],
            [50.0, 60.0, np.sqrt(6100.0)],
            [70.0, 80.0, np.sqrt(11300.0)],
        ]
    ))


def test_2d_catalog_marks_only_primary_ready_and_defaults_to_u_magnitude() -> None:
    provider = build_result_provider(_source(), _two_dimensional_result())
    catalog = provider.catalog()

    assert catalog.source == provider.source
    states = {
        (
            item.descriptor.field_id.variable,
            item.descriptor.field_id.position,
        ): item.state
        for item in catalog.fields
    }
    assert states[(ResultVariable.U, FieldPosition.NODE)] is FieldState.READY
    assert states[(ResultVariable.RF, FieldPosition.NODE)] is FieldState.READY
    for position in (
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
        FieldPosition.NODE_REGION,
        FieldPosition.RESOLVED_NODAL,
    ):
        assert states[(ResultVariable.S, position)] is FieldState.LAZY
    assert catalog.default_selection.component == "Magnitude"
    assert (
        catalog.default_selection.field_key.request.field_id.variable
        is ResultVariable.U
    )


def test_primary_locations_use_true_ids_coordinates_and_displacements() -> None:
    provider = build_result_provider(_source(), _two_dimensional_result())
    field_data = provider.field(
        provider.resolve_request(
            _request(ResultVariable.U, FieldPosition.NODE)
        )
    )

    assert tuple(location.node_id for location in field_data.locations) == (
        10,
        20,
        30,
        50,
    )
    assert field_data.locations[0].coordinates == (1.0, 1.5, 0.0)
    assert field_data.locations[0].displacement == (1.0, 2.0, 0.0)


def test_3d_continuum_primary_fields_have_three_actual_components() -> None:
    provider = build_result_provider(_source(), _three_dimensional_result())

    assert provider.profile.family is ResultModelFamily.SOLID_CONTINUUM
    u = provider.field(
        provider.resolve_request(
            _request(ResultVariable.U, FieldPosition.NODE)
        )
    )
    rf = provider.field(
        provider.resolve_request(
            _request(ResultVariable.RF, FieldPosition.NODE)
        )
    )
    assert u.descriptor.components == ("U1", "U2", "U3")
    assert rf.descriptor.components == ("RF1", "RF2", "RF3")
    assert u.values[0] == pytest.approx([1.0, 2.0, 3.0, np.sqrt(14.0)])
    assert rf.values[0] == pytest.approx(
        [101.0, 102.0, 103.0, np.sqrt(101**2 + 102**2 + 103**2)]
    )


def test_beam_primary_mapping_uses_u_ur_rf_rm_dof_labels() -> None:
    provider = build_result_provider(_source(), _beam_result())

    assert provider.profile.family is ResultModelFamily.BEAM
    assert _field_values(provider, ResultVariable.U) == pytest.approx(np.asarray(
        [
            [1.0, 2.0, 3.0, np.sqrt(14.0)],
            [7.0, 8.0, 9.0, np.sqrt(194.0)],
        ]
    ))
    assert _field_values(provider, ResultVariable.UR) == pytest.approx(
        np.asarray([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]])
    )
    assert _field_values(provider, ResultVariable.RF) == pytest.approx(np.asarray(
        [
            [101.0, 102.0, 103.0, np.sqrt(101**2 + 102**2 + 103**2)],
            [107.0, 108.0, 109.0, np.sqrt(107**2 + 108**2 + 109**2)],
        ]
    ))
    assert _field_values(provider, ResultVariable.RM) == pytest.approx(
        np.asarray([[104.0, 105.0, 106.0], [110.0, 111.0, 112.0]])
    )
    assert provider.snapshot.topology.nodal_displacements == pytest.approx(
        np.asarray([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    )


def test_provider_deep_owns_model_result_step_and_nested_element_state() -> None:
    result = _beam_result()
    provider = build_result_provider(_source(), result)
    before_topology = deepcopy(provider.snapshot.topology.connectivity)
    before_u = _field_values(provider, ResultVariable.U)
    before_rf = _field_values(provider, ResultVariable.RF)
    before_props = deepcopy(
        provider._owned_result.model.mesh.elements[0].props
    )

    assert provider._owned_result is not result
    assert provider._owned_result.model is not result.model
    assert provider._owned_result.step is provider._owned_result.model.steps[0]
    assert provider._owned_result.step is not result.step
    assert provider._owned_result.U.flags.writeable is False
    assert provider._owned_result.reactions.flags.writeable is False

    result.U[:] = -1000.0
    result.reactions[:] = 2000.0
    result.model.mesh.nodes[0].x = 999.0
    result.model.mesh.elements[0].node_ids[:] = [30, 10]
    result.model.mesh.elements[0].props["width"] = 500.0
    result.step.line_loads = (LineLoad(70, (8.0, 9.0, 10.0)),)

    assert provider.snapshot.topology.connectivity == before_topology
    assert _field_values(provider, ResultVariable.U) == pytest.approx(before_u)
    assert _field_values(provider, ResultVariable.RF) == pytest.approx(before_rf)
    assert provider._owned_result.model.mesh.nodes[0].x == 3.0
    assert provider._owned_result.model.mesh.elements[0].props == before_props
    assert provider._owned_result.step.line_loads[0].vector == (
        0.0,
        -2.0,
        0.0,
    )


def test_public_array_access_cannot_mutate_provider_snapshot() -> None:
    provider = build_result_provider(_source(), _three_dimensional_result())
    field_data = provider.field(
        provider.resolve_request(
            _request(ResultVariable.U, FieldPosition.NODE)
        )
    )
    values = field_data.values
    coordinates = provider.snapshot.topology.node_coordinates

    with pytest.raises(ValueError):
        values[0, 0] = 999.0
    with pytest.raises(ValueError):
        coordinates[0, 0] = 999.0
    values.setflags(write=True)
    coordinates.setflags(write=True)
    values[0, 0] = 999.0
    coordinates[0, 0] = 999.0

    assert field_data.values[0, 0] == 1.0
    assert provider.snapshot.topology.node_coordinates[0, 0] == 0.0


def _mixed_solid_truss_result() -> ModelResult:
    nodes = [
        Node3D(10 + index * 10, float(index), float(index % 2), 0.0)
        for index in range(8)
    ]
    mesh = Mesh3D(
        nodes=nodes,
        elements=[
            Element3D(
                500,
                [node.id for node in nodes],
                "Hex8",
                props={"E": 20.0, "nu": 0.3},
            ),
            Element3D(
                200,
                [nodes[0].id, nodes[1].id],
                "Truss2",
                props={"E": 20.0, "A": 2.0},
            ),
        ],
    )
    return ModelResult(
        model=FEMModel(mesh=mesh),
        step=None,
        U=np.arange(float(mesh.num_dofs)),
        reactions=np.arange(float(mesh.num_dofs)) + 100.0,
    )


def test_common_dof_mixed_model_publishes_primary_without_partial_stress() -> None:
    provider = build_result_provider(_source(), _mixed_solid_truss_result())
    catalog = provider.catalog()

    assert provider.profile.family is ResultModelFamily.MIXED_UNSUPPORTED
    assert provider.profile.primary_compatible is True
    assert provider.profile.stress_compatible is False
    assert tuple(
        item.descriptor.field_id.variable
        for item in catalog.fields
    ) == (ResultVariable.U, ResultVariable.RF)
    assert all(
        item.state is FieldState.READY
        for item in catalog.fields
    )
    assert tuple(item.code for item in catalog.diagnostics) == (
        "result.catalog.stress_family_unsupported",
    )
    assert catalog.diagnostics[0].path == (
        "results",
        "catalog",
        "variables",
        "S",
    )
    assert provider.snapshot.topology.element_ids == (500, 200)
    assert provider.snapshot.topology.element_types == ("Hex8", "Truss2")


def test_incompatible_dof_mix_fails_base_provider_construction() -> None:
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(1, [1, 2], "Truss2"),
            Element3D(2, [1, 2], "Beam2"),
        ],
        dofs_per_node=6,
    )
    result = ModelResult(
        FEMModel(mesh=mesh),
        None,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )

    with pytest.raises(ValueError, match="common primary DOF"):
        build_result_provider(_source(), result)


@pytest.mark.parametrize(
    ("connectivity", "message"),
    (
        ([10, 20, 30], "must contain 4"),
        ([10, 20, 30, 30], "must not repeat"),
    ),
)
def test_topology_rejects_wrong_or_repeated_element_connectivity(
    connectivity: list[int],
    message: str,
) -> None:
    result = _two_dimensional_result()
    result.model.mesh.elements[0].node_ids = connectivity

    with pytest.raises(ValueError, match=message):
        build_result_provider(_source(), result)


def test_request_resolution_and_lookup_use_the_complete_key() -> None:
    provider = build_result_provider(_source(), _two_dimensional_result())
    resolved_request = _request(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(25.0),
    )
    key = provider.resolve_request(resolved_request)

    assert key.request is resolved_request
    assert key.recovery_contract == 1
    assert provider.field_status(key).state is FieldState.LAZY
    with pytest.raises(KeyError):
        provider.field(key)
    with pytest.raises(KeyError):
        provider.field_status(
            FieldMaterializationKey(key.request, recovery_contract=2)
        )


def test_contextually_unknown_field_and_truss_gauss_order_fail_closed() -> None:
    beam = build_result_provider(_source(), _beam_result())
    with pytest.raises(KeyError):
        beam.resolve_request(
            _request(ResultVariable.S, FieldPosition.CENTROID)
        )

    result = _mixed_solid_truss_result()
    result.model.mesh.elements = [result.model.mesh.elements[1]]
    result.model.mesh.rebuild_dof_map()
    truss = build_result_provider(_source(), result)
    with pytest.raises(ValueError, match="gauss_order"):
        truss.resolve_request(
            _request(
                ResultVariable.S,
                FieldPosition.CENTROID,
                gauss_order=1,
            )
        )


def test_build_provider_requires_exact_source_and_result_types() -> None:
    result = _two_dimensional_result()
    with pytest.raises(TypeError, match="source"):
        build_result_provider(object(), result)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ModelResult"):
        build_result_provider(_source(), object())  # type: ignore[arg-type]
