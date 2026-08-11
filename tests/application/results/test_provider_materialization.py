from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fem.application.results import _materializers
from fem.application.results.data import (
    FieldState,
    ResultMaterializationPatch,
)
from fem.application.results.fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    field_materialization_sort_key,
)
from fem.application.results.provider import build_result_provider
from fem.core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import AnalysisStep, FEMModel, LineLoad
from fem.core.result import ModelResult
from fem.elements import get_element_capabilities, get_element_kernel
from fem.post.averaging import NodalAveragingPolicy
from tests.helpers.mesh_builders import make_hex8_stiffness_mesh
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)


def _source(suffix: str = "1") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result-{suffix}",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=2,
        step_name="Step-1",
        run_id=f"run-{suffix}",
    )


def _request(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
    section_point_number: int | None = None,
) -> FieldRequest:
    return FieldRequest(
        ResultFieldId(variable, position, section_point_number),
        averaging_policy=policy,
        gauss_order=gauss_order,
    )


def _key(
    provider,
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
    section_point_number: int | None = None,
):
    return provider.resolve_request(
        _request(
            variable,
            position,
            policy=policy,
            gauss_order=gauss_order,
            section_point_number=section_point_number,
        )
    )


def _assert_same_field(first, second) -> None:
    assert first.descriptor == second.descriptor
    assert first.source == second.source
    assert first.key == second.key
    assert first.locations == second.locations
    assert first.values == pytest.approx(second.values)


def _quad4_result() -> ModelResult:
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                10,
                [1, 2, 3, 4],
                "Quad4",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "plane_type": "stress",
                    "thickness": 1.0,
                },
            )
        ],
    )
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 0)] = 0.1
    displacement[mesh.global_dof(3, 0)] = 0.1
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        displacement,
        np.zeros(mesh.num_dofs),
    )


def _hex8_result() -> ModelResult:
    mesh = make_hex8_stiffness_mesh()
    displacement = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = 0.01 * node.x
        displacement[mesh.global_dof(node.id, 1)] = 0.02 * node.y
        displacement[mesh.global_dof(node.id, 2)] = 0.03 * node.z
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        displacement,
        np.zeros(mesh.num_dofs),
    )


def _loaded_beam_result() -> ModelResult:
    properties = {
        "E": 100.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 2.0,
        "width": 1.0,
    }
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 4.0, 0.0, 0.0),
        ],
        elements=[Element3D(10, [1, 2], "Beam2", properties)],
        dofs_per_node=6,
    )
    step = AnalysisStep(
        "Step-1",
        line_loads=(LineLoad(10, (0.0, 12.0, 0.0), "local"),),
    )
    model = FEMModel(mesh=mesh, steps=[step])
    return ModelResult(
        model,
        step,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )


def _truss_chain_result() -> ModelResult:
    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 1.0, 0.0, 0.0),
            Node3D(30, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                101,
                [10, 20],
                "Truss2",
                {"E": 100.0, "area": 1.0},
            ),
            Element3D(
                205,
                [20, 30],
                "Truss2",
                {"E": 100.0, "area": 1.0},
            ),
        ],
    )
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )


def _beam_chain_result() -> ModelResult:
    properties = {
        "E": 100.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 2.0,
        "width": 1.0,
    }
    mesh = Mesh3D(
        nodes=[
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 1.0, 0.0, 0.0),
            Node3D(30, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(101, [10, 20], "Beam2", properties),
            Element3D(205, [20, 30], "Beam2", properties),
        ],
        dofs_per_node=6,
    )
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )


def _mixed_continuum_result(
    element_types: tuple[str, ...],
) -> ModelResult:
    capabilities = tuple(
        get_element_capabilities(element_type)
        for element_type in element_types
    )
    spatial_dimension = capabilities[0].spatial_dimension
    nodes = []
    elements = []
    next_node_id = 1
    for element_index, (element_type, capability) in enumerate(
        zip(element_types, capabilities, strict=True),
        start=1,
    ):
        node_ids = tuple(
            range(next_node_id, next_node_id + capability.node_count)
        )
        next_node_id += capability.node_count
        if spatial_dimension == 2:
            nodes.extend(
                Node2D(
                    node_id,
                    float(node_id),
                    float(element_index),
                )
                for node_id in node_ids
            )
            elements.append(
                Element2D(element_index, list(node_ids), element_type)
            )
        else:
            nodes.extend(
                Node3D(
                    node_id,
                    float(node_id),
                    float(element_index),
                    0.0,
                )
                for node_id in node_ids
            )
            elements.append(
                Element3D(element_index, list(node_ids), element_type)
            )
    mesh = (
        Mesh2D(nodes=nodes, elements=elements)
        if spatial_dimension == 2
        else Mesh3D(nodes=nodes, elements=elements)
    )
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )


def test_continuum_materializes_every_position_as_one_atomic_ordered_patch() -> None:
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    keys = (
        _key(
            provider,
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(),
        ),
        _key(provider, ResultVariable.S, FieldPosition.NODE_REGION),
        _key(provider, ResultVariable.S, FieldPosition.ELEMENT_NODAL),
        _key(provider, ResultVariable.S, FieldPosition.CENTROID),
        _key(provider, ResultVariable.S, FieldPosition.INTEGRATION_POINT),
    )

    patch = provider.materialize((*keys, keys[0], keys[2]))

    assert tuple(field.key for field in patch.fields) == tuple(
        sorted(set(keys), key=field_materialization_sort_key)
    )
    by_position = {
        field.key.request.field_id.position: field for field in patch.fields
    }
    assert {
        position: len(field.locations)
        for position, field in by_position.items()
    } == {
        FieldPosition.INTEGRATION_POINT: 3,
        FieldPosition.CENTROID: 3,
        FieldPosition.ELEMENT_NODAL: 9,
        FieldPosition.NODE_REGION: 8,
        FieldPosition.RESOLVED_NODAL: 9,
    }
    assert {
        position: field.descriptor.association
        for position, field in by_position.items()
    } == {
        FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
        FieldPosition.CENTROID: FieldAssociation.ELEMENT,
        FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
        FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
        FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    }
    for field in patch.fields:
        assert field.values.shape[1] == 8
        assert np.isfinite(field.values).all()
        assert all(len(location.coordinates) == 3 for location in field.locations)
        assert all(
            location.displacement is not None
            and len(location.displacement) == 3
            for location in field.locations
        )


def test_solid_continuum_materializes_every_position_with_six_components() -> None:
    provider = build_result_provider(_source("solid"), _hex8_result())
    keys = tuple(
        _key(
            provider,
            ResultVariable.S,
            position,
            policy=(
                NodalAveragingPolicy()
                if position is FieldPosition.RESOLVED_NODAL
                else None
            ),
        )
        for position in (
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
        )
    )

    patch = provider.materialize(keys)

    by_position = {
        field.key.request.field_id.position: field for field in patch.fields
    }
    assert {
        position: len(field.locations)
        for position, field in by_position.items()
    } == {
        FieldPosition.INTEGRATION_POINT: 8,
        FieldPosition.CENTROID: 1,
        FieldPosition.ELEMENT_NODAL: 8,
        FieldPosition.NODE_REGION: 8,
        FieldPosition.RESOLVED_NODAL: 8,
    }
    for field in patch.fields:
        assert field.descriptor.components == (
            "S11",
            "S22",
            "S33",
            "S12",
            "S23",
            "S13",
        )
        assert field.values.shape == (len(field.locations), 10)
        assert np.isfinite(field.values).all()
        assert all(len(location.coordinates) == 3 for location in field.locations)
        assert all(
            location.displacement is not None
            and len(location.displacement) == 3
            for location in field.locations
        )


def test_apply_creates_same_generation_draft_and_leaves_old_provider_unchanged() -> None:
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    key = _key(provider, ResultVariable.S, FieldPosition.CENTROID)
    patch = provider.materialize((key,))

    draft = provider.apply(patch)

    assert draft is not provider
    assert draft.snapshot.generation == provider.snapshot.generation == 0
    assert provider.field_status(key).state is FieldState.LAZY
    assert draft.field_status(key).state is FieldState.READY
    with pytest.raises(KeyError):
        provider.field(key)
    assert draft.field(key).key == key
    cache_hit = draft.materialize((key, key))
    assert cache_hit.fields == ()
    assert draft.apply(cache_hit) is draft


def test_continuum_batch_reuses_one_recovery_for_all_default_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    keys = tuple(
        _key(
            provider,
            ResultVariable.S,
            position,
            policy=(
                NodalAveragingPolicy()
                if position is FieldPosition.RESOLVED_NODAL
                else None
            ),
        )
        for position in (
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
        )
    )
    original = _materializers.StressRecovery
    calls = []

    class CountingRecovery(original):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("gauss_order"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_materializers, "StressRecovery", CountingRecovery)

    provider.materialize(keys)

    assert calls == [None]


def test_policy_specific_resolved_keys_coexist_and_keep_exact_row_identity() -> None:
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    raw_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    averaged_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )

    patch = provider.materialize((averaged_key, raw_key))
    draft = provider.apply(patch)
    raw = draft.field(raw_key)
    averaged = draft.field(averaged_key)

    assert raw.key != averaged.key
    assert len(raw.locations) == 9
    assert len(averaged.locations) == 8
    raw_center = [
        location for location in raw.locations if location.node_id == 1
    ]
    averaged_center = [
        location for location in averaged.locations if location.node_id == 1
    ]
    assert [(item.element_id, item.local_node, item.averaged) for item in raw_center] == [
        (1, 1, False),
        (2, 1, False),
        (3, 1, False),
    ]
    assert [
        (item.element_id, item.local_node, item.averaged)
        for item in averaged_center
    ] == [
        (None, None, True),
        (3, 1, False),
    ]


def test_custom_gauss_keys_coexist_and_invalid_position_batch_is_atomic() -> None:
    provider = build_result_provider(_source(), _quad4_result())
    ip_one = _key(
        provider,
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
        gauss_order=1,
    )
    ip_two = _key(
        provider,
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
        gauss_order=2,
    )

    draft = provider.apply(provider.materialize((ip_two, ip_one)))

    assert len(draft.field(ip_one).locations) == 1
    assert len(draft.field(ip_two).locations) == 4
    assert ip_one != ip_two

    invalid_en = FieldMaterializationKey(
        _request(
            ResultVariable.S,
            FieldPosition.ELEMENT_NODAL,
            gauss_order=1,
        ),
        recovery_contract=1,
    )
    with pytest.raises(ValueError, match="gauss_order"):
        provider.materialize((ip_one, invalid_en))
    assert provider.field_status(ip_one).state is FieldState.LAZY
    assert provider.snapshot.generation == 0
    valid_en = _key(
        provider,
        ResultVariable.S,
        FieldPosition.ELEMENT_NODAL,
    )
    assert len(provider.materialize((valid_en,)).fields) == 1


@pytest.mark.parametrize(
    ("element_types", "position", "allowed", "rejected"),
    (
        (
            ("Quad4",),
            FieldPosition.INTEGRATION_POINT,
            (1, 2),
            (3,),
        ),
        (
            ("Quad4",),
            FieldPosition.ELEMENT_NODAL,
            (2,),
            (1, 3),
        ),
        (
            ("Quad8",),
            FieldPosition.RESOLVED_NODAL,
            (2, 3),
            (1,),
        ),
        (("Tri6",), FieldPosition.CENTROID, (3,), (1, 2)),
        (("Tri3",), FieldPosition.CENTROID, (), (1, 2, 3)),
        (("Tet4",), FieldPosition.INTEGRATION_POINT, (), (1, 2)),
        (("Tet10",), FieldPosition.CENTROID, (), (2, 3)),
        (("Hex8",), FieldPosition.ELEMENT_NODAL, (2,), (1, 3)),
        (("Hex20",), FieldPosition.NODE_REGION, (3,), (1, 2)),
        (
            ("Quad4", "Quad8"),
            FieldPosition.INTEGRATION_POINT,
            (2,),
            (1, 3),
        ),
        (
            ("Tri6", "Quad4"),
            FieldPosition.CENTROID,
            (),
            (2, 3),
        ),
        (
            ("Hex8", "Hex20"),
            FieldPosition.INTEGRATION_POINT,
            (),
            (2, 3),
        ),
    ),
)
def test_resolve_request_enforces_contextual_common_gauss_order(
    element_types: tuple[str, ...],
    position: FieldPosition,
    allowed: tuple[int, ...],
    rejected: tuple[int, ...],
) -> None:
    provider = build_result_provider(
        _source("-".join(element_types)),
        _mixed_continuum_result(element_types),
    )
    policy = (
        NodalAveragingPolicy()
        if position is FieldPosition.RESOLVED_NODAL
        else None
    )

    assert _key(
        provider,
        ResultVariable.S,
        position,
        policy=policy,
    ).request.gauss_order is None
    for order in allowed:
        assert _key(
            provider,
            ResultVariable.S,
            position,
            policy=policy,
            gauss_order=order,
        ).request.gauss_order == order
    for order in rejected:
        with pytest.raises(ValueError, match="gauss_order"):
            _key(
                provider,
                ResultVariable.S,
                position,
                policy=policy,
                gauss_order=order,
            )


def test_truss_one_recovery_splits_le_and_s_with_exact_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(
        _source(),
        make_truss_field_characterization_result(),
    )
    le_key = _key(provider, ResultVariable.LE, FieldPosition.CENTROID)
    stress_key = _key(provider, ResultVariable.S, FieldPosition.CENTROID)
    original = _materializers.truss.recover
    calls = []

    def counted_recover(mesh, displacement, *, checkpoint=None):
        calls.append((mesh, displacement))
        return original(
            mesh,
            displacement,
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(_materializers.truss, "recover", counted_recover)

    patch = provider.materialize((stress_key, le_key, stress_key))
    draft = provider.apply(patch)

    assert calls and len(calls) == 1
    assert tuple(field.key for field in patch.fields) == (le_key, stress_key)
    le = draft.field(le_key)
    stress = draft.field(stress_key)
    assert le.descriptor.columns == ("LE11",)
    assert stress.descriptor.columns == ("S11", "Mises")
    assert le.values == pytest.approx(np.asarray([[0.1]]))
    assert stress.values == pytest.approx(np.asarray([[10.0, 10.0]]))
    assert le.locations[0].element_id == 30
    assert le.locations[0].coordinates == (1.0, 0.0, 0.0)
    assert le.locations[0].displacement == (0.1, 0.0, 0.0)


def test_batch_and_incremental_materialization_are_numerically_identical() -> None:
    result = make_truss_field_characterization_result()
    batch_provider = build_result_provider(_source(), result)
    incremental_provider = build_result_provider(_source(), result)
    le_key = _key(batch_provider, ResultVariable.LE, FieldPosition.CENTROID)
    stress_key = _key(batch_provider, ResultVariable.S, FieldPosition.CENTROID)

    batch = batch_provider.apply(
        batch_provider.materialize((le_key, stress_key))
    )
    incremental = incremental_provider.apply(
        incremental_provider.materialize((le_key,))
    )
    incremental = incremental.apply(incremental.materialize((stress_key,)))

    _assert_same_field(batch.field(le_key), incremental.field(le_key))
    _assert_same_field(batch.field(stress_key), incremental.field(stress_key))


def test_beam_one_integration_recovery_materializes_requested_section_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(
        _source(),
        make_beam_field_characterization_result(),
    )
    point_one_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
        section_point_number=1,
    )
    point_four_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
        section_point_number=4,
    )
    original = _materializers.beam.recover_integration_point_s11
    calls = []

    def counted_recover(result, *, checkpoint=None):
        calls.append(result)
        return original(result, checkpoint=checkpoint)

    monkeypatch.setattr(
        _materializers.beam,
        "recover_integration_point_s11",
        counted_recover,
    )

    draft = provider.apply(
        provider.materialize((point_four_key, point_one_key))
    )

    assert len(calls) == 1
    for key, number in ((point_one_key, 1), (point_four_key, 4)):
        field = draft.field(key)
        assert len(field.locations) == 1
        location = field.locations[0]
        assert (location.element_id, location.integration_point) == (30, 1)
        assert location.section_point is not None
        assert location.section_point.number == number
        assert field.descriptor.columns == (
            "S11",
            "S22",
            "S12",
            "Mises",
            "MaxPrincipal",
            "MidPrincipal",
            "MinPrincipal",
        )
        assert field.descriptor.association is FieldAssociation.INTEGRATION_POINT


def test_beam_lazy_integration_recovery_uses_owned_result_after_caller_mutation() -> None:
    result = _loaded_beam_result()
    provider = build_result_provider(_source(), result)
    point_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
        section_point_number=1,
    )

    result.step.line_loads = ()
    result.model.steps[0].line_loads = ()
    result.U[:] = 500.0
    result.model.mesh.elements[0].props["height"] = 1000.0
    field = provider.apply(
        provider.materialize((point_key,))
    ).field(point_key)

    assert field.values[:, 0] == pytest.approx([0.0])
    assert field.locations[0].integration_point == 1


def test_apply_rejects_foreign_unknown_collision_and_ready_overwrite() -> None:
    source = _source()
    continuum = build_result_provider(
        source,
        make_continuum_nodal_semantics_result(),
    )
    truss = build_result_provider(
        source,
        make_truss_field_characterization_result(),
    )
    le_key = _key(truss, ResultVariable.LE, FieldPosition.CENTROID)
    stress_key = _key(truss, ResultVariable.S, FieldPosition.CENTROID)
    truss_patch = truss.materialize((le_key, stress_key))

    with pytest.raises(KeyError):
        continuum.apply(
            ResultMaterializationPatch(source, (truss_patch.fields[0],))
        )
    with pytest.raises(ValueError, match="descriptor"):
        continuum.apply(
            ResultMaterializationPatch(source, (truss_patch.fields[1],))
        )
    draft = truss.apply(truss_patch)
    with pytest.raises(ValueError, match="READY"):
        draft.apply(truss_patch)
    foreign = ResultMaterializationPatch(
        source=replace(source, result_id="foreign"),
        fields=(),
    )
    with pytest.raises(ValueError, match="source"):
        continuum.apply(foreign)


def test_post_exception_does_not_poison_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(
        _source(),
        make_truss_field_characterization_result(),
    )
    key = _key(provider, ResultVariable.S, FieldPosition.CENTROID)
    original = _materializers.truss.recover
    calls = 0

    def fail_once(mesh, displacement, *, checkpoint=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected recovery failure")
        return original(
            mesh,
            displacement,
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(_materializers.truss, "recover", fail_once)

    with pytest.raises(RuntimeError, match="injected recovery failure"):
        provider.materialize((key,))
    assert provider.field_status(key).state is FieldState.LAZY
    assert len(provider.materialize((key,)).fields) == 1
    assert calls == 2


class _Cancelled(RuntimeError):
    pass


class _CancellationSwitch:
    def __init__(self) -> None:
        self.cancelled = False

    def checkpoint(self) -> None:
        if self.cancelled:
            raise _Cancelled("cancelled")


def test_continuum_element_loop_cancellation_returns_no_patch_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )
    keys = (
        _key(provider, ResultVariable.S, FieldPosition.INTEGRATION_POINT),
        _key(provider, ResultVariable.S, FieldPosition.CENTROID),
    )
    before = provider.snapshot
    cancellation = _CancellationSwitch()
    kernel_type = type(get_element_kernel("Tri3"))
    original = kernel_type.integration_point_stress
    completed_elements: list[int] = []
    arm_once = True

    def counted(self, mesh, element, *args, **kwargs):
        nonlocal arm_once
        values = original(
            self,
            mesh,
            element,
            *args,
            **kwargs,
        )
        completed_elements.append(int(element.id))
        if arm_once:
            cancellation.cancelled = True
            arm_once = False
        return values

    monkeypatch.setattr(
        kernel_type,
        "integration_point_stress",
        counted,
    )

    with pytest.raises(_Cancelled, match="cancelled"):
        provider.materialize(keys, cancellation=cancellation)

    assert completed_elements == [1]
    assert provider.snapshot is before
    assert all(
        provider.field_status(key).state is FieldState.LAZY for key in keys
    )
    cancellation.cancelled = False
    patch = provider.materialize(keys, cancellation=cancellation)
    assert len(patch.fields) == 2
    assert completed_elements == [1, 1, 2, 3]


def test_truss_element_loop_cancellation_returns_no_patch_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(_source("truss"), _truss_chain_result())
    key = _key(provider, ResultVariable.S, FieldPosition.CENTROID)
    before = provider.snapshot
    cancellation = _CancellationSwitch()
    kernel_type = type(get_element_kernel("Truss2"))
    original = kernel_type.element_stress
    completed_elements: list[int] = []
    arm_once = True

    def counted(self, mesh, element, displacement, lookup):
        nonlocal arm_once
        values = original(
            self,
            mesh,
            element,
            displacement,
            lookup,
        )
        completed_elements.append(int(element.id))
        if arm_once:
            cancellation.cancelled = True
            arm_once = False
        return values

    monkeypatch.setattr(kernel_type, "element_stress", counted)

    with pytest.raises(_Cancelled, match="cancelled"):
        provider.materialize((key,), cancellation=cancellation)

    assert completed_elements == [101]
    assert provider.snapshot is before
    assert provider.field_status(key).state is FieldState.LAZY
    cancellation.cancelled = False
    patch = provider.materialize((key,), cancellation=cancellation)
    assert tuple(field.key for field in patch.fields) == (key,)
    assert completed_elements == [101, 101, 205]


def test_beam_element_loop_cancellation_returns_no_patch_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_result_provider(_source("beam"), _beam_chain_result())
    keys = (
        _key(
            provider,
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
            section_point_number=1,
        ),
        _key(
            provider,
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
            section_point_number=4,
        ),
    )
    before = provider.snapshot
    cancellation = _CancellationSwitch()
    kernel_type = type(get_element_kernel("Beam2"))
    original = kernel_type.local_integration_point_forces
    completed_elements: list[int] = []
    arm_once = True

    def counted(
        self,
        mesh,
        element,
        displacement,
        lookup,
    ):
        nonlocal arm_once
        values = original(
            self,
            mesh,
            element,
            displacement,
            lookup,
        )
        completed_elements.append(int(element.id))
        if arm_once:
            cancellation.cancelled = True
            arm_once = False
        return values

    monkeypatch.setattr(kernel_type, "local_integration_point_forces", counted)

    with pytest.raises(_Cancelled, match="cancelled"):
        provider.materialize(keys, cancellation=cancellation)

    assert completed_elements == [101]
    assert provider.snapshot is before
    assert all(
        provider.field_status(key).state is FieldState.LAZY
        for key in keys
    )
    cancellation.cancelled = False
    patch = provider.materialize(keys, cancellation=cancellation)
    assert {field.key for field in patch.fields} == set(keys)
    assert completed_elements == [101, 101, 205]


def test_callable_cancellation_convention_and_invalid_probe() -> None:
    provider = build_result_provider(
        _source(),
        make_truss_field_characterization_result(),
    )
    key = _key(provider, ResultVariable.S, FieldPosition.CENTROID)
    calls = []

    patch = provider.materialize(
        (key,),
        cancellation=lambda: calls.append(True),
    )

    assert patch.fields
    assert len(calls) >= 3
    with pytest.raises(TypeError, match="cancellation"):
        provider.materialize((key,), cancellation=object())
