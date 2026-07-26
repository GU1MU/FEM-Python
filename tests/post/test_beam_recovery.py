from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel, LineLoad
from fem.core.result import ModelResult
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    get_element_kernel,
    resolve_beam_frame,
)
from fem.post.stress import beam


class _RecoveryCancelled(RuntimeError):
    pass


def _beam_properties() -> dict[str, float | str]:
    return {
        "E": 100.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 2.0,
        "width": 1.0,
    }


def _chain_result(*, with_isolated_node: bool = True) -> ModelResult:
    nodes = [
        Node3D(10, 0.0, 0.0, 0.0),
        Node3D(20, 1.0, 0.0, 0.0),
        Node3D(30, 2.0, 0.0, 0.0),
    ]
    if with_isolated_node:
        nodes.append(Node3D(40, 3.0, 0.0, 0.0))
    mesh = Mesh3D(
        nodes=nodes,
        elements=[
            Element3D(101, [10, 20], "Beam2", _beam_properties()),
            Element3D(205, [20, 30], "Beam2", _beam_properties()),
        ],
        dofs_per_node=6,
    )
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(20, 0)] = 0.1
    displacement[mesh.global_dof(30, 0)] = 0.1
    return ModelResult(
        FEMModel(mesh=mesh),
        AnalysisStep("Load"),
        displacement,
        np.zeros(mesh.num_dofs),
    )


def test_section_end_rows_keep_element_end_provenance_and_mesh_order() -> None:
    result = _chain_result()

    field = beam.recover_section_end_stress(result)

    assert field.position == "section_end"
    assert field.component_names == ("S11Max", "S11Min", "S11AbsMax")
    assert field.node_order == (10, 20, 30, 40)
    assert [
        (row.element_id, row.local_node, row.node_id)
        for row in field.rows
    ] == [
        (101, 1, 10),
        (101, 2, 20),
        (205, 1, 20),
        (205, 2, 30),
    ]
    assert [row.coordinates for row in field.rows] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    assert [row.displacement for row in field.rows] == [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    ]
    assert [
        (row.s11_max, row.s11_min, row.s11_abs_max)
        for row in field.rows
    ] == pytest.approx([
        (10.0, 10.0, 10.0),
        (10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ])
    assert [row.axial_force for row in field.rows] == pytest.approx(
        [20.0, 20.0, 0.0, 0.0]
    )
    assert all(row.moment_y == row.moment_z == 0.0 for row in field.rows)
    assert field.rows[0].values() == {
        "S11Max": pytest.approx(10.0),
        "S11Min": pytest.approx(10.0),
        "S11AbsMax": pytest.approx(10.0),
    }

    with pytest.raises(FrozenInstanceError):
        field.rows[0].node_id = 999


def test_canonical_node_envelope_omits_isolated_nodes_and_keeps_shared_extrema() -> None:
    field = beam.recover_section_end_stress(_chain_result())

    envelope = beam.section_node_envelope(field)

    assert envelope.position == "section_node_envelope"
    assert envelope.component_names == ("S11Max", "S11Min", "S11AbsMax")
    assert [row.node_id for row in envelope.rows] == [10, 20, 30]
    assert [
        (row.s11_max, row.s11_min, row.s11_abs_max)
        for row in envelope.rows
    ] == pytest.approx([
        (10.0, 10.0, 10.0),
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 0.0),
    ])
    assert envelope.rows[1].coordinates == (1.0, 0.0, 0.0)
    assert envelope.rows[1].displacement == (0.1, 0.0, 0.0)

    legacy = beam.nodal_envelope(_chain_result())
    assert [row.node_id for row in legacy] == [10, 20, 30, 40]
    assert legacy[-1] == beam.Beam2NodalStress(40, 0.0, 0.0, 0.0)


def test_section_end_recovery_cancels_after_one_element_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _chain_result(with_isolated_node=False)
    kernel_type = type(get_element_kernel("Beam2"))
    original = kernel_type.local_end_actions
    completed_elements: list[int] = []

    def counted(
        self,
        mesh,
        element,
        displacement,
        local_load,
        lookup,
    ):
        actions = original(
            self,
            mesh,
            element,
            displacement,
            local_load,
            lookup,
        )
        completed_elements.append(int(element.id))
        return actions

    def checkpoint() -> None:
        if completed_elements:
            raise _RecoveryCancelled("cancelled after one Beam2 element")

    monkeypatch.setattr(kernel_type, "local_end_actions", counted)

    with pytest.raises(_RecoveryCancelled, match="one Beam2 element"):
        beam.recover_section_end_stress(
            result,
            checkpoint=checkpoint,
        )

    assert completed_elements == [101]
    retried = beam.recover_section_end_stress(result)
    assert [
        (row.element_id, row.local_node)
        for row in retried.rows
    ] == [(101, 1), (101, 2), (205, 1), (205, 2)]
    assert completed_elements == [101, 101, 205]


@pytest.mark.parametrize(
    "cancel_at",
    (
        pytest.param(2, id="contribution-loop"),
        pytest.param(6, id="node-loop"),
    ),
)
def test_node_envelope_cancels_inside_each_loop_and_retries(
    cancel_at: int,
) -> None:
    field = beam.recover_section_end_stress(_chain_result())
    checkpoint_calls = 0

    def checkpoint() -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls == cancel_at:
            raise _RecoveryCancelled("cancelled during Beam envelope")

    with pytest.raises(_RecoveryCancelled, match="Beam envelope"):
        beam.section_node_envelope(
            field,
            checkpoint=checkpoint,
        )

    assert checkpoint_calls == cancel_at
    retried = beam.section_node_envelope(field)
    assert [row.node_id for row in retried.rows] == [10, 20, 30]


def test_section_end_recovery_preserves_arbitrary_integer_mesh_ids() -> None:
    mesh = Mesh3D(
        nodes=[
            Node3D(0, 0.0, 0.0, 0.0),
            Node3D(-2, 1.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(-10, [0, -2], "Beam2", _beam_properties()),
        ],
        dofs_per_node=6,
    )
    result = ModelResult(
        FEMModel(mesh=mesh),
        AnalysisStep("Load"),
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )

    field = beam.recover_section_end_stress(result)

    assert field.node_order == tuple(mesh.node_ids) == (-2, 0)
    assert [
        (row.element_id, row.local_node, row.node_id)
        for row in field.rows
    ] == [(-10, 1, 0), (-10, 2, -2)]


def test_section_end_recovery_includes_distributed_load_context() -> None:
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 4.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(10, [1, 2], "Beam2", _beam_properties()),
        ],
        dofs_per_node=6,
    )
    model = FEMModel(mesh=mesh)
    unloaded = ModelResult(
        model,
        AnalysisStep("unloaded"),
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )
    loaded = ModelResult(
        model,
        AnalysisStep(
            "loaded",
            line_loads=(LineLoad(10, (0.0, 12.0, 0.0), "local"),),
        ),
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )

    unloaded_rows = beam.recover_section_end_stress(unloaded).rows
    loaded_rows = beam.recover_section_end_stress(loaded).rows

    assert [row.s11_abs_max for row in unloaded_rows] == [0.0, 0.0]
    assert [row.moment_z for row in loaded_rows] == pytest.approx(
        [16.0, 16.0]
    )
    assert loaded_rows[0].s11_abs_max > 0.0
    assert loaded_rows[1].s11_abs_max > 0.0
    assert [
        (row.maximum, row.minimum, row.absolute_maximum)
        for row in beam.nodal_envelope(loaded)
    ] == pytest.approx([
        (
            loaded_rows[0].s11_max,
            loaded_rows[0].s11_min,
            loaded_rows[0].s11_abs_max,
        ),
        (
            loaded_rows[1].s11_max,
            loaded_rows[1].s11_min,
            loaded_rows[1].s11_abs_max,
        ),
    ])


def test_reversed_connectivity_swaps_local_end_identity_but_keeps_node_values() -> None:
    result = _chain_result(with_isolated_node=False)
    forward = beam.recover_section_end_stress(result)
    result.model.mesh.elements[0].node_ids = [20, 10]

    reversed_field = beam.recover_section_end_stress(result)

    assert [
        (row.local_node, row.node_id)
        for row in reversed_field.rows[:2]
    ] == [(1, 20), (2, 10)]
    forward_by_node = {
        row.node_id: (row.s11_max, row.s11_min, row.s11_abs_max)
        for row in forward.rows[:2]
    }
    reversed_by_node = {
        row.node_id: (row.s11_max, row.s11_min, row.s11_abs_max)
        for row in reversed_field.rows[:2]
    }
    assert reversed_by_node == pytest.approx(forward_by_node)


def test_explicit_orientation_global_line_load_matches_local_end_actions() -> None:
    properties = _beam_properties()
    properties[BEAM_LOCAL_Y_REFERENCE_KEY] = (0.0, 1.0, 0.0)
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 3.0, 6.0),
        ],
        elements=[Element3D(10, [1, 2], "Beam2", properties)],
        dofs_per_node=6,
    )
    model = FEMModel(mesh=mesh)
    frame = resolve_beam_frame(mesh, mesh.elements[0])
    local_vector = np.asarray((2.0, 3.0, 4.0))
    global_vector = frame.rotation.T @ local_vector

    def recover(line_load: LineLoad) -> beam.BeamEndStressField:
        return beam.recover_section_end_stress(
            ModelResult(
                model,
                AnalysisStep("Load", line_loads=(line_load,)),
                np.zeros(mesh.num_dofs),
                np.zeros(mesh.num_dofs),
            )
        )

    local = recover(LineLoad(10, local_vector, "local"))
    global_field = recover(LineLoad(10, global_vector, "global"))

    global_values = [
        (
            row.axial_force,
            row.moment_y,
            row.moment_z,
            row.s11_max,
            row.s11_min,
            row.s11_abs_max,
        )
        for row in global_field.rows
    ]
    local_values = [
        (
            row.axial_force,
            row.moment_y,
            row.moment_z,
            row.s11_max,
            row.s11_min,
            row.s11_abs_max,
        )
        for row in local.rows
    ]
    assert np.allclose(global_values, local_values)


def test_section_end_recovery_rejects_non_model_result_and_mixed_family() -> None:
    with pytest.raises(TypeError, match="ModelResult"):
        beam.recover_section_end_stress(object())

    result = _chain_result(with_isolated_node=False)
    result.model.mesh.elements[1].type = "Truss2"

    with pytest.raises(ValueError, match="only Beam2"):
        beam.recover_section_end_stress(result)


def test_section_end_recovery_rejects_non_six_dof_and_invalid_topology() -> None:
    seven_dof_mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2", _beam_properties())],
        dofs_per_node=7,
    )
    seven_dof_result = ModelResult(
        FEMModel(mesh=seven_dof_mesh),
        AnalysisStep("Load"),
        np.zeros(seven_dof_mesh.num_dofs),
        np.zeros(seven_dof_mesh.num_dofs),
    )
    with pytest.raises(ValueError, match="exactly six DOFs"):
        beam.recover_section_end_stress(seven_dof_result)

    invalid_mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
        elements=[Element3D(1, [1, 2, 3], "Beam2", _beam_properties())],
        dofs_per_node=6,
    )
    invalid_result = ModelResult(
        FEMModel(mesh=invalid_mesh),
        AnalysisStep("Load"),
        np.zeros(invalid_mesh.num_dofs),
        np.zeros(invalid_mesh.num_dofs),
    )
    with pytest.raises(ValueError, match="exactly two nodes"):
        beam.recover_section_end_stress(invalid_result)


def test_section_end_recovery_rejects_invalid_section_and_end_action_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _chain_result(with_isolated_node=False)
    result.model.mesh.elements[0].props.pop("E")
    with pytest.raises(KeyError, match="missing property E"):
        beam.recover_section_end_stress(result)

    result = _chain_result(with_isolated_node=False)
    kernel_type = type(get_element_kernel("Beam2"))
    monkeypatch.setattr(
        kernel_type,
        "local_end_actions",
        lambda *args, **kwargs: np.zeros((1, 3)),
    )
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        beam.recover_section_end_stress(result)
