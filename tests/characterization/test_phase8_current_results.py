from __future__ import annotations

import numpy as np
import pytest

from fem.post import stress
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data
from fem_gui.visualization.stress_adapter import build_stress_render_geometry
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)


def _records_for_node(records: tuple[object, ...], node_id: int) -> list[object]:
    return [record for record in records if record.node_id == node_id]


def _rendered_rows_for_node(rendered: object, node_id: int) -> list[tuple[float, int | None]]:
    return [
        (
            float(rendered.values[point_index]),
            rendered.point_index_to_element_id[point_index],
        )
        for point_index, rendered_node_id in rendered.point_index_to_node_id.items()
        if rendered_node_id == node_id
    ]


@pytest.mark.parametrize(
    ("threshold", "expected_gui_center"),
    [
        (0.0, [(10.0, 1), (30.0, 2), (50.0, None)]),
        (75.0, [(10.0, 1), (30.0, 2), (50.0, None)]),
        (100.0, [(20.0, None), (50.0, None)]),
    ],
)
def test_three_current_continuum_nodal_oracles_are_known_to_diverge(
    threshold: float,
    expected_gui_center: list[tuple[float, int | None]],
) -> None:
    """Freeze the three current nodal meanings before Phase 8 unifies them."""

    result = make_continuum_nodal_semantics_result()
    mesh = result.model.mesh
    geometry = build_model_geometry(result.model)

    recovery = stress.StressRecovery(mesh, result.U)
    recovered_nodal = recovery.collect(stress.StressPosition.NODAL)
    recovered_center = _records_for_node(recovered_nodal.records, 1)

    raw = stress.field.collect(mesh, result.U)
    resolved = stress.field.resolve(raw, threshold=threshold)
    resolved_center = _records_for_node(resolved.rows, 1)

    gui_data = build_result_data(result, geometry)
    rendered = build_stress_render_geometry(
        geometry,
        gui_data,
        "NODAL:S11",
        threshold=threshold,
    )
    gui_center = _rendered_rows_for_node(rendered, 1)

    # Current oracle: StressRecovery always averages each region and drops
    # element/local-node provenance.
    assert sorted(record.components[0] for record in recovered_center) == pytest.approx(
        [20.0, 50.0]
    )
    assert len({record.region_key for record in recovered_center}) == 2
    assert all(
        record.elem_id is None and record.local_node is None
        for record in recovered_center
    )

    # Known divergence: field.resolve sees more than one region at the node
    # and therefore preserves every raw contribution at every threshold.
    assert [record.components[0] for record in resolved_center] == pytest.approx(
        [10.0, 30.0, 50.0]
    )
    assert [
        (record.elem_id, record.local_node, record.averaged)
        for record in resolved_center
    ] == [(1, 1, False), (2, 1, False), (3, 1, False)]

    # Known divergence: the GUI resolves each (node, region) independently.
    # Its single-sample second region loses element provenance even at 0%.
    assert gui_center == pytest.approx(expected_gui_center)

    # Known divergence: the core resolver synthesizes an isolated zero row,
    # StressRecovery omits it, and the GUI scalar cache carries NaN while its
    # render topology omits the isolated point.
    assert _records_for_node(recovered_nodal.records, 8) == []
    isolated = _records_for_node(resolved.rows, 8)
    assert len(isolated) == 1
    assert isolated[0].components == (0.0, 0.0, 0.0)
    assert isolated[0].averaged
    assert isolated[0].elem_id is isolated[0].local_node is None
    isolated_index = geometry.node_id_to_point_index[8]
    center_index = geometry.node_id_to_point_index[1]
    assert np.isnan(gui_data.fields["NODAL:S11"].values[center_index])
    assert np.isnan(gui_data.fields["NODAL:S11"].values[isolated_index])
    assert 8 not in rendered.point_index_to_node_id.values()

    # Known divergence: a one-sample node retains exact provenance in
    # field.resolve, while StressRecovery and the GUI present a region value.
    recovered_single = _records_for_node(recovered_nodal.records, 2)
    resolved_single = _records_for_node(resolved.rows, 2)
    gui_single = _rendered_rows_for_node(rendered, 2)
    assert len(recovered_single) == 1
    assert recovered_single[0].components[0] == pytest.approx(10.0)
    assert recovered_single[0].elem_id is recovered_single[0].local_node is None
    assert [
        (record.components[0], record.elem_id, record.local_node, record.averaged)
        for record in resolved_single
    ] == [(10.0, 1, 2, False)]
    assert gui_single == pytest.approx([(10.0, None)])


def test_current_truss_gui_field_keys_order_and_numeric_oracle() -> None:
    result = make_truss_field_characterization_result()
    data = build_result_data(result, build_model_geometry(result.model))

    assert list(data.fields) == [
        "U1",
        "U2",
        "U3",
        "U",
        "RF1",
        "RF2",
        "RF3",
        "RF",
        "CENTROID:LE11",
        "CENTROID:S11",
        "CENTROID:Mises",
    ]
    assert data.fields["U1"].values == pytest.approx([0.0, 0.2])
    assert data.fields["RF"].values == pytest.approx(
        [np.sqrt(14.0), np.sqrt(77.0)]
    )
    assert data.fields["CENTROID:LE11"].values == pytest.approx([0.1])
    assert data.fields["CENTROID:S11"].values == pytest.approx([10.0])
    assert data.fields["CENTROID:Mises"].values == pytest.approx([10.0])
    assert data.element_stress == {
        30: {"LE11": 0.1, "S11": 10.0, "Mises": 10.0}
    }
    assert data.nodal_stress == {}


def test_current_beam_gui_field_keys_order_and_numeric_oracle() -> None:
    result = make_beam_field_characterization_result()
    data = build_result_data(result, build_model_geometry(result.model))

    # This exact insertion order is the current GUI contract. In particular,
    # rotations still use R1/R2/R3; Phase 8's target contract uses UR.
    assert list(data.fields) == [
        "U1",
        "U2",
        "U3",
        "U",
        "RF1",
        "RF2",
        "RF3",
        "RF",
        "R1",
        "RM1",
        "R2",
        "RM2",
        "R3",
        "RM3",
        "NODAL:S11Max",
        "NODAL:S11Min",
        "NODAL:S11AbsMax",
    ]
    assert data.fields["U1"].values == pytest.approx([0.0, 0.2])
    assert data.fields["U2"].values == pytest.approx([0.0, 0.02])
    assert data.fields["U"].values == pytest.approx([0.0, np.sqrt(0.0404)])
    assert data.fields["R1"].values == pytest.approx([0.0, 0.0])
    assert data.fields["R2"].values == pytest.approx([0.0, 0.0])
    assert data.fields["R3"].values == pytest.approx([0.0, 0.02])
    assert data.fields["RF1"].values == pytest.approx([1.0, 7.0])
    assert data.fields["RF2"].values == pytest.approx([2.0, 8.0])
    assert data.fields["RF3"].values == pytest.approx([3.0, 9.0])
    assert data.fields["RF"].values == pytest.approx(
        [np.sqrt(14.0), np.sqrt(194.0)]
    )
    assert data.fields["RM1"].values == pytest.approx([4.0, 10.0])
    assert data.fields["RM2"].values == pytest.approx([5.0, 11.0])
    assert data.fields["RM3"].values == pytest.approx([6.0, 12.0])

    assert data.nodal_stress == {
        10: {"S11Max": 11.0, "S11Min": 9.0, "S11AbsMax": 11.0},
        20: {"S11Max": 11.0, "S11Min": 9.0, "S11AbsMax": 11.0},
    }
    assert data.fields["NODAL:S11Max"].values == pytest.approx([11.0, 11.0])
    assert data.fields["NODAL:S11Min"].values == pytest.approx([9.0, 9.0])
    assert data.fields["NODAL:S11AbsMax"].values == pytest.approx([11.0, 11.0])
    assert data.stress_position_label("NODAL") == "节点包络"
