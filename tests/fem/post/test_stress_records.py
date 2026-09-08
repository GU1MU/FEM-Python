
import numpy as np
import pytest

from fem.core.mesh import (
    Element2D,
    Mesh2D,
    Node2D,
)
from fem.elements import get_element_kernel
from fem.post import (
    stress,
)
from tests.helpers.mesh_builders import (
    make_mixed_tri3_quad4_mesh,
)


def _stress_contribution(node_id, elem_id, components, region=None, weight=1.0):
    if region is None:
        region = stress.field.StressRegionKey(("material", "steel"), ("solid",))
    return stress.field.ElementNodalStressContribution(
        node_id=node_id,
        elem_id=elem_id,
        local_node=1,
        components=tuple(components),
        weight=weight,
        region_key=region,
    )


def test_nodal_stress_field_collects_ordered_element_contributions():
    mesh = make_mixed_tri3_quad4_mesh()

    raw = stress.field.collect(mesh, np.zeros(mesh.num_dofs))

    assert raw.component_names == ("sig_x", "sig_y", "tau_xy")
    assert tuple(raw.contributions_by_node) == tuple(mesh.node_ids)
    assert [
        (item.elem_id, item.local_node, item.weight)
        for item in raw.contributions_by_node[2]
    ] == [(1, 2, 1.0), (2, 1, 1.0)]
    assert raw.contributions_by_node[2][0].plane_type == "stress"
    assert raw.contributions_by_node[2][0].poisson_ratio == pytest.approx(0.25)
    assert raw.contributions_by_node[5][0].components == pytest.approx((0.0, 0.0, 0.0))


def test_canonical_stress_positions_include_plane_strain_s33():
    props = {
        "E": 100.0,
        "nu": 0.25,
        "plane_type": "strain",
        "thickness": 1.0,
    }
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[Element2D(1, [1, 2, 3], "Tri3", props)],
    )
    U = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.02])
    recovery = stress.StressRecovery(mesh, U)
    assert stress.collect_stress(
        mesh,
        U,
        position="integration_point",
    ) == recovery.collect(stress.StressPosition.INTEGRATION_POINT)

    expected_counts = {
        stress.StressPosition.INTEGRATION_POINT: 1,
        stress.StressPosition.CENTROID: 1,
        stress.StressPosition.ELEMENT_NODAL: 3,
        stress.StressPosition.NODAL: 3,
    }
    for position, expected_count in expected_counts.items():
        stress_field = recovery.collect(position)
        assert stress_field.component_names == ("S11", "S22", "S33", "S12")
        assert len(stress_field.records) == expected_count
        for record in stress_field.records:
            assert record.components == pytest.approx((2.0, 2.8, 1.2, 0.0))
            assert record.invariants.mises == pytest.approx(np.sqrt(1.92))
            assert (
                record.invariants.max_principal,
                record.invariants.mid_principal,
                record.invariants.min_principal,
            ) == pytest.approx((2.8, 2.0, 1.2))


def test_nodal_mises_is_derived_after_tensor_component_averaging():
    props = {
        "E": 100.0,
        "nu": 0.25,
        "plane_type": "stress",
        "thickness": 1.0,
    }
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, 1.0, 1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3", dict(props)),
            Element2D(2, [2, 4, 3], "Tri3", dict(props)),
        ],
    )
    U = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    recovery = stress.StressRecovery(mesh, U)
    element_nodal = recovery.collect(stress.StressPosition.ELEMENT_NODAL)
    nodal = recovery.collect(stress.StressPosition.NODAL)

    contributions = [
        record for record in element_nodal.records if record.node_id == 2
    ]
    averaged = next(record for record in nodal.records if record.node_id == 2)
    expected_components = np.mean(
        [record.components for record in contributions],
        axis=0,
    )
    s11, s22, _s33, s12 = expected_components
    expected_mises = np.sqrt(s11**2 - s11*s22 + s22**2 + 3.0*s12**2)

    assert averaged.components == pytest.approx(expected_components)
    assert averaged.invariants.mises == pytest.approx(expected_mises)
    assert averaged.invariants.mises != pytest.approx(
        np.mean([record.invariants.mises for record in contributions])
    )


def test_quad4_centroid_uses_integration_point_interpolation_not_direct_b_evaluation():
    props = {
        "E": 100.0,
        "nu": 0.25,
        "plane_type": "stress",
        "thickness": 1.0,
    }
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 1.7, 1.2),
            Node2D(4, -0.2, 0.8),
        ],
        elements=[Element2D(1, [1, 2, 3, 4], "Quad4", props)],
    )
    U = np.array([0.0, 0.0, 0.1, 0.02, 0.17, 0.14, -0.04, 0.08])
    recovery = stress.StressRecovery(mesh, U)
    integration_points = recovery.collect(stress.StressPosition.INTEGRATION_POINT)
    centroid = recovery.collect(stress.StressPosition.CENTROID).records[0]

    expected = np.mean(
        [record.components for record in integration_points.records],
        axis=0,
    )
    direct = get_element_kernel("Quad4").stress_at(
        mesh,
        mesh.elements[0],
        U,
        0.0,
        0.0,
    )

    assert centroid.components == pytest.approx(expected)
    assert centroid.components[:2] != pytest.approx(direct[:2])


def test_nodal_stress_resolution_uses_inclusive_component_threshold():
    contributions = {
        1: (
            _stress_contribution(1, 1, (10.0, 0.0, 0.0)),
            _stress_contribution(1, 2, (20.0, 0.0, 0.0)),
        ),
        2: (_stress_contribution(2, 1, (0.0, 0.0, 0.0)),),
        3: (_stress_contribution(3, 2, (40.0, 0.0, 0.0)),),
    }
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node=contributions,
        node_ids=(1, 2, 3),
    )

    exact = stress.field.resolve(raw, threshold=25.0)
    below = stress.field.resolve(raw, threshold=24.9)

    exact_rows = [row for row in exact.rows if row.node_id == 1]
    below_rows = [row for row in below.rows if row.node_id == 1]
    assert len(exact_rows) == 1
    assert exact_rows[0].components == pytest.approx((15.0, 0.0, 0.0))
    assert exact_rows[0].elem_id is None
    assert exact_rows[0].local_node is None
    assert exact_rows[0].averaged is True
    assert [row.components for row in below_rows] == [
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
    ]


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 100.1, np.nan, "75", True],
)
def test_nodal_stress_resolution_rejects_invalid_thresholds(threshold):
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node={1: ()},
        node_ids=(1,),
    )

    with pytest.raises(ValueError, match="threshold"):
        stress.field.resolve(raw, threshold=threshold)


def test_nodal_stress_resolution_handles_zero_and_full_thresholds():
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node={
            1: (
                _stress_contribution(1, 1, (5.0, 2.0, 1.0)),
                _stress_contribution(1, 2, (5.0, 2.0, 1.0)),
            ),
            2: (_stress_contribution(2, 1, (0.0, 0.0, 0.0)),),
            3: (_stress_contribution(3, 2, (10.0, 4.0, 2.0)),),
        },
        node_ids=(1, 2, 3),
    )

    zero_rows = [row for row in stress.field.resolve(raw, 0.0).rows if row.node_id == 1]
    full_rows = [row for row in stress.field.resolve(raw, 100.0).rows if row.node_id == 1]

    assert len(zero_rows) == 2
    assert all(not row.averaged for row in zero_rows)
    assert len(full_rows) == 1
    assert full_rows[0].averaged


@pytest.mark.parametrize(
    "second_region",
    [
        stress.field.StressRegionKey(("material", "aluminum"), ("solid",)),
        stress.field.StressRegionKey(("material", "steel"), ("shell",)),
    ],
)
def test_nodal_stress_resolution_preserves_region_boundaries(second_region):
    first_region = stress.field.StressRegionKey(("material", "steel"), ("solid",))
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node={
            1: (
                _stress_contribution(1, 1, (1.0, 2.0, 3.0), first_region),
                _stress_contribution(1, 2, (1.0, 2.0, 3.0), second_region),
            )
        },
        node_ids=(1,),
    )

    rows = stress.field.resolve(raw, 100.0).rows

    assert [(row.elem_id, row.averaged) for row in rows] == [(1, False), (2, False)]


def test_nodal_stress_resolution_requires_every_component_to_pass():
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node={
            1: (
                _stress_contribution(1, 1, (10.0, 0.0, 0.0)),
                _stress_contribution(1, 2, (20.0, 10.0, 0.0)),
            ),
            2: (_stress_contribution(2, 1, (0.0, 0.0, 0.0)),),
            3: (_stress_contribution(3, 2, (100.0, 10.0, 0.0)),),
        },
        node_ids=(1, 2, 3),
    )

    rows = [row for row in stress.field.resolve(raw, 20.0).rows if row.node_id == 1]

    assert [row.elem_id for row in rows] == [1, 2]


def test_nodal_stress_resolution_uses_weights_and_emits_unconnected_zero():
    raw = stress.field.NodalStressField(
        component_names=("sig_x", "sig_y", "tau_xy"),
        contributions_by_node={
            1: (
                _stress_contribution(1, 1, (10.0, 0.0, 0.0), weight=1.0),
                _stress_contribution(1, 2, (20.0, 0.0, 0.0), weight=3.0),
            ),
            2: (),
        },
        node_ids=(1, 2),
    )

    rows = stress.field.resolve(raw, 100.0).rows

    assert rows[0].components == pytest.approx((17.5, 0.0, 0.0))
    assert rows[1].components == (0.0, 0.0, 0.0)
    assert rows[1].elem_id is None
    assert rows[1].local_node is None
    assert rows[1].averaged is True
