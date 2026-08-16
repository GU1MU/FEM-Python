import csv

import numpy as np
import pytest

from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.elements import get_element_kernel
from fem.post import displacement, path, stress, vtk
from fem.post.stress import dispatch
from fem.post.polar import convert_nodal_solution_into_polar_coord
from fem.post.vtk.polar import convert_nodal_displacement
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_mixed_hex20_tet10_mesh,
    make_mixed_hex8_hex20_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tet4_tet10_mesh,
    make_mixed_tri3_quad4_mesh,
    make_mixed_tri6_quad8_mesh,
    make_tri6_stiffness_mesh,
    make_unit_hex8_mesh,
)
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.result_builders import make_zero_result


def _affine_solid_displacement(mesh):
    U = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = 0.01 * node.x + 0.02 * node.y + 0.03 * node.z
        U[mesh.global_dof(node.id, 1)] = -0.02 * node.x + 0.04 * node.y + 0.01 * node.z
        U[mesh.global_dof(node.id, 2)] = 0.03 * node.x - 0.01 * node.y + 0.05 * node.z
    return U


def _make_beam_dispatch_mesh():
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2")],
        dofs_per_node=6,
    )


def _write_current_element_stress(
    mesh,
    displacement,
    path,
    element_type=None,
    gauss_order=None,
):
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        stress.element.by_type(
            type_keys[0],
            mesh,
            displacement,
            path,
            gauss_order,
        )
        return
    stress.element.mixed(
        type_keys,
        mesh,
        displacement,
        path,
        gauss_order,
    )


def _write_current_nodal_stress(
    mesh,
    displacement,
    path,
    element_type=None,
    gauss_order=None,
    threshold=75.0,
):
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        stress.nodal.by_type(
            type_keys[0],
            mesh,
            displacement,
            path,
            gauss_order,
            threshold,
        )
        return
    stress.nodal.mixed(
        type_keys,
        mesh,
        displacement,
        path,
        gauss_order,
        threshold,
    )


def test_nodal_stress_field_collects_ordered_element_contributions():
    mesh = make_mixed_tri3_quad4_mesh()

    assert hasattr(stress, "field")
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


def test_recovered_plane_stress_writers_match_legacy_exports(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    mesh.elements[0].props["plane_type"] = "strain"
    mesh.elements[1].props = dict(mesh.elements[0].props)
    displacement = np.asarray([
        0.0,
        0.001,
        0.01,
        0.002,
        0.013,
        0.018,
        -0.002,
        0.016,
        0.021,
        -0.004,
    ])
    recovered = stress.StressRecovery(mesh, displacement).collect(
        stress.StressPosition.ELEMENT_NODAL
    )
    csv_recovered = stress.collect_plane_element_nodal(mesh, displacement)

    expected_element = tmp_path / "expected_element.csv"
    actual_element = tmp_path / "actual_element.csv"
    stress.element.mixed(
        ("tri3", "quad4"),
        mesh,
        displacement,
        expected_element,
    )
    stress.element.write_plane_element_nodal(
        mesh,
        recovered,
        actual_element,
    )

    expected_nodal = tmp_path / "expected_nodal.csv"
    actual_nodal = tmp_path / "actual_nodal.csv"
    legacy_raw = stress.nodal_from_stress_field(mesh, recovered)
    legacy_resolved = stress.field.resolve(legacy_raw, threshold=100.0)
    stress.nodal._write_resolved(
        mesh,
        legacy_resolved,
        expected_nodal,
    )
    stress.nodal.write_recovered(
        mesh,
        csv_recovered,
        actual_nodal,
        threshold=100.0,
    )

    expected_canonical = tmp_path / "expected_canonical.csv"
    actual_canonical = tmp_path / "actual_canonical.csv"
    stress.export.csv(
        mesh,
        displacement,
        expected_canonical,
        position=stress.StressPosition.ELEMENT_NODAL,
    )
    stress.export.write_csv(recovered, actual_canonical)

    assert actual_element.read_bytes() == expected_element.read_bytes()
    assert actual_nodal.read_bytes() == expected_nodal.read_bytes()
    assert actual_canonical.read_bytes() == expected_canonical.read_bytes()


def test_recovered_solid_nodal_writer_matches_legacy_resolution(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[1].props = dict(mesh.elements[0].props)
    displacement = np.arange(mesh.num_dofs, dtype=float) ** 2 * 0.001
    recovered = stress.StressRecovery(mesh, displacement).collect(
        stress.StressPosition.ELEMENT_NODAL
    )
    expected = tmp_path / "expected_solid_nodal.csv"
    actual = tmp_path / "actual_solid_nodal.csv"

    legacy_raw = stress.nodal_from_stress_field(mesh, recovered)
    legacy_resolved = stress.field.resolve(legacy_raw, threshold=100.0)
    stress.nodal._write_resolved(mesh, legacy_resolved, expected)
    stress.nodal.write_recovered(
        mesh,
        recovered,
        actual,
        threshold=100.0,
    )

    assert actual.read_bytes() == expected.read_bytes()


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
            s11, s22, s33, _s12 = record.components
            assert s33 == pytest.approx(0.25 * (s11 + s22))
            expected = stress.invariants.derive_stress_invariants(
                record.components,
                stress_field.component_names,
            )
            assert record.invariants == expected


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
    expected_invariants = stress.invariants.derive_stress_invariants(
        expected_components,
        nodal.component_names,
    )

    assert averaged.components == pytest.approx(expected_components)
    assert averaged.invariants.mises == pytest.approx(expected_invariants.mises)
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


def test_canonical_csv_defaults_to_integration_points_and_writes_s33(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    output = tmp_path / "stress.csv"

    stress.export.csv(mesh, np.zeros(mesh.num_dofs), output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["position"] for row in rows} == {"integration_point"}
    assert all(row["integration_point"] for row in rows)
    assert all(row["S33"] for row in rows)


def test_deprecated_stress_export_wrappers_emit_explicit_warnings(tmp_path):
    mesh = make_unit_hex8_mesh()
    displacement = np.zeros(mesh.num_dofs)

    with pytest.warns(
        DeprecationWarning,
        match=r"stress\.export\.element\(\) is deprecated",
    ):
        stress.export.element(
            mesh,
            displacement,
            tmp_path / "compat-element.csv",
        )
    with pytest.warns(
        DeprecationWarning,
        match=r"stress\.export\.nodal\(\) is deprecated",
    ):
        stress.export.nodal(
            mesh,
            displacement,
            tmp_path / "compat-nodal.csv",
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

    assert hasattr(stress.field, "resolve")
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
    [-0.1, 100.1, np.nan, np.inf, -np.inf, "75", None, True],
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


def test_nodal_stress_csv_preserves_material_boundary_contributions(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(
                1,
                [1, 2, 3],
                "Tri3",
                {"E": 100.0, "nu": 0.25, "plane_type": "stress", "thickness": 1.0},
            ),
            Element2D(
                2,
                [3, 4, 5],
                "Tri3",
                {"E": 200.0, "nu": 0.3, "plane_type": "stress", "thickness": 1.0},
            ),
        ],
    )
    csv_path = tmp_path / "material_boundary.csv"

    _write_current_nodal_stress(
        mesh,
        np.zeros(mesh.num_dofs),
        csv_path,
        threshold=100.0,
    )

    with csv_path.open("r", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["node_id", "x", "y", "elem_id", "local_node", "averaged", "sig_x", "sig_y", "tau_xy", "mises"],
            ["1", "0.0", "0.0", "1", "1", "false", "0.0", "0.0", "0.0", "0.0"],
            ["2", "1.0", "0.0", "1", "2", "false", "0.0", "0.0", "0.0", "0.0"],
            ["3", "0.0", "1.0", "1", "3", "false", "0.0", "0.0", "0.0", "0.0"],
            ["3", "0.0", "1.0", "2", "1", "false", "0.0", "0.0", "0.0", "0.0"],
            ["4", "-1.0", "0.0", "2", "2", "false", "0.0", "0.0", "0.0", "0.0"],
            ["5", "0.0", "-1.0", "2", "3", "false", "0.0", "0.0", "0.0", "0.0"],
        ]


def test_vtk_duplicates_points_for_unaveraged_nodal_stress_rows(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [3, 4, 5], "Tri3"),
        ],
    )
    displacement_path = tmp_path / "displacement.csv"
    nodal_stress_path = tmp_path / "nodal_stress.csv"
    vtk_path = tmp_path / "split.vtk"
    displacement_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,0,0\n2,1,0,0,0\n3,0,1,3,4\n4,-1,0,0,0\n5,0,-1,0,0\n",
        encoding="utf-8",
    )
    nodal_stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy,mises\n"
        "1,0,0,1,1,false,0,0,0,0\n"
        "2,1,0,1,2,false,0,0,0,0\n"
        "3,0,1,1,3,false,10,0,0,10\n"
        "3,0,1,2,1,false,20,0,0,20\n"
        "4,-1,0,2,2,false,0,0,0,0\n"
        "5,0,-1,2,3,false,0,0,0,0\n",
        encoding="utf-8",
    )

    vtk.export.from_csv(
        mesh,
        displacement_path,
        None,
        vtk_path,
        nodal_stress_path,
    )

    lines = vtk_path.read_text(encoding="utf-8").splitlines()
    cells_index = lines.index("CELLS 2 8")
    displacement_index = lines.index("VECTORS displacement float")
    sig_x_index = lines.index("SCALARS sig_x float 1")
    assert "POINTS 6 float" in lines
    assert "POINT_DATA 6" in lines
    assert lines[cells_index + 1 : cells_index + 3] == ["3 0 1 2", "3 3 4 5"]
    assert lines[displacement_index + 1 : displacement_index + 7] == [
        "0.0 0.0 0.0",
        "0.0 0.0 0.0",
        "3.0 4.0 0.0",
        "3.0 4.0 0.0",
        "0.0 0.0 0.0",
        "0.0 0.0 0.0",
    ]
    assert lines[sig_x_index + 2 : sig_x_index + 8] == [
        "0.0", "0.0", "10.0", "20.0", "0.0", "0.0"
    ]


def test_vtk_from_result_uses_threshold_for_csv_and_topology(tmp_path):
    props = {"E": 100.0, "nu": 0.25, "plane_type": "stress", "thickness": 1.0}
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3", dict(props)),
            Element2D(2, [3, 4, 5], "Tri3", dict(props)),
        ],
    )

    vtk.export.from_result(
        make_zero_result(mesh, "threshold_zero"),
        output_dir=tmp_path,
        threshold=0.0,
    )

    stress_path = tmp_path / "threshold_zero_nodal_stress.csv"
    with stress_path.open("r", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    shared_rows = [row for row in csv_rows if row["node_id"] == "3"]
    vtk_lines = (tmp_path / "threshold_zero.vtk").read_text(encoding="utf-8").splitlines()
    assert [(row["elem_id"], row["averaged"]) for row in shared_rows] == [
        ("1", "false"),
        ("2", "false"),
    ]
    assert "POINTS 6 float" in vtk_lines


@pytest.mark.parametrize(
    ("header", "row", "message"),
    (
        ("x,y,elem_id,local_node,averaged,sig_x", "0,0,,,true,10", "node_id"),
        ("node_id,x,y,local_node,averaged,sig_x", "1,0,0,,true,10", "elem_id"),
        ("node_id,x,y,elem_id,averaged,sig_x", "1,0,0,,true,10", "local_node"),
        ("node_id,x,y,elem_id,local_node,sig_x", "1,0,0,,,10", "averaged"),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "bad,0,0,,,true,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,bad,1,false,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,bad,false,10",
            "expected an integer",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,1,maybe,10",
            "expected true or false",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,,1,false,10",
            "missing elem_id",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,,false,10",
            "missing local_node",
        ),
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x",
            "1,0,0,1,0,false,10",
            "one-based",
        ),
    ),
)
def test_nodal_stress_reader_enforces_current_metadata_contract(
    tmp_path,
    header,
    row,
    message,
):
    stress_path = tmp_path / "nodal_stress.csv"
    stress_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        vtk.fields.read_nodal_stress_rows(stress_path)


def test_nodal_stress_reader_accepts_averaged_rows_without_provenance(tmp_path):
    stress_path = tmp_path / "averaged_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )

    data = vtk.fields.read_nodal_stress_rows(stress_path)

    assert [(row.node_id, row.elem_id, row.local_node, row.averaged) for row in data.rows] == [
        (1, None, None, True),
        (2, None, None, True),
    ]
    assert [row.values["sig_x"] for row in data.rows] == [10.0, 20.0]


@pytest.mark.parametrize(
    "consumer",
    ("vtk", "path", "polar"),
)
def test_nodal_stress_consumers_reject_malformed_current_rows(tmp_path, consumer):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "invalid_current_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,0,0,1,1,not-bool,10,2,3\n",
        encoding="utf-8",
    )

    def call():
        if consumer == "vtk":
            return vtk.fields.read_nodal_stress_rows(stress_path)
        if consumer == "path":
            return path.extract_path_data(
                mesh,
                1,
                2,
                2,
                "sig_x",
                path=tmp_path / "path.csv",
                stress_csv_path=stress_path,
            )
        return convert_nodal_solution_into_polar_coord(
            stress_path,
            (0.0, 0.0),
            tmp_path / "polar.csv",
        )

    with pytest.raises(ValueError, match="averaged"):
        call()


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (
            vtk.fields.NodalStressCsvRow(1, 1, None, False, {"sig_x": 10.0}),
            r"node 1.*requires.*provenance.*missing local_node",
        ),
        (
            vtk.fields.NodalStressCsvRow(1, 1, 2, False, {"sig_x": 10.0}),
            r"node 1.*element 1.*local node 2.*connectivity",
        ),
    ),
)
def test_vtk_validates_single_nonaveraged_row_provenance(row, message):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )

    with pytest.raises(ValueError, match=message):
        vtk.cells.build_result(mesh, (row,))


def test_vtk_uses_shared_zero_fallback_for_incident_element_without_raw_row():
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [3, 4, 5], "Tri3"),
        ],
    )
    row = vtk.fields.NodalStressCsvRow(3, 1, 3, False, {"sig_x": 10.0})

    topology = vtk.cells.build_result(mesh, (row,))
    point_fields = vtk.fields.point_fields(
        vtk.fields.NodalStressCsv(("sig_x",), (row,)),
        topology.point_rows,
    )

    selected_point = topology.cells[0][3]
    fallback_point = topology.cells[1][1]
    assert len(topology.cells) == 2
    assert selected_point != fallback_point
    assert topology.point_node_ids[selected_point] == 3
    assert topology.point_node_ids[fallback_point] == 3
    assert topology.point_rows[selected_point] is row
    assert topology.point_rows[fallback_point] is None
    assert point_fields["sig_x"][selected_point] == pytest.approx(10.0)
    assert point_fields["sig_x"][fallback_point] == pytest.approx(0.0)


def test_vtk_uses_single_boundary_raw_row_for_its_element_local_point():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )
    row = vtk.fields.NodalStressCsvRow(2, 1, 2, False, {"sig_x": 10.0})

    topology = vtk.cells.build_result(mesh, (row,))

    point_index = topology.cells[0][2]
    assert topology.point_node_ids[point_index] == 2
    assert topology.point_rows[point_index] is row


def test_vtk_rejects_repeated_nodal_stress_rows_without_provenance():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3")],
    )
    rows = (
        vtk.fields.NodalStressCsvRow(1, 1, 1, False, {"sig_x": 10.0}),
        vtk.fields.NodalStressCsvRow(1, None, None, False, {"sig_x": 20.0}),
    )
    with pytest.raises(ValueError, match="requires.*provenance"):
        vtk.cells.build_result(mesh, rows)


def test_polar_conversion_preserves_distinct_repeated_nodal_rows():
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 1.0)],
        elements=[],
    )
    data = vtk.fields.NodalStressCsv(
        ("sig_x", "sig_y", "tau_xy", "mises"),
        (
            vtk.fields.NodalStressCsvRow(
                1, 1, 1, False, {"sig_x": 10.0, "sig_y": 0.0, "tau_xy": 0.0, "mises": 10.0}
            ),
            vtk.fields.NodalStressCsvRow(
                1, 2, 1, False, {"sig_x": 20.0, "sig_y": 0.0, "tau_xy": 0.0, "mises": 20.0}
            ),
        ),
    )

    converted = vtk.polar.convert_nodal_stress_rows(mesh, data, (0.0, 0.0))

    assert [row.values["sig_t"] for row in converted.rows] == pytest.approx([10.0, 20.0])
    assert [(row.elem_id, row.local_node) for row in converted.rows] == [(1, 1), (2, 1)]


def test_polar_csv_conversion_reports_non_numeric_center_with_context(tmp_path):
    center = ("bad-radius", 0.0)

    with pytest.raises(ValueError) as exc_info:
        convert_nodal_solution_into_polar_coord(
            tmp_path / "unused.csv",
            center,
            tmp_path / "unused_polar.csv",
        )

    message = str(exc_info.value)
    assert repr(center) in message
    assert "2 numeric values" in message
    assert "expected numeric x/y" in message


def test_polar_csv_conversion_reports_missing_numeric_value_with_context(tmp_path):
    csv_path = tmp_path / "incomplete_displacement.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        convert_nodal_solution_into_polar_coord(
            csv_path,
            (0.0, 0.0),
            tmp_path / "incomplete_polar.csv",
        )

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field uy" in message
    assert "raw value None" in message
    assert "expected a numeric value" in message


def test_path_stress_entrypoint_accepts_current_metadata(tmp_path):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "current_nodal_stress.csv"
    out_path = tmp_path / "nodes.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )

    path.extract_nodes_data(
        mesh,
        [1, 2],
        ["sig_x"],
        path=out_path,
        stress_csv_path=stress_path,
    )

    with out_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["sig_x"]) for row in rows] == [10.0, 20.0]


@pytest.mark.parametrize("target", ("elem_id", "local_node", "averaged"))
def test_path_stress_entrypoints_reject_metadata_targets(tmp_path, target):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "current_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )
    calls = (
        lambda: path.extract_path_data(
            mesh,
            1,
            2,
            2,
            target,
            path=tmp_path / "path.csv",
            stress_csv_path=stress_path,
        ),
        lambda: path.extract_nodes_data(
            mesh,
            [1],
            [target],
            path=tmp_path / "nodes.csv",
            stress_csv_path=stress_path,
        ),
    )

    for call in calls:
        with pytest.raises(ValueError, match=rf"target {target} not found"):
            call()


def test_path_stress_reader_rejects_duplicate_node_rows(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    stress_path = tmp_path / "duplicate_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,1,1,false,10\n"
        "1,0,0,2,1,false,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        path.extract_nodes_data(
            mesh,
            [1],
            ["sig_x"],
            path=tmp_path / "nodes.csv",
            stress_csv_path=stress_path,
        )

    message = str(exc_info.value)
    assert str(stress_path) in message
    assert "node_id 1" in message
    assert "multiple element-nodal contributions" in message
    assert "explicit selection or averaging" in message


def test_path_general_reader_keeps_last_duplicate_row_and_invalid_scalar_zero(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    disp_path = tmp_path / "duplicate_nodal_displacement.csv"
    out_path = tmp_path / "nodes.csv"
    disp_path.write_text(
        "node_id,x,y,ux,uy\n"
        "1,0,0,10,0\n"
        "1,0,0,invalid,0\n",
        encoding="utf-8",
    )

    path.extract_nodes_data(
        mesh,
        [1],
        ["ux"],
        path=out_path,
        disp_csv_path=disp_path,
    )

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["ux"]) == 0.0


def test_path_nodal_reader_reports_invalid_node_id_with_context(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    csv_path = tmp_path / "invalid_node_id.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\nbad,0,0,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        path.extract_nodes_data(
            mesh,
            [1],
            ["ux"],
            path=tmp_path / "nodes.csv",
            disp_csv_path=csv_path,
        )

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field node_id" in message
    assert "raw value 'bad'" in message
    assert "expected an integer" in message


def test_polar_csv_conversion_accepts_current_stress_metadata(tmp_path):
    csv_path = tmp_path / "current_nodal_stress.csv"
    out_path = tmp_path / "current_polar.csv"
    csv_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,1,1,false,10,2,3\n",
        encoding="utf-8",
    )

    convert_nodal_solution_into_polar_coord(csv_path, (0.0, 0.0), out_path)

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["sig_r"]) == pytest.approx(10.0)
    assert float(row["sig_t"]) == pytest.approx(2.0)
    assert float(row["tau_rt"]) == pytest.approx(3.0)


def test_averaged_whitespace_provenance_is_accepted_across_stress_entries(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 1.0, 0.0)], elements=[])
    stress_path = tmp_path / "averaged_whitespace_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,   ,   , TRUE ,10,2,3\n"
        "1,1,0,   ,   , TRUE ,20,4,6\n"
        "1,1,0, 1 , 2 ,false,30,6,9\n",
        encoding="utf-8",
    )

    vtk_data = vtk.fields.read_nodal_stress_rows(stress_path)
    assert [(row.elem_id, row.local_node, row.averaged) for row in vtk_data.rows] == [
        (None, None, True),
        (None, None, True),
        (1, 2, False),
    ]

    path_stress_path = tmp_path / "path_averaged_whitespace_nodal_stress.csv"
    path_stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x,sig_y,tau_xy\n"
        "1,1,0,   ,   , TRUE ,10,2,3\n",
        encoding="utf-8",
    )
    path_out = tmp_path / "nodes.csv"
    path.extract_nodes_data(
        mesh,
        [1],
        ["sig_x"],
        path=path_out,
        stress_csv_path=path_stress_path,
    )
    with path_out.open("r", encoding="utf-8") as stream:
        path_row = next(csv.DictReader(stream))
    assert float(path_row["sig_x"]) == pytest.approx(10.0)

    polar_out = tmp_path / "polar.csv"
    convert_nodal_solution_into_polar_coord(
        stress_path,
        (0.0, 0.0),
        polar_out,
    )
    with polar_out.open("r", encoding="utf-8") as stream:
        polar_rows = list(csv.DictReader(stream))
    assert [float(row["sig_r"]) for row in polar_rows] == pytest.approx(
        [10.0, 20.0, 30.0]
    )
    assert [float(row["sig_t"]) for row in polar_rows] == pytest.approx([2.0, 4.0, 6.0])
    assert [float(row["tau_rt"]) for row in polar_rows] == pytest.approx([3.0, 6.0, 9.0])


def test_polar_csv_conversion_keeps_displacement_schema_unchanged(tmp_path):
    csv_path = tmp_path / "displacement.csv"
    out_path = tmp_path / "polar_displacement.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\n1,0,1,2,0\n",
        encoding="utf-8",
    )

    convert_nodal_solution_into_polar_coord(csv_path, (0.0, 0.0), out_path)

    with out_path.open("r", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["ur"]) == pytest.approx(0.0)
    assert float(row["ut"]) == pytest.approx(-2.0)


def test_isolated_node_export_uses_averaged_zero_row_and_reaches_vtk(tmp_path):
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, 2.0, 2.0),
        ],
        elements=[
            Element2D(
                1,
                [1, 2, 3],
                "Tri3",
                {"E": 100.0, "nu": 0.25, "thickness": 1.0, "plane_type": "stress"},
            )
        ],
    )

    vtk.export.from_result(
        make_zero_result(mesh, "isolated_node"),
        output_dir=tmp_path,
    )

    data = vtk.fields.read_nodal_stress_rows(
        tmp_path / "isolated_node_nodal_stress.csv"
    )
    isolated_row = next(row for row in data.rows if row.node_id == 4)
    vtk_text = (tmp_path / "isolated_node.vtk").read_text(encoding="utf-8")
    assert isolated_row.elem_id is None
    assert isolated_row.local_node is None
    assert isolated_row.averaged is True
    assert set(isolated_row.values.values()) == {0.0}
    assert "POINTS 4 float" in vtk_text


def test_polar_displacement_fills_mesh_nodes_and_ignores_unknown_ids():
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 1.0, 0.0),
            Node2D(2, 0.0, 1.0),
            Node2D(3, -1.0, 0.0),
            Node2D(4, 0.0, 0.0),
        ],
        elements=[],
    )

    polar_values = convert_nodal_displacement(
        mesh,
        {
            1: {"ux": 2.0, "uy": 0.0, "rz": 0.5},
            2: {"ux": 0.0, "uy": 3.0, "rz": 0.0},
            4: {"ux": 4.0},
            999: {"ux": 9.0, "uy": 9.0, "rz": 9.0},
        },
        [0.0, 0.0],
    )

    assert set(polar_values) == {1, 2, 3, 4}
    assert polar_values[1] == {"ux": 2.0, "uy": 0.0, "rz": 0.5}
    assert polar_values[2] == {"ux": 3.0, "uy": 0.0, "rz": 0.0}
    assert polar_values[3] == {"ux": 0.0, "uy": 0.0, "rz": 0.0}
    assert polar_values[4] == {"ux": 4.0, "uy": 0.0, "rz": 0.0}


def test_stress_export_infers_single_element_type_from_mesh(tmp_path):
    mesh = make_unit_hex8_mesh()
    elem_path = tmp_path / "test_post_stress_element.csv"
    nodal_path = tmp_path / "test_post_stress_nodal.csv"

    _write_current_element_stress(mesh, np.zeros(mesh.num_dofs), elem_path)
    _write_current_nodal_stress(mesh, np.zeros(mesh.num_dofs), nodal_path)

    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == 2
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == 9


def test_explicit_hex8_nodal_stress_subset_exports_mixed_mesh_to_vtk(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    U = _affine_solid_displacement(mesh)
    disp_path = tmp_path / "mixed_subset_nodal_displacement.csv"
    stress_path = tmp_path / "mixed_subset_nodal_stress.csv"
    vtk_path = tmp_path / "mixed_subset.vtk"

    displacement.export.nodal(mesh, U, disp_path)
    _write_current_nodal_stress(
        mesh,
        U,
        stress_path,
        element_type="hex8",
    )

    nodal_data = vtk.fields.read_nodal_stress_rows(stress_path)
    topology = vtk.cells.build_result(mesh, nodal_data.rows)
    point_fields = vtk.fields.point_fields(nodal_data, topology.point_rows)
    hex8_point = topology.cells[0][2]
    tet4_fallback_point = topology.cells[1][1]

    assert len(topology.cells) == 2
    assert topology.point_node_ids[hex8_point] == 2
    assert topology.point_node_ids[tet4_fallback_point] == 2
    assert topology.point_rows[hex8_point].elem_id == 1
    assert topology.point_rows[hex8_point].local_node == 2
    assert topology.point_rows[tet4_fallback_point] is None
    assert abs(point_fields["sig_x"][hex8_point]) > 0.0
    assert point_fields["sig_x"][tet4_fallback_point] == pytest.approx(0.0)

    vtk.export.from_csv(
        mesh,
        disp_path,
        None,
        vtk_path,
        stress_path,
    )

    vtk_text = vtk_path.read_text(encoding="utf-8")
    assert "\nCELLS 2 " in vtk_text
    assert "\nCELL_TYPES 2\n" in vtk_text


def test_dispatch_rejects_reduced_integration_hex20():
    element_type = "C3D20R"
    mesh = make_hex20_stiffness_mesh()
    mesh.elements[0].type = element_type

    assert dispatch.type_key_from_name(element_type) is None
    with pytest.raises(
        ValueError,
        match=rf"Unsupported stress element type: '{element_type}'",
    ):
        dispatch.resolve_type_keys(mesh, None)


def test_hex20_stress_exports_write_one_element_and_twenty_nodes(tmp_path):
    mesh = make_hex20_stiffness_mesh(curved=True)
    mesh.elements[0].type = "C3D20"
    U = _affine_solid_displacement(mesh)
    elem_path = tmp_path / "hex20_element_stress.csv"
    nodal_path = tmp_path / "hex20_nodal_stress.csv"

    _write_current_element_stress(mesh, U, elem_path)
    _write_current_nodal_stress(mesh, U, nodal_path)

    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert len(elem_rows) == 2
    assert len(nodal_rows) == 21
    rows_by_node = {int(row[0]): row for row in nodal_rows[1:]}
    expected = get_element_kernel(mesh.elements[0].type).nodal_stress(
        mesh,
        mesh.elements[0],
        U,
    )
    for local_index in (0, 8):
        node_id = mesh.elements[0].node_ids[local_index]
        exported = np.array([float(value) for value in rows_by_node[node_id][7:13]])
        assert not np.allclose(expected[local_index], 0.0)
        assert np.allclose(exported, expected[local_index])


def test_mixed_solid_element_export_uses_hex20_and_tet4_centroids(tmp_path):
    hex20_mesh = make_hex20_stiffness_mesh(curved=True)
    mesh = Mesh3D(
        nodes=[*hex20_mesh.nodes, Node3D(21, 2.0, 0.0, 0.0)],
        elements=[
            hex20_mesh.elements[0],
            Element3D(
                2,
                [2, 21, 3, 6],
                "Tet4",
                {"E": 120.0, "nu": 0.25},
            ),
        ],
    )
    U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
    csv_path = tmp_path / "mixed_hex20_tet4_element_stress.csv"

    _write_current_element_stress(mesh, U, csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    expected = [
        get_element_kernel(mesh.elements[0].type).stress_at(
            mesh, mesh.elements[0], U, 0.0, 0.0, 0.0
        ),
        get_element_kernel(mesh.elements[1].type).stress_at(
            mesh, mesh.elements[1], U, 0.25, 0.25, 0.25
        ),
    ]

    assert [row[0] for row in rows[1:]] == ["1", "2"]
    for row, expected_stress in zip(rows[1:], expected):
        assert np.allclose([float(value) for value in row[1:7]], expected_stress)


def test_vtk_cells_support_hex20_in_abaqus_node_order(tmp_path):
    mesh = make_hex20_stiffness_mesh(curved=True)

    vtk_cells, cell_types, elems_for_cell = vtk.cells.build(mesh)

    assert vtk_cells == [[20, *range(20)]]
    assert cell_types == [25]
    assert elems_for_cell == mesh.elements

    result = make_zero_result(mesh, "hex20_vtk")
    vtk.export.from_result(result, output_dir=tmp_path)
    vtk_text = (tmp_path / "hex20_vtk.vtk").read_text(encoding="utf-8")

    assert "20 " + " ".join(str(index) for index in range(20)) in vtk_text
    assert "CELL_TYPES 1" in vtk_text
    assert "\n25\n" in vtk_text


def test_vtk_cells_reject_reduced_integration_hex20_without_type_25():
    element_type = "C3D20R"
    mesh = make_hex20_stiffness_mesh()
    mesh.elements[0].type = element_type

    with pytest.raises(
        ValueError,
        match=rf"Unsupported element type for VTK export: {element_type}",
    ):
        vtk.cells.build(mesh)


def test_vtk_export_from_result_materializes_missing_csvs(tmp_path):
    result = make_zero_result(make_unit_hex8_mesh(), "vtk_auto")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_auto_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_auto_element_stress.csv").exists()
    assert (tmp_path / "vtk_auto_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_auto.vtk").exists()


def test_vtk_export_from_result_overwrites_derived_csvs(tmp_path):
    result = make_zero_result(make_unit_hex8_mesh(), "vtk_overwrite")
    stale_disp = tmp_path / "vtk_overwrite_nodal_displacement.csv"
    stale_disp.write_text(
        "node_id,x,y,z,ux,uy,uz\n1,0,0,0,999,999,999\n",
        encoding="utf-8",
    )

    vtk.export.from_result(result, output_dir=tmp_path)

    assert "999" not in stale_disp.read_text(encoding="utf-8")


def test_vtk_export_from_result_skips_unsupported_nodal_stress(tmp_path):
    result = make_zero_result(make_simple_truss_mesh(E=100.0, area=1.0), "vtk_truss")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "vtk_truss_nodal_displacement.csv").exists()
    assert (tmp_path / "vtk_truss_element_stress.csv").exists()
    assert not (tmp_path / "vtk_truss_nodal_stress.csv").exists()
    assert (tmp_path / "vtk_truss.vtk").exists()


def test_vtk_element_stress_reader_averages_repeated_element_rows(tmp_path):
    from fem.post.vtk import fields

    csv_path = tmp_path / "test_vtk_element_stress_average.csv"
    csv_path.write_text(
        "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises\n"
        "1,1,1,1,2,3,4\n"
        "1,2,2,3,4,5,6\n",
        encoding="utf-8",
    )

    fields_by_name = fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)
    assert fields_by_name["mises"][1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "row", "expected"),
    (
        ("node_id", "bad,1,2,3,4", "expected an integer"),
        ("ux", "1,bad,2,3,4", "expected a numeric value"),
        ("uy", "1,1,bad,3,4", "expected a numeric value"),
        ("uz", "1,1,2,bad,4", "expected a numeric value"),
        ("rz", "1,1,2,3,bad", "expected a numeric value"),
    ),
)
def test_vtk_displacement_reader_reports_invalid_leaf_with_context(
    tmp_path,
    field,
    row,
    expected,
):
    mesh = Mesh3D(nodes=[Node3D(1, 0.0, 0.0, 0.0)], elements=[])
    csv_path = tmp_path / f"invalid_displacement_{field}.csv"
    csv_path.write_text(
        "node_id,ux,uy,uz,rz\n" + row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        vtk.fields.read_displacement(mesh, csv_path)

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert f"field {field}" in message
    assert "raw value 'bad'" in message
    assert expected in message


def test_vtk_element_stress_reader_reports_invalid_elem_id_with_context(tmp_path):
    csv_path = tmp_path / "invalid_element_stress_elem_id.csv"
    csv_path.write_text(
        "elem_id,sig_x\nbad,10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        vtk.fields.read_element_stress(csv_path)

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field elem_id" in message
    assert "raw value 'bad'" in message
    assert "expected an integer" in message


def test_vtk_element_stress_reader_keeps_invalid_scalar_zero_in_average(tmp_path):
    csv_path = tmp_path / "invalid_element_stress_scalar.csv"
    csv_path.write_text(
        "elem_id,sig_x\n1,4\n1,invalid\n",
        encoding="utf-8",
    )

    fields_by_name = vtk.fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)


def test_direct_post_exports_create_parent_dirs_and_beam_uses_six_components(tmp_path):
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2")],
        dofs_per_node=6,
    )
    output_path = tmp_path / "nested" / "beam_displacement.csv"

    displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), output_path)

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    for component in ("ux", "uy", "uz", "rx", "ry", "rz"):
        assert component in header


@pytest.mark.parametrize(
    (
        "mesh_builder",
        "expected_keys",
        "expected_group",
        "element_supported",
        "nodal_supported",
        "gauss_order",
    ),
    (
        (make_hex20_stiffness_mesh, ("hex20",), "solid", True, True, 3),
        (make_mixed_hex8_tet4_mesh, ("hex8", "tet4"), "solid", True, True, None),
        (make_mixed_hex8_hex20_mesh, ("hex8", "hex20"), "solid", True, True, None),
        (make_mixed_hex20_tet10_mesh, ("hex20", "tet10"), "solid", True, True, None),
        (make_mixed_tri3_quad4_mesh, ("tri3", "quad4"), "plane", True, True, None),
        (make_mixed_tri6_quad8_mesh, ("tri6", "quad8"), "plane", True, True, None),
        (_make_beam_dispatch_mesh, ("beam2",), "line", False, True, None),
    ),
)
def test_post_stress_dispatch_supports_current_type_groups(
    mesh_builder,
    expected_keys,
    expected_group,
    element_supported,
    nodal_supported,
    gauss_order,
):
    mesh = mesh_builder()

    assert dispatch.resolve_type_keys(mesh, None) == expected_keys
    assert dispatch.stress_group_for_keys(expected_keys) == expected_group
    assert dispatch.element_stress_supported(expected_keys) is element_supported
    assert dispatch.nodal_stress_supported(expected_keys) is nodal_supported
    if gauss_order is not None:
        assert dispatch.default_gauss_order(expected_keys[0]) == gauss_order


def test_vtk_reader_parses_beam2_nodal_stress_csv_as_three_scalars(tmp_path):
    path = tmp_path / "beam_nodal_stress.csv"
    path.write_text(
        "node_id,x,y,z,axial_stress_max,axial_stress_min,axial_stress_abs_max\n"
        "1,0,0,0,12,-4,12\n"
        "2,1,0,0,8,-6,8\n",
        encoding="utf-8",
    )

    data = vtk.fields.read_nodal_stress_rows(path)

    assert data.field_names == (
        "axial_stress_max",
        "axial_stress_min",
        "axial_stress_abs_max",
    )
    assert [row.node_id for row in data.rows] == [1, 2]
    assert all(row.averaged for row in data.rows)
    assert data.rows[0].values == {
        "axial_stress_max": 12.0,
        "axial_stress_min": -4.0,
        "axial_stress_abs_max": 12.0,
    }


@pytest.mark.parametrize(
    ("mesh_builder", "name", "element_row_count"),
    (
        (make_mixed_hex8_tet4_mesh, "mixed_solid", 3),
        (make_mixed_tri3_quad4_mesh, "mixed_plane", 8),
    ),
)
def test_mixed_stress_exports_write_element_and_nodal_rows(
    tmp_path,
    mesh_builder,
    name,
    element_row_count,
):
    mesh = mesh_builder()
    elem_path = tmp_path / f"{name}_element_stress.csv"
    nodal_path = tmp_path / f"{name}_nodal_stress.csv"

    _write_current_element_stress(mesh, np.zeros(mesh.num_dofs), elem_path)
    _write_current_nodal_stress(mesh, np.zeros(mesh.num_dofs), nodal_path)
    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == element_row_count
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == sum(len(elem.node_ids) for elem in mesh.elements) + 1
    assert {row[0] for row in nodal_rows[1:]} == {str(node.id) for node in mesh.nodes}


def test_stress_exports_cover_higher_order_mixed_types(tmp_path):
    solid_mesh = make_mixed_tet4_tet10_mesh()
    plane_mesh = make_mixed_tri6_quad8_mesh()
    solid_elem = tmp_path / "mixed_tet_element_stress.csv"
    solid_nodal = tmp_path / "mixed_tet_nodal_stress.csv"
    solid_vtk = make_zero_result(solid_mesh, "mixed_tet_vtk")
    plane_elem = tmp_path / "mixed_quad_element_stress.csv"
    plane_nodal = tmp_path / "mixed_quad_nodal_stress.csv"

    _write_current_element_stress(
        solid_mesh,
        np.zeros(solid_mesh.num_dofs),
        solid_elem,
    )
    _write_current_nodal_stress(
        solid_mesh,
        np.zeros(solid_mesh.num_dofs),
        solid_nodal,
    )
    vtk.export.from_result(solid_vtk, output_dir=tmp_path)
    _write_current_element_stress(
        plane_mesh,
        np.zeros(plane_mesh.num_dofs),
        plane_elem,
    )
    _write_current_nodal_stress(
        plane_mesh,
        np.zeros(plane_mesh.num_dofs),
        plane_nodal,
    )

    with solid_elem.open("r", encoding="utf-8") as f:
        solid_elem_rows = list(csv.reader(f))
    with solid_nodal.open("r", encoding="utf-8") as f:
        solid_nodal_rows = list(csv.reader(f))
    with plane_elem.open("r", encoding="utf-8") as f:
        plane_elem_rows = list(csv.reader(f))
    with plane_nodal.open("r", encoding="utf-8") as f:
        plane_nodal_rows = list(csv.reader(f))
    vtk_text = (tmp_path / "mixed_tet_vtk.vtk").read_text(encoding="utf-8")

    assert [row[0] for row in solid_elem_rows[1:]] == ["1", "2"]
    assert len(solid_nodal_rows) == len(solid_mesh.nodes) + 1
    assert len(plane_elem_rows) == 15
    assert len(plane_nodal_rows) == len(plane_mesh.nodes) + 1
    assert "\n10\n" in vtk_text
    assert "\n24\n" in vtk_text


@pytest.mark.parametrize(
    ("mesh_builder", "name", "element_rows", "nodal_rows", "cell_types"),
    (
        (make_mixed_hex8_hex20_mesh, "mixed_hex8_hex20", 3, 29, [12, 25]),
        (make_mixed_hex20_tet10_mesh, "mixed_hex20_tet10", 3, 31, [25, 24]),
    ),
)
def test_mixed_hex20_stress_and_vtk_exports_have_exact_rows_and_cell_types(
    tmp_path,
    mesh_builder,
    name,
    element_rows,
    nodal_rows,
    cell_types,
):
    mesh = mesh_builder()
    U = _affine_solid_displacement(mesh)
    element_path = tmp_path / f"{name}_direct_element_stress.csv"
    nodal_path = tmp_path / f"{name}_direct_nodal_stress.csv"

    _write_current_element_stress(mesh, U, element_path)
    _write_current_nodal_stress(mesh, U, nodal_path)
    vtk.export.from_result(make_zero_result(mesh, name), output_dir=tmp_path)

    with element_path.open("r", encoding="utf-8") as f:
        element_stress_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_stress_rows = list(csv.reader(f))
    vtk_lines = (tmp_path / f"{name}.vtk").read_text(encoding="utf-8").splitlines()
    cell_types_index = vtk_lines.index("CELL_TYPES 2")

    assert len(element_stress_rows) == element_rows
    assert len(nodal_stress_rows) == nodal_rows
    assert [int(value) for value in vtk_lines[cell_types_index + 1 : cell_types_index + 3]] == cell_types
    rows_by_node = {int(row[0]): row for row in nodal_stress_rows[1:]}
    hex20_elem = next(
        elem
        for elem in mesh.elements
        if dispatch.type_key_from_name(elem.type) == "hex20"
    )
    expected = get_element_kernel(hex20_elem.type).nodal_stress(mesh, hex20_elem, U)
    for local_index in (0, 8):
        node_id = hex20_elem.node_ids[local_index]
        exported = np.array([float(value) for value in rows_by_node[node_id][7:13]])
        assert not np.allclose(expected[local_index], 0.0)
        assert np.allclose(exported, expected[local_index])


def test_vtk_cells_support_tri6_quadratic_triangle(tmp_path):
    result = make_zero_result(make_tri6_stiffness_mesh(), "tri6_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)
    vtk_text = (tmp_path / "tri6_vtk.vtk").read_text(encoding="utf-8")

    assert "CELL_TYPES 1" in vtk_text
    assert "\n22\n" in vtk_text


def test_vtk_cells_support_mixed_tri3_quad4_topology(tmp_path):
    result = make_zero_result(make_mixed_tri3_quad4_mesh(), "mixed_plane_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)

    vtk_lines = (tmp_path / "mixed_plane_vtk.vtk").read_text(
        encoding="utf-8"
    ).splitlines()
    cell_types_index = vtk_lines.index("CELL_TYPES 2")

    assert [int(value) for value in vtk_lines[cell_types_index + 1 : cell_types_index + 3]] == [
        5,
        9,
    ]


def test_vtk_export_from_result_materializes_mixed_stress_csvs(tmp_path):
    result = make_zero_result(make_mixed_hex8_tet4_mesh(), "mixed_vtk")

    vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "mixed_vtk_nodal_displacement.csv").exists()
    assert (tmp_path / "mixed_vtk_element_stress.csv").exists()
    assert (tmp_path / "mixed_vtk_nodal_stress.csv").exists()
    vtk_text = (tmp_path / "mixed_vtk.vtk").read_text(encoding="utf-8")

    assert "CELL_TYPES 2" in vtk_text
    assert "\n12\n" in vtk_text
    assert "\n10\n" in vtk_text


def test_vtk_cells_report_unsupported_element_type(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[1].type = "UnsupportedSolid"
    result = make_zero_result(mesh, "unsupported_vtk")

    with pytest.raises(ValueError, match="Unsupported element type for VTK export: UnsupportedSolid"):
        vtk.export.from_result(result, output_dir=tmp_path)
