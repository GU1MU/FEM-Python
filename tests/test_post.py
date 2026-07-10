import csv
import importlib
import sys

import numpy as np
import pytest

import fem.post as post
from fem.core.mesh import (
    BeamMesh2D,
    Element2D,
    Element3D,
    HexMesh3D,
    Node2D,
    Node3D,
    PlaneMesh2D,
)
from fem.elements import get_element_kernel
from fem.post import displacement, path, polar, stress, vtk
from fem.post.stress import dispatch
from fem.post.polar import convert_nodal_solution_into_polar_coord
from fem.post.vtk.polar import convert_nodal_displacement
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_mixed_hex20_tet10_mesh,
    make_mixed_hex8_hex20_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_quad4_quad8_mesh,
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
    assert rows[1].averaged is False


def test_nodal_stress_csv_preserves_material_boundary_contributions(tmp_path):
    mesh = PlaneMesh2D(
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
                "Tri3Plane",
                {"E": 100.0, "nu": 0.25, "plane_type": "stress", "thickness": 1.0},
            ),
            Element2D(
                2,
                [3, 4, 5],
                "Tri3Plane",
                {"E": 200.0, "nu": 0.3, "plane_type": "stress", "thickness": 1.0},
            ),
        ],
    )
    csv_path = tmp_path / "material_boundary.csv"

    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path, threshold=100.0)

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
    mesh = PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3Plane"),
            Element2D(2, [3, 4, 5], "Tri3Plane"),
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
    mesh = PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3Plane", dict(props)),
            Element2D(2, [3, 4, 5], "Tri3Plane", dict(props)),
        ],
    )

    vtk.export.from_result(
        make_zero_result(mesh, "threshold_zero"),
        output_dir=tmp_path,
        threshold=0.0,
    )

    csv_rows = list(
        csv.DictReader(
            (tmp_path / "threshold_zero_nodal_stress.csv").open(
                "r", encoding="utf-8"
            )
        )
    )
    shared_rows = [row for row in csv_rows if row["node_id"] == "3"]
    vtk_lines = (tmp_path / "threshold_zero.vtk").read_text(encoding="utf-8").splitlines()
    assert [(row["elem_id"], row["averaged"]) for row in shared_rows] == [
        ("1", "false"),
        ("2", "false"),
    ]
    assert "POINTS 6 float" in vtk_lines


def test_vtk_accepts_legacy_single_row_nodal_stress_csv(tmp_path):
    mesh = PlaneMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3Plane")],
    )
    displacement_path = tmp_path / "legacy_displacement.csv"
    stress_path = tmp_path / "legacy_nodal_stress.csv"
    vtk_path = tmp_path / "legacy.vtk"
    displacement_path.write_text(
        "node_id,x,y,ux,uy\n1,0,0,0,0\n2,1,0,0,0\n3,0,1,0,0\n",
        encoding="utf-8",
    )
    stress_path.write_text(
        "node_id,x,y,sig_x,sig_y,tau_xy,mises\n"
        "1,0,0,1,0,0,1\n2,1,0,2,0,0,2\n3,0,1,3,0,0,3\n",
        encoding="utf-8",
    )

    vtk.export.from_csv(mesh, displacement_path, None, vtk_path, stress_path)

    lines = vtk_path.read_text(encoding="utf-8").splitlines()
    sig_x_index = lines.index("SCALARS sig_x float 1")
    assert "POINTS 3 float" in lines
    assert lines[sig_x_index + 2 : sig_x_index + 5] == ["1.0", "2.0", "3.0"]
    assert vtk.fields.read_nodal_stress(stress_path)["sig_x"] == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }


def test_vtk_rejects_ambiguous_repeated_nodal_stress_rows(tmp_path):
    stress_path = tmp_path / "repeated_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,1,1,false,10\n1,0,0,2,1,false,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeated node_id 1"):
        vtk.fields.read_nodal_stress(stress_path)

    mesh = PlaneMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0), Node2D(3, 0.0, 1.0)],
        elements=[Element2D(1, [1, 2, 3], "Tri3Plane")],
    )
    rows = (
        vtk.fields.NodalStressCsvRow(1, 1, 1, False, {"sig_x": 10.0}),
        vtk.fields.NodalStressCsvRow(1, None, None, False, {"sig_x": 20.0}),
    )
    with pytest.raises(ValueError, match="requires.*provenance"):
        vtk.cells.build_result(mesh, rows)


def test_polar_conversion_preserves_distinct_repeated_nodal_rows():
    mesh = PlaneMesh2D(
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


def test_vtk_export_lives_inside_post_package():
    mesh = PlaneMesh2D(
        nodes=[Node2D(1, 1.0, 0.0), Node2D(2, 0.0, 1.0)],
        elements=[],
    )

    polar_values = convert_nodal_displacement(
        mesh,
        {
            1: {"ux": 2.0, "uy": 0.0, "rz": 0.5},
            2: {"ux": 0.0, "uy": 3.0, "rz": 0.0},
        },
        [0.0, 0.0],
    )

    assert polar_values[1]["ux"] == pytest.approx(2.0)
    assert polar_values[1]["uy"] == pytest.approx(0.0)
    assert polar_values[1]["rz"] == pytest.approx(0.5)
    assert polar_values[2]["ux"] == pytest.approx(3.0)
    assert polar_values[2]["uy"] == pytest.approx(0.0)

    sys.modules.pop("fem.vtk_export", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.vtk_export")
    sys.modules.pop("fem.post.vtk_export", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.post.vtk_export")


def test_post_package_exposes_submodules_without_function_facade():
    assert hasattr(post, "__path__")
    assert post.displacement is displacement
    assert post.path is path
    assert post.polar is polar
    assert post.stress is stress
    assert post.vtk is vtk
    assert not hasattr(post, "export_nodal_displacements_csv")
    assert not hasattr(post, "export_hex8_element_stress_csv")
    assert hasattr(displacement, "__path__")
    assert callable(displacement.export.nodal)
    assert not hasattr(displacement, "export_nodal_displacement")
    assert callable(path.extract_path_data)
    assert callable(convert_nodal_solution_into_polar_coord)
    assert hasattr(stress, "__path__")
    assert callable(stress.dispatch.resolve_type_key)
    assert callable(stress.element.by_type)
    assert callable(stress.export.element)
    assert callable(stress.export.nodal)
    assert callable(stress.invariants.von_mises_3d)
    assert callable(stress.nodal.by_type)
    assert not hasattr(stress, "export_hex8_element_stress")
    assert not hasattr(stress, "_compute_hex8_element_stress_at_point")
    assert hasattr(vtk, "__path__")
    assert hasattr(vtk, "cells")
    assert callable(vtk.export.from_csv)
    assert hasattr(vtk, "fields")
    assert hasattr(vtk, "polar")
    assert hasattr(vtk, "writer")
    assert not hasattr(vtk, "export_from_csv_3d")

    for old_module in (
        "fem.post.displacement_export",
        "fem.post.path_export",
        "fem.post.stress_export",
        "fem.post.vtk_export",
    ):
        sys.modules.pop(old_module, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)


def test_stress_export_infers_single_element_type_from_mesh(tmp_path):
    mesh = make_unit_hex8_mesh()
    elem_path = tmp_path / "test_post_stress_element.csv"
    nodal_path = tmp_path / "test_post_stress_nodal.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)

    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == 2
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == 9


def test_dispatch_supports_hex20_stress_exports():
    mesh = make_hex20_stiffness_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("hex20",)
    assert dispatch.type_key_from_name("C3D20") == "hex20"
    assert "hex20" in dispatch.ELEMENT_STRESS_KEYS
    assert "hex20" in dispatch.NODAL_STRESS_KEYS
    assert dispatch.TYPE_GROUPS["hex20"] == "solid"
    assert dispatch.default_gauss_order("hex20") == 3


@pytest.mark.parametrize("element_type", ["C3D20R", "c3D20r"])
def test_dispatch_rejects_reduced_integration_hex20_alias(element_type):
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

    stress.export.element(mesh, U, elem_path)
    stress.export.nodal(mesh, U, nodal_path)

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
    mesh = HexMesh3D(
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

    stress.export.element(mesh, U, csv_path)

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


@pytest.mark.parametrize("element_type", ["C3D20R", "c3D20r"])
def test_vtk_cells_reject_reduced_integration_hex20_without_type_25(element_type):
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
        "elem_id,node_id,local_node,sig_x,sig_y,tau_xy,mises_stress\n"
        "1,1,1,1,2,3,4\n"
        "1,2,2,3,4,5,6\n",
        encoding="utf-8",
    )

    fields_by_name = fields.read_element_stress(csv_path)

    assert fields_by_name["sig_x"][1] == pytest.approx(2.0)
    assert fields_by_name["mises_stress"][1] == pytest.approx(5.0)


def test_direct_post_exports_create_parent_dirs_and_beam_uses_rz(tmp_path):
    mesh = BeamMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[Element2D(1, [1, 2], "Beam2D")],
    )
    output_path = tmp_path / "nested" / "beam_displacement.csv"

    displacement.export.nodal(mesh, np.zeros(mesh.num_dofs), output_path)

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert "rz" in header
    assert "uz" not in header


def test_dispatch_resolves_compatible_mixed_solid_type_keys():
    mesh = make_mixed_hex8_tet4_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("hex8", "tet4")
    assert dispatch.stress_group_for_keys(("hex8", "tet4")) == "solid"
    assert dispatch.element_stress_supported(("hex8", "tet4"))
    assert dispatch.nodal_stress_supported(("hex8", "tet4"))


def test_dispatch_supports_mixed_hex20_solid_type_keys():
    hex_mesh = make_mixed_hex8_hex20_mesh()
    tet_mesh = make_mixed_hex20_tet10_mesh()

    assert dispatch.resolve_type_keys(hex_mesh, None) == ("hex8", "hex20")
    assert dispatch.resolve_type_keys(tet_mesh, None) == ("hex20", "tet10")
    assert dispatch.stress_group_for_keys(("hex8", "hex20")) == "solid"
    assert dispatch.element_stress_supported(("hex20", "tet10"))
    assert dispatch.nodal_stress_supported(("hex20", "tet10"))


def test_dispatch_resolves_compatible_mixed_plane_type_keys():
    mesh = make_mixed_tri3_quad4_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("tri3", "quad4")
    assert dispatch.stress_group_for_keys(("tri3", "quad4")) == "plane"


def test_dispatch_resolves_compatible_mixed_quadratic_plane_type_keys():
    mesh = make_mixed_tri6_quad8_mesh()

    assert dispatch.resolve_type_keys(mesh, None) == ("tri6", "quad8")
    assert dispatch.stress_group_for_keys(("tri6", "quad8")) == "plane"
    assert dispatch.element_stress_supported(("tri6", "quad8"))
    assert dispatch.nodal_stress_supported(("tri6", "quad8"))


def test_element_stress_export_writes_mixed_solid_rows(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    csv_path = tmp_path / "mixed_element_stress.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "elem_id"
    assert len(rows) == len(mesh.elements) + 1
    assert [row[0] for row in rows[1:]] == ["1", "2"]


def test_nodal_stress_export_writes_mixed_solid_nodes(tmp_path):
    mesh = make_mixed_hex8_tet4_mesh()
    csv_path = tmp_path / "mixed_nodal_stress.csv"

    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "node_id"
    assert len(rows) == sum(len(elem.node_ids) for elem in mesh.elements) + 1
    assert {row[0] for row in rows[1:]} == {str(node.id) for node in mesh.nodes}


def test_stress_exports_write_mixed_plane_rows_and_nodes(tmp_path):
    mesh = make_mixed_tri3_quad4_mesh()
    elem_path = tmp_path / "mixed_plane_element_stress.csv"
    nodal_path = tmp_path / "mixed_plane_nodal_stress.csv"

    stress.export.element(mesh, np.zeros(mesh.num_dofs), elem_path)
    stress.export.nodal(mesh, np.zeros(mesh.num_dofs), nodal_path)
    with elem_path.open("r", encoding="utf-8") as f:
        elem_rows = list(csv.reader(f))
    with nodal_path.open("r", encoding="utf-8") as f:
        nodal_rows = list(csv.reader(f))

    assert elem_rows[0][0] == "elem_id"
    assert len(elem_rows) == 8
    assert nodal_rows[0][0] == "node_id"
    assert len(nodal_rows) == sum(len(elem.node_ids) for elem in mesh.elements) + 1


def test_stress_exports_cover_higher_order_mixed_types(tmp_path):
    solid_mesh = make_mixed_tet4_tet10_mesh()
    plane_mesh = make_mixed_tri6_quad8_mesh()
    solid_elem = tmp_path / "mixed_tet_element_stress.csv"
    solid_nodal = tmp_path / "mixed_tet_nodal_stress.csv"
    solid_vtk = make_zero_result(solid_mesh, "mixed_tet_vtk")
    plane_elem = tmp_path / "mixed_quad_element_stress.csv"
    plane_nodal = tmp_path / "mixed_quad_nodal_stress.csv"

    stress.export.element(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_elem)
    stress.export.nodal(solid_mesh, np.zeros(solid_mesh.num_dofs), solid_nodal)
    vtk.export.from_result(solid_vtk, output_dir=tmp_path)
    stress.export.element(plane_mesh, np.zeros(plane_mesh.num_dofs), plane_elem)
    stress.export.nodal(plane_mesh, np.zeros(plane_mesh.num_dofs), plane_nodal)

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

    stress.export.element(mesh, U, element_path)
    stress.export.nodal(mesh, U, nodal_path)
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
