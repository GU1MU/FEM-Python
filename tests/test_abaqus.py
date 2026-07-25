import numpy as np
import pytest

from fem import abaqus, materials, post
from fem.boundary.step import boundary_for_step, get_step
from fem.core.model import (
    DisplacementConstraint,
    ElementEdge,
    ElementFace,
    GravityLoad,
    NodalLoad,
    OutputRequest,
    SectionAssignment,
)
from fem.core.mesh import Mesh3D
from fem.solvers import static_linear
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.file_builders import write_inp


def _assert_pressure_points_inward(model, bc):
    node_lookup = {node.id: node for node in model.mesh.nodes}
    elem = model.mesh.elements[0]
    elem_xyz = np.array(
        [
            [node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
            for node_id in elem.node_ids
        ],
        dtype=float,
    )
    elem_center = elem_xyz.mean(axis=0)

    for face, traction in zip(model.surfaces["LOADED"].faces, bc.surface_tractions):
        face_xyz = np.array(
            [
                [node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
                for node_id in face.node_ids
            ],
            dtype=float,
        )
        inward = elem_center - face_xyz.mean(axis=0)
        assert np.dot(np.array(traction.vector), inward) > 0.0


def test_abaqus_read_builds_model_with_sets_surfaces_materials_and_steps(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_model.inp",
        [
            "*Heading",
            "** minimal model reader fixture",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Nset, nset=FIXED",
            "1,4,5,8",
            "*Nset, nset=TIP",
            "2,3,6,7",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=TIP_FACE",
            "SOLID, S4",
            "*Material, name=STEEL",
            "*Density",
            "7.85",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD, nlgeom=NO",
            "*Static",
            "1., 1., 1e-05, 1.",
            "*Boundary",
            "FIXED, 1, 3, 0.",
            "*Cload",
            "TIP, 3, -50.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert model.name == "test_abaqus_model"
    assert model.mesh.num_nodes == 8
    assert model.mesh.elements[0].type == "Hex8"
    assert "E" not in model.mesh.elements[0].props
    assert model.materials["STEEL"].properties["E"] == 210.0
    assert model.materials["STEEL"].properties["nu"] == 0.3
    assert model.materials["STEEL"].properties["rho"] == 7.85
    assert model.node_sets["FIXED"].node_ids == (1, 4, 5, 8)
    assert model.element_sets["SOLID"].element_ids == (1,)
    assert model.surfaces["TIP_FACE"].faces[0] == ElementFace(1, 5, (2, 3, 7, 6))
    assert model.sections[0] == SectionAssignment("SOLID", "STEEL")
    assert model.steps[0].name == "LOAD"
    assert model.steps[0].boundaries[0] == DisplacementConstraint("FIXED", 1, 3, 0.0)
    assert model.steps[0].cloads[0] == NodalLoad("TIP", 3, -50.0)

    bc = boundary_for_step(model, "LOAD")
    assert len(bc.prescribed_displacements) == 12
    assert sum(bc.nodal_forces.values()) == pytest.approx(-200.0)


@pytest.mark.parametrize(
    ("filename", "step_lines"),
    [
        ("test_perforated_plate_pressure.inp", ("*Dsload", "Surf-right, P, 2.")),
        ("test_perforated_plate_disp.inp", ("*Boundary", "Set-right, 1, 1, 0.05")),
    ],
)
def test_abaqus_read_perforated_plate_style_inputs_without_data_files(
    tmp_path,
    filename,
    step_lines,
):
    inp_path = write_perforated_plate_style_inp(tmp_path, filename, step_lines)
    model = abaqus.read(inp_path)

    assert model.mesh.num_nodes == 6
    assert len(model.mesh.elements) == 2
    assert "Surf-right" in model.edges
    assert model.edges["Surf-right"].edges[0] == ElementEdge(2, 1, (3, 6))


def test_abaqus_read_preserves_2d_solid_section_thickness(tmp_path):
    inp_path = write_perforated_plate_style_inp(
        tmp_path,
        "plate_with_thickness.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=("2.5,",),
    )

    model = abaqus.read(inp_path)

    assert model.sections[0].properties == {"thickness": 2.5}
    materials.apply_sections(model)
    assert {
        element.props["thickness"]
        for element in model.mesh.elements
    } == {2.5}


def test_abaqus_solid_section_thickness_affects_2d_stiffness(tmp_path):
    thin_path = write_perforated_plate_style_inp(
        tmp_path,
        "plate_thickness_1.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=("1.,",),
    )
    thick_path = write_perforated_plate_style_inp(
        tmp_path,
        "plate_thickness_2.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=("2.,",),
    )

    thin = static_linear.solve(abaqus.read(thin_path))
    thick = static_linear.solve(abaqus.read(thick_path))

    thin_displacement = thin.nodal_displacement(3, 1)
    thick_displacement = thick.nodal_displacement(3, 1)
    assert thin_displacement != 0.0
    assert thick_displacement == pytest.approx(thin_displacement / 2.0)


def test_abaqus_rejects_solid_section_thickness_for_3d_elements(tmp_path):
    path = write_inp(
        tmp_path,
        "solid_thickness_3d.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "2.5,",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )

    with pytest.raises(ValueError, match="two-dimensional CPS/CPE"):
        abaqus.read(path)


@pytest.mark.parametrize(
    ("filename", "step_lines", "expected_load_type"),
    [
        ("test_perforated_plate_pressure.inp", ("*Dsload", "Surf-right, P, 2."), "pressure"),
        (
            "test_perforated_plate_disp.inp",
            ("*Boundary", "Set-right, 1, 1, 0.05"),
            "displacement",
        ),
    ],
)
def test_abaqus_builds_boundary_for_perforated_plate_style_inputs_without_data_files(
    tmp_path,
    filename,
    step_lines,
    expected_load_type,
):
    inp_path = write_perforated_plate_style_inp(tmp_path, filename, step_lines)
    model = abaqus.read(inp_path)
    bc = boundary_for_step(model)

    assert len(bc.prescribed_displacements) > 0
    if expected_load_type == "pressure":
        assert len(bc.edge_tractions) > 0
    else:
        assert any(abs(value) > 1e-12 for value in bc.prescribed_displacements.values())


def test_abaqus_solves_and_exports_perforated_plate_displacement_input_without_data_files(
    tmp_path,
):
    path = write_perforated_plate_style_inp(
        tmp_path,
        "test_perforated_plate_disp.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    model = abaqus.read(path)
    model.name = "perforated_plate_disp"

    result = static_linear.solve(model)
    post.vtk.export.from_result(result, output_dir=tmp_path)

    assert (tmp_path / "perforated_plate_disp_nodal_displacement.csv").exists()
    assert (tmp_path / "perforated_plate_disp_element_stress.csv").exists()
    assert (tmp_path / "perforated_plate_disp_nodal_stress.csv").exists()
    assert (tmp_path / "perforated_plate_disp.vtk").exists()


def test_abaqus_read_hides_internal_element_sets_without_breaking_surfaces_or_sections(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_internal_element_sets.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=_PickedSet7",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=_Surface_Pressure_Load_1_Face_S4",
            "1",
            "*Surface, type=ELEMENT, name=Surface_Pressure_Load_1_Face",
            "_Surface_Pressure_Load_1_Face_S4, S4",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=_PickedSet7, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "Surface_Pressure_Load_1_Face, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert model.element_sets == {}
    assert model.surfaces["Surface_Pressure_Load_1_Face"].faces[0] == ElementFace(
        1,
        5,
        (2, 3, 7, 6),
    )

    bc = boundary_for_step(model, "LOAD")
    assert len(bc.surface_tractions) == 1

    materials.apply_sections(model)
    assert model.mesh.elements[0].props["material"] == "STEEL"


def test_abaqus_sections_use_part_element_ids_when_assembly_set_reuses_name(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_section_part_elset_scope.inp",
        [
            "*Part, name=BLOCK",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8",
            "1, 1,2,3,4,5,6,7,8",
            "2, 1,2,3,4,5,6,7,8",
            "*Elset, elset=Set-1, generate",
            "1, 2, 1",
            "*Solid Section, elset=Set-1, material=STEEL",
            "*End Part",
            "*Assembly, name=Assembly",
            "*Instance, name=BLOCK-1, part=BLOCK",
            "*End Instance",
            "*Elset, elset=Set-1, instance=BLOCK-1",
            "1",
            "*End Assembly",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
        ],
    )

    model = abaqus.read(path)

    materials.apply_sections(model)

    assert [elem.props.get("material") for elem in model.mesh.elements] == ["STEEL", "STEEL"]
    assert [elem.props.get("E") for elem in model.mesh.elements] == [210.0, 210.0]


def test_abaqus_read_inherits_initial_boundaries_across_steps(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_initial_steps.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "*Element, type=CPS3, elset=SOLID",
            "1, 1,2,1",
            "*Nset, nset=FIXED",
            "1",
            "*Nset, nset=TIP",
            "2",
            "*Elset, elset=SOLID",
            "1",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "FIXED, 1, 2, 0.",
            "*Step, name=STEP-1",
            "*Static",
            "*Cload",
            "TIP, 1, 10.",
            "*End Step",
            "*Step, name=STEP-2",
            "*Static",
            "*Cload",
            "TIP, 2, 20.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert [step.name for step in model.steps] == ["Initial", "STEP-1", "STEP-2"]
    assert get_step(model).name == "STEP-1"
    step2_bc = boundary_for_step(model, "STEP-2")
    assert len(step2_bc.prescribed_displacements) == 2
    assert sum(step2_bc.nodal_forces.values()) == pytest.approx(20.0)


def test_abaqus_read_prefers_assembly_node_set_over_part_set_for_load_targets(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_scoped_node_sets.inp",
        [
            "*Part, name=BLOCK",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Nset, nset=LOADSET, generate",
            "1, 4, 1",
            "*Elset, elset=SOLID",
            "1",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*End Part",
            "*Assembly, name=Assembly",
            "*Instance, name=BLOCK-1, part=BLOCK",
            "*End Instance",
            "*Nset, nset=LOADSET, instance=BLOCK-1",
            "2",
            "*Nset, nset=FIXED, instance=BLOCK-1",
            "1",
            "*End Assembly",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Step, name=LOAD",
            "*Static",
            "*Boundary",
            "FIXED, 1, 3, 0.",
            "*Cload",
            "LOADSET, 2, -1000.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    assert model.node_sets["LOADSET"].node_ids == (2,)
    assert len(bc.nodal_forces) == 1
    assert sum(bc.nodal_forces.values()) == pytest.approx(-1000.0)


def test_abaqus_read_converts_dsload_and_dload_pressure_to_surface_tractions(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_surface_loads.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=TIP_FACE",
            "SOLID, S4",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "TIP_FACE, P, 2.",
            "*Dload",
            "SOLID, P4, 3.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    assert len(model.steps[0].surface_loads) == 2
    assert "TIP_FACE" in model.surfaces
    assert len(bc.surface_tractions) == 2
    assert np.allclose(bc.surface_tractions[0].vector, (-2.0, 0.0, 0.0))
    assert np.allclose(bc.surface_tractions[1].vector, (-3.0, 0.0, 0.0))


def test_abaqus_parse_preserves_blank_gravity_target_and_trims_trailing_separators(
    tmp_path,
):
    path = write_inp(
        tmp_path,
        "test_abaqus_parse_blank_gravity_target.inp",
        [
            "*Step, name=LOAD",
            "*Dload",
            ", GRAV, 9810., 0., -1., 0., ,",
            "*End Step",
        ],
    )

    deck = abaqus.parse_file(path)

    load = deck.steps[0].distributed_loads[0]
    assert load.target is None
    assert load.label == "GRAV"
    assert load.magnitude == pytest.approx(9810.0)
    assert load.extra == (0.0, -1.0, 0.0)
    assert load.source == "dload"


def test_abaqus_read_builds_global_set_and_element_gravity_without_topology(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_gravity_targets.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Step, name=LOAD",
            "*Dload",
            ", GRAV, 10., 0., -2., 0.",
            "SOLID, GRAV, 3., 0., 0., -4.",
            "1, GRAV, 5., 1., 0., 0.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert model.steps[0].gravity_loads == (
        GravityLoad((0.0, -10.0, 0.0)),
        GravityLoad((0.0, 0.0, -3.0), target="SOLID"),
        GravityLoad((5.0, 0.0, 0.0), target=1),
    )
    assert model.steps[0].surface_loads == ()
    assert model.steps[0].edge_loads == ()
    assert model.surfaces == {}
    assert model.edges == {}


def test_abaqus_read_converts_in_plane_gravity_for_2d_model(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_gravity.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 0., 1.",
            "*Element, type=CPS3, elset=SOLID",
            "1, 1,2,3",
            "*Step, name=LOAD",
            "*Dload",
            ", GRAV, 10., 3., 4., 0.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert model.steps[0].gravity_loads == (GravityLoad((6.0, 8.0)),)
    assert model.surfaces == {}
    assert model.edges == {}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (", GRAV, 1., 0., -1.", r"GRAV requires.*3 direction components"),
        (", GRAV, 1., 0., -1., 0., 2.", r"GRAV requires.*3 direction components"),
        (", GRAV, bad, 0., -1., 0.", r"GRAV.*must be numeric"),
        (", GRAV, 1., 0., , 0.", r"GRAV.*must be numeric"),
    ],
    ids=["too-few-components", "too-many-components", "magnitude", "empty-component"],
)
def test_abaqus_parse_reports_gravity_specific_record_errors(tmp_path, record, message):
    path = write_inp(
        tmp_path,
        "test_abaqus_invalid_gravity_record.inp",
        [
            "*Step, name=LOAD",
            "*Dload",
            record,
            "*End Step",
        ],
    )

    with pytest.raises(ValueError, match=message):
        abaqus.parse_file(path)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (", GRAV, nan, 1., 0., 0.", r"GRAV magnitude must be finite"),
        (", GRAV, 1., nan, 0., 0.", r"GRAV direction components must be finite"),
        (", GRAV, 1., 0., 0., 0.", r"GRAV direction vector must be nonzero"),
    ],
    ids=["magnitude", "direction", "zero-direction"],
)
def test_abaqus_read_rejects_invalid_gravity_values(tmp_path, record, message):
    path = write_inp(
        tmp_path,
        "test_abaqus_invalid_gravity_value.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Step, name=LOAD",
            "*Dload",
            record,
            "*End Step",
        ],
    )

    with pytest.raises(ValueError, match=message):
        abaqus.read(path)


def test_abaqus_read_rejects_out_of_plane_gravity_for_2d_model(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_out_of_plane_gravity.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 0., 1.",
            "*Element, type=CPS3, elset=SOLID",
            "1, 1,2,3",
            "*Step, name=LOAD",
            "*Dload",
            ", GRAV, 10., 0., 0., 1.",
            "*End Step",
        ],
    )

    with pytest.raises(ValueError, match=r"GRAV out-of-plane acceleration.*2D"):
        abaqus.read(path)


def test_abaqus_read_rejects_dsload_gravity_before_surface_lookup(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_dsload_gravity.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Step, name=LOAD",
            "*Dsload",
            "MISSING_SURFACE, GRAV, 10., 0., -1., 0.",
            "*End Step",
        ],
    )

    with pytest.raises(ValueError, match=r"DSLOAD GRAV.*DLOAD"):
        abaqus.read(path)


def test_abaqus_2d_pressure_load_builds_edge_traction(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_pressure_load.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1,2,3,4",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=RIGHT_EDGE",
            "SOLID, S2",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "RIGHT_EDGE, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)
    bc = boundary_for_step(model, "LOAD")

    assert "RIGHT_EDGE" in model.edges
    assert "RIGHT_EDGE" not in model.surfaces
    assert model.edges["RIGHT_EDGE"].edges[0] == ElementEdge(1, 1, (2, 3))
    assert len(model.steps[0].edge_loads) == 1
    assert model.steps[0].edge_loads[0].load_type == "pressure"
    assert len(bc.edge_tractions) == 1
    assert bc.edge_tractions[0].elem_id == 1
    assert bc.edge_tractions[0].local_index == 1
    assert np.allclose(bc.edge_tractions[0].vector, (-2.0, 0.0))


def test_abaqus_2d_dload_pressure_generates_edge_load(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_dload_pressure.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1,2,3,4",
            "*Step, name=load",
            "*DLOAD",
            "SOLID, P2, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)
    bc = boundary_for_step(model)

    assert len(model.edges) == 1
    generated_name = next(iter(model.edges))
    assert generated_name.startswith("__DLOAD_0_load_0")
    assert model.edges[generated_name].edges == (ElementEdge(1, 1, (2, 3)),)
    assert len(model.steps[0].edge_loads) == 1
    assert model.steps[0].edge_loads[0].edge == generated_name
    assert len(bc.edge_tractions) == 1
    assert np.allclose(bc.edge_tractions[0].vector, (-2.0, 0.0))


def test_abaqus_2d_trvec_load_builds_edge_traction(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_trvec_load.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1,2,3,4",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=RIGHT_EDGE",
            "SOLID, S2",
            "*Step, name=LOAD",
            "*DSLOAD",
            "RIGHT_EDGE, TRVEC, 5., 0., -2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)
    bc = boundary_for_step(model, "LOAD")

    assert len(model.steps[0].edge_loads) == 1
    assert model.steps[0].edge_loads[0].load_type == "traction"
    assert model.steps[0].edge_loads[0].vector == (0.0, -5.0)
    assert len(bc.edge_tractions) == 1
    assert np.allclose(bc.edge_tractions[0].vector, (0.0, -5.0))


def test_abaqus_3d_surface_loads_remain_surfaces(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_3d_surface_stays_surface.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Surface, type=ELEMENT, name=FACE",
            "SOLID, S4",
            "*Step, name=load",
            "*DSLOAD",
            "FACE, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert "FACE" in model.surfaces
    assert "FACE" not in model.edges
    assert len(model.steps[0].surface_loads) == 1
    assert model.steps[0].edge_loads == ()


def test_abaqus_read_projects_trshr_direction_to_surface_tangent(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_trshr_surface_load.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=TOP",
            "SOLID, S2",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "TOP, TRSHR, 10., 2., 0., 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    assert len(model.steps[0].surface_loads) == 1
    assert model.steps[0].surface_loads[0].load_type == "shear_traction"
    assert len(bc.surface_tractions) == 1
    assert np.allclose(bc.surface_tractions[0].vector, (10.0, 0.0, 0.0))


def test_abaqus_read_keeps_gravity_and_trshr_in_the_same_step(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_gravity_and_trshr.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Surface, type=ELEMENT, name=TOP",
            "SOLID, S2",
            "*Step, name=LOAD",
            "*Static",
            "*Dload",
            ", GRAV, 9.81, 0., -1., 0.",
            "*Dsload",
            "TOP, TRSHR, 10., 1., 0., 0.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    step = model.steps[0]
    assert step.gravity_loads == (GravityLoad((0.0, -9.81, 0.0)),)
    assert len(step.surface_loads) == 1
    assert step.surface_loads[0].load_type == "shear_traction"
    assert set(model.surfaces) == {"TOP"}
    assert not any(name.startswith("__DLOAD") for name in model.surfaces)


def test_abaqus_trshr_rejects_nonplanar_faces(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_trshr_nonplanar_face.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.2",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=WARPED_TOP",
            "SOLID, S2",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "WARPED_TOP, TRSHR, 10., 1., 0., 0.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    with pytest.raises(ValueError, match="non-planar"):
        boundary_for_step(model, "LOAD")


@pytest.mark.parametrize(
    ("element_type", "node_line", "surface_label", "expected_edge"),
    [
        ("CPS3", "1, 1,2,3", "S1", ElementEdge(1, 0, (1, 2))),
        ("CPS3", "1, 1,2,3", "S2", ElementEdge(1, 1, (2, 3))),
        ("CPS3", "1, 1,2,3", "S3", ElementEdge(1, 2, (3, 1))),
        ("CPS6", "1, 1,2,3,4,5,6", "S1", ElementEdge(1, 0, (1, 4, 2))),
        ("CPS6", "1, 1,2,3,4,5,6", "S2", ElementEdge(1, 1, (2, 5, 3))),
        ("CPS6", "1, 1,2,3,4,5,6", "S3", ElementEdge(1, 2, (3, 6, 1))),
        ("CPS4", "1, 1,2,3,4", "S1", ElementEdge(1, 0, (1, 2))),
        ("CPS4", "1, 1,2,3,4", "S2", ElementEdge(1, 1, (2, 3))),
        ("CPS4", "1, 1,2,3,4", "S3", ElementEdge(1, 2, (3, 4))),
        ("CPS4", "1, 1,2,3,4", "S4", ElementEdge(1, 3, (4, 1))),
    ],
)
def test_abaqus_read_maps_2d_surface_labels_to_edges(
    tmp_path,
    element_type,
    node_line,
    surface_label,
    expected_edge,
):
    path = write_inp(
        tmp_path,
        "test_abaqus_2d_surface_labels.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            f"*Element, type={element_type}, elset=SOLID",
            node_line,
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=EDGE_SURFACE",
            f"SOLID, {surface_label}",
        ],
    )

    model = abaqus.read(path)

    assert model.edges["EDGE_SURFACE"].edges[0] == expected_edge


def test_abaqus_read_cps6_model_solves(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_cps6_solve.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 2., 0.",
            "3, 0., 1.",
            "4, 1., 0.",
            "5, 1., 0.5",
            "6, 0., 0.5",
            "*Element, type=CPS6, elset=SOLID",
            "1, 1,2,3,4,5,6",
            "*Elset, elset=SOLID",
            "1",
            "*Nset, nset=FIXED",
            "1,3,6",
            "*Nset, nset=TIP",
            "2",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Boundary",
            "FIXED, 1, 2, 0.",
            "*Cload",
            "TIP, 1, 1.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)
    result = static_linear.solve(model, "LOAD")

    assert model.mesh.elements[0].type == "Tri6"
    assert np.all(np.isfinite(result.U))


@pytest.mark.parametrize(
    ("surface_label", "expected_edge"),
    [
        ("S1", ElementEdge(1, 0, (1, 5, 2))),
        ("S2", ElementEdge(1, 1, (2, 6, 3))),
        ("S3", ElementEdge(1, 2, (3, 7, 4))),
        ("S4", ElementEdge(1, 3, (4, 8, 1))),
    ],
)
def test_abaqus_read_maps_cps8_surface_labels_to_edges(
    tmp_path,
    surface_label,
    expected_edge,
):
    path = write_inp(
        tmp_path,
        "test_abaqus_cps8_surface_labels.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 2., 0.",
            "3, 2., 2.",
            "4, 0., 2.",
            "5, 1., 0.",
            "6, 2., 1.",
            "7, 1., 2.",
            "8, 0., 1.",
            "*Element, type=CPS8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=EDGE_SURFACE",
            f"SOLID, {surface_label}",
        ],
    )

    model = abaqus.read(path)

    assert model.edges["EDGE_SURFACE"].edges[0] == expected_edge


@pytest.mark.parametrize(
    ("surface_name", "expected_face"),
    [
        ("FACE_1", ElementFace(1, 3, (1, 2, 3))),
        ("FACE_2", ElementFace(1, 2, (1, 2, 4))),
        ("FACE_3", ElementFace(1, 0, (2, 3, 4))),
        ("FACE_4", ElementFace(1, 1, (1, 3, 4))),
    ],
)
def test_abaqus_read_maps_tetra_face_labels_to_local_faces(tmp_path, surface_name, expected_face):
    path = write_inp(
        tmp_path,
        "test_abaqus_tet_face_labels.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=FACE_1",
            "SOLID, S1",
            "*Surface, type=ELEMENT, name=FACE_2",
            "SOLID, S2",
            "*Surface, type=ELEMENT, name=FACE_3",
            "SOLID, S3",
            "*Surface, type=ELEMENT, name=FACE_4",
            "SOLID, S4",
        ],
    )

    model = abaqus.read(path)

    assert model.surfaces[surface_name].faces[0] == expected_face


@pytest.mark.parametrize(
    ("surface_name", "expected_face"),
    [
        ("FACE_1", ElementFace(1, 0, (1, 4, 3, 2))),
        ("FACE_2", ElementFace(1, 1, (5, 6, 7, 8))),
        ("FACE_3", ElementFace(1, 2, (1, 2, 6, 5))),
        ("FACE_4", ElementFace(1, 5, (2, 3, 7, 6))),
        ("FACE_5", ElementFace(1, 3, (3, 4, 8, 7))),
        ("FACE_6", ElementFace(1, 4, (1, 5, 8, 4))),
    ],
)
def test_abaqus_read_maps_hex_face_labels_to_local_faces(tmp_path, surface_name, expected_face):
    path = write_inp(
        tmp_path,
        "test_abaqus_hex_face_labels.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=FACE_1",
            "SOLID, S1",
            "*Surface, type=ELEMENT, name=FACE_2",
            "SOLID, S2",
            "*Surface, type=ELEMENT, name=FACE_3",
            "SOLID, S3",
            "*Surface, type=ELEMENT, name=FACE_4",
            "SOLID, S4",
            "*Surface, type=ELEMENT, name=FACE_5",
            "SOLID, S5",
            "*Surface, type=ELEMENT, name=FACE_6",
            "SOLID, S6",
        ],
    )

    model = abaqus.read(path)

    assert model.surfaces[surface_name].faces[0] == expected_face


def test_abaqus_pressure_points_into_hex_element_for_all_faces(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_hex_pressure_direction.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=LOADED",
            "SOLID, S1",
            "SOLID, S2",
            "SOLID, S3",
            "SOLID, S4",
            "SOLID, S5",
            "SOLID, S6",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "LOADED, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    _assert_pressure_points_inward(model, bc)


@pytest.mark.parametrize(
    ("surface_name", "expected_face"),
    [
        ("FACE_1", ElementFace(1, 3, (1, 2, 3, 5, 6, 7))),
        ("FACE_2", ElementFace(1, 2, (1, 2, 4, 5, 9, 8))),
        ("FACE_3", ElementFace(1, 0, (2, 3, 4, 6, 10, 9))),
        ("FACE_4", ElementFace(1, 1, (1, 3, 4, 7, 10, 8))),
    ],
)
def test_abaqus_read_maps_tet10_face_labels_to_local_faces(tmp_path, surface_name, expected_face):
    path = write_inp(
        tmp_path,
        "test_abaqus_tet10_face_labels.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "5, 0.5, 0., 0.",
            "6, 0.5, 0.5, 0.",
            "7, 0., 0.5, 0.",
            "8, 0., 0., 0.5",
            "9, 0.5, 0., 0.5",
            "10, 0., 0.5, 0.5",
            "*Element, type=C3D10, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=FACE_1",
            "SOLID, S1",
            "*Surface, type=ELEMENT, name=FACE_2",
            "SOLID, S2",
            "*Surface, type=ELEMENT, name=FACE_3",
            "SOLID, S3",
            "*Surface, type=ELEMENT, name=FACE_4",
            "SOLID, S4",
        ],
    )

    model = abaqus.read(path)

    assert model.surfaces[surface_name].faces[0] == expected_face


def test_abaqus_pressure_points_into_tet10_element_for_all_faces(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_tet10_pressure_direction.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "5, 0.5, 0., 0.",
            "6, 0.5, 0.5, 0.",
            "7, 0., 0.5, 0.",
            "8, 0., 0., 0.5",
            "9, 0.5, 0., 0.5",
            "10, 0., 0.5, 0.5",
            "*Element, type=C3D10, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=LOADED",
            "SOLID, S1",
            "SOLID, S2",
            "SOLID, S3",
            "SOLID, S4",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "LOADED, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    _assert_pressure_points_inward(model, bc)


def test_abaqus_pressure_points_into_tetra_element_for_all_faces(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_tet_pressure_direction.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4, elset=SOLID",
            "1, 1,2,3,4",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=LOADED",
            "SOLID, S1",
            "SOLID, S2",
            "SOLID, S3",
            "SOLID, S4",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "LOADED, P, 2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    _assert_pressure_points_inward(model, bc)


def test_abaqus_read_accumulates_repeated_sets_and_scales_trvec_loads(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_sets_trvec.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "*Element, type=C3D8, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8",
            "*Nset, nset=FIXED",
            "1,4",
            "*Nset, nset=FIXED",
            "5,8",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=TIP_FACE",
            "SOLID, S6",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Dsload",
            "TIP_FACE, TRVEC, 10., 0., 0., -2.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    bc = boundary_for_step(model, "LOAD")
    assert model.node_sets["FIXED"].node_ids == (1, 4, 5, 8)
    assert bc.surface_tractions[0].vector == (0.0, 0.0, -10.0)


def test_abaqus_read_stores_output_requests_on_steps(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_output_requests.inp",
        [
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "*Element, type=CPS3, elset=SOLID",
            "1, 1,2,1",
            "*Material, name=STEEL",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=OUTPUT",
            "*Static",
            "*Output, field, variable=PRESELECT",
            "*Node Output",
            "U, RF",
            "*Element Output, directions=YES",
            "S, E",
            "*Output, history, variable=PRESELECT",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    outputs = model.steps[0].outputs
    assert outputs[0] == OutputRequest("field", "preselect", ("PRESELECT",), {"variable": "PRESELECT"})
    assert outputs[1] == OutputRequest("field", "node", ("U", "RF"), {})
    assert outputs[2] == OutputRequest("field", "element", ("S", "E"), {"directions": "YES"})
    assert outputs[3] == OutputRequest("history", "preselect", ("PRESELECT",), {"variable": "PRESELECT"})


def test_abaqus_parse_accumulates_wrapped_c3d20_connectivity(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_wrapped_c3d20.inp",
        [
            "*Element, type=C3D20, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "11,12,13,14,15,16,17,18,19,20",
        ],
    )

    deck = abaqus.parse_file(path)

    assert len(deck.elements) == 1
    assert deck.elements[0].id == 1
    assert deck.elements[0].type == "C3D20"
    assert deck.elements[0].node_ids == tuple(range(1, 21))
    assert deck.element_sets["SOLID"] == [1]


def test_abaqus_parse_rejects_incomplete_wrapped_c3d20_connectivity(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_incomplete_c3d20.inp",
        [
            "*Element, type=C3D20",
            "1, 1,2,3,4,5,6,7,8,9,10",
        ],
    )

    with pytest.raises(ValueError, match="Incomplete C3D20 connectivity record"):
        abaqus.parse_file(path)


def test_abaqus_parse_rejects_incomplete_c3d20_before_next_keyword(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_incomplete_c3d20_before_keyword.inp",
        [
            "*Element, type=C3D20",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "*Nset, nset=AFTER_ELEMENT",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="Incomplete C3D20 connectivity record"):
        abaqus.parse_file(path)


def test_abaqus_parse_consumes_two_wrapped_c3d20_records_from_one_block(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_two_wrapped_c3d20_records.inp",
        [
            "*Element, type=C3D20, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "11,12,13,14,15,16,17,18,19,20",
            "2, 21,22,23,24,25,26,27,28,29,30",
            "31,32,33,34,35,36,37,38,39,40",
        ],
    )

    deck = abaqus.parse_file(path)

    assert [(element.id, element.node_ids) for element in deck.elements] == [
        (1, tuple(range(1, 21))),
        (2, tuple(range(21, 41))),
    ]
    assert deck.element_sets["SOLID"] == [1, 2]


def test_abaqus_parse_clears_pending_values_between_supported_element_blocks(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_adjacent_supported_element_blocks.inp",
        [
            "*Element, type=C3D4, elset=TETS",
            "1, 1,2,3,4",
            "*Element, type=C3D8, elset=HEXES",
            "2, 5,6,7,8,9,10,11,12",
        ],
    )

    deck = abaqus.parse_file(path)

    assert [(element.id, element.type, element.node_ids) for element in deck.elements] == [
        (1, "C3D4", (1, 2, 3, 4)),
        (2, "C3D8", (5, 6, 7, 8, 9, 10, 11, 12)),
    ]
    assert deck.element_sets["TETS"] == [1]
    assert deck.element_sets["HEXES"] == [2]


def test_abaqus_parse_keeps_unsupported_element_type_records_independent(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_unsupported_element_type_records.inp",
        [
            "*Element, type=U99, elset=CUSTOM",
            "1, 10,11,12",
            "2, 20,21,22,23",
        ],
    )

    deck = abaqus.parse_file(path)

    assert [(element.id, element.node_ids) for element in deck.elements] == [
        (1, (10, 11, 12)),
        (2, (20, 21, 22, 23)),
    ]
    assert deck.element_sets["CUSTOM"] == [1, 2]


def test_abaqus_read_builds_and_solves_full_c3d20_model(tmp_path):
    path = write_inp(
        tmp_path,
        "test_abaqus_full_c3d20.inp",
        [
            "*Heading",
            "** one quadratic unit-cube element",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 1., 1., 0.",
            "4, 0., 1., 0.",
            "5, 0., 0., 1.",
            "6, 1., 0., 1.",
            "7, 1., 1., 1.",
            "8, 0., 1., 1.",
            "9, 0.5, 0., 0.",
            "10, 1., 0.5, 0.",
            "11, 0.5, 1., 0.",
            "12, 0., 0.5, 0.",
            "13, 0.5, 0., 1.",
            "14, 1., 0.5, 1.",
            "15, 0.5, 1., 1.",
            "16, 0., 0.5, 1.",
            "17, 0., 0., 0.5",
            "18, 1., 0., 0.5",
            "19, 1., 1., 0.5",
            "20, 0., 1., 0.5",
            "*Element, type=C3D20, elset=SOLID",
            "1, 1,2,3,4,5,6,7,8,9,10",
            "11,12,13,14,15,16,17,18,19,20",
            "*Nset, nset=FIXED",
            "1,4,5,8,12,16,17,20",
            "*Elset, elset=SOLID",
            "1",
            "*Surface, type=ELEMENT, name=FACE_1",
            "SOLID, S1",
            "*Surface, type=ELEMENT, name=FACE_2",
            "SOLID, S2",
            "*Surface, type=ELEMENT, name=FACE_3",
            "SOLID, S3",
            "*Surface, type=ELEMENT, name=FACE_4",
            "SOLID, S4",
            "*Surface, type=ELEMENT, name=FACE_5",
            "SOLID, S5",
            "*Surface, type=ELEMENT, name=FACE_6",
            "SOLID, S6",
            "*Material, name=STEEL",
            "*Density",
            "2.",
            "*Elastic",
            "210., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Step, name=LOAD",
            "*Static",
            "*Boundary",
            "FIXED, ENCASTRE",
            "*Dsload",
            "FACE_2, P, 1.",
            "FACE_2, TRSHR, 2., 1., 0., 0.",
            "*Dload",
            ", GRAV, 3., 0., -1., 0.",
            "*End Step",
        ],
    )

    model = abaqus.read(path)

    assert isinstance(model.mesh, Mesh3D)
    assert model.mesh.elements[0].type == "Hex20"
    assert model.mesh.elements[0].node_ids == list(range(1, 21))
    assert model.node_sets["FIXED"].node_ids == (1, 4, 5, 8, 12, 16, 17, 20)
    assert model.element_sets["SOLID"].element_ids == (1,)
    assert model.sections == [SectionAssignment("SOLID", "STEEL")]
    expected_faces = {
        "FACE_1": ElementFace(1, 0, (1, 4, 3, 2, 12, 11, 10, 9)),
        "FACE_2": ElementFace(1, 1, (5, 6, 7, 8, 13, 14, 15, 16)),
        "FACE_3": ElementFace(1, 2, (1, 2, 6, 5, 9, 18, 13, 17)),
        "FACE_4": ElementFace(1, 5, (2, 3, 7, 6, 10, 19, 14, 18)),
        "FACE_5": ElementFace(1, 3, (3, 4, 8, 7, 11, 20, 15, 19)),
        "FACE_6": ElementFace(1, 4, (1, 5, 8, 4, 17, 16, 20, 12)),
    }
    assert {
        name: surface.faces[0]
        for name, surface in model.surfaces.items()
    } == expected_faces
    assert all(
        len(surface.faces) == 1 and len(surface.faces[0].node_ids) == 8
        for surface in model.surfaces.values()
    )
    step = model.steps[0]
    assert step.name == "LOAD"
    assert step.procedure == "static"
    assert step.boundaries == (DisplacementConstraint("FIXED", 1, 3, 0.0),)
    assert step.surface_loads[0].surface == "FACE_2"
    assert step.surface_loads[0].load_type == "pressure"
    assert step.surface_loads[0].magnitude == pytest.approx(1.0)
    assert step.surface_loads[1].load_type == "shear_traction"
    assert step.surface_loads[1].magnitude == pytest.approx(2.0)
    assert step.gravity_loads == (GravityLoad((0.0, -3.0, 0.0)),)
    assert not any(name.startswith("__DLOAD") for name in model.surfaces)

    materials.apply_sections(model)
    assert model.mesh.elements[0].props["material"] == "STEEL"
    assert model.mesh.elements[0].props["E"] == 210.0
    assert model.mesh.elements[0].props["nu"] == 0.3
    assert model.mesh.elements[0].props["rho"] == 2.0

    result = static_linear.solve(model, "LOAD")
    assert np.all(np.isfinite(result.U))
    assert result.reactions[0::3].sum() == pytest.approx(-2.0, abs=1e-9)
    assert result.reactions[1::3].sum() == pytest.approx(6.0, abs=1e-9)
    assert result.reactions[2::3].sum() == pytest.approx(1.0, abs=1e-9)
