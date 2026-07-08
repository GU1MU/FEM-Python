import importlib
import importlib.util
import sys

import numpy as np
import pytest

from fem import materials, selection
from fem.core import mesh as core_mesh
from fem.core import dof
from fem.core import model as core_model
from fem.core.dof import DofMap
from fem.core.mesh import HexMesh3D, Node2D, Node3D, PlaneMesh2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementFace,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    SectionAssignment,
    Surface,
)
from tests.helpers.mesh_builders import (
    make_dof_order_meshes,
    make_minimal_hex_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri3_quad4_mesh,
    make_selection_hex_mesh,
    make_selection_mixed_plane_mesh,
    make_selection_quad_mesh,
)


def test_dof_map_handles_mesh_nodes_independent_of_dimension():
    nodes = [Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)]

    dof_map = DofMap.from_nodes(nodes, dofs_per_node=2)

    assert dof_map.node_ids == [10, 20]
    assert dof_map.node_dofs(10) == [0, 1]
    assert dof_map.node_dofs(20) == [2, 3]
    assert dof_map.element_dofs([20, 10]) == [2, 3, 0, 1]


def test_dof_map_rejects_duplicate_nodes_and_invalid_components():
    nodes = [Node2D(1, 0.0, 0.0), Node2D(1, 1.0, 0.0)]

    with pytest.raises(ValueError):
        DofMap.from_nodes(nodes, dofs_per_node=2)

    dof_map = DofMap.from_nodes([Node2D(1, 0.0, 0.0)], dofs_per_node=2)
    with pytest.raises(IndexError):
        dof_map.global_dof(1, -1)
    with pytest.raises(IndexError):
        dof_map.global_dof(1, 2)


def test_core_dof_exposes_dof_map_without_legacy_dof_manager():
    assert dof.DofMap is DofMap
    assert not hasattr(dof, "DofManager2D")
    assert not hasattr(dof, "DofManager3D")

    sys.modules.pop("fem.dof_manager", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.dof_manager")


def test_meshes_expose_dof_interface_without_dof_manager_access():
    assert core_mesh.HexMesh3D is HexMesh3D
    sys.modules.pop("fem.mesh", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fem.mesh")

    for mesh in make_dof_order_meshes():
        assert hasattr(mesh, "dof_map")
        assert not hasattr(mesh, "dof_manager")
        assert mesh.node_ids == [10, 20]
        assert mesh.node_dofs(10) == list(range(mesh.dofs_per_node))
        assert mesh.element_dofs(mesh.elements[0])[:mesh.dofs_per_node] == mesh.node_dofs(20)


def test_core_model_stores_sets_surfaces_materials_and_sections():
    mesh = make_minimal_hex_mesh()
    node_set = NodeSet("FIXED", [1, 2])
    element_set = ElementSet("SOLID", [1])
    surface = Surface("LOAD", [ElementFace(1, 0, [1, 2])])
    material = MaterialDefinition("STEEL", {"E": 210.0, "nu": 0.3})
    section = SectionAssignment("SOLID", "STEEL")
    model = FEMModel(
        mesh=mesh,
        node_sets={node_set.name: node_set},
        element_sets={element_set.name: element_set},
        surfaces={surface.name: surface},
        materials={material.name: material},
        sections=[section],
        name="job",
    )

    assert model.name == "job"
    assert model.node_sets["FIXED"].node_ids == (1, 2)
    assert model.element_sets["SOLID"].element_ids == (1,)
    assert model.surfaces["LOAD"].faces[0].local_index == 0
    assert model.materials["STEEL"].properties["E"] == 210.0
    assert model.sections[0].element_set == "SOLID"


def test_core_model_stores_analysis_steps():
    step = AnalysisStep(
        "load",
        procedure="static",
        boundaries=[DisplacementConstraint("FIXED", 1, 3, 0.0)],
        cloads=[NodalLoad("TIP", 3, -100.0)],
        metadata={"nlgeom": "NO"},
    )
    model = FEMModel(mesh=make_minimal_hex_mesh(), steps=[step])

    assert model.steps[0].name == "load"
    assert model.steps[0].boundaries[0].target == "FIXED"
    assert model.steps[0].cloads[0].component == 3
    assert model.steps[0].metadata["nlgeom"] == "NO"


def test_model_element_info_returns_type_material_and_properties_by_element_id():
    mesh = make_mixed_hex8_tet4_mesh()
    model = FEMModel(mesh=mesh, name="mixed_info")
    model.element_sets["hexes"] = ElementSet("hexes", (1,))
    model.element_sets["tets"] = ElementSet("tets", (2,))
    steel = materials.linear_elastic.material("steel", E=210.0, nu=0.3)
    aluminum = materials.linear_elastic.material("aluminum", E=120.0, nu=0.25)
    materials.add(model, steel)
    materials.add(model, aluminum)
    materials.assign(model, "steel", "hexes", rho=7.85)
    materials.assign(model, "aluminum", "tets", rho=2.7)

    info = core_model.model_element_info(model, 2)

    assert isinstance(info, core_model.ElementInfo)
    assert info.elem_id == 2
    assert info.element_type == "Tet4"
    assert info.type == "Tet4"
    assert info.node_ids == (2, 9, 3, 6)
    assert info.material == "aluminum"
    assert info.section_type == "solid"
    assert info.element_sets == ("tets",)
    assert info.properties["material"] == "aluminum"
    assert info.properties["E"] == 120.0
    assert info.properties["nu"] == 0.25
    assert info.properties["rho"] == 2.7
    assert "material" not in mesh.elements[1].props


def test_model_element_info_raises_for_unknown_element_id():
    model = FEMModel(mesh=make_mixed_hex8_tet4_mesh())

    with pytest.raises(KeyError, match="element 99 is not defined"):
        core_model.model_element_info(model, 99)


def test_core_model_has_no_solver_or_boundary_pipeline_methods():
    forbidden = (
        "boundary",
        "from_mesh",
        "add_node_set",
        "add_element_set",
        "add_surface",
        "add_material",
        "assign_section",
        "add_step",
        "add_displacement",
        "add_nodal_load",
        "add_surface_traction",
        "add_surface_pressure",
        "add_output_request",
        "get_step",
        "boundary_for_step",
        "assemble_stiffness",
        "load_vector",
        "solve",
        "run",
        "run_all",
    )
    for name in forbidden:
        assert not hasattr(FEMModel, name)


def test_selection_package_exposes_nodes_edges_and_faces_only():
    assert hasattr(selection, "nodes")
    assert hasattr(selection, "edges")
    assert hasattr(selection, "elements")
    assert hasattr(selection, "faces")
    assert importlib.util.find_spec("fem.helper") is None


def test_nodes_select_2d_and_3d_coordinates():
    mesh2d = PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 2.0),
        ],
        elements=[],
    )
    mesh3d = HexMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 1.0, 2.0, 3.0),
        ],
        elements=[],
    )

    assert selection.nodes.by_x(mesh2d, 1.0) == [2, 3]
    assert selection.nodes.by_coord(mesh2d, x=1.0, y=2.0) == [3]
    assert selection.nodes.by_z(mesh3d, 3.0) == [3]
    assert selection.nodes.by_coord(mesh3d, x=1.0, z=0.0) == [2]


def test_edges_select_boundary_edges_by_coordinate():
    mesh = make_selection_quad_mesh()

    assert len(selection.edges.boundary(mesh)) == 4
    assert selection.edges.by_x(mesh, 0.0) == [(1, 3, [4, 1])]
    assert selection.edges.by_y(mesh, 1.0) == [(1, 2, [3, 4])]


def test_elements_select_by_id_and_type_and_build_sets():
    mesh = make_selection_mixed_plane_mesh()

    assert selection.elements.all(mesh) == [1, 2]
    assert selection.elements.by_type(mesh, "quad4") == [2]
    assert selection.elements.by_ids(mesh, [2, 3]) == [2]
    assert selection.elements.set_by_type(mesh, "QUADS", "quad4") == ElementSet("QUADS", (2,))


def test_faces_select_boundary_faces_by_coordinate():
    mesh = make_selection_hex_mesh()

    assert len(selection.faces.boundary(mesh)) == 6
    assert selection.faces.by_z(mesh, 4.0) == [(1, 1, [5, 6, 7, 8])]
    assert selection.faces.by_x(mesh, 2.0) == [(1, 5, [2, 3, 7, 6])]


def test_selection_can_build_model_sets_and_surfaces():
    mesh = make_selection_hex_mesh()

    fixed = selection.nodes.set_by_x(mesh, "FIXED", 0.0)
    load_surface = selection.faces.surface_by_x(mesh, "LOAD", 2.0)

    assert isinstance(fixed, NodeSet)
    assert fixed.node_ids == (1, 4, 5, 8)
    assert isinstance(load_surface, Surface)
    assert load_surface.faces == (ElementFace(1, 5, (2, 3, 7, 6)),)


def test_materials_package_exposes_linear_elastic_module_only():
    assert hasattr(materials, "linear_elastic")
    assert not hasattr(materials, "compute_plane_stress_matrix")
    assert not hasattr(materials, "compute_plane_strain_matrix")
    assert not hasattr(materials, "compute_plane_elastic_matrix")
    assert not hasattr(materials, "compute_3d_elastic_matrix")


def test_linear_elastic_constitutive_matrices():
    E = 210.0
    nu = 0.3

    plane_stress = materials.linear_elastic.plane_stress_matrix(E, nu)
    plane_matrix = materials.linear_elastic.plane_matrix(E, nu, "stress")
    solid = materials.linear_elastic.solid_3d_matrix(E, nu)

    assert plane_stress.shape == (3, 3)
    assert np.allclose(plane_matrix, plane_stress)
    assert solid.shape == (6, 6)
    assert plane_stress[0, 1] == pytest.approx(E * nu / (1.0 - nu ** 2))


def test_mixed_hex8_tet4_mesh_keeps_element_types_and_3d_dofs():
    mesh = make_mixed_hex8_tet4_mesh()

    assert mesh.dofs_per_node == 3
    assert [elem.type for elem in mesh.elements] == ["Hex8", "Tet4"]
    assert mesh.elements[0].id == 1
    assert mesh.elements[1].id == 2
    assert mesh.num_dofs == 27


def test_mixed_tri3_quad4_mesh_keeps_element_types_and_2d_dofs():
    mesh = make_mixed_tri3_quad4_mesh()

    assert mesh.dofs_per_node == 2
    assert [elem.type for elem in mesh.elements] == ["Tri3Plane", "Quad4Plane"]
    assert mesh.num_dofs == 10
