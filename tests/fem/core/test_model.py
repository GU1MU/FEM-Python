from dataclasses import replace

import pytest

from fem import materials, steps
from fem.core import model as core_model
from fem.post import result_region_key_for_element
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    Edge,
    EdgeLoad,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    OutputRequest,
    SectionAssignment,
    Surface,
)
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh, make_mixed_tri3_quad4_mesh, make_selection_hex_mesh


def test_model_records_preserve_set_and_topology_order_after_source_mutation():
    node_ids = [2, 1]
    element_ids = [1]
    edge_nodes = [2, 1]
    face_nodes = [1, 4, 3, 2]
    edge_records = [ElementEdge(1, 0, edge_nodes), ElementEdge(1, 1, [2, 3])]
    face_records = [ElementFace(1, 0, face_nodes), ElementFace(1, 1, [5, 6, 7, 8])]
    material = MaterialDefinition("STEEL", {"E": 210.0, "nu": 0.3})
    section = SectionAssignment("SOLID", "STEEL")
    model = FEMModel(
        mesh=make_selection_hex_mesh(),
        node_sets={"FIXED": NodeSet("FIXED", node_ids)},
        element_sets={"SOLID": ElementSet("SOLID", element_ids)},
        edges={"LINE_LOAD": Edge("LINE_LOAD", edge_records)},
        surfaces={"FACE_LOAD": Surface("FACE_LOAD", face_records)},
        materials={"STEEL": material},
        sections=[section],
        name="job",
    )
    for source in (node_ids, element_ids, edge_nodes, face_nodes, edge_records, face_records):
        source.clear()

    assert model.name == "job"
    assert model.node_sets["FIXED"].node_ids == (2, 1)
    assert model.element_sets["SOLID"].element_ids == (1,)
    assert model.edges["LINE_LOAD"].edges == (
        ElementEdge(1, 0, (2, 1)), ElementEdge(1, 1, (2, 3)),
    )
    assert model.surfaces["FACE_LOAD"].faces == (
        ElementFace(1, 0, (1, 4, 3, 2)), ElementFace(1, 1, (5, 6, 7, 8)),
    )
    assert model.materials["STEEL"] == material
    assert model.sections == [section]


def test_analysis_step_owns_ordered_record_sequences_and_metadata_snapshot():
    boundaries = [DisplacementConstraint("FIXED", 1, 3, 0.0)]
    loads = [NodalLoad("TIP", 3, -100.0), NodalLoad("TIP", 1, 20.0)]
    edge_loads = [EdgeLoad("LINE_LOAD", (1.0, 0.0), load_type="traction")]
    gravity_loads = [GravityLoad((0.0, 0.0, -9.81))]
    outputs = [OutputRequest("field", "node", ("U",))]
    metadata = {"nlgeom": "NO"}
    step = AnalysisStep(
        "load", boundaries=boundaries, cloads=loads, edge_loads=edge_loads,
        gravity_loads=gravity_loads, outputs=outputs, metadata=metadata,
    )
    for source in (boundaries, loads, edge_loads, gravity_loads, outputs):
        source.clear()
    metadata["nlgeom"] = "YES"

    assert step.boundaries == (DisplacementConstraint("FIXED", 1, 3, 0.0),)
    assert step.cloads == (NodalLoad("TIP", 3, -100.0), NodalLoad("TIP", 1, 20.0))
    assert step.edge_loads == (EdgeLoad("LINE_LOAD", (1.0, 0.0), load_type="traction"),)
    assert step.gravity_loads == (GravityLoad((0.0, 0.0, -9.81)),)
    assert step.outputs == (OutputRequest("field", "node", ("U",)),)
    assert step.metadata == {"nlgeom": "NO"}


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


def test_apply_sections_groups_equivalent_assignments_into_the_same_result_region():
    mesh = make_mixed_tri3_quad4_mesh()
    model = FEMModel(mesh=mesh)
    model.element_sets["triangles"] = ElementSet("triangles", (1,))
    model.element_sets["quadrilaterals"] = ElementSet("quadrilaterals", (2,))
    materials.add(
        model,
        MaterialDefinition(
            "steel",
            {"E": 210.0, "nu": 0.3, "metadata": {"grade": "A"}},
        ),
    )
    materials.assign(
        model,
        "steel",
        "triangles",
        section_type="plane",
        plane_type="stress",
        thickness=1.5,
    )
    materials.assign(
        model,
        "steel",
        "quadrilaterals",
        section_type="plane",
        plane_type="stress",
        thickness=1.5,
    )

    materials.apply_sections(model)

    first_region, second_region = [
        result_region_key_for_element(element) for element in mesh.elements
    ]
    assert first_region == second_region
    assert len({first_region, second_region}) == 1

    model.sections[1] = replace(
        model.sections[1],
        properties={**model.sections[1].properties, "thickness": 2.0},
    )
    materials.apply_sections(model)

    changed_region = result_region_key_for_element(mesh.elements[1])
    assert changed_region.material_signature == first_region.material_signature
    assert changed_region.section_signature != first_region.section_signature
    assert len({first_region, changed_region}) == 2


def test_model_element_info_raises_for_unknown_element_id():
    model = FEMModel(mesh=make_mixed_hex8_tet4_mesh())

    with pytest.raises(KeyError, match="element 99 is not defined"):
        core_model.model_element_info(model, 99)


def test_steps_add_edge_load_helpers():
    step = AnalysisStep("load")
    edge = Edge("TOP", [ElementEdge(1, 2, [3, 4])])

    traction = steps.edge_traction(step, edge, (1.0, -2.0))
    pressure = steps.edge_pressure(step, "TOP", 3.0)

    assert traction == EdgeLoad("TOP", (1.0, -2.0), load_type="traction")
    assert pressure == EdgeLoad("TOP", magnitude=3.0, load_type="pressure")
    assert step.edge_loads == (traction, pressure)


def test_gravity_load_owns_acceleration_and_step_helper_appends_records():
    acceleration = [0.0, -9.81, 0.0]
    step = AnalysisStep("load", gravity_loads=[GravityLoad(acceleration)])
    acceleration[1] = 0.0

    targeted = steps.gravity(step, (1.0, 0.0, 0.0), target="BALLAST")

    assert step.gravity_loads[0].acceleration == (0.0, -9.81, 0.0)
    assert targeted == GravityLoad((1.0, 0.0, 0.0), "BALLAST")
    assert step.gravity_loads == (
        GravityLoad((0.0, -9.81, 0.0)),
        targeted,
    )
