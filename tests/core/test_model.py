from copy import copy, deepcopy
from dataclasses import replace
import pickle

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
    OutputSourceEvidence,
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


def test_output_request_preserves_exact_variable_spelling_order_and_duplicates():
    request = OutputRequest(
        "FIELD",
        "NODE",
        (value for value in ("rf", "U", "rf", "CustomVariable")),
    )

    assert request.kind == "field"
    assert request.target == "node"
    assert request.variables == ("rf", "U", "rf", "CustomVariable")


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((1, "node", ()), "kind"),
        (("field", object(), ()), "target"),
        ((" ", "node", ()), "kind"),
        (("field", "\t", ()), "target"),
        (("field", "node", "U"), "variables"),
        (("field", "node", ("U", 1)), r"variables\[1\]"),
    ),
)
def test_output_request_rejects_nonexact_or_blank_intrinsic_strings(
    arguments,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        OutputRequest(*arguments)


def test_output_request_rejects_string_and_dict_subclasses():
    class StringSubclass(str):
        pass

    class DictSubclass(dict):
        pass

    with pytest.raises(TypeError, match="kind"):
        OutputRequest(StringSubclass("field"), "node", ("U",))
    with pytest.raises(TypeError, match=r"variables\[0\]"):
        OutputRequest("field", "node", (StringSubclass("U"),))
    with pytest.raises(TypeError, match="keys"):
        OutputRequest(
            "field",
            "node",
            ("U",),
            {StringSubclass("frequency"): 1},
        )
    with pytest.raises(TypeError, match="exact dict"):
        OutputRequest("field", "node", ("U",), DictSubclass())


def test_output_request_metadata_is_strict_deep_owned_and_immutable():
    thresholds = [0, 75, 100]
    nested = {"thresholds": thresholds}
    metadata = {
        "averaging": nested,
        "enabled": True,
        "note": None,
        "scale": 1.5,
    }

    request = OutputRequest("field", "element", ("S",), metadata)
    thresholds[1] = 80
    nested["late"] = "caller-owned"
    metadata["new"] = False

    assert request.metadata == {
        "averaging": {"thresholds": (0, 75, 100)},
        "enabled": True,
        "note": None,
        "scale": 1.5,
    }
    assert copy(request.metadata) is request.metadata
    assert deepcopy(request.metadata) is request.metadata
    assert pickle.loads(pickle.dumps(request)) == request

    with pytest.raises(TypeError):
        request.metadata["new"] = False
    with pytest.raises(TypeError):
        request.metadata["averaging"]["new"] = False
    with pytest.raises(TypeError):
        request.metadata["averaging"]["thresholds"][0] = 1


def test_output_request_can_reuse_its_already_frozen_metadata() -> None:
    request = OutputRequest(
        "field",
        "node",
        ("U",),
        {"frequency": 1},
    )

    updated = replace(request, variables=("RF",))

    assert updated.metadata is request.metadata
    assert updated.variables == ("RF",)


@pytest.mark.parametrize(
    "metadata",
    (
        {1: "non-string-key"},
        {"tuple": (1, 2)},
        {"custom": object()},
        {"nan": float("nan")},
        {"positive_infinity": float("inf")},
        {"negative_infinity": float("-inf")},
    ),
)
def test_output_request_metadata_rejects_values_outside_strict_finite_json(
    metadata,
):
    with pytest.raises((TypeError, ValueError)):
        OutputRequest("field", "node", ("U",), metadata)


def test_output_request_metadata_rejects_cyclic_json_containers():
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict

    for metadata in ({"cycle": cyclic_list}, cyclic_dict):
        with pytest.raises(ValueError, match="cyclic"):
            OutputRequest("field", "node", ("U",), metadata)


def test_output_source_evidence_is_exact_deeply_immutable_and_owned():
    parent_parameters = [["Frequency", "1"]]
    parent_flags = ["FIELD"]
    child_parameters = [["NSET", "Tip"]]
    child_flags = ["FutureFlag"]

    evidence = OutputSourceEvidence(
        "ABAQUS",
        parent_parameters,
        parent_flags,
        child_parameters,
        child_flags,
    )
    request = OutputRequest(
        "field",
        "node",
        ("u", "u"),
        {"frequency": "1"},
        evidence,
    )
    parent_parameters[0][1] = "2"
    parent_flags.append("LATE")
    child_parameters.clear()
    child_flags.clear()

    assert evidence.source_kind == "abaqus"
    assert evidence.parent_parameters == (("Frequency", "1"),)
    assert evidence.parent_flags == ("FIELD",)
    assert evidence.child_parameters == (("NSET", "Tip"),)
    assert evidence.child_flags == ("FutureFlag",)
    assert request.source_evidence is evidence
    assert deepcopy(request) == request

    with pytest.raises(TypeError, match="source_evidence"):
        OutputRequest("field", "node", ("U",), {}, object())


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
