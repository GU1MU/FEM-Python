from __future__ import annotations

from types import SimpleNamespace

import pytest

from fem.application import (
    AuthoringStatus,
    RegionRef,
    describe_model_capabilities,
    describe_native_authoring_capabilities,
    describe_region_capabilities,
    require_region_kind,
)
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    NodeSet,
    Surface,
)
from fem.mesh.settings import MeshSettings


def _element(
    element_id: int,
    element_type: str,
    node_ids: tuple[int, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=element_id,
        type=element_type,
        node_ids=node_ids,
        props={},
    )


def _model(*elements: SimpleNamespace) -> SimpleNamespace:
    element_ids = tuple(element.id for element in elements)
    node_ids = tuple(
        dict.fromkeys(
            node_id for element in elements for node_id in element.node_ids
        )
    )
    return SimpleNamespace(
        mesh=SimpleNamespace(elements=list(elements)),
        node_sets={"SAME": NodeSet("SAME", node_ids[:1])},
        element_sets={"SAME": ElementSet("SAME", element_ids)},
        edges={
            "SAME": Edge(
                "SAME",
                (
                    ElementEdge(
                        elements[0].id,
                        0,
                        elements[0].node_ids[:2],
                    ),
                ),
            )
        },
        surfaces={
            "SAME": Surface(
                "SAME",
                (
                    ElementFace(
                        elements[0].id,
                        0,
                        elements[0].node_ids[:3],
                    ),
                ),
            )
        },
        metadata={},
    )


def test_line_element_report_separates_topology_and_spatial_dimension() -> None:
    model = _model(_element(1, "Beam2", (1, 2)))

    report = describe_model_capabilities(model)
    region = report.region(RegionRef("element_set", "SAME"))

    assert report.canonical_element_types == ("Beam2",)
    assert report.topological_dimension == 1
    assert report.spatial_dimension == 3
    assert report.dof_labels == (
        "U1",
        "U2",
        "U3",
        "UR1",
        "UR2",
        "UR3",
    )
    assert report.force_labels == (
        "Fx",
        "Fy",
        "Fz",
        "Mx",
        "My",
        "Mz",
    )
    assert region.section_presets == (
        "rectangle",
        "solid_circle",
        "hollow_circle",
    )
    assert region.distributed_load_kinds == ("line",)
    assert region.status is AuthoringStatus.LIMITED
    assert {
        item.code for item in region.diagnostics
    } == {"beam.orientation.assumed"}


def test_same_name_regions_remain_distinct_typed_references() -> None:
    model = _model(_element(1, "Tri3", (1, 2, 3)))

    reports = {
        kind: describe_region_capabilities(
            model,
            RegionRef(kind, "SAME"),
        )
        for kind in ("node_set", "element_set", "edge", "surface")
    }

    assert {item.region for item in reports.values()} == {
        RegionRef("node_set", "SAME"),
        RegionRef("element_set", "SAME"),
        RegionRef("edge", "SAME"),
        RegionRef("surface", "SAME"),
    }
    assert all(item.compatible for item in reports.values())


def test_same_family_mixed_region_uses_safe_common_contract() -> None:
    model = _model(
        _element(1, "Tri3", (1, 2, 3)),
        _element(2, "Quad4", (2, 3, 4, 5)),
    )

    region = describe_region_capabilities(
        model,
        RegionRef("element_set", "SAME"),
    )

    assert region.canonical_element_types == ("Tri3", "Quad4")
    assert not region.homogeneous
    assert region.compatible
    assert region.families == ("plane_continuum",)
    assert region.section_families == ("solid",)
    assert region.distributed_load_kinds == ("edge",)
    assert region.diagnostics == ()


def test_cross_family_or_dof_conflict_fails_closed() -> None:
    model = _model(
        _element(1, "Truss2", (1, 2)),
        _element(2, "Beam2", (2, 3)),
    )

    region = describe_region_capabilities(
        model,
        RegionRef("element_set", "SAME"),
    )

    assert not region.compatible
    assert region.status is AuthoringStatus.UNAVAILABLE
    assert region.section_families == ()
    assert region.distributed_load_kinds == ()
    assert region.diagnostics[0].code == (
        "model.capability.unsupported_mix"
    )


def test_unknown_region_and_element_type_fail_closed() -> None:
    model = _model(_element(1, "Future42", (1, 2)))

    report = describe_model_capabilities(model)
    missing = report.region(RegionRef("element_set", "MISSING"))

    assert not report.compatible
    assert report.status is AuthoringStatus.UNAVAILABLE
    assert report.diagnostics[0].code == (
        "model.capability.unsupported_mix"
    )
    assert not missing.compatible
    assert missing.diagnostics[0].code == "step.reference.invalid"


def test_authoring_command_validates_region_kind_before_string_dto() -> None:
    region = RegionRef("element_set", "BEAMS")

    assert require_region_kind(region, "element_set") == "BEAMS"
    with pytest.raises(ValueError, match="requires region kind"):
        require_region_kind(RegionRef("node_set", "BEAMS"), "element_set")
    with pytest.raises(TypeError, match="RegionRef"):
        require_region_kind("BEAMS", "element_set")


@pytest.mark.parametrize(
    ("shape", "order", "canonical", "dimension"),
    (
        ("triangle", 1, "Tri3", 2),
        ("quadrilateral", 2, "Quad8", 2),
        ("tetrahedron", 2, "Tet10", 3),
        ("hexahedron", 1, "Hex8", 3),
    ),
)
def test_native_authoring_uses_catalog_for_mesh_settings(
    shape: str,
    order: int,
    canonical: str,
    dimension: int,
) -> None:
    report = describe_native_authoring_capabilities(
        object(),
        MeshSettings(1.0, order=order, cell_shape=shape),
    )

    assert report.canonical_element_types == (canonical,)
    assert report.topological_dimension == dimension
    assert report.spatial_dimension == dimension
    assert report.operation("section.create").status is (
        AuthoringStatus.ENABLED
    )


def test_output_request_create_is_unavailable_with_stable_reason() -> None:
    report = describe_model_capabilities(
        _model(_element(1, "Tri3", (1, 2, 3)))
    )

    create = report.operation("output_request.create")
    existing = report.operation("output_request.existing")

    assert create.status is AuthoringStatus.UNAVAILABLE
    assert existing.status is AuthoringStatus.READ_ONLY
    assert create.diagnostics[0].code == "output.request.not_executed"
