from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fem.application.capabilities as capabilities_module
from fem.abaqus import read
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
from fem.core.model import LineLoad, SectionAssignment
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY
from fem.mesh.settings import MeshSettings


_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


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
    assert region.status is AuthoringStatus.ENABLED
    assert region.diagnostics == ()
    assert region.status_for("section.rectangle") is (
        AuthoringStatus.LIMITED
    )
    assert region.status_for("load.line.local") is (
        AuthoringStatus.LIMITED
    )
    assert region.status_for("load.line.global") is (
        AuthoringStatus.ENABLED
    )
    assert region.status_for("section.solid_circle") is (
        AuthoringStatus.ENABLED
    )
    assert {
        item.code
        for item in region.diagnostics_for("section.rectangle")
    } == {"beam.orientation.assumed"}


def test_installed_rectangle_orientation_is_contextual() -> None:
    model = read(_FIXTURES / "beam2_rectangle_uniform_load.inp")
    target = RegionRef("element_set", "BEAM")
    model.sections[0].properties.pop(
        BEAM_LOCAL_Y_REFERENCE_KEY,
        None,
    )

    automatic = describe_model_capabilities(model)
    automatic_region = automatic.region(target)

    assert automatic.status is AuthoringStatus.LIMITED
    assert automatic_region.status is AuthoringStatus.LIMITED
    assert {
        item.code for item in automatic.diagnostics
    } == {"beam.orientation.assumed"}

    model.sections[0].properties[
        BEAM_LOCAL_Y_REFERENCE_KEY
    ] = (0.0, 1.0, 0.0)
    explicit = describe_model_capabilities(model)
    explicit_region = explicit.region(target)

    assert explicit.status is AuthoringStatus.ENABLED
    assert explicit_region.status is AuthoringStatus.ENABLED
    assert explicit_region.status_for("section.rectangle") is (
        AuthoringStatus.ENABLED
    )
    assert explicit_region.status_for("load.line.local") is (
        AuthoringStatus.ENABLED
    )


def test_circle_with_global_load_has_no_irrelevant_orientation_warning() -> None:
    model = read(_FIXTURES / "beam2_rectangle_uniform_load.inp")
    original = model.sections[0]
    model.sections = [
        SectionAssignment(
            original.element_set,
            original.material,
            "solid_circle",
            {"radius": 0.05},
        )
    ]
    selected = next(
        item for item in model.steps if item.name == "UniformLoad"
    )
    selected.line_loads = (
        LineLoad("BEAM", (0.0, -500.0, 0.0), "global"),
    )

    report = describe_model_capabilities(model)
    region = report.region(RegionRef("element_set", "BEAM"))

    assert report.status is AuthoringStatus.ENABLED
    assert region.status is AuthoringStatus.ENABLED
    assert report.diagnostics == ()
    assert region.status_for("section.rectangle") is (
        AuthoringStatus.LIMITED
    )
    assert region.status_for("load.line.local") is (
        AuthoringStatus.LIMITED
    )


def test_mixed_explicit_and_automatic_region_reports_only_automatic_ids() -> None:
    model = read(_FIXTURES / "beam2_rectangle_uniform_load.inp")
    model.element_sets["HEAD"] = ElementSet("HEAD", (1, 2))
    model.element_sets["TAIL"] = ElementSet("TAIL", (3, 4))
    original = model.sections[0]
    properties = dict(original.properties)
    properties.pop(BEAM_LOCAL_Y_REFERENCE_KEY, None)
    model.sections = [
        SectionAssignment(
            "HEAD",
            original.material,
            "rectangle",
            {
                **properties,
                BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
            },
        ),
        SectionAssignment(
            "TAIL",
            original.material,
            "rectangle",
            properties,
        ),
    ]

    region = describe_region_capabilities(
        model,
        RegionRef("element_set", "BEAM"),
    )
    warning = next(
        item
        for item in region.diagnostics_for("load.line.local")
        if item.code == "beam.orientation.assumed"
    )

    assert region.status_for("load.line.local") is (
        AuthoringStatus.LIMITED
    )
    assert warning.details_dict()["element_ids"] == (3, 4)


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
    assert report.operation("output_request.create").status is (
        AuthoringStatus.ENABLED
    )
    assert report.operation("output_request.existing").status is (
        AuthoringStatus.READ_ONLY
    )


def test_native_and_realized_output_support_share_result_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    original = capabilities_module.ResultCapabilityCatalog.from_profile

    def from_profile(profile):
        catalog = original(profile)
        observed.append(catalog)
        return catalog

    monkeypatch.setattr(
        capabilities_module.ResultCapabilityCatalog,
        "from_profile",
        from_profile,
    )
    realized = _model(_element(1, "Tri3", (1, 2, 3)))
    realized.mesh.dofs_per_node = 2

    native_report = describe_native_authoring_capabilities(
        object(),
        MeshSettings(1.0, order=1, cell_shape="triangle"),
    )
    realized_report = describe_model_capabilities(realized)

    assert len(observed) == 2
    assert observed[0].profile == observed[1].profile
    assert observed[0].entries == observed[1].entries
    assert native_report.operation(
        "output_request.create"
    ).status is AuthoringStatus.ENABLED
    assert realized_report.operation(
        "output_request.create"
    ).status is AuthoringStatus.ENABLED
    assert native_report.operation(
        "output_request.existing"
    ).status is AuthoringStatus.READ_ONLY
    assert realized_report.operation(
        "output_request.existing"
    ).status is AuthoringStatus.READ_ONLY


def test_output_request_create_is_enabled_when_catalog_has_candidates() -> None:
    model = _model(_element(1, "Tri3", (1, 2, 3)))
    model.mesh.dofs_per_node = 2
    report = describe_model_capabilities(model)

    create = report.operation("output_request.create")
    existing = report.operation("output_request.existing")

    assert create.status is AuthoringStatus.ENABLED
    assert existing.status is AuthoringStatus.READ_ONLY
    assert create.diagnostics == ()
    assert existing.diagnostics == ()


def test_output_capability_uses_canonical_result_projection_with_open_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = read(_FIXTURES / "truss2_tension.inp")
    observed = []
    original = capabilities_module.project_output_request

    def project(request, catalog, *, request_index):
        projection = original(
            request,
            catalog,
            request_index=request_index,
        )
        observed.append((catalog, projection))
        return projection

    monkeypatch.setattr(
        capabilities_module,
        "project_output_request",
        project,
    )

    report = describe_model_capabilities(model)
    create = report.operation("output_request.create")
    existing = report.operation("output_request.existing")

    assert capabilities_module.output_execution_installed is True
    assert len(observed) == 2
    assert len({id(catalog) for catalog, _projection in observed}) == 1
    assert observed[0][0].profile.family.value == "truss"
    assert all(projection.executable for _catalog, projection in observed)
    assert create.status is AuthoringStatus.ENABLED
    assert existing.status is AuthoringStatus.READ_ONLY
    assert create.diagnostics == ()
    assert existing.diagnostics == ()
