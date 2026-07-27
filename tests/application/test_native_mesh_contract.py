from __future__ import annotations

import pytest

from fem.application.native_mesh_contract import (
    NativeMeshContractError,
    describe_native_mesh_contract,
    require_complete_native_mesh_contract,
)
from fem.application.capabilities import (
    AuthoringStatus,
    RegionRef,
    describe_native_authoring_capabilities,
)
from fem.application.definitions import NamedRegion
from fem.geometry.references import LogicalEntityRef
from fem.geometry import RectangleGeometry, WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings


def _wire() -> WireGeometry:
    return WireGeometry(
        "Wire",
        (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0)),
        (WireMember("M1", "P1", "P2"),),
    )


@pytest.mark.parametrize("line_element_type", ("Truss2", "Beam2"))
def test_line_contract_is_complete_and_registry_driven(line_element_type: str) -> None:
    contract = describe_native_mesh_contract(
        _wire(),
        MeshSettings(
            0.25,
            cell_shape="line",
            line_element_type=line_element_type,
        ),
    )

    assert contract.complete
    assert contract.dimension == 1
    assert contract.canonical_element_type == line_element_type
    assert contract.line_element_type == line_element_type
    assert require_complete_native_mesh_contract(
        _wire(),
        MeshSettings(0.25, cell_shape="line", line_element_type=line_element_type),
    ) == contract


def test_incomplete_wire_contract_does_not_infer_a_formulation() -> None:
    contract = describe_native_mesh_contract(_wire(), None)

    assert not contract.complete
    assert contract.canonical_element_type is None
    assert contract.line_element_type is None
    with pytest.raises(NativeMeshContractError, match="explicit line_element_type"):
        require_complete_native_mesh_contract(_wire(), None)


@pytest.mark.parametrize(
    ("settings", "dimension", "canonical"),
    (
        (MeshSettings(0.5, cell_shape="triangle"), 2, "Tri3"),
        (MeshSettings(0.5, cell_shape="quadrilateral"), 2, "Quad4"),
    ),
)
def test_continuum_contracts_preserve_existing_shape_mapping(
    settings: MeshSettings,
    dimension: int,
    canonical: str,
) -> None:
    contract = describe_native_mesh_contract(
        RectangleGeometry("Plate", 2.0, 1.0),
        settings,
    )

    assert contract.dimension == dimension
    assert contract.canonical_element_type == canonical
    assert contract.complete


def test_line_shape_is_rejected_for_a_continuum_recipe() -> None:
    with pytest.raises(NativeMeshContractError, match="not supported"):
        describe_native_mesh_contract(
            RectangleGeometry("Plate", 2.0, 1.0),
            MeshSettings(0.5, cell_shape="line", line_element_type="Truss2"),
        )


def test_incomplete_wire_capability_reports_a_blocking_formulation_diagnostic() -> None:
    report = describe_native_authoring_capabilities(_wire(), None)

    assert report.diagnostics[0].code == "native.line.formulation_required"
    assert report.status is AuthoringStatus.UNAVAILABLE
    assert report.canonical_element_types == ()
    assert report.output_request_catalog is None


def test_beam_wire_capabilities_expose_beam_member_and_point_operations() -> None:
    report = describe_native_authoring_capabilities(
        _wire(),
        MeshSettings(0.25, cell_shape="line", line_element_type="Beam2"),
        named_regions=(
            NamedRegion("Root", (LogicalEntityRef("point:P1"),)),
            NamedRegion("Member", (LogicalEntityRef("edge:M1"),)),
        ),
    )

    member = report.region(RegionRef("element_set", "Member"))
    root = report.region(RegionRef("node_set", "Root"))
    assert report.dofs_per_node == 6
    assert member.operation("section.assignment").status is AuthoringStatus.ENABLED
    assert member.operation("load.line.global").status is AuthoringStatus.ENABLED
    assert member.operation("load.line.local").status is AuthoringStatus.LIMITED
    assert root.operation("boundary.displacement").status is AuthoringStatus.ENABLED
