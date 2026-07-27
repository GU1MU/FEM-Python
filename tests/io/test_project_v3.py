from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application.definitions import (
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.elements import BeamOrientation
from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.io.project_v3 import (
    ProjectV3DecodeError,
    decode_project_v3,
    dumps_project_v3,
    encode_project_v3,
    load_project_v3,
    loads_project_v3,
    save_project_v3,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings


def _wire_recipe() -> WireGeometry:
    return WireGeometry(
        "Wire",
        (
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0, 0.0),
            WirePoint("P3", 2.0, 0.5, 0.0),
        ),
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3"),
        ),
    )


def _wire_snapshot(
    line_element_type: str | None = "Truss2",
    *,
    with_definitions: bool = True,
) -> ProjectSnapshot:
    recipe = _wire_recipe()
    settings = (
        None
        if line_element_type is None
        else MeshSettings(
            0.5,
            cell_shape="line",
            local_controls=(
                LocalMeshControl(LogicalEntityRef("edge:M1"), 0.1),
            ),
            line_element_type=line_element_type,
        )
    )
    named_regions = (
        NamedRegion("Root", (LogicalEntityRef("point:P1"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:P3"),)),
        NamedRegion("Members", (LogicalEntityRef("edge:M1"),)),
    )
    materials = (
        MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),
    )
    sections = (
        SectionDefinition("TrussSection", "Steel", "truss", {"area": 0.01}),
    )
    assignments = (RegionAssignment("TrussSection", "DOMAIN"),)
    steps = (
        AnalysisStep(
            "Load",
            boundaries=(DisplacementConstraint("Root", 1, 3),),
            cloads=(NodalLoad("Tip", 2, -10.0),),
        ),
    )
    if line_element_type == "Beam2":
        sections = (
            SectionDefinition(
                "BeamSection",
                "Steel",
                "rectangle",
                {"width": 0.1, "height": 0.2},
            ),
        )
        assignments = (
            RegionAssignment(
                "BeamSection",
                "DOMAIN",
                BeamOrientation((0.0, 0.0, 1.0)),
            ),
        )
        steps = (
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint("Root", 1, 6),),
                cloads=(
                    NodalLoad("Tip", 2, -10.0),
                    NodalLoad("Tip", 3, 5.0),
                ),
                line_loads=(LineLoad("Members", (0.0, -1.0, 0.0)),),
            ),
        )
    if not with_definitions:
        materials = ()
        sections = ()
        assignments = ()
        steps = ()
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("WirePart", "WireBody"),),
        geometry_recipe=recipe,
        mesh_settings=settings,
        feature_history=derive_feature_history(recipe),
        named_regions=named_regions,
        material_definitions=materials,
        section_definitions=sections,
        region_assignments=assignments,
        analysis_definitions=steps,
    )


def _continuum_snapshot() -> ProjectSnapshot:
    recipe = RectangleGeometry("Plate", 4.0, 2.0)
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(0.5),
        feature_history=derive_feature_history(recipe),
    )


def test_v3_truss_wire_roundtrip_preserves_regions_definitions_and_links() -> None:
    original = _wire_snapshot()

    payload = encode_project_v3(original)
    reopened = decode_project_v3(payload)

    assert payload["schema"] == 3
    assert reopened == original
    assert dumps_project_v3(reopened) == dumps_project_v3(original)
    entities = payload["project"]["authoring"]["logical_topology"]["signature"][
        "entities"
    ]
    member = next(item for item in entities if item["logical_id"] == "edge:M1")
    assert member["topology_links"] == ["point:P1", "point:P2"]


def test_v3_beam_wire_roundtrip_preserves_orientation_and_line_load() -> None:
    original = _wire_snapshot("Beam2")

    reopened = loads_project_v3(dumps_project_v3(original))

    assert reopened == original
    assert reopened.region_assignments[0].beam_orientation == BeamOrientation(
        (0.0, 0.0, 1.0)
    )
    assert reopened.analysis_definitions[0].line_loads == (
        LineLoad("Members", (0.0, -1.0, 0.0)),
    )


def test_v3_incomplete_wire_persists_nullable_mesh_settings() -> None:
    original = _wire_snapshot(None, with_definitions=False)

    payload = encode_project_v3(original)
    assert payload["project"]["authoring"]["mesh_settings"] is None
    assert loads_project_v3(dumps_project_v3(original)) == original


def test_v3_continuum_requires_nullable_line_element_type_field() -> None:
    original = _continuum_snapshot()

    payload = encode_project_v3(original)
    settings = payload["project"]["authoring"]["mesh_settings"]
    assert settings["line_element_type"] is None
    assert decode_project_v3(payload) == original

    broken = deepcopy(payload)
    del broken["project"]["authoring"]["mesh_settings"]["line_element_type"]
    with pytest.raises(ProjectV3DecodeError, match="缺少必需字段.*line_element_type"):
        decode_project_v3(broken)


def test_v3_rejects_unknown_fields_and_stale_topology_links() -> None:
    payload = encode_project_v3(_wire_snapshot())

    unknown = deepcopy(payload)
    unknown["project"]["authoring"]["geometry"]["extra"] = True
    with pytest.raises(ProjectV3DecodeError, match="未知字段.*extra"):
        decode_project_v3(unknown)

    stale = deepcopy(payload)
    member = next(
        item
        for item in stale["project"]["authoring"]["logical_topology"][
            "signature"
        ]["entities"]
        if item["logical_id"] == "edge:M1"
    )
    member["topology_links"] = ["point:P2", "point:P3"]
    with pytest.raises(ProjectV3DecodeError, match="topology fingerprint"):
        decode_project_v3(stale)

    duplicate = deepcopy(payload)
    duplicate_entities = duplicate["project"]["authoring"]["logical_topology"][
        "signature"
    ]["entities"]
    duplicate_entities.append(deepcopy(duplicate_entities[0]))
    with pytest.raises(
        ProjectV3DecodeError,
        match="duplicates another topology entity",
    ):
        decode_project_v3(duplicate)


def test_v3_save_is_atomic_and_reopens_with_source_path(tmp_path) -> None:
    target = save_project_v3(tmp_path / "wire.femproj", _wire_snapshot())

    reopened = load_project_v3(target)
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert reopened.source_path == target
    assert reopened.geometry_recipe == _wire_recipe()
