from __future__ import annotations

import numpy as np
import pytest

from fem.application import (
    ModelDefinitions,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
    compile_model_definitions,
)
from fem.application.preflight import run_static_preflight
from fem.application.preprocessing import generate_fem_model
from fem.application.feature_history import derive_feature_history
from fem.application.results import (
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
    build_result_provider,
    prepare_result_export_snapshot,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.io.result_csv import read_result_csv, write_result_csv
from fem.io.result_vtk import read_result_vtk, write_result_vtk
from fem.io.project_v3 import load_project_v3, save_project_v3
from fem.mesh.settings import MeshSettings
from fem.solvers.static_linear import solve


def _connected_wire() -> WireGeometry:
    return WireGeometry(
        "Connected",
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


def _wire_regions() -> tuple[NamedRegion, ...]:
    return (
        NamedRegion("Root", (LogicalEntityRef("point:P1"),)),
        NamedRegion("Joint", (LogicalEntityRef("point:P2"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:P3"),)),
        NamedRegion("Member1", (LogicalEntityRef("edge:M1"),)),
        NamedRegion("Member2", (LogicalEntityRef("edge:M2"),)),
    )


def test_connected_wire_truss_has_stable_point_and_member_sets(real_gmsh) -> None:
    del real_gmsh

    model = generate_fem_model(
        _connected_wire(),
        MeshSettings(0.4, cell_shape="line", line_element_type="Truss2"),
        named_regions=_wire_regions(),
    )

    assert model.mesh.dofs_per_node == 3
    assert {element.type for element in model.mesh.elements} == {"Truss2"}
    assert len(model.node_sets["Joint"].node_ids) == 1
    member_ids = {
        *model.element_sets["Member1"].element_ids,
        *model.element_sets["Member2"].element_ids,
    }
    assert member_ids == set(model.element_sets["DOMAIN"].element_ids)
    assert set(model.element_sets["Member1"].element_ids).isdisjoint(
        model.element_sets["Member2"].element_ids
    )
    assert not model.edges
    assert not model.surfaces


def test_coincident_disconnected_wire_points_remain_distinct(real_gmsh) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "CoincidentDisconnected",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 0.0, 0.0, 0.0),
            WirePoint("C", 1.0, 0.0, 0.0),
            WirePoint("D", 1.0, 1.0, 0.0),
        ),
        (
            WireMember("AC", "A", "C"),
            WireMember("BD", "B", "D"),
        ),
    )
    regions = (
        NamedRegion("A", (LogicalEntityRef("point:A"),)),
        NamedRegion("B", (LogicalEntityRef("point:B"),)),
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(0.4, cell_shape="line", line_element_type="Truss2"),
        named_regions=regions,
    )

    assert model.node_sets["A"].node_ids
    assert model.node_sets["B"].node_ids
    assert set(model.node_sets["A"].node_ids).isdisjoint(
        model.node_sets["B"].node_ids
    )


def test_crossing_without_declared_joint_does_not_create_a_shared_member_set(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "Crossing",
        (
            WirePoint("A", -1.0, -1.0, 0.0),
            WirePoint("B", 1.0, 1.0, 0.0),
            WirePoint("C", -1.0, 1.0, 0.0),
            WirePoint("D", 1.0, -1.0, 0.0),
        ),
        (
            WireMember("AB", "A", "B"),
            WireMember("CD", "C", "D"),
        ),
    )
    regions = (
        NamedRegion("AB", (LogicalEntityRef("edge:AB"),)),
        NamedRegion("CD", (LogicalEntityRef("edge:CD"),)),
    )

    model = generate_fem_model(
        recipe,
        MeshSettings(0.5, cell_shape="line", line_element_type="Beam2"),
        named_regions=regions,
    )

    assert model.mesh.dofs_per_node == 6
    assert {element.type for element in model.mesh.elements} == {"Beam2"}
    assert set(model.element_sets["AB"].element_ids).isdisjoint(
        model.element_sets["CD"].element_ids
    )
    assert not model.edges
    assert not model.surfaces


def test_truss2_headless_vertical_slice_matches_axial_bar_solution(real_gmsh) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "Bar",
        (
            WirePoint("RootPoint", 0.0, 0.0, 0.0),
            WirePoint("TipPoint", 1.0, 0.0, 0.0),
        ),
        (WireMember("BarMember", "RootPoint", "TipPoint"),),
    )
    named_regions = (
        NamedRegion("Root", (LogicalEntityRef("point:RootPoint"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:TipPoint"),)),
    )
    step = AnalysisStep(
        "Load",
        boundaries=(
            DisplacementConstraint("Root", 1, 3),
            DisplacementConstraint("Tip", 2, 3),
        ),
        cloads=(NodalLoad("Tip", 1, 10.0),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(2.0, cell_shape="line", line_element_type="Truss2"),
        named_regions=named_regions,
    )
    definitions = ModelDefinitions(
        materials=(MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        sections=(
            SectionDefinition("BarSection", "Steel", "truss", {"area": 0.01}),
        ),
        assignments=(RegionAssignment("BarSection", "DOMAIN"),),
        steps=(step,),
    )
    compiled = compile_model_definitions(model, definitions).require_model()
    report = run_static_preflight(compiled, "Load")
    assert report.passed, tuple(diagnostic.message for diagnostic in report.errors)

    result = solve(compiled, "Load")
    tip_id = compiled.node_sets["Tip"].node_ids[0]
    expected = 10.0 * 1.0 / (210000.0 * 0.01)
    assert result.nodal_displacement(tip_id, component=1) == pytest.approx(expected)
    assert np.isfinite(result.U).all()


def test_beam2_headless_vertical_slice_has_spatial_dofs_and_bending_response(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "Cantilever",
        (
            WirePoint("RootPoint", 0.0, 0.0, 0.0),
            WirePoint("TipPoint", 1.0, 0.0, 0.0),
        ),
        (WireMember("BeamMember", "RootPoint", "TipPoint"),),
    )
    named_regions = (
        NamedRegion("Root", (LogicalEntityRef("point:RootPoint"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:TipPoint"),)),
        NamedRegion("Member", (LogicalEntityRef("edge:BeamMember"),)),
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Root", 1, 6),),
        cloads=(
            NodalLoad("Tip", 2, -10.0),
            NodalLoad("Tip", 3, 5.0),
        ),
        line_loads=(LineLoad("Member", (0.0, -1.0, 0.0)),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(2.0, cell_shape="line", line_element_type="Beam2"),
        named_regions=named_regions,
    )
    definitions = ModelDefinitions(
        materials=(MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        sections=(
            SectionDefinition(
                "BeamSection",
                "Steel",
                "rectangle",
                {"width": 0.1, "height": 0.2},
            ),
        ),
        assignments=(RegionAssignment("BeamSection", "DOMAIN"),),
        steps=(step,),
    )
    compiled = compile_model_definitions(model, definitions).require_model()
    report = run_static_preflight(compiled, "Load")
    assert report.passed, tuple(diagnostic.message for diagnostic in report.errors)

    result = solve(compiled, "Load")
    tip_id = compiled.node_sets["Tip"].node_ids[0]
    assert compiled.mesh.dofs_per_node == 6
    assert np.isfinite(result.U).all()
    assert abs(result.nodal_displacement(tip_id, component=2)) > 0.0
    assert abs(result.nodal_displacement(tip_id, component=3)) > 0.0


def test_beam2_result_provider_csv_and_vtk_exports_preserve_line_topology(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "ExportBeam",
        (
            WirePoint("RootPoint", 0.0, 0.0, 0.0),
            WirePoint("TipPoint", 1.0, 0.0, 0.0),
        ),
        (WireMember("BeamMember", "RootPoint", "TipPoint"),),
    )
    named_regions = (
        NamedRegion("Root", (LogicalEntityRef("point:RootPoint"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:TipPoint"),)),
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Root", 1, 6),),
        cloads=(NodalLoad("Tip", 2, -10.0),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(2.0, cell_shape="line", line_element_type="Beam2"),
        named_regions=named_regions,
    )
    compiled = compile_model_definitions(
        model,
        ModelDefinitions(
            materials=(MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            sections=(
                SectionDefinition(
                    "BeamSection",
                    "Steel",
                    "rectangle",
                    {"width": 0.1, "height": 0.2},
                ),
            ),
            assignments=(RegionAssignment("BeamSection", "DOMAIN"),),
            steps=(step,),
        ),
    ).require_model()
    result = solve(compiled, "Load")
    provider = build_result_provider(
        ResultSourceKey(
            "result-beam",
            "session-beam",
            "artifact-beam",
            1,
            "Load",
            "run-beam",
        ),
        result,
    )
    field = next(
        item
        for item in provider.snapshot.fields
        if item.key.request.field_id.variable is ResultVariable.U
    )
    selection = ScalarFieldSelection(field.key, field.descriptor.columns[0])
    export = prepare_result_export_snapshot(provider.snapshot, selection)

    csv_path = write_result_csv(tmp_path / "beam.csv", export)
    vtk_path = write_result_vtk(tmp_path / "beam.vtk", export)
    csv_readback = read_result_csv(csv_path)
    vtk_readback = read_result_vtk(vtk_path)

    assert csv_readback.records
    assert vtk_readback.cells
    assert vtk_readback.cell_types == (3,)


def test_schema3_reopened_truss_regenerates_and_solves_without_old_artifacts(
    real_gmsh,
    tmp_path,
) -> None:
    del real_gmsh
    recipe = WireGeometry(
        "ReopenedBar",
        (
            WirePoint("RootPoint", 0.0, 0.0, 0.0),
            WirePoint("TipPoint", 1.0, 0.0, 0.0),
        ),
        (WireMember("BarMember", "RootPoint", "TipPoint"),),
    )
    settings = MeshSettings(
        2.0,
        cell_shape="line",
        line_element_type="Truss2",
    )
    named_regions = (
        NamedRegion("Root", (LogicalEntityRef("point:RootPoint"),)),
        NamedRegion("Tip", (LogicalEntityRef("point:TipPoint"),)),
    )
    step = AnalysisStep(
        "Load",
        boundaries=(
            DisplacementConstraint("Root", 1, 3),
            DisplacementConstraint("Tip", 2, 3),
        ),
        cloads=(NodalLoad("Tip", 1, 10.0),),
    )
    snapshot = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=settings,
        feature_history=derive_feature_history(recipe),
        named_regions=named_regions,
        material_definitions=(
            MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),
        ),
        section_definitions=(
            SectionDefinition("BarSection", "Steel", "truss", {"area": 0.01}),
        ),
        region_assignments=(RegionAssignment("BarSection", "DOMAIN"),),
        analysis_definitions=(step,),
    )
    target = save_project_v3(tmp_path / "reopened-bar.femproj", snapshot)
    reopened = load_project_v3(target)

    session = ModelSession()
    assert session.replace_from_snapshot(reopened).accepted
    task = session.prepare_mesh_generation()
    generated = generate_fem_model(task)
    assert session.accept_generated_model(task.token, generated).accepted
    validation = session.prepare_validation("Load")
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert report.passed, tuple(diagnostic.message for diagnostic in report.errors)
    assert session.accept_validation(validation.token, report).accepted
    solve_task = session.prepare_solve("Load", "Reopened-Run")
    assert session.begin_run(solve_task.token).accepted
    result = solve(solve_task.model, solve_task.step_name)
    assert session.accept_run_succeeded(
        solve_task.token,
        build_solve_result_bundle(solve_task, result),
    ).accepted
    assert session.current_result() is not None
