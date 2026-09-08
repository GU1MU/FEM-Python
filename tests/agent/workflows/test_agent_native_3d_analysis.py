from __future__ import annotations

import numpy as np
import pytest

from fem.application import NamedRegion, run_static_preflight
from fem.application.preprocessing import generate_fem_model
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    OutputRequest,
    SectionAssignment,
    SurfaceLoad,
)
from fem.geometry import ExtrudedGeometry, LogicalEntityRef, RectangleGeometry
from fem.materials.linear_elastic import material
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear


def _analysis_model(load_type: str, *, order: int = 1):
    recipe = ExtrudedGeometry(RectangleGeometry("Oracle block", 1.0, 1.0), 1.0)
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.35,
            order=order,
            cell_shape="tetrahedron",
            strict_cell_shape=True,
        ),
        named_regions=(
            NamedRegion("Body", (LogicalEntityRef("body:domain"),)),
            NamedRegion("Fixed", (LogicalEntityRef("face:side/left"),)),
            NamedRegion("Loaded", (LogicalEntityRef("face:side/right"),)),
        ),
    )
    steel = material("Steel", 1000.0, 0.0)
    model.materials[steel.name] = steel
    model.sections.append(SectionAssignment("Body", steel.name, "solid"))
    surface_load = (
        SurfaceLoad("Loaded", (10.0, 0.0, 0.0), load_type="traction")
        if load_type == "traction"
        else SurfaceLoad("Loaded", magnitude=10.0, load_type="pressure")
    )
    model.steps.append(
        AnalysisStep(
            "Static",
            boundaries=(
                DisplacementConstraint(
                    "Fixed",
                    1,
                    3,
                    0.0,
                    target_kind="surface",
                ),
            ),
            surface_loads=(surface_load,),
            outputs=(
                OutputRequest("field", "node", ("U", "RF"), name="Nodal"),
                OutputRequest("field", "element", ("S",), name="Stress"),
            ),
            metadata={"nlgeom": False},
        )
    )
    return model


def test_tet10_runs_preflight_and_real_solver(real_gmsh) -> None:
    del real_gmsh
    model = _analysis_model("traction", order=2)

    report = run_static_preflight(model, "Static")
    result = static_linear.solve(model, "Static")

    assert report.passed, report.diagnostics
    assert {element.type for element in model.mesh.elements} == {"Tet10"}
    assert np.all(np.isfinite(result.U))
    assert float(result.reactions[0::3].sum()) == pytest.approx(-10.0, abs=1.0e-9)


def test_preflight_diagnoses_uncovered_solid_and_rigid_body_dofs(
    real_gmsh,
) -> None:
    del real_gmsh
    uncovered = _analysis_model("traction")
    uncovered.sections.clear()
    uncovered_report = run_static_preflight(uncovered, "Static")

    underconstrained = _analysis_model("traction")
    underconstrained.steps[0].boundaries = (
        DisplacementConstraint(
            "Fixed",
            1,
            1,
            0.0,
            target_kind="surface",
        ),
    )
    rigid_body_report = run_static_preflight(underconstrained, "Static")

    assert not uncovered_report.passed
    assert "definition.section.missing" in {
        diagnostic.code for diagnostic in uncovered_report.diagnostics
    }
    assert not rigid_body_report.passed
    assert "static.stiffness.singular" in {
        diagnostic.code for diagnostic in rigid_body_report.diagnostics
    }


@pytest.mark.parametrize(
    ("load_type", "expected_displacement", "expected_reaction"),
    (
        ("traction", 0.01, -10.0),
        ("pressure", -0.01, 10.0),
    ),
)
def test_3d_tension_compression_oracle_and_reaction_balance(
    real_gmsh,
    load_type: str,
    expected_displacement: float,
    expected_reaction: float,
) -> None:
    del real_gmsh
    model = _analysis_model(load_type)

    report = run_static_preflight(model, "Static")
    result = static_linear.solve(model, "Static")
    loaded_ids = model.node_sets["Loaded"].node_ids
    loaded_displacements = np.array(
        [result.nodal_displacement(node_id, 1) for node_id in loaded_ids],
    )

    assert report.passed, report.diagnostics
    assert loaded_displacements == pytest.approx(expected_displacement, abs=1.0e-10)
    assert float(result.reactions[0::3].sum()) == pytest.approx(
        expected_reaction,
        abs=1.0e-9,
    )
    assert float(result.reactions[1::3].sum()) == pytest.approx(0.0, abs=1.0e-9)
    assert float(result.reactions[2::3].sum()) == pytest.approx(0.0, abs=1.0e-9)
