from __future__ import annotations

from datetime import datetime, timezone
import math


import numpy as np

from fem.application.results import (
    FieldPosition,
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    execute_output_requests,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    NodalLoad,
    OutputRequest,
)
from fem.elements import (
    BEAM_FRAME_FIELD_KEY,
    BeamFrameField,
)
from fem.solvers import static_linear


_LENGTH = 2.0
_AXIAL_FORCE = 12.0
_LOCAL_Y_FORCE = 3.0
_LOCAL_Z_FORCE = -2.0
_TORQUE = 1.5
_POINT_COLUMNS = (
    "S11",
    "S22",
    "S12",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)
_SECTION_CASES = (
    ("rectangle", {"height": 0.4, "width": 0.2}),
    ("solid_circle", {"radius": 0.2}),
    ("hollow_circle", {"outer_radius": 0.2, "inner_radius": 0.1}),
)


def _inline_cantilever(
    section_type: str,
    dimensions: dict[str, float],
) -> FEMModel:
    frame = BeamFrameField.from_rotations(
        _LENGTH,
        np.eye(3),
        np.eye(3),
    )
    mesh = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, _LENGTH, 0.0, 0.0),
        ),
        elements=(
            Element3D(
                10,
                (1, 2),
                "Beam2",
                {
                    "E": 210.0e9,
                    "nu": 0.3,
                    "section_type": section_type,
                    **dimensions,
                    BEAM_FRAME_FIELD_KEY: frame,
                },
            ),
        ),
        dofs_per_node=6,
    )
    return FEMModel(
        mesh=mesh,
        name=f"inline-{section_type}-cantilever",
        steps=(
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=(
                    NodalLoad(2, 1, _AXIAL_FORCE),
                    NodalLoad(2, 2, _LOCAL_Y_FORCE),
                    NodalLoad(2, 3, _LOCAL_Z_FORCE),
                    NodalLoad(2, 4, _TORQUE),
                ),
            ),
        ),
    )


def _solve_and_request_stress(
    section_type: str,
    dimensions: dict[str, float],
):
    result = static_linear.solve(
        _inline_cantilever(section_type, dimensions),
        "Load",
    )
    source = ResultSourceKey(
        result_id=f"result-{section_type}",
        session_id="section-point-session",
        artifact_id=f"artifact-{section_type}",
        model_revision=1,
        step_name="Load",
        run_id=f"run-{section_type}",
    )
    provider = build_result_provider(source, result)
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("SF", "SM", "S")),),
    )
    return result, outcome.provider_draft, outcome


def _independent_integration_point_oracle(
    section_type: str,
    dimensions: dict[str, float],
) -> dict[int, tuple[float, ...]]:
    if section_type == "rectangle":
        height = dimensions["height"]
        width = dimensions["width"]
        area = height * width
        iyy = width * height**3 / 12.0
        izz = height * width**3 / 12.0
        points = (
            (width / 2.0, height / 2.0),
            (-width / 2.0, height / 2.0),
            (-width / 2.0, -height / 2.0),
            (width / 2.0, -height / 2.0),
        )
    else:
        outer_radius = (
            dimensions["radius"]
            if section_type == "solid_circle"
            else dimensions["outer_radius"]
        )
        inner_radius = (
            0.0 if section_type == "solid_circle" else dimensions["inner_radius"]
        )
        area = math.pi * (outer_radius**2 - inner_radius**2)
        iyy = izz = math.pi * (outer_radius**4 - inner_radius**4) / 4.0
        point_radius = (
            outer_radius
            if section_type == "solid_circle"
            else (outer_radius + inner_radius) / 2.0
        )
        points = (
            (point_radius, 0.0),
            (0.0, point_radius),
            (-point_radius, 0.0),
            (0.0, -point_radius),
        )

    axial = _AXIAL_FORCE / area
    moment_y = -_LOCAL_Z_FORCE * _LENGTH / 2.0
    moment_z = _LOCAL_Y_FORCE * _LENGTH / 2.0
    point_values: dict[int, tuple[float, ...]] = {}
    for number, (local_y, local_z) in enumerate(points, start=1):
        s11 = axial + moment_y * local_z / iyy - moment_z * local_y / izz
        values = (s11,)
        point_values[number] = values

    return point_values


def _stress_fields(provider):
    return tuple(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.variable is ResultVariable.S
        and field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is not None
    )


def _field(provider, variable: ResultVariable):
    return next(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.variable is variable
        and field.key.request.field_id.position
        is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is None
    )


def _archive_snapshot(provider, outcome, section_type: str):
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    return ResultArchiveSnapshot(
        archive_id=f"archive-{section_type}",
        created_at=now,
        producer_version="section-point-test",
        origin=ResultArchiveOrigin(
            model_name=f"inline-{section_type}-cantilever",
            model_fingerprint="0" * 64,
        ),
        run=ResultArchiveRun(
            "Job-1",
            "Load",
            now,
            output_report=outcome.report,
        ),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=ResultArchiveModelProjection(
            provider.snapshot.topology,
        ),
    )
