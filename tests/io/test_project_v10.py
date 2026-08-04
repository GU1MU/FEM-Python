from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from fem.application import ModelSession, UnitContext
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    OutputRequest,
)
from fem.geometry import RectangleGeometry
from fem.io.project import CURRENT_PROJECT_SCHEMA, decode_project, encode_project
from fem.io.project_v9 import (
    ProjectV9DecodeError,
    decode_project_v9,
    encode_project_v9,
)
from fem.io.project_v10 import (
    ProjectV10DecodeError,
    decode_project_v10,
    encode_project_v10,
)


def _snapshot(*, named: bool):
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-板",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("实体-板", 10.0, 4.0),
        part_name="部件-板",
    )
    step = AnalysisStep(
        "分析步-静力",
        boundaries=(
            DisplacementConstraint(
                "边-固定端",
                1,
                2,
                0.0,
                "edge",
                "位移-固定端" if named else None,
            ),
        ),
        edge_loads=(
            EdgeLoad(
                "边-加载端",
                (12.0, 0.0),
                load_type="traction",
                name="载荷-拉伸" if named else None,
            ),
        ),
        outputs=(
            OutputRequest(
                "field",
                "node",
                ("U", "RF"),
                name="结果请求-位移反力" if named else None,
            ),
        ),
        metadata={"nlgeom": False},
    )
    return replace(
        session.prepare_project_save().snapshot,
        analysis_definitions=(step,),
    )


def test_a5_schema_v10_round_trip_preserves_named_analysis_identity() -> None:
    payload = encode_project_v10(_snapshot(named=True))
    step = decode_project_v10(payload).analysis_definitions[0]

    assert CURRENT_PROJECT_SCHEMA == 12
    assert payload["schema"] == 10
    encoded = payload["project"]["authoring"]["definitions"]["steps"][0]
    assert encoded["boundaries"][0]["name"] == "位移-固定端"
    assert encoded["edge_loads"][0]["name"] == "载荷-拉伸"
    assert encoded["outputs"][0]["name"] == "结果请求-位移反力"
    assert step.boundaries[0].name == "位移-固定端"
    assert step.edge_loads[0].name == "载荷-拉伸"
    assert step.outputs[0].name == "结果请求-位移反力"
    assert decode_project_v10(payload).unit_context == UnitContext("mm", "N", "MPa")


def test_a5_anonymous_v9_project_migrates_deterministic_names() -> None:
    payload = encode_project_v9(_snapshot(named=False))
    first = decode_project(payload).snapshot.analysis_definitions[0]
    second = decode_project(payload).snapshot.analysis_definitions[0]

    assert first.boundaries[0].name == second.boundaries[0].name
    assert first.edge_loads[0].name == second.edge_loads[0].name
    assert first.outputs[0].name == second.outputs[0].name
    assert first.boundaries[0].name.startswith("位移-兼容-")
    assert first.edge_loads[0].name.startswith("载荷-兼容-")
    assert first.outputs[0].name.startswith("结果请求-兼容-")

    saved = encode_project(
        replace(
            _snapshot(named=False),
            analysis_definitions=(first,),
        )
    )
    reopened = decode_project(saved).snapshot.analysis_definitions[0]
    assert reopened.boundaries[0].name == first.boundaries[0].name
    assert reopened.edge_loads[0].name == first.edge_loads[0].name
    assert reopened.outputs[0].name == first.outputs[0].name


def test_a5_v9_remains_strict_and_v10_requires_exact_name_fields() -> None:
    v9 = encode_project_v9(_snapshot(named=False))
    widened = deepcopy(v9)
    widened["project"]["authoring"]["definitions"]["steps"][0][
        "boundaries"
    ][0]["name"] = "位移-固定端"
    with pytest.raises(ProjectV9DecodeError, match="未知字段|字段"):
        decode_project_v9(widened)

    v10 = encode_project_v10(_snapshot(named=True))
    missing = deepcopy(v10)
    del missing["project"]["authoring"]["definitions"]["steps"][0][
        "edge_loads"
    ][0]["name"]
    with pytest.raises(ProjectV10DecodeError, match="缺少字段 'name'"):
        decode_project_v10(missing)

    unknown = deepcopy(v10)
    unknown["project"]["authoring"]["definitions"]["steps"][0][
        "outputs"
    ][0]["future"] = True
    with pytest.raises(ProjectV10DecodeError, match="未知字段|包含"):
        decode_project_v10(unknown)


def test_a5_pressure_sign_and_unit_context_survive_reopen() -> None:
    snapshot = _snapshot(named=True)
    step = snapshot.analysis_definitions[0]
    pressure_step = replace(
        step,
        edge_loads=(
            EdgeLoad(
                "边-加载端",
                magnitude=-3.5,
                load_type="pressure",
                name="载荷-外向压力",
            ),
        ),
    )
    reopened = decode_project(
        encode_project(
            replace(snapshot, analysis_definitions=(pressure_step,))
        )
    ).snapshot

    load = reopened.analysis_definitions[0].edge_loads[0]
    assert load.magnitude == -3.5
    assert load.load_type == "pressure"
    assert reopened.unit_context.force == "N"
    assert reopened.unit_context.length == "mm"
