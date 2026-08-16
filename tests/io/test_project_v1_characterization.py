from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fem.application import NativePart
from fem.core.immutable_json import thaw_json_mapping
from fem.core.model import OutputRequest, OutputSourceEvidence
from fem.io import (
    ProjectDecodeError,
    ProjectEncodeError,
)
from fem.io._project_codec import loads_json_strict
from fem.io.project_v1 import (
    ProjectV1DecodeError,
    ProjectV1EncodeError,
    ProjectV1Error,
    decode_project_v1,
    dumps_project_v1,
    encode_project_v1,
    load_project_v1,
    loads_project_v1,
    save_project_v1,
)


FIXTURES = Path(__file__).parents[1] / "helpers" / "fixtures" / "femproj" / "v1"


def test_static_minimal_v1_fixture_uses_legacy_defaults() -> None:
    path = FIXTURES / "minimal_rectangle.femproj"

    snapshot = load_project_v1(path)

    assert snapshot.source_path == path
    assert snapshot.parts == (NativePart(),)
    assert snapshot.mesh_settings is None
    assert snapshot.feature_history[0].name == "Base-1"
    assert snapshot.named_regions == ()


def test_static_full_v1_fixture_is_payload_and_file_bytes_golden(
    tmp_path,
) -> None:
    fixture = FIXTURES / "full_rectangle_canonical.femproj"
    golden_bytes = fixture.read_bytes()
    parsed_payload = loads_json_strict(golden_bytes)

    snapshot = load_project_v1(fixture)

    assert encode_project_v1(snapshot) == parsed_payload
    assert (dumps_project_v1(snapshot) + "\n").encode("utf-8") == golden_bytes
    assert list(parsed_payload) == [
        "schema",
        "logical_topology_version",
        "source",
        "parts",
        "geometry",
        "mesh_settings",
        "feature_history",
        "named_regions",
        "materials",
        "sections",
        "assignments",
        "steps",
    ]
    assert snapshot.geometry_recipe.name == "矩形板"
    assert snapshot.analysis_definitions[0].line_loads == ()
    assert all(
        output.source_evidence is None
        for step in snapshot.analysis_definitions
        for output in step.outputs
    )

    target = save_project_v1(tmp_path / "golden.femproj", snapshot)

    assert target.read_bytes() == golden_bytes


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            '{"schema":1,"schema":1,"source":"native",'
            '"geometry":{"type":"RectangleGeometry","name":"R",'
            '"width":1,"height":1}}',
            "重复键",
        ),
        (
            '{"schema":1,"source":"native",'
            '"geometry":{"type":"RectangleGeometry","name":"R",'
            '"width":1e9999,"height":1}}',
            "非有限数值",
        ),
    ],
)
def test_v1_loader_uses_shared_strict_parser(document, message) -> None:
    with pytest.raises(ProjectV1DecodeError, match=message):
        loads_project_v1(document)


def test_v1_errors_preserve_legacy_and_generic_catch_contracts() -> None:
    assert issubclass(ProjectV1DecodeError, ProjectV1Error)
    assert issubclass(ProjectV1DecodeError, ProjectDecodeError)
    assert issubclass(ProjectV1EncodeError, ProjectV1Error)
    assert issubclass(ProjectV1EncodeError, ProjectEncodeError)


def test_v1_missing_output_variables_migrates_to_empty_owned_tuple() -> None:
    payload = loads_json_strict(
        (FIXTURES / "full_rectangle_canonical.femproj").read_bytes()
    )
    del payload["steps"][0]["outputs"][0]["variables"]

    snapshot = decode_project_v1(payload)
    request = snapshot.analysis_definitions[0].outputs[0]

    assert request.variables == ()
    assert request.source_evidence is None


def test_v1_writer_rejects_source_evidence_without_touching_target(
    tmp_path,
) -> None:
    snapshot = load_project_v1(FIXTURES / "full_rectangle_canonical.femproj")
    step = snapshot.analysis_definitions[0]
    original = step.outputs[0]
    output = OutputRequest(
        original.kind,
        original.target,
        original.variables,
        thaw_json_mapping(original.metadata),
        OutputSourceEvidence(
            "abaqus",
            (("frequency", "1"),),
            ("field",),
            (),
            (),
        ),
    )
    guarded = replace(
        snapshot,
        analysis_definitions=(
            replace(step, outputs=(output, *step.outputs[1:])),
            *snapshot.analysis_definitions[1:],
        ),
    )
    target = tmp_path / "existing.femproj"
    target.write_text("old-target", encoding="utf-8")

    with pytest.raises(
        ProjectV1EncodeError,
        match=r"source_evidence.*v1",
    ):
        save_project_v1(target, guarded)

    assert target.read_text(encoding="utf-8") == "old-target"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
