from __future__ import annotations

from pathlib import Path

import pytest

from fem.application import NativePart
from fem.io import (
    ProjectDecodeError,
    ProjectEncodeError,
)
from fem.io._project_codec import loads_json_strict
from fem.io.project_v1 import (
    ProjectV1DecodeError,
    ProjectV1EncodeError,
    ProjectV1Error,
    dumps_project_v1,
    encode_project_v1,
    load_project_v1,
    loads_project_v1,
    save_project_v1,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "femproj" / "v1"


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
