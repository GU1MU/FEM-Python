from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    ExtrudedGeometry,
    SketchGeometry,
    SketchRectangle,
)
from fem.io.project import decode_project, encode_project
from fem.io.project_v1 import encode_project_v1
from fem.io.project_v2 import encode_project_v2
from fem.io.project_v3 import (
    ProjectV3EncodeError,
    decode_project_v3,
    encode_project_v3,
)
from fem.io.project_v4 import (
    ProjectV4DecodeError,
    decode_project_v4,
    encode_project_v4,
)
from tests.geometry.test_profile_extrusion import (
    profile_face_id,
    two_profile_sketch,
)


def _snapshot(recipe: ExtrudedGeometry) -> ProjectSnapshot:
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )


def _single_profile_base() -> SketchGeometry:
    sketch = two_profile_sketch()
    return SketchGeometry(
        "Single",
        sketch.plane,
        sketch.points[:4],
        sketch.curves[:4],
    )


def _legacy_profile_base() -> SketchGeometry:
    return SketchGeometry(
        "Legacy",
        (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
    )


def test_v4_selected_profiles_roundtrip_canonical_ids() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")
    original = _snapshot(
        ExtrudedGeometry(sketch, 3.0, (second, first))
    )

    payload = encode_project_v4(original)
    reopened = decode_project_v4(payload)

    source_ids = payload["project"]["authoring"]["geometry"][
        "source_face_ids"
    ]
    assert payload["schema"] == 4
    assert source_ids == list(original.geometry_recipe.source_face_ids)
    assert reopened == original


def test_v4_alias_input_is_canonicalized_on_decode() -> None:
    sketch = _single_profile_base()
    primary = profile_face_id(sketch, "L1")
    original = _snapshot(
        ExtrudedGeometry(sketch, 2.0, (primary,))
    )
    payload = encode_project_v4(original)
    payload["project"]["authoring"]["geometry"]["source_face_ids"] = [
        "face:domain"
    ]

    reopened = decode_project_v4(payload)

    assert reopened.geometry_recipe.source_face_ids == (primary,)


@pytest.mark.parametrize(
    ("logical_id", "path_pattern"),
    (
        ("edge:L1", r"source_face_ids\[0\].*face"),
        ("face:missing", r"source_face_ids\[0\].*失效"),
    ),
)
def test_v4_bad_source_face_has_path_aware_diagnostic(
    logical_id: str,
    path_pattern: str,
) -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    payload = encode_project_v4(
        _snapshot(ExtrudedGeometry(sketch, 2.0, (first,)))
    )
    broken = deepcopy(payload)
    broken["project"]["authoring"]["geometry"]["source_face_ids"] = [
        logical_id
    ]

    with pytest.raises(ProjectV4DecodeError, match=path_pattern):
        decode_project_v4(broken)


def test_v3_writer_rejects_selected_source_faces_but_legacy_migrates_empty() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    selected = _snapshot(ExtrudedGeometry(sketch, 2.0, (first,)))

    with pytest.raises(ProjectV3EncodeError, match="source_face_ids"):
        encode_project_v3(selected)

    legacy = _snapshot(ExtrudedGeometry(_single_profile_base(), 2.0))
    loaded = decode_project(encode_project_v3(legacy))
    assert loaded.source_schema == 3
    geometry = loaded.snapshot.geometry_recipe
    assert isinstance(geometry, ExtrudedGeometry)
    assert geometry.source_face_ids


def test_v3_unproven_strict_extrusion_fingerprint_migrates_to_exact() -> None:
    legacy = _snapshot(ExtrudedGeometry(_single_profile_base(), 2.0))
    payload = encode_project_v3(legacy)
    payload["project"]["authoring"]["logical_topology"] = {
        "contract": 2,
        "signature": {
            "dimension": 3,
            "exact": False,
            "entities": [
                {
                    "kind": "body",
                    "logical_id": "body:result",
                    "semantic_role": "result.unproven",
                    "selectable": False,
                    "topology_links": [],
                }
            ],
        },
    }

    reopened = decode_project_v3(payload)

    assert reopened.geometry_recipe.source_face_ids == ()


@pytest.mark.parametrize(
    ("encode", "decode"),
    (
        (encode_project_v3, decode_project_v3),
        (encode_project_v4, decode_project_v4),
    ),
)
def test_nested_legacy_sketch_extrusion_roundtrips_canonical_tree(
    encode,
    decode,
) -> None:
    original = _snapshot(
        ExtrudedGeometry(_legacy_profile_base(), 2.0)
    )

    reopened = decode(encode(original))

    assert reopened.geometry_recipe.base.is_strict
    assert reopened.geometry_recipe.source_face_ids == ()


def test_v4_nested_legacy_source_alias_is_recanonicalized() -> None:
    original = _snapshot(
        ExtrudedGeometry(
            _legacy_profile_base(),
            2.0,
            ("face:domain",),
        )
    )

    reopened = decode_project_v4(encode_project_v4(original))

    assert reopened.geometry_recipe.base.is_strict
    assert len(reopened.geometry_recipe.source_face_ids) == 1
    assert reopened.geometry_recipe.source_face_ids[0].startswith(
        "face:profile/"
    )


@pytest.mark.parametrize(
    "legacy_encoder",
    (encode_project_v1, encode_project_v2, encode_project_v3),
)
def test_old_schema_legacy_extrusion_can_save_as_v5_and_reopen(
    legacy_encoder,
) -> None:
    original = _snapshot(
        ExtrudedGeometry(_legacy_profile_base(), 2.0)
    )
    migrated = decode_project(legacy_encoder(original)).snapshot

    reopened = decode_project(encode_project(migrated)).snapshot

    geometry = reopened.geometry_recipe
    assert isinstance(geometry, ExtrudedGeometry)
    assert geometry.base.is_strict
    assert len(geometry.source_face_ids) == 1
