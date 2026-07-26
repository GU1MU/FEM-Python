from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

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
    EdgeLoad,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    OutputSourceEvidence,
    SurfaceLoad,
)
from fem.elements import BeamOrientation
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)
from fem.geometry.references import LogicalEntityRef
from fem.io._project_errors import ProjectDecodeError, ProjectEncodeError
from fem.io._project_codec import (
    ProjectFieldCodecPolicy,
    decode_step_field,
    encode_step_field,
)
from fem.io.project_v2 import (
    ProjectV2DecodeError,
    ProjectV2EncodeError,
    decode_assignment_v2,
    decode_project_v2,
    dumps_project_v2,
    encode_assignment_v2,
    encode_project_v2,
    loads_project_v2,
    save_project_v2,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff


_CURRENT_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v2",
    decode_error=ProjectV2DecodeError,
    encode_error=ProjectV2EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
)


def _snapshot() -> ProjectSnapshot:
    recipe = RectangleGeometry("板", 10.0, 5.0)
    fixed = NamedRegion(
        "Fixed",
        (LogicalEntityRef("edge:left"),),
    )
    loaded = NamedRegion(
        "Loaded",
        (LogicalEntityRef("edge:right"),),
    )
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart("零件", "主体"),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:right"),
                    0.25,
                    MeshSizeFalloff("global_size", 0.0, 2.0),
                ),
            ),
        ),
        feature_history=derive_feature_history(recipe),
        named_regions=(loaded, fixed),
        material_definitions=(
            MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),
        ),
        section_definitions=(
            SectionDefinition("Solid", "Steel", "solid", {"thickness": 1.0}),
        ),
        region_assignments=(RegionAssignment("Solid", "DOMAIN"),),
        analysis_definitions=(
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
                edge_loads=(EdgeLoad("Loaded", (0.0, -1.0)),),
                gravity_loads=(GravityLoad((0.0, -9.81), None),),
                metadata={"increments": [1, 2]},
            ),
        ),
    )


def test_v2_native_authoring_round_trip_is_canonical_and_utf8():
    original = _snapshot()

    dumped = dumps_project_v2(original)
    reopened = loads_project_v2(dumped)

    assert reopened == original
    assert dumped.endswith("\n")
    assert "\\u96f6" not in dumped
    assert "\"feature_history\"" not in dumped
    assert "\"local_size\"" not in dumped
    assert "\"entity_id\"" not in dumped
    assert dumps_project_v2(reopened) == dumped


@pytest.mark.parametrize(
    "recipe",
    (
        RectangleGeometry("Rectangle", 4.0, 2.0),
        DiskGeometry("Disk", 2.0),
        PlateWithHoleGeometry("Plate", 10.0, 8.0, 4.0, 3.0, 1.0),
        BoxGeometry("Box", 4.0, 3.0, 2.0),
        CylinderGeometry("Cylinder", 2.0, 5.0),
        SketchGeometry(
            "Sketch",
            (SketchRectangle("material", 0.0, 0.0, 4.0, 2.0),),
        ),
        MovedGeometry(
            RectangleGeometry("Moved", 4.0, 2.0),
            1.0,
            -2.0,
            0.0,
        ),
        RotatedGeometry(
            RectangleGeometry("Rotated", 4.0, 2.0),
            "z",
            30.0,
        ),
        ExtrudedGeometry(
            RectangleGeometry("Extruded", 4.0, 2.0),
            3.0,
        ),
        BooleanGeometry(
            "Cut",
            "cut",
            RectangleGeometry("Object", 10.0, 8.0),
            MovedGeometry(DiskGeometry("Tool", 1.0), 4.0, 3.0),
        ),
    ),
    ids=(
        "rectangle",
        "disk",
        "plate-with-hole",
        "box",
        "cylinder",
        "sketch",
        "moved",
        "rotated",
        "extruded",
        "boolean",
    ),
)
def test_v2_every_current_geometry_recipe_round_trips_with_topology(recipe):
    snapshot = _geometry_only_snapshot(recipe)

    encoded = encode_project_v2(snapshot)
    reopened = decode_project_v2(encoded)
    reencoded = encode_project_v2(reopened)

    assert reopened.geometry_recipe == recipe
    assert (
        reencoded["project"]["authoring"]["logical_topology"]
        == encoded["project"]["authoring"]["logical_topology"]
    )
    assert reencoded == encoded


def test_v2_nonexact_recipe_without_geometry_references_round_trips():
    payload = _nonexact_payload()

    reopened = decode_project_v2(payload)

    assert reopened.geometry_recipe == _nonexact_recipe()
    assert encode_project_v2(reopened) == payload


@pytest.mark.parametrize(
    ("target", "falloff"),
    (
        (
            LogicalEntityRef("edge:outer-loop"),
            MeshSizeFalloff("global_size", 0.0, 2.0),
        ),
        (
            LogicalEntityRef("edge:hole-loop"),
            MeshSizeFalloff("target_radius", 0.25, 2.0),
        ),
    ),
    ids=("global-size", "target-radius"),
)
def test_v2_current_falloff_profiles_round_trip(target, falloff):
    recipe = PlateWithHoleGeometry(
        "Plate",
        10.0,
        8.0,
        4.0,
        3.0,
        1.0,
    )
    snapshot = replace(
        _geometry_only_snapshot(recipe),
        mesh_settings=MeshSettings(
            1.0,
            local_controls=(LocalMeshControl(target, 0.25, falloff),),
        ),
    )

    reopened = decode_project_v2(encode_project_v2(snapshot))

    assert reopened.mesh_settings == snapshot.mesh_settings
    assert dumps_project_v2(reopened) == dumps_project_v2(snapshot)


def test_shared_current_step_field_codecs_round_trip_every_load_shape():
    step = AnalysisStep(
        "AllFields",
        boundaries=(DisplacementConstraint("Nodes", 1, 2, 0.0),),
        cloads=(NodalLoad("Nodes", 2, -3.0),),
        edge_loads=(EdgeLoad("Edge", (1.0, 2.0)),),
        surface_loads=(SurfaceLoad("Surface", (0.0, 0.0, -1.0)),),
        line_loads=(LineLoad("Beam", (0.0, -5.0), "local"),),
        gravity_loads=(
            GravityLoad((0.0, -9.81), None),
            GravityLoad((0.0, -9.81), "DOMAIN"),
        ),
        outputs=(
            OutputRequest(
                "field",
                "all",
                ("U", "S"),
                {"frequency": 1},
            ),
        ),
        metadata={"increments": 10},
    )

    encoded = encode_step_field(
        step,
        "step",
        policy=_CURRENT_FIELD_POLICY,
    )
    reopened = decode_step_field(
        encoded,
        "step",
        policy=_CURRENT_FIELD_POLICY,
    )

    assert reopened == step
    assert reopened.outputs[0].source_evidence is None
    assert (
        encode_step_field(
            reopened,
            "step",
            policy=_CURRENT_FIELD_POLICY,
        )
        == encoded
    )


def test_v2_full_project_round_trips_capability_compatible_loads():
    base = _snapshot()
    step = AnalysisStep(
        "Compatible2D",
        boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
        cloads=(NodalLoad("Fixed", 1, 3.0),),
        edge_loads=(EdgeLoad("Loaded", (0.0, -1.0)),),
        gravity_loads=(
            GravityLoad((0.0, -9.81), None),
            GravityLoad((0.0, -9.81), "DOMAIN"),
        ),
        outputs=(OutputRequest("field", "all", ("U",), {}),),
    )
    snapshot_2d = replace(base, analysis_definitions=(step,))
    box = BoxGeometry("Box", 4.0, 3.0, 2.0)
    snapshot_3d = replace(
        _geometry_only_snapshot(box),
        named_regions=(
            NamedRegion(
                "LoadedSurface",
                (LogicalEntityRef("face:front"),),
            ),
        ),
        analysis_definitions=(
            AnalysisStep(
                "Compatible3D",
                surface_loads=(
                    SurfaceLoad(
                        "LoadedSurface",
                        (0.0, 0.0, -1.0),
                    ),
                ),
            ),
        ),
    )

    for snapshot in (snapshot_2d, snapshot_3d):
        reopened = decode_project_v2(encode_project_v2(snapshot))
        assert reopened == snapshot


def test_v2_nested_geometry_fields_are_required_without_legacy_defaults():
    base = _snapshot()
    recipe = MovedGeometry(
        base.geometry_recipe,
        dx=1.0,
        dy=2.0,
        dz=0.0,
    )
    snapshot = replace(
        base,
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )
    payload = encode_project_v2(snapshot)
    del payload["project"]["authoring"]["geometry"]["dz"]

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"\$\.project\.authoring\.geometry "
            r"缺少必需字段：dz"
        ),
    ):
        decode_project_v2(payload)


def test_v2_nested_geometry_unknown_fields_are_rejected():
    base = _snapshot()
    recipe = MovedGeometry(
        base.geometry_recipe,
        dx=1.0,
        dy=2.0,
        dz=0.0,
    )
    snapshot = replace(
        base,
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )
    payload = encode_project_v2(snapshot)
    payload["project"]["authoring"]["geometry"]["legacy_offset"] = 0.0

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"\$\.project\.authoring\.geometry "
            r"包含 v2 未知字段：legacy_offset"
        ),
    ):
        decode_project_v2(payload)


def test_v2_topology_evidence_is_recomputed_and_fail_closed():
    payload = encode_project_v2(_snapshot())
    payload["project"]["authoring"]["logical_topology"]["signature"]["entities"][0][
        "semantic_role"
    ] = "tampered"

    with pytest.raises(ProjectV2DecodeError, match="fingerprint"):
        decode_project_v2(payload)


def test_v2_topology_entity_array_order_is_semantically_normalized():
    payload = encode_project_v2(_snapshot())
    entities = payload["project"]["authoring"]["logical_topology"]["signature"][
        "entities"
    ]
    entities.reverse()

    reopened = decode_project_v2(payload)

    assert reopened.geometry_recipe == _snapshot().geometry_recipe
    assert encode_project_v2(reopened) == encode_project_v2(_snapshot())


@pytest.mark.parametrize(
    ("mutation", "expected_path"),
    (
        (
            lambda signature: signature.__setitem__("dimension", 3),
            "logical_topology.signature.dimension",
        ),
        (
            lambda signature: signature.__setitem__("exact", False),
            "logical_topology.signature.exact",
        ),
        (
            lambda signature: signature["entities"][0].__setitem__(
                "semantic_role",
                "tampered",
            ),
            r"logical_topology.signature.entities\[0\].semantic_role",
        ),
        (
            lambda signature: signature["entities"][0].__setitem__(
                "selectable",
                not signature["entities"][0]["selectable"],
            ),
            r"logical_topology.signature.entities\[0\].selectable",
        ),
        (
            lambda signature: signature["entities"].pop(),
            "logical_topology.signature.entities",
        ),
        (
            lambda signature: signature["entities"].append(
                {
                    "kind": "point",
                    "logical_id": "point:extra",
                    "semantic_role": "extra",
                    "selectable": True,
                }
            ),
            r"logical_topology.signature.entities\[\d+\].logical_id",
        ),
    ),
)
def test_v2_topology_mismatch_reports_narrow_json_path(
    mutation,
    expected_path,
):
    payload = encode_project_v2(_snapshot())
    signature = payload["project"]["authoring"]["logical_topology"][
        "signature"
    ]
    mutation(signature)

    with pytest.raises(ProjectV2DecodeError, match=expected_path):
        decode_project_v2(payload)


def test_v2_assignment_field_codec_preserves_explicit_orientation():
    assignment = RegionAssignment(
        "Beam",
        "BeamRegion",
        BeamOrientation((0.0, 1.0, 0.0)),
    )

    encoded = encode_assignment_v2(assignment)
    decoded = decode_assignment_v2(encoded)

    assert decoded == assignment
    assert encoded == {
        "section_name": "Beam",
        "region_name": "BeamRegion",
        "beam_orientation": {
            "type": "local_y_reference",
            "vector": [0.0, 1.0, 0.0],
        },
    }


def test_v2_rejects_noncanonical_feature_projection_and_runtime_model():
    original = _snapshot()
    bad_history = replace(original, feature_history=())
    bad_model = replace(original, model=object())

    with pytest.raises(ProjectV2EncodeError, match="feature_history"):
        encode_project_v2(bad_history)
    with pytest.raises(ProjectV2EncodeError, match="runtime model"):
        encode_project_v2(bad_model)


def test_v2_encoder_rejects_noncanonical_definitions():
    original = _snapshot()
    section = original.section_definitions[0]
    noncanonical = replace(
        original,
        section_definitions=(
            replace(section, section_type=" SOLID "),
        ),
    )

    with pytest.raises(ProjectV2EncodeError, match="canonical"):
        encode_project_v2(noncanonical)


def test_v2_decoder_installs_normalized_definitions():
    payload = encode_project_v2(_snapshot())
    definitions = payload["project"]["authoring"]["definitions"]
    definitions["materials"][0]["name"] = " Steel "
    definitions["sections"][0].update(
        {
            "name": " Solid ",
            "material": " Steel ",
            "section_type": " SOLID ",
        }
    )
    definitions["assignments"][0].update(
        {
            "section_name": " Solid ",
            "region_name": " DOMAIN ",
        }
    )
    definitions["steps"][0]["name"] = " Load "

    reopened = decode_project_v2(payload)

    assert reopened.material_definitions[0].name == "Steel"
    assert reopened.section_definitions[0].name == "Solid"
    assert reopened.section_definitions[0].material == "Steel"
    assert reopened.section_definitions[0].section_type == "solid"
    assert reopened.region_assignments[0].section_name == "Solid"
    assert reopened.region_assignments[0].region_name == "DOMAIN"
    assert reopened.analysis_definitions[0].name == "Load"


@pytest.mark.parametrize(
    "case",
    (
        "material.properties",
        "section.section_type",
        "step.procedure",
        "boundary.value",
        "cload.component",
        "edge_load.magnitude",
        "surface_load.vector",
        "line_load.coordinate_system",
        "gravity_load.target",
        "output.variables",
    ),
)
def test_v2_schema_owned_definition_fields_are_required(case):
    payload = encode_project_v2(_snapshot())
    expected_path, missing = _remove_required_definition_field(
        payload,
        case,
    )

    with pytest.raises(
        ProjectV2DecodeError,
        match=rf"{expected_path} 缺少必需字段：{missing}",
    ):
        decode_project_v2(payload)


def _remove_required_definition_field(payload, case):
    definitions = payload["project"]["authoring"]["definitions"]
    step = definitions["steps"][0]
    base = r"\$\.project\.authoring\.definitions"
    if case == "material.properties":
        del definitions["materials"][0]["properties"]
        return rf"{base}\.materials\[0\]", "properties"
    if case == "section.section_type":
        del definitions["sections"][0]["section_type"]
        return rf"{base}\.sections\[0\]", "section_type"
    if case == "step.procedure":
        del step["procedure"]
        return rf"{base}\.steps\[0\]", "procedure"

    collection, missing, seed = {
        "boundary.value": (
            "boundaries",
            "value",
            None,
        ),
        "cload.component": (
            "cloads",
            "component",
            {"target": "Fixed", "component": 1, "value": 1.0},
        ),
        "edge_load.magnitude": (
            "edge_loads",
            "magnitude",
            None,
        ),
        "surface_load.vector": (
            "surface_loads",
            "vector",
            {
                "surface": "Surface",
                "vector": [0.0, 1.0],
                "magnitude": None,
                "load_type": "traction",
            },
        ),
        "line_load.coordinate_system": (
            "line_loads",
            "coordinate_system",
            {
                "target": "Line",
                "vector": [0.0, 1.0],
                "coordinate_system": "global",
            },
        ),
        "gravity_load.target": (
            "gravity_loads",
            "target",
            None,
        ),
        "output.variables": (
            "outputs",
            "variables",
            {
                "kind": "field",
                "target": "all",
                "variables": ["U"],
                "metadata": {},
            },
        ),
    }[case]
    if seed is not None:
        step[collection] = [seed]
    del step[collection][0][missing]
    return rf"{base}\.steps\[0\]\.{collection}\[0\]", missing


@pytest.mark.parametrize("schema", (True, 2.0, "2"))
def test_v2_schema_requires_a_strict_integer(schema):
    payload = encode_project_v2(_snapshot())
    payload["schema"] = schema

    with pytest.raises(ProjectV2DecodeError, match="严格整数"):
        decode_project_v2(payload)


def test_v2_rejects_duplicate_json_keys_and_nonfinite_numbers():
    dumped = dumps_project_v2(_snapshot())
    duplicate = dumped.replace('"schema": 2', '"schema": 2, "schema": 2', 1)
    nonfinite = dumped.replace('"size": 1.0', '"size": NaN', 1)

    with pytest.raises(ProjectV2DecodeError, match="重复键"):
        loads_project_v2(duplicate)
    with pytest.raises(ProjectV2DecodeError, match="非有限"):
        loads_project_v2(nonfinite)


def test_v2_huge_numeric_integer_has_path_aware_decode_error():
    payload = encode_project_v2(_snapshot())
    payload["project"]["authoring"]["mesh_settings"]["size"] = 10**10000

    with pytest.raises(
        ProjectV2DecodeError,
        match=r"mesh_settings\.size 必须是有限实数",
    ):
        decode_project_v2(payload)


def test_v2_huge_numeric_integer_has_path_aware_encode_error():
    original = _snapshot()
    settings = original.mesh_settings
    assert settings is not None
    object.__setattr__(settings, "size", 10**10000)

    with pytest.raises(
        ProjectV2EncodeError,
        match=r"snapshot\.mesh_settings\.size 必须是有限实数",
    ):
        encode_project_v2(original)


def test_v2_malformed_named_region_reference_has_item_path_and_cause():
    payload = encode_project_v2(_snapshot())
    payload["project"]["authoring"]["named_regions"][0]["references"][0] = (
        "not-a-logical-id"
    )

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"named_regions\[0\]\.references\[0\] 无效：logical_id"
        ),
    ) as caught:
        decode_project_v2(payload)

    assert isinstance(caught.value.__cause__, ValueError)


def test_v2_unknown_named_reference_has_contextual_path_and_cause():
    payload = encode_project_v2(_snapshot())
    payload["project"]["authoring"]["named_regions"][0]["references"][0] = (
        "edge:missing"
    )

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"named_regions\[0\]\.references\[0\] 无效："
            r"unknown logical reference"
        ),
    ) as caught:
        decode_project_v2(payload)

    assert isinstance(caught.value.__cause__, ValueError)


def test_v2_wrong_kind_local_control_has_contextual_path_and_cause():
    payload = encode_project_v2(_snapshot())
    payload["project"]["authoring"]["mesh_settings"]["local_controls"][0][
        "target"
    ] = "body:domain"

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"mesh_settings\.local_controls\[0\]\.target 无效："
            r"logical reference kind 'body' is not allowed"
        ),
    ) as caught:
        decode_project_v2(payload)

    assert isinstance(caught.value.__cause__, ValueError)


def test_v2_unselectable_named_reference_has_contextual_path_and_cause():
    base = _snapshot()
    recipe = SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 0.5, 1.5, 0.5),
        ),
    )
    snapshot = replace(
        base,
        geometry_recipe=recipe,
        mesh_settings=None,
        feature_history=derive_feature_history(recipe),
        named_regions=(),
        material_definitions=(),
        section_definitions=(),
        region_assignments=(),
        analysis_definitions=(),
    )
    payload = encode_project_v2(snapshot)
    payload["project"]["authoring"]["named_regions"] = [
        {"name": "Unselectable", "references": ["face:result"]}
    ]

    with pytest.raises(
        ProjectV2DecodeError,
        match=(
            r"named_regions\[0\]\.references\[0\] 无效："
            r"logical reference 'face:result' is not selectable"
        ),
    ) as caught:
        decode_project_v2(payload)

    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "named_region",
            (
                r"named_regions\[0\]\.references\[0\] 无效："
                r"logical reference 'face:result' is not selectable"
            ),
        ),
        (
            "local_control",
            (
                r"mesh_settings\.local_controls\[0\]\.target 无效："
                r"logical reference 'face:result' is not selectable"
            ),
        ),
        (
            "assignment",
            r"non-exact topology cannot carry NamedRegion",
        ),
        (
            "named_step_target",
            r"non-exact topology cannot carry NamedRegion",
        ),
    ),
)
def test_v2_nonexact_recipe_rejects_every_reference_consumer(
    case,
    expected,
):
    payload = _nonexact_payload()
    authoring = payload["project"]["authoring"]
    if case == "named_region":
        authoring["named_regions"] = [
            {"name": "Ambiguous", "references": ["face:result"]}
        ]
    elif case == "local_control":
        authoring["mesh_settings"] = {
            "size": 1.0,
            "order": 1,
            "cell_shape": "triangle",
            "local_controls": [
                {
                    "target": "face:result",
                    "size": 0.25,
                    "falloff": {
                        "reference": "global_size",
                        "start_factor": 0.0,
                        "end_factor": 2.0,
                    },
                }
            ],
        }
    elif case == "assignment":
        authoring["definitions"].update(
            {
                "materials": [{"name": "Steel", "properties": {}}],
                "sections": [
                    {
                        "name": "Solid",
                        "material": "Steel",
                        "section_type": "solid",
                        "properties": {},
                    }
                ],
                "assignments": [
                    {
                        "section_name": "Solid",
                        "region_name": "DOMAIN",
                        "beam_orientation": None,
                    }
                ],
            }
        )
    else:
        authoring["definitions"]["steps"] = [
            {
                "name": "Load",
                "procedure": "static",
                "metadata": {},
                "boundaries": [
                    {
                        "target": "BOTTOM",
                        "first_component": 1,
                        "last_component": 1,
                        "value": 0.0,
                    }
                ],
                "cloads": [],
                "edge_loads": [],
                "surface_loads": [],
                "line_loads": [],
                "gravity_loads": [],
                "outputs": [],
            }
        ]

    with pytest.raises(ProjectV2DecodeError, match=expected) as caught:
        decode_project_v2(payload)

    assert isinstance(caught.value.__cause__, ValueError)


def _nonexact_payload():
    recipe = _nonexact_recipe()
    return encode_project_v2(_geometry_only_snapshot(recipe))


def _nonexact_recipe():
    return SketchGeometry(
        "Composite",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 0.5, 1.5, 0.5),
        ),
    )


def _geometry_only_snapshot(recipe):
    return replace(
        _snapshot(),
        geometry_recipe=recipe,
        mesh_settings=None,
        feature_history=derive_feature_history(recipe),
        named_regions=(),
        material_definitions=(),
        section_definitions=(),
        region_assignments=(),
        analysis_definitions=(),
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("missing-format", r"\$ 缺少必需字段：format"),
        ("invalid-format", r"\$\.format 必须精确等于"),
        ("missing-schema", r"\$ 缺少必需字段：schema"),
        ("schema-zero", r"v2 decoder 不能读取 schema 0"),
        ("schema-one", r"v2 decoder 不能读取 schema 1"),
        ("schema-future", r"v2 decoder 不能读取 schema 3"),
        ("unknown-root", r"\$ 包含 v2 未知字段：legacy"),
        (
            "unknown-project",
            r"\$\.project 包含 v2 未知字段：legacy",
        ),
        (
            "unknown-authoring",
            r"\$\.project\.authoring 包含 v2 未知字段：legacy",
        ),
        (
            "unknown-definitions",
            r"definitions 包含 v2 未知字段：legacy",
        ),
        ("part-shape", r"authoring\.part 必须是 JSON object"),
        ("part-name-empty", r"authoring\.part\.name 不能为空"),
        ("part-name-type", r"authoring\.part\.name 必须是字符串"),
        ("part-body-empty", r"authoring\.part\.body_name 不能为空"),
        ("part-body-type", r"authoring\.part\.body_name 必须是字符串"),
        (
            "geometry-discriminator",
            r"authoring\.geometry\.type 是未知几何类型",
        ),
    ),
)
def test_v2_envelope_and_authoring_negative_matrix(case, expected):
    payload = encode_project_v2(_snapshot())
    _mutate_v2_envelope(payload, case)

    with pytest.raises(ProjectV2DecodeError, match=expected):
        decode_project_v2(payload)


def _mutate_v2_envelope(payload, case):
    project = payload["project"]
    authoring = project["authoring"]
    if case == "missing-format":
        del payload["format"]
    elif case == "invalid-format":
        payload["format"] = "legacy-project"
    elif case == "missing-schema":
        del payload["schema"]
    elif case == "schema-zero":
        payload["schema"] = 0
    elif case == "schema-one":
        payload["schema"] = 1
    elif case == "schema-future":
        payload["schema"] = 3
    elif case == "unknown-root":
        payload["legacy"] = None
    elif case == "unknown-project":
        project["legacy"] = None
    elif case == "unknown-authoring":
        authoring["legacy"] = None
    elif case == "unknown-definitions":
        authoring["definitions"]["legacy"] = []
    elif case == "part-shape":
        authoring["part"] = []
    elif case == "part-name-empty":
        authoring["part"]["name"] = ""
    elif case == "part-name-type":
        authoring["part"]["name"] = 1
    elif case == "part-body-empty":
        authoring["part"]["body_name"] = ""
    elif case == "part-body-type":
        authoring["part"]["body_name"] = 1
    else:
        authoring["geometry"]["type"] = "LegacyGeometry"


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "discriminator",
            r"beam_orientation\.type 只接受 'local_y_reference'",
        ),
        (
            "vector-length",
            r"beam_orientation\.vector 必须恰有三个分量",
        ),
        (
            "vector-value",
            r"beam_orientation\.vector\[1\] 必须是有限实数",
        ),
    ),
)
def test_v2_orientation_negative_matrix(case, expected):
    payload = encode_project_v2(_snapshot())
    assignment = payload["project"]["authoring"]["definitions"][
        "assignments"
    ][0]
    assignment["beam_orientation"] = {
        "type": "local_y_reference",
        "vector": [0.0, 1.0, 0.0],
    }
    if case == "discriminator":
        assignment["beam_orientation"]["type"] = "global_vector"
    elif case == "vector-length":
        assignment["beam_orientation"]["vector"] = [0.0, 1.0]
    else:
        assignment["beam_orientation"]["vector"][1] = "up"

    with pytest.raises(ProjectV2DecodeError, match=expected):
        decode_project_v2(payload)


@pytest.mark.parametrize(
    ("case", "expected", "has_cause"),
    (
        ("contract-bool", r"logical_topology\.contract 必须是严格整数", False),
        ("contract-float", r"logical_topology\.contract 必须是严格整数", False),
        ("contract-string", r"logical_topology\.contract 必须是严格整数", False),
        ("contract-unknown", r"logical_topology\.contract 不支持：999", False),
        (
            "topology-unknown",
            r"logical_topology 包含 v2 未知字段：legacy",
            False,
        ),
        (
            "signature-missing",
            r"logical_topology 缺少必需字段：signature",
            False,
        ),
        (
            "signature-unknown",
            r"logical_topology\.signature 包含 v2 未知字段：legacy",
            False,
        ),
        (
            "duplicate-entity",
            r"logical_topology 无效：topology fingerprint contains duplicate",
            True,
        ),
        (
            "entity-unknown",
            r"logical_topology\.signature\.entities\[0\] "
            r"包含 v2 未知字段：legacy",
            False,
        ),
        (
            "kind-prefix",
            r"logical_topology\.signature\.entities\[0\] 无效："
            r"fingerprint entity kind",
            True,
        ),
    ),
)
def test_v2_topology_shape_negative_matrix(case, expected, has_cause):
    payload = encode_project_v2(_snapshot())
    topology = payload["project"]["authoring"]["logical_topology"]
    signature = topology["signature"]
    entities = signature["entities"]
    if case == "contract-bool":
        topology["contract"] = True
    elif case == "contract-float":
        topology["contract"] = 2.0
    elif case == "contract-string":
        topology["contract"] = "2"
    elif case == "contract-unknown":
        topology["contract"] = 999
    elif case == "topology-unknown":
        topology["legacy"] = None
    elif case == "signature-missing":
        del topology["signature"]
    elif case == "signature-unknown":
        signature["legacy"] = None
    elif case == "duplicate-entity":
        entities.append(deepcopy(entities[0]))
    elif case == "entity-unknown":
        entities[0]["legacy"] = None
    else:
        entities[0]["kind"] = "face"

    with pytest.raises(ProjectV2DecodeError, match=expected) as caught:
        decode_project_v2(payload)

    assert (caught.value.__cause__ is not None) is has_cause


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("zero-parts", r"必须且只能包含一个 part"),
        ("multiple-parts", r"必须且只能包含一个 part"),
        ("imported-source", r"source_kind 本阶段只接受 'native'"),
    ),
)
def test_v2_encoder_source_and_part_negative_matrix(case, expected):
    snapshot = _snapshot()
    if case == "zero-parts":
        snapshot = replace(snapshot, parts=())
    elif case == "multiple-parts":
        snapshot = replace(
            snapshot,
            parts=(NativePart(), NativePart("Part-2", "Body-2")),
        )
    else:
        snapshot = replace(snapshot, source_kind="imported")

    with pytest.raises(ProjectV2EncodeError, match=expected):
        encode_project_v2(snapshot)


@pytest.mark.parametrize(
    ("case", "expected", "has_cause"),
    (
        (
            "duplicate-material",
            r"definitions\.materials 包含忽略大小写后重复的名称",
            False,
        ),
        (
            "broken-material",
            r"definitions\.sections\[0\]\.material 无效："
            r"references missing material",
            True,
        ),
        (
            "broken-section",
            r"definitions\.assignments\[0\]\.section_name 无效："
            r"references missing section",
            True,
        ),
        (
            "broken-region",
            r"definitions\.assignments\[0\]\.region_name",
            True,
        ),
        (
            "beam-assignment",
            r"definitions\.assignments\[0\] section type 'beam' "
            r"is incompatible",
            True,
        ),
        (
            "line-load-capability",
            r"definitions\.steps\[0\]\.line_loads\[0\]\.target",
            True,
        ),
        (
            "surface-load-dimension",
            r"definitions\.steps\[0\]\.surface_loads\[0\]\.surface",
            True,
        ),
        (
            "reserved-orientation",
            r"definitions\.materials\[0\]\.properties"
            r"\.beam_local_y_reference 无效",
            True,
        ),
    ),
)
def test_v2_definition_link_and_capability_negative_matrix(
    case,
    expected,
    has_cause,
):
    payload = encode_project_v2(_snapshot())
    definitions = payload["project"]["authoring"]["definitions"]
    step = definitions["steps"][0]
    if case == "duplicate-material":
        duplicate = deepcopy(definitions["materials"][0])
        duplicate["name"] = duplicate["name"].swapcase()
        definitions["materials"].append(duplicate)
    elif case == "broken-material":
        definitions["sections"][0]["material"] = "Missing"
    elif case == "broken-section":
        definitions["assignments"][0]["section_name"] = "Missing"
    elif case == "broken-region":
        definitions["assignments"][0]["region_name"] = "Missing"
    elif case == "beam-assignment":
        definitions["sections"][0]["section_type"] = "beam"
    elif case == "line-load-capability":
        step["line_loads"] = [
            {
                "target": "Loaded",
                "vector": [0.0, -1.0],
                "coordinate_system": "global",
            }
        ]
    elif case == "surface-load-dimension":
        step["surface_loads"] = [
            {
                "surface": "Loaded",
                "vector": [0.0, -1.0],
                "magnitude": None,
                "load_type": "traction",
            }
        ]
    else:
        definitions["materials"][0]["properties"][
            "beam_local_y_reference"
        ] = [0.0, 1.0, 0.0]

    with pytest.raises(ProjectV2DecodeError, match=expected) as caught:
        decode_project_v2(payload)

    assert (caught.value.__cause__ is not None) is has_cause


@pytest.mark.parametrize(
    ("collection", "seed"),
    (
        ("boundaries", None),
        (
            "cloads",
            {"target": "Fixed", "component": 1, "value": 1.0},
        ),
        (
            "line_loads",
            {
                "target": "Loaded",
                "vector": [0.0, -1.0],
                "coordinate_system": "global",
            },
        ),
        (
            "gravity_loads",
            {"acceleration": [0.0, -9.81], "target": "DOMAIN"},
        ),
    ),
)
def test_v2_integer_analysis_targets_are_rejected_at_field_path(
    collection,
    seed,
):
    payload = encode_project_v2(_snapshot())
    step = payload["project"]["authoring"]["definitions"]["steps"][0]
    if seed is not None:
        step[collection] = [seed]
    step[collection][0]["target"] = 1
    expected = (
        rf"definitions\.steps\[0\]\.{collection}\[0\]\.target "
        r"必须是 non-empty stable region name"
    )

    with pytest.raises(ProjectV2DecodeError, match=expected):
        decode_project_v2(payload)


@pytest.mark.parametrize(
    ("case", "expected", "has_cause"),
    (
        (
            "missing",
            r"local_controls\[0\] 缺少必需字段：falloff",
            False,
        ),
        (
            "unknown-reference",
            r"local_controls\[0\]\.falloff 无效："
            r"mesh-size falloff reference",
            True,
        ),
        (
            "bool-factor",
            r"falloff\.start_factor 必须是有限实数",
            False,
        ),
        (
            "string-factor",
            r"falloff\.start_factor 必须是有限实数",
            False,
        ),
        (
            "nonfinite-factor",
            r"falloff\.start_factor 必须是有限实数",
            False,
        ),
        (
            "negative-factor",
            r"local_controls\[0\]\.falloff 无效："
            r"mesh-size falloff requires 0 <= start_factor < end_factor",
            True,
        ),
        (
            "unordered-range",
            r"local_controls\[0\]\.falloff 无效："
            r"mesh-size falloff requires 0 <= start_factor < end_factor",
            True,
        ),
        (
            "duplicate-profile",
            r"mesh_settings 无效：同一个几何实体和 falloff profile",
            True,
        ),
        (
            "target-radius-noncircle",
            r"local_controls\[0\]\.target 无效："
            r"logical target .* has no proven circular-hole radius",
            True,
        ),
    ),
)
def test_v2_falloff_negative_matrix(case, expected, has_cause):
    payload = encode_project_v2(_snapshot())
    settings = payload["project"]["authoring"]["mesh_settings"]
    control = settings["local_controls"][0]
    falloff = control["falloff"]
    if case == "missing":
        del control["falloff"]
    elif case == "unknown-reference":
        falloff["reference"] = "exponential"
    elif case == "bool-factor":
        falloff["start_factor"] = True
    elif case == "string-factor":
        falloff["start_factor"] = "0"
    elif case == "nonfinite-factor":
        falloff["start_factor"] = float("inf")
    elif case == "negative-factor":
        falloff["start_factor"] = -0.1
    elif case == "unordered-range":
        falloff["start_factor"] = falloff["end_factor"]
    elif case == "duplicate-profile":
        settings["local_controls"].append(deepcopy(control))
    else:
        falloff["reference"] = "target_radius"

    with pytest.raises(ProjectV2DecodeError, match=expected) as caught:
        decode_project_v2(payload)

    assert (caught.value.__cause__ is not None) is has_cause


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "unknown",
            r"snapshot\.named_regions\[0\]\.references\[0\] 无效："
            r"unknown logical reference",
        ),
        (
            "wrong-kind",
            r"snapshot\.mesh_settings\.local_controls\[0\]\.target 无效："
            r"logical reference kind 'body' is not allowed",
        ),
        (
            "unselectable",
            r"snapshot\.named_regions\[0\]\.references\[0\] 无效："
            r"logical reference 'face:result' is not selectable",
        ),
    ),
)
def test_v2_encode_contextual_reference_paths_and_causes(case, expected):
    if case == "unknown":
        snapshot = replace(
            _geometry_only_snapshot(
                RectangleGeometry("Rectangle", 4.0, 2.0)
            ),
            named_regions=(
                NamedRegion(
                    "Unknown",
                    (LogicalEntityRef("edge:missing"),),
                ),
            ),
        )
    elif case == "wrong-kind":
        control = LocalMeshControl(
            LogicalEntityRef("edge:right"),
            0.25,
        )
        object.__setattr__(
            control,
            "target",
            LogicalEntityRef("body:domain"),
        )
        snapshot = replace(
            _geometry_only_snapshot(
                RectangleGeometry("Rectangle", 4.0, 2.0)
            ),
            mesh_settings=MeshSettings(1.0, local_controls=(control,)),
        )
    else:
        snapshot = replace(
            _geometry_only_snapshot(_nonexact_recipe()),
            named_regions=(
                NamedRegion(
                    "Unselectable",
                    (LogicalEntityRef("face:result"),),
                ),
            ),
        )

    with pytest.raises(ProjectV2EncodeError, match=expected) as caught:
        encode_project_v2(snapshot)

    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "broken-material",
            r"snapshot\.section_definitions\[0\]\.material 无效",
        ),
        (
            "broken-section",
            r"snapshot\.region_assignments\[0\]\.section_name 无效",
        ),
        (
            "broken-region",
            r"snapshot\.region_assignments\[0\]\.region_name",
        ),
        (
            "beam-assignment",
            r"snapshot\.region_assignments\[0\] section type 'beam'",
        ),
        (
            "line-load",
            r"snapshot\.analysis_definitions\[0\]\.line_loads\[0\]\.target",
        ),
        (
            "surface-load",
            r"snapshot\.analysis_definitions\[0\]\.surface_loads\[0\]\.surface",
        ),
        (
            "reserved-orientation",
            r"snapshot\.material_definitions\[0\]\.properties"
            r"\.beam_local_y_reference 无效",
        ),
    ),
)
def test_v2_encode_definition_context_paths_and_causes(case, expected):
    snapshot = _snapshot()
    section = snapshot.section_definitions[0]
    assignment = snapshot.region_assignments[0]
    step = snapshot.analysis_definitions[0]
    material = snapshot.material_definitions[0]
    if case == "broken-material":
        snapshot = replace(
            snapshot,
            section_definitions=(replace(section, material="Missing"),),
        )
    elif case == "broken-section":
        snapshot = replace(
            snapshot,
            region_assignments=(
                replace(assignment, section_name="Missing"),
            ),
        )
    elif case == "broken-region":
        snapshot = replace(
            snapshot,
            region_assignments=(
                replace(assignment, region_name="Missing"),
            ),
        )
    elif case == "beam-assignment":
        snapshot = replace(
            snapshot,
            section_definitions=(
                replace(section, section_type="beam"),
            ),
        )
    elif case == "line-load":
        snapshot = replace(
            snapshot,
            analysis_definitions=(
                replace(
                    step,
                    line_loads=(
                        LineLoad("Loaded", (0.0, -1.0)),
                    ),
                ),
            ),
        )
    elif case == "surface-load":
        snapshot = replace(
            snapshot,
            analysis_definitions=(
                replace(
                    step,
                    surface_loads=(
                        SurfaceLoad("Loaded", (0.0, -1.0)),
                    ),
                ),
            ),
        )
    else:
        snapshot = replace(
            snapshot,
            material_definitions=(
                replace(
                    material,
                    properties={
                        **material.properties,
                        "beam_local_y_reference": [0.0, 1.0, 0.0],
                    },
                ),
            ),
        )

    with pytest.raises(ProjectV2EncodeError, match=expected) as caught:
        encode_project_v2(snapshot)

    assert isinstance(caught.value.__cause__, ValueError)


def test_v2_errors_participate_in_generic_hierarchy():
    assert issubclass(ProjectV2DecodeError, ProjectDecodeError)
    assert issubclass(ProjectV2EncodeError, ProjectEncodeError)


def test_v2_atomic_writer_preserves_old_target_on_encode_failure(tmp_path):
    target = tmp_path / "project.femproj"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(ProjectV2EncodeError):
        save_project_v2(target, replace(_snapshot(), feature_history=()))

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_v2_writer_rejects_source_evidence_without_touching_target(tmp_path):
    snapshot = _snapshot()
    step = snapshot.analysis_definitions[0]
    output = OutputRequest(
        "field",
        "node",
        ("u", "u"),
        {"frequency": "1"},
        OutputSourceEvidence(
            "abaqus",
            (("frequency", "1"),),
            ("field",),
            (("nset", "Tip"),),
            ("futureflag",),
        ),
    )
    guarded = replace(
        snapshot,
        analysis_definitions=(replace(step, outputs=(output,)),),
    )
    target = tmp_path / "project.femproj"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(
        ProjectV2EncodeError,
        match=r"source_evidence.*v2",
    ):
        save_project_v2(target, guarded)

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_v2_json_key_order_is_independent_of_metadata_insertion_order():
    first = _snapshot()
    step = first.analysis_definitions[0]
    second = replace(
        first,
        analysis_definitions=(
            replace(step, metadata={"z": 2, "a": 1}),
        ),
    )
    third = replace(
        first,
        analysis_definitions=(
            replace(step, metadata={"a": 1, "z": 2}),
        ),
    )

    assert dumps_project_v2(second) == dumps_project_v2(third)
    json.loads(dumps_project_v2(second))
