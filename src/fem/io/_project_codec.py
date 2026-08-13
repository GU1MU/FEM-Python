"""Shared strict JSON and crash-safe file helpers for project codecs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, TypeVar

from fem.application.definitions import (
    RegionAssignment,
    SectionDefinition,
)
from fem.core.immutable_json import thaw_json_mapping
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.elements import BeamOrientation
from fem.geometry.recipes import (
    BooleanBodyContext,
    BooleanGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    FaceSeedConnectionProof,
    FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    FaceSketchBooleanStepProof,
    FaceSketchWorkplaneStrategy,
    MovedGeometry,
    MultiBodyGeometry,
    PartBooleanContext,
    PathSweptGeometry,
    PlateWithHoleGeometry,
    PlanarBooleanContext,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchArc,
    SketchAngleDimension,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchConcentricConstraint,
    SketchConstraint,
    SketchDistanceDimension,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchExternalCoincidence,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchGeometry,
    SketchFixedConstraint,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchPerpendicularConstraint,
    SketchRadiusDimension,
    SketchRectangle,
    SketchTangentConstraint,
    SketchVerticalConstraint,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.geometry.recipe_analysis import legacy_sketch_to_strict
from fem.geometry.references import LogicalEntityRef, logical_ref_sort_key

from ._project_errors import (
    ProjectDecodeError,
    ProjectEncodeError,
)
from ._atomic_text import atomic_write_verified_text


_VerifiedT = TypeVar("_VerifiedT")
_SKETCH_CONSTRAINT_CODEC = ContextVar("sketch_constraint_codec", default=False)


@contextmanager
def sketch_constraint_codec():
    """Enable schema-v13 sketch constraint fields for nested geometry codecs."""

    token = _SKETCH_CONSTRAINT_CODEC.set(True)
    try:
        yield
    finally:
        _SKETCH_CONSTRAINT_CODEC.reset(token)


@dataclass(frozen=True, slots=True)
class ProjectFieldCodecPolicy:
    """Version policy for shared current-authoring field codecs."""

    version_label: str
    decode_error: type[ProjectDecodeError]
    encode_error: type[ProjectEncodeError]
    require_current_fields: bool
    assignment_orientation: bool
    allow_wire_geometry: bool = False
    allow_strict_sketch: bool = False
    extrusion_source_faces: bool = False
    displacement_region_targets: bool = False
    body_force_loads: bool = False
    allow_multi_body: bool = False
    allow_planar_boolean: bool = False
    allow_part_boolean: bool = False
    allow_revolved_geometry: bool = False
    allow_path_swept_geometry: bool = False
    allow_face_sketch_boolean: bool = False


def loads_json_strict(
    data: str | bytes | bytearray,
    *,
    error_type: type[ProjectDecodeError] = ProjectDecodeError,
    document_label: str = "项目",
) -> Any:
    """Parse one UTF-8 JSON document with strict project-file semantics."""

    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise error_type(
                f"{document_label}文件不是有效的 UTF-8 文本"
            ) from exc
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("data 必须是 str、bytes 或 bytearray")

    def unique_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise error_type(
                    f"{document_label} JSON 包含重复键：{key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise error_type(
            f"{document_label} JSON 包含非有限数值：{value}"
        )

    def strict_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise error_type(
                f"{document_label} JSON 包含非有限数值：{value}"
            )
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=strict_float,
        )
    except ProjectDecodeError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise error_type(
            f"{document_label} JSON 无效：{exc}"
        ) from exc


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    final_newline: bool = False,
    separators: tuple[str, str] | None = None,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
    error_message: str = "项目包含无法编码的值",
) -> str:
    """Serialize JSON without ASCII escaping or non-finite numbers."""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
            separators=separators,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise error_type(f"{error_message}：{exc}") from exc
    if final_newline:
        return serialized + "\n"
    return serialized


def dumps_canonical_json(
    value: Any,
    *,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
    error_message: str = "项目包含无法编码的值",
) -> str:
    """Return canonical current-project JSON with one final LF."""

    return dumps_json(
        value,
        indent=2,
        sort_keys=True,
        final_newline=True,
        error_type=error_type,
        error_message=error_message,
    )


def dumps_compact_canonical_json(
    value: Any,
    *,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
    error_message: str = "项目包含无法编码的值",
) -> str:
    """Return deterministic compact JSON with one final LF."""

    return dumps_json(
        value,
        indent=None,
        sort_keys=True,
        final_newline=True,
        separators=(",", ":"),
        error_type=error_type,
        error_message=error_message,
    )


def unwrap_project_snapshot(
    snapshot: Any,
    *,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
) -> Any:
    """Return a detached copy of a project or save snapshot."""

    try:
        from fem.application.session import (
            ProjectSaveSnapshot,
            ProjectSnapshot,
        )
    except ImportError as exc:
        raise error_type(
            "fem.application.session 尚未提供项目 snapshot"
        ) from exc

    if type(snapshot) is ProjectSaveSnapshot:
        project = snapshot.snapshot
    elif type(snapshot) is ProjectSnapshot:
        project = snapshot
    else:
        raise error_type(
            "save/encode 需要 ProjectSnapshot 或 ProjectSaveSnapshot"
        )
    if type(project) is not ProjectSnapshot:
        raise error_type("保存 snapshot 未包含有效的 ProjectSnapshot")
    return deepcopy(project)


def borrow_project_snapshot(
    snapshot: Any,
    *,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
) -> Any:
    """Borrow one already-detached snapshot for synchronous read-only encoding."""

    try:
        from fem.application.session import (
            ProjectSaveSnapshot,
            ProjectSnapshot,
        )
    except ImportError as exc:
        raise error_type(
            "fem.application.session 尚未提供项目 snapshot"
        ) from exc

    if type(snapshot) is ProjectSaveSnapshot:
        project = snapshot._snapshot_for_persistence()
    elif type(snapshot) is ProjectSnapshot:
        project = snapshot
    else:
        raise error_type(
            "save/encode 需要 ProjectSnapshot 或 ProjectSaveSnapshot"
        )
    if type(project) is not ProjectSnapshot:
        raise error_type("保存 snapshot 未包含有效的 ProjectSnapshot")
    return project


def atomic_write_project(
    path: str | Path,
    serialized: str,
    *,
    verifier: Callable[[Path], _VerifiedT],
    semantic_encoder: Callable[[_VerifiedT], Any],
    expected_semantic: Any,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
    mismatch_message: str = "临时项目文件校验后与保存 snapshot 不一致",
    checkpoint: Callable[[], Any] | None = None,
    replace_func: Callable[[str | Path, str | Path], Any] | None = None,
    unlink_func: Callable[[Path], Any] | None = None,
) -> Path:
    """Install serialized project text through the shared atomic foundation."""

    return atomic_write_verified_text(
        path,
        serialized,
        verifier=verifier,
        semantic_encoder=semantic_encoder,
        expected_semantic=expected_semantic,
        error_type=error_type,
        mismatch_message=mismatch_message,
        checkpoint=checkpoint,
        replace_func=os.replace if replace_func is None else replace_func,
        unlink_func=unlink_func,
        _mkstemp_func=tempfile.mkstemp,
        _fdopen_func=os.fdopen,
        _fsync_func=os.fsync,
        _close_func=os.close,
    )


_PRIMITIVE_GEOMETRY_FIELDS: dict[
    type[Any],
    tuple[str, tuple[str, ...], tuple[str, ...]],
] = {
    RectangleGeometry: (
        "RectangleGeometry",
        ("name",),
        ("width", "height"),
    ),
    DiskGeometry: ("DiskGeometry", ("name",), ("radius",)),
    BoxGeometry: (
        "BoxGeometry",
        ("name",),
        ("width", "depth", "height"),
    ),
    CylinderGeometry: (
        "CylinderGeometry",
        ("name",),
        ("radius", "height"),
    ),
    PlateWithHoleGeometry: (
        "PlateWithHoleGeometry",
        ("name",),
        ("width", "height", "hole_x", "hole_y", "hole_radius"),
    ),
}
_PRIMITIVE_GEOMETRY_TYPES = {
    wire_type: (recipe_type, string_fields, number_fields)
    for recipe_type, (
        wire_type,
        string_fields,
        number_fields,
    ) in _PRIMITIVE_GEOMETRY_FIELDS.items()
}


def decode_geometry_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> Any:
    """Decode one geometry recipe under an explicit wire-version policy."""

    data = _field_mapping(value, path, policy.decode_error)
    if "type" not in data:
        raise policy.decode_error(f"{path} 缺少必需字段：type")
    kind = _field_string(data["type"], f"{path}.type", policy.decode_error)
    primitive = _PRIMITIVE_GEOMETRY_TYPES.get(kind)
    if primitive is not None:
        recipe_type, string_fields, number_fields = primitive
        _field_keys(
            data,
            path,
            required={"type", *string_fields, *number_fields},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        arguments = {
            field_name: _field_string(
                data[field_name],
                f"{path}.{field_name}",
                policy.decode_error,
            )
            for field_name in string_fields
        }
        arguments.update(
            {
                field_name: _field_number(
                    data[field_name],
                    f"{path}.{field_name}",
                    policy.decode_error,
                    policy=policy,
                )
                for field_name in number_fields
            }
        )
        return _field_construct(recipe_type, path, policy, **arguments)
    if kind == "WireGeometry":
        if not policy.allow_wire_geometry:
            raise policy.decode_error(
                f"{path}.type 的几何类型无法由 {policy.version_label} "
                "无损解码：'WireGeometry'"
            )
        _field_keys(
            data,
            path,
            required={"type", "name", "points", "members"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        points = tuple(
            _decode_wire_point_field(
                item,
                f"{path}.points[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_array(
                    data["points"],
                    f"{path}.points",
                    policy.decode_error,
                )
            )
        )
        members = tuple(
            _decode_wire_member_field(
                item,
                f"{path}.members[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_array(
                    data["members"],
                    f"{path}.members",
                    policy.decode_error,
                )
            )
        )
        return _field_construct(
            WireGeometry,
            path,
            policy,
            _field_string(data["name"], f"{path}.name", policy.decode_error),
            points,
            members,
        )
    if kind == "SketchGeometry":
        if "contours" in data:
            _field_keys(
                data,
                path,
                required={"type", "name", "contours"},
                optional=set(),
                policy=policy,
                error_type=policy.decode_error,
            )
            contours = tuple(
                decode_contour_field(
                    item,
                    f"{path}.contours[{index}]",
                    policy=policy,
                )
                for index, item in enumerate(
                    _field_array(
                        data["contours"],
                        f"{path}.contours",
                        policy.decode_error,
                    )
                )
            )
            return _field_construct(
                SketchGeometry,
                path,
                policy,
                _field_string(
                    data["name"],
                    f"{path}.name",
                    policy.decode_error,
                ),
                contours,
            )
        if not policy.allow_strict_sketch:
            raise policy.decode_error(
                f"{path} 的 curve-based sketch 无法由 "
                f"{policy.version_label} 解码"
            )
        required = {"type", "name", "plane", "points", "curves"}
        if _SKETCH_CONSTRAINT_CODEC.get():
            required.add("constraints")
        _field_keys(
            data,
            path,
            required=required,
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        plane = _decode_sketch_plane_field(
            data["plane"],
            f"{path}.plane",
            policy=policy,
        )
        points = tuple(
            _decode_sketch_point_field(
                item,
                f"{path}.points[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_array(data["points"], f"{path}.points", policy.decode_error)
            )
        )
        curves = tuple(
            _decode_sketch_curve_field(
                item,
                f"{path}.curves[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_array(data["curves"], f"{path}.curves", policy.decode_error)
            )
        )
        constraints = (
            tuple(
                _decode_sketch_constraint_field(
                    item,
                    f"{path}.constraints[{index}]",
                    policy=policy,
                )
                for index, item in enumerate(
                    _field_array(
                        data["constraints"],
                        f"{path}.constraints",
                        policy.decode_error,
                    )
                )
            )
            if _SKETCH_CONSTRAINT_CODEC.get()
            else ()
        )
        return _field_construct(
            SketchGeometry,
            path,
            policy,
            _field_string(data["name"], f"{path}.name", policy.decode_error),
            plane,
            points,
            curves,
            constraints,
        )
    if kind == "FaceSketchBooleanGeometry":
        if not policy.allow_face_sketch_boolean:
            raise policy.decode_error(
                f"{path}.type 的几何类型无法由 {policy.version_label} "
                "无损解码：'FaceSketchBooleanGeometry'"
            )
        _field_keys(
            data,
            path,
            required={
                "type",
                "base",
                "feature_id",
                "name",
                "support_face_id",
                "workplane_strategy",
                "sketch",
                "operation",
                "direction",
                "distance",
                "participating_profile_ids",
                "external_references",
                "external_coincidences",
                "step_proofs",
            },
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        sketch = decode_geometry_field(
            data["sketch"], f"{path}.sketch", policy=policy
        )
        if type(sketch) is not SketchGeometry or not sketch.is_strict:
            raise policy.decode_error(f"{path}.sketch 必须是严格平面草图")
        result = _field_construct(
            FaceSketchBooleanGeometry,
            path,
            policy,
            decode_geometry_field(data["base"], f"{path}.base", policy=policy),
            _field_string(data["feature_id"], f"{path}.feature_id", policy.decode_error),
            _field_string(data["name"], f"{path}.name", policy.decode_error),
            _field_string(
                data["support_face_id"], f"{path}.support_face_id", policy.decode_error
            ),
            _decode_face_sketch_workplane_strategy(
                data["workplane_strategy"], f"{path}.workplane_strategy", policy=policy
            ),
            sketch,
            _field_enum(
                data["operation"], f"{path}.operation", FaceSketchBooleanOperation, policy
            ),
            _field_enum(
                data["direction"], f"{path}.direction", FaceSketchBooleanDirection, policy
            ),
            _field_number(data["distance"], f"{path}.distance", policy.decode_error, policy=policy),
            tuple(
                _field_string(item, f"{path}.participating_profile_ids[{index}]", policy.decode_error)
                for index, item in enumerate(
                    _field_array(data["participating_profile_ids"], f"{path}.participating_profile_ids", policy.decode_error)
                )
            ),
            tuple(
                _decode_face_sketch_external_reference(
                    item, f"{path}.external_references[{index}]", policy=policy
                )
                for index, item in enumerate(
                    _field_array(data["external_references"], f"{path}.external_references", policy.decode_error)
                )
            ),
            tuple(
                _decode_face_sketch_external_coincidence(
                    item, f"{path}.external_coincidences[{index}]", policy=policy
                )
                for index, item in enumerate(
                    _field_array(data["external_coincidences"], f"{path}.external_coincidences", policy.decode_error)
                )
            ),
            tuple(
                _decode_face_sketch_step_proof(
                    item, f"{path}.step_proofs[{index}]", policy=policy
                )
                for index, item in enumerate(
                    _field_array(data["step_proofs"], f"{path}.step_proofs", policy.decode_error)
                )
            ),
        )
        if tuple(item.profile_id for item in result.step_proofs) != result.participating_profile_ids:
            raise policy.decode_error(f"{path}.step_proofs 必须完整覆盖参与轮廓并保持稳定顺序")
        if any(not item.result_entities or not item.topology_mappings for item in result.step_proofs):
            raise policy.decode_error(f"{path}.step_proofs 包含不完整的布尔证明")
        return result
    if kind == "MovedGeometry":
        required = {"type", "base", "dx", "dy"}
        optional = {"dz"}
        if policy.require_current_fields:
            required.add("dz")
            optional.clear()
        _field_keys(
            data,
            path,
            required=required,
            optional=optional,
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            MovedGeometry,
            path,
            policy,
            decode_geometry_field(
                data["base"],
                f"{path}.base",
                policy=policy,
            ),
            _field_number(
                data["dx"],
                f"{path}.dx",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["dy"],
                f"{path}.dy",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["dz"] if "dz" in data else 0.0,
                f"{path}.dz",
                policy.decode_error,
                policy=policy,
            ),
        )
    if kind == "RotatedGeometry":
        _field_keys(
            data,
            path,
            required={"type", "base", "axis", "angle_degrees"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            RotatedGeometry,
            path,
            policy,
            decode_geometry_field(
                data["base"],
                f"{path}.base",
                policy=policy,
            ),
            _field_string(
                data["axis"],
                f"{path}.axis",
                policy.decode_error,
            ),
            _field_number(
                data["angle_degrees"],
                f"{path}.angle_degrees",
                policy.decode_error,
                policy=policy,
            ),
        )
    if kind == "ExtrudedGeometry":
        source_field = {"source_face_ids"} if policy.extrusion_source_faces else set()
        _field_keys(
            data,
            path,
            required={"type", "base", "height", *source_field},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        base = decode_geometry_field(
            data["base"],
            f"{path}.base",
            policy=policy,
        )
        height = _field_number(
            data["height"],
            f"{path}.height",
            policy.decode_error,
            policy=policy,
        )
        source_face_ids: tuple[str, ...] = ()
        if policy.extrusion_source_faces:
            values = _field_array(
                data["source_face_ids"],
                f"{path}.source_face_ids",
                policy.decode_error,
            )
            parsed_ids: list[str] = []
            seen_ids: set[str] = set()
            for index, item in enumerate(values):
                item_path = f"{path}.source_face_ids[{index}]"
                logical_id = _field_string(
                    item,
                    item_path,
                    policy.decode_error,
                )
                try:
                    reference = LogicalEntityRef(logical_id)
                except (TypeError, ValueError) as error:
                    raise policy.decode_error(
                        f"{item_path} 不是有效 logical ID：{error}"
                    ) from error
                if reference.kind != "face":
                    raise policy.decode_error(
                        f"{item_path} 必须引用 face logical ID"
                    )
                if logical_id in seen_ids:
                    raise policy.decode_error(
                        f"{item_path} 与前一项重复：{logical_id!r}"
                    )
                seen_ids.add(logical_id)
                parsed_ids.append(logical_id)
            source_face_ids = tuple(parsed_ids)
        try:
            return ExtrudedGeometry(base, height, source_face_ids)
        except (TypeError, ValueError) as error:
            logical_id = getattr(error, "logical_id", None)
            if logical_id in source_face_ids:
                error_path = (
                    f"{path}.source_face_ids"
                    f"[{source_face_ids.index(logical_id)}]"
                )
            else:
                error_path = path
            raise policy.decode_error(f"{error_path} 无效：{error}") from error
    if kind == "RevolvedGeometry":
        if not policy.allow_revolved_geometry:
            raise policy.decode_error(f"{path}.type 是未知几何类型：{kind!r}")
        _field_keys(
            data,
            path,
            required={"type", "base", "axis", "angle_degrees", "source_face_ids"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            RevolvedGeometry,
            path,
            policy,
            decode_geometry_field(data["base"], f"{path}.base", policy=policy),
            _field_string(data["axis"], f"{path}.axis", policy.decode_error),
            _field_number(
                data["angle_degrees"], f"{path}.angle_degrees",
                policy.decode_error, policy=policy,
            ),
            tuple(
                _field_string(
                    item, f"{path}.source_face_ids[{index}]", policy.decode_error,
                )
                for index, item in enumerate(
                    _field_array(
                        data["source_face_ids"], f"{path}.source_face_ids",
                        policy.decode_error,
                    )
                )
            ),
        )
    if kind == "PathSweptGeometry":
        if not policy.allow_path_swept_geometry:
            raise policy.decode_error(f"{path}.type 是未知几何类型：{kind!r}")
        _field_keys(
            data,
            path,
            required={
                "type", "base", "path", "source_face_ids", "frame_strategy"
            },
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        path_recipe = decode_geometry_field(
            data["path"], f"{path}.path", policy=policy,
        )
        if type(path_recipe) is not WireGeometry:
            raise policy.decode_error(f"{path}.path 必须是 WireGeometry")
        return _field_construct(
            PathSweptGeometry,
            path,
            policy,
            decode_geometry_field(data["base"], f"{path}.base", policy=policy),
            path_recipe,
            tuple(
                _field_string(
                    item, f"{path}.source_face_ids[{index}]", policy.decode_error,
                )
                for index, item in enumerate(
                    _field_array(
                        data["source_face_ids"], f"{path}.source_face_ids",
                        policy.decode_error,
                    )
                )
            ),
            _field_string(
                data["frame_strategy"], f"{path}.frame_strategy", policy.decode_error,
            ),
        )
    if kind == "BooleanGeometry":
        optional = set()
        if policy.allow_multi_body:
            optional.add("body_context")
        if policy.allow_planar_boolean:
            optional.add("planar_context")
        if policy.allow_part_boolean:
            optional.add("part_context")
        _field_keys(
            data,
            path,
            required={"type", "name", "operation", "object", "tool"},
            optional=optional,
            policy=policy,
            error_type=policy.decode_error,
        )
        result = _field_construct(
            BooleanGeometry,
            path,
            policy,
            _field_string(
                data["name"],
                f"{path}.name",
                policy.decode_error,
            ),
            _field_string(
                data["operation"],
                f"{path}.operation",
                policy.decode_error,
            ),
            decode_geometry_field(
                data["object"],
                f"{path}.object",
                policy=policy,
            ),
            decode_geometry_field(
                data["tool"],
                f"{path}.tool",
                policy=policy,
            ),
            (
                _decode_boolean_body_context(
                    data["body_context"],
                    f"{path}.body_context",
                    policy=policy,
                )
                if "body_context" in data
                else None
            ),
            (
                _decode_planar_boolean_context(
                    data["planar_context"],
                    f"{path}.planar_context",
                    policy=policy,
                )
                if "planar_context" in data
                else None
            ),
            (
                _decode_part_boolean_context(
                    data["part_context"],
                    f"{path}.part_context",
                    policy=policy,
                )
                if "part_context" in data
                else None
            ),
        )
        return result
    if kind == "MultiBodyGeometry":
        if not policy.allow_multi_body:
            raise policy.decode_error(
                f"{path}.type 是未知几何类型：{kind!r}"
            )
        _field_keys(
            data,
            path,
            required={
                "type",
                "name",
                "bodies",
                "retired_body_ids",
                "retired_boolean_feature_ids",
            },
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        body_values = _field_array(
            data["bodies"],
            f"{path}.bodies",
            policy.decode_error,
        )
        bodies: list[SolidBody] = []
        for index, value in enumerate(body_values):
            body_path = f"{path}.bodies[{index}]"
            body_data = _field_mapping(value, body_path, policy.decode_error)
            _field_keys(
                body_data,
                body_path,
                required={"id", "name", "recipe"},
                optional=set(),
                policy=policy,
                error_type=policy.decode_error,
            )
            bodies.append(
                _field_construct(
                    SolidBody,
                    body_path,
                    policy,
                    _field_string(
                        body_data["id"],
                        f"{body_path}.id",
                        policy.decode_error,
                    ),
                    _field_string(
                        body_data["name"],
                        f"{body_path}.name",
                        policy.decode_error,
                    ),
                    decode_geometry_field(
                        body_data["recipe"],
                        f"{body_path}.recipe",
                        policy=policy,
                    ),
                )
            )
        if tuple(body.id for body in bodies) != tuple(
            sorted(
                (body.id for body in bodies),
                key=lambda value: int(value[1:]),
            )
        ):
            raise policy.decode_error(
                f"{path}.bodies must use canonical Body ID order"
            )
        retired_body_ids = tuple(
            _field_string(
                value,
                f"{path}.retired_body_ids[{index}]",
                policy.decode_error,
            )
            for index, value in enumerate(
                _field_array(
                    data["retired_body_ids"],
                    f"{path}.retired_body_ids",
                    policy.decode_error,
                )
            )
        )
        if retired_body_ids != tuple(
            sorted(set(retired_body_ids), key=lambda value: int(value[1:]))
        ):
            raise policy.decode_error(
                f"{path}.retired_body_ids must be canonical and unique"
            )
        retired_feature_ids = tuple(
            _field_string(
                value,
                f"{path}.retired_boolean_feature_ids[{index}]",
                policy.decode_error,
            )
            for index, value in enumerate(
                _field_array(
                    data["retired_boolean_feature_ids"],
                    f"{path}.retired_boolean_feature_ids",
                    policy.decode_error,
                )
            )
        )
        if retired_feature_ids != tuple(
            sorted(set(retired_feature_ids), key=lambda value: int(value[2:]))
        ):
            raise policy.decode_error(
                f"{path}.retired_boolean_feature_ids must be canonical "
                "and unique"
            )
        return _field_construct(
            MultiBodyGeometry,
            path,
            policy,
            _field_string(data["name"], f"{path}.name", policy.decode_error),
            tuple(bodies),
            retired_body_ids,
            retired_feature_ids,
        )
    raise policy.decode_error(f"{path}.type 是未知几何类型：{kind!r}")


def decode_contour_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchRectangle | SketchCircle:
    data = _field_mapping(value, path, policy.decode_error)
    if "type" not in data:
        raise policy.decode_error(f"{path} 缺少必需字段：type")
    kind = _field_string(data["type"], f"{path}.type", policy.decode_error)
    if kind == "rectangle":
        _field_keys(
            data,
            path,
            required={"type", "operation", "x", "y", "width", "height"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            SketchRectangle,
            path,
            policy,
            _field_string(
                data["operation"],
                f"{path}.operation",
                policy.decode_error,
            ),
            _field_number(
                data["x"],
                f"{path}.x",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["y"],
                f"{path}.y",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["width"],
                f"{path}.width",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["height"],
                f"{path}.height",
                policy.decode_error,
                policy=policy,
            ),
        )
    if kind == "circle":
        _field_keys(
            data,
            path,
            required={"type", "operation", "x", "y", "radius"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            SketchCircle,
            path,
            policy,
            _field_string(
                data["operation"],
                f"{path}.operation",
                policy.decode_error,
            ),
            _field_number(
                data["x"],
                f"{path}.x",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["y"],
                f"{path}.y",
                policy.decode_error,
                policy=policy,
            ),
            _field_number(
                data["radius"],
                f"{path}.radius",
                policy.decode_error,
                policy=policy,
            ),
        )
    raise policy.decode_error(f"{path}.type 是未知草图轮廓：{kind!r}")


def _decode_sketch_plane_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchPlane:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"origin", "x_direction", "y_direction"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        SketchPlane,
        path,
        policy,
        _field_decode_number_array(data["origin"], f"{path}.origin", policy),
        _field_decode_number_array(
            data["x_direction"],
            f"{path}.x_direction",
            policy,
        ),
        _field_decode_number_array(
            data["y_direction"],
            f"{path}.y_direction",
            policy,
        ),
    )


def _decode_face_sketch_workplane_strategy(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> FaceSketchWorkplaneStrategy:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"seed_axis", "sign", "origin_rule"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        FaceSketchWorkplaneStrategy,
        path,
        policy,
        _field_string(data["seed_axis"], f"{path}.seed_axis", policy.decode_error),
        _field_integer(
            data["sign"],
            f"{path}.sign",
            policy.decode_error,
            policy=policy,
        ),
        _field_string(data["origin_rule"], f"{path}.origin_rule", policy.decode_error),
    )


def _decode_face_sketch_external_reference(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchExternalReference:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"id", "source_logical_id", "derived_type"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        SketchExternalReference,
        path,
        policy,
        _field_string(data["id"], f"{path}.id", policy.decode_error),
        _field_construct(
            LogicalEntityRef,
            f"{path}.source_logical_id",
            policy,
            _field_string(
                data["source_logical_id"],
                f"{path}.source_logical_id",
                policy.decode_error,
            ),
        ),
        _field_enum(
            data["derived_type"],
            f"{path}.derived_type",
            SketchExternalReferenceType,
            policy,
        ),
    )


def _decode_face_sketch_external_coincidence(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchExternalCoincidence:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"point_id", "reference_id"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        SketchExternalCoincidence,
        path,
        policy,
        _field_string(data["point_id"], f"{path}.point_id", policy.decode_error),
        _field_string(
            data["reference_id"], f"{path}.reference_id", policy.decode_error
        ),
    )


def _decode_face_sketch_step_proof(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> FaceSketchBooleanStepProof:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={
            "profile_id",
            "result_entities",
            "topology_mappings",
            "connection_proof",
        },
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    raw_entities = _field_array(
        data["result_entities"], f"{path}.result_entities", policy.decode_error
    )
    raw_mappings = _field_array(
        data["topology_mappings"], f"{path}.topology_mappings", policy.decode_error
    )
    lineage = _decode_boolean_body_context(
        {
            "feature_id": "BF1",
            "target_body_id": "B1",
            "tool_body_id": "B2",
            "tool_body_name": "Face sketch proof",
            "result_entities": sorted(
                raw_entities,
                key=lambda item: logical_ref_sort_key(
                    LogicalEntityRef(item["logical_id"])
                ),
            ),
            "topology_mappings": sorted(
                raw_mappings,
                key=lambda item: (
                    item["source"],
                    logical_ref_sort_key(
                        LogicalEntityRef(item["source_logical_id"])
                    ),
                    logical_ref_sort_key(
                        LogicalEntityRef(item["target_logical_id"])
                    ),
                    item["relation"],
                ),
            ),
        },
        path,
        policy=policy,
    )
    if lineage is None:
        raise policy.decode_error(f"{path} 的布尔谱系不能为空")
    entities_by_id = {item.logical_id: item for item in lineage.result_entities}
    entities = tuple(entities_by_id[item["logical_id"]] for item in raw_entities)
    mappings_by_key = {
        (
            item.source,
            item.source_logical_id,
            item.target_logical_id,
            item.relation,
        ): item
        for item in lineage.topology_mappings
    }
    mappings = tuple(
        mappings_by_key[
            (
                item["source"],
                item["source_logical_id"],
                item["target_logical_id"],
                item["relation"],
            )
        ]
        for item in raw_mappings
    )
    raw_connection = data["connection_proof"]
    connection = None
    if raw_connection is not None:
        connection_data = _field_mapping(
            raw_connection, f"{path}.connection_proof", policy.decode_error
        )
        _field_keys(
            connection_data,
            f"{path}.connection_proof",
            required={"support_face_id", "tool_start_face_id", "overlap_area"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        connection = _field_construct(
            FaceSeedConnectionProof,
            f"{path}.connection_proof",
            policy,
            _field_string(
                connection_data["support_face_id"],
                f"{path}.connection_proof.support_face_id",
                policy.decode_error,
            ),
            _field_string(
                connection_data["tool_start_face_id"],
                f"{path}.connection_proof.tool_start_face_id",
                policy.decode_error,
            ),
            _field_number(
                connection_data["overlap_area"],
                f"{path}.connection_proof.overlap_area",
                policy.decode_error,
                policy=policy,
            ),
        )
    return _field_construct(
        FaceSketchBooleanStepProof,
        path,
        policy,
        _field_string(data["profile_id"], f"{path}.profile_id", policy.decode_error),
        entities,
        mappings,
        connection,
    )


def _decode_boolean_body_context(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> BooleanBodyContext | None:
    if value is None:
        return None
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={
            "feature_id",
            "target_body_id",
            "tool_body_id",
            "tool_body_name",
            "result_entities",
            "topology_mappings",
        },
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    entities: list[BooleanLineageEntity] = []
    for index, item in enumerate(
        _field_array(
            data["result_entities"],
            f"{path}.result_entities",
            policy.decode_error,
        )
    ):
        item_path = f"{path}.result_entities[{index}]"
        item_data = _field_mapping(item, item_path, policy.decode_error)
        _field_keys(
            item_data,
            item_path,
            required={"kind", "logical_id", "semantic_role", "topology_links"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        links = tuple(
            _field_string(
                link,
                f"{item_path}.topology_links[{link_index}]",
                policy.decode_error,
            )
            for link_index, link in enumerate(
                _field_array(
                    item_data["topology_links"],
                    f"{item_path}.topology_links",
                    policy.decode_error,
                )
            )
        )
        if links != tuple(sorted(set(links))):
            raise policy.decode_error(
                f"{item_path}.topology_links must be canonical and unique"
            )
        entities.append(
            _field_construct(
                BooleanLineageEntity,
                item_path,
                policy,
                _field_string(
                    item_data["kind"],
                    f"{item_path}.kind",
                    policy.decode_error,
                ),
                _field_string(
                    item_data["logical_id"],
                    f"{item_path}.logical_id",
                    policy.decode_error,
                ),
                _field_string(
                    item_data["semantic_role"],
                    f"{item_path}.semantic_role",
                    policy.decode_error,
                ),
                links,
            )
        )
    mappings: list[BooleanLineageMapping] = []
    for index, item in enumerate(
        _field_array(
            data["topology_mappings"],
            f"{path}.topology_mappings",
            policy.decode_error,
        )
    ):
        item_path = f"{path}.topology_mappings[{index}]"
        item_data = _field_mapping(item, item_path, policy.decode_error)
        _field_keys(
            item_data,
            item_path,
            required={
                "source",
                "source_logical_id",
                "target_logical_id",
                "relation",
            },
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        mappings.append(
            _field_construct(
                BooleanLineageMapping,
                item_path,
                policy,
                *(
                    _field_string(
                        item_data[field_name],
                        f"{item_path}.{field_name}",
                        policy.decode_error,
                    )
                    for field_name in (
                        "source",
                        "source_logical_id",
                        "target_logical_id",
                        "relation",
                    )
                ),
            )
        )
    canonical_entities = tuple(
        sorted(
            entities,
            key=lambda item: logical_ref_sort_key(
                LogicalEntityRef(item.logical_id)
            ),
        )
    )
    if tuple(entities) != canonical_entities:
        raise policy.decode_error(
            f"{path}.result_entities must use canonical logical order"
        )
    canonical_mappings = tuple(
        sorted(
            mappings,
            key=lambda item: (
                item.source,
                logical_ref_sort_key(
                    LogicalEntityRef(item.source_logical_id)
                ),
                logical_ref_sort_key(
                    LogicalEntityRef(item.target_logical_id)
                ),
                item.relation,
            ),
        )
    )
    if tuple(mappings) != canonical_mappings:
        raise policy.decode_error(
            f"{path}.topology_mappings must use canonical logical order"
        )
    return _field_construct(
        BooleanBodyContext,
        path,
        policy,
        *(
            _field_string(
                data[field_name],
                f"{path}.{field_name}",
                policy.decode_error,
            )
            for field_name in (
                "feature_id",
                "target_body_id",
                "tool_body_id",
                "tool_body_name",
            )
        ),
        tuple(entities),
        tuple(mappings),
    )


def _decode_planar_boolean_context(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> PlanarBooleanContext | None:
    if value is None:
        return None
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={
            "feature_id",
            "target_face_id",
            "tool_face_ids",
            "result_entities",
            "topology_mappings",
        },
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    synthetic = {
        "feature_id": "BF1",
        "target_body_id": "B1",
        "tool_body_id": "B2",
        "tool_body_name": "Planar Tool",
        "result_entities": data["result_entities"],
        "topology_mappings": data["topology_mappings"],
    }
    lineage = _decode_boolean_body_context(
        synthetic,
        path,
        policy=policy,
    )
    if lineage is None:
        raise policy.decode_error(f"{path} must not be null")
    tool_face_ids = tuple(
        _field_string(
            item,
            f"{path}.tool_face_ids[{index}]",
            policy.decode_error,
        )
        for index, item in enumerate(
            _field_array(
                data["tool_face_ids"],
                f"{path}.tool_face_ids",
                policy.decode_error,
            )
        )
    )
    return _field_construct(
        PlanarBooleanContext,
        path,
        policy,
        _field_string(
            data["feature_id"],
            f"{path}.feature_id",
            policy.decode_error,
        ),
        _field_string(
            data["target_face_id"],
            f"{path}.target_face_id",
            policy.decode_error,
        ),
        tool_face_ids,
        lineage.result_entities,
        lineage.topology_mappings,
    )


def _decode_part_boolean_context(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> PartBooleanContext | None:
    if value is None:
        return None
    from fem.geometry.part_boolean import namespace_part_boolean_context
    from fem.geometry.part_namespace import strip_part_logical_id

    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={
            "feature_id",
            "target_part_id",
            "tool_part_id",
            "result_part_id",
            "result_entities",
            "topology_mappings",
        },
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    values = {
        field_name: _field_string(
            data[field_name],
            f"{path}.{field_name}",
            policy.decode_error,
        )
        for field_name in (
            "feature_id",
            "target_part_id",
            "tool_part_id",
            "result_part_id",
        )
    }
    source_parts = {
        "target": values["target_part_id"],
        "tool": values["tool_part_id"],
    }
    try:
        local_entities: list[dict[str, Any]] = []
        for index, raw in enumerate(
            _field_array(
                data["result_entities"],
                f"{path}.result_entities",
                policy.decode_error,
            )
        ):
            item_path = f"{path}.result_entities[{index}]"
            item = _field_mapping(raw, item_path, policy.decode_error)
            _field_keys(
                item,
                item_path,
                required={
                    "kind",
                    "logical_id",
                    "semantic_role",
                    "topology_links",
                },
                optional=set(),
                policy=policy,
                error_type=policy.decode_error,
            )
            local_entities.append(
                {
                    "kind": item["kind"],
                    "logical_id": strip_part_logical_id(
                        values["result_part_id"],
                        _field_string(
                            item["logical_id"],
                            f"{item_path}.logical_id",
                            policy.decode_error,
                        ),
                    ),
                    "semantic_role": item["semantic_role"],
                    "topology_links": [
                        strip_part_logical_id(
                            values["result_part_id"],
                            _field_string(
                                link,
                                f"{item_path}.topology_links[{link_index}]",
                                policy.decode_error,
                            ),
                        )
                        for link_index, link in enumerate(
                            _field_array(
                                item["topology_links"],
                                f"{item_path}.topology_links",
                                policy.decode_error,
                            )
                        )
                    ],
                }
            )
        local_mappings: list[dict[str, Any]] = []
        for index, raw in enumerate(
            _field_array(
                data["topology_mappings"],
                f"{path}.topology_mappings",
                policy.decode_error,
            )
        ):
            item_path = f"{path}.topology_mappings[{index}]"
            item = _field_mapping(raw, item_path, policy.decode_error)
            _field_keys(
                item,
                item_path,
                required={
                    "source",
                    "source_logical_id",
                    "target_logical_id",
                    "relation",
                },
                optional=set(),
                policy=policy,
                error_type=policy.decode_error,
            )
            source = _field_string(
                item["source"],
                f"{item_path}.source",
                policy.decode_error,
            )
            if source not in source_parts:
                raise policy.decode_error(
                    f"{item_path}.source must be target or tool"
                )
            local_mappings.append(
                {
                    "source": source,
                    "source_logical_id": strip_part_logical_id(
                        source_parts[source],
                        _field_string(
                            item["source_logical_id"],
                            f"{item_path}.source_logical_id",
                            policy.decode_error,
                        ),
                    ),
                    "target_logical_id": strip_part_logical_id(
                        values["result_part_id"],
                        _field_string(
                            item["target_logical_id"],
                            f"{item_path}.target_logical_id",
                            policy.decode_error,
                        ),
                    ),
                    "relation": item["relation"],
                }
            )
        lineage = _decode_boolean_body_context(
            {
                "feature_id": "BF1",
                "target_body_id": "B1",
                "tool_body_id": "B2",
                "tool_body_name": "Part Tool",
                "result_entities": local_entities,
                "topology_mappings": local_mappings,
            },
            path,
            policy=policy,
        )
        if lineage is None:
            raise policy.decode_error(f"{path} must not be null")
        return namespace_part_boolean_context(
            feature_id=values["feature_id"],
            target_part_id=values["target_part_id"],
            tool_part_id=values["tool_part_id"],
            result_part_id=values["result_part_id"],
            result_entities=lineage.result_entities,
            topology_mappings=lineage.topology_mappings,
        )
    except policy.decode_error:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise policy.decode_error(f"{path} 无效：{error}") from error


def _decode_sketch_point_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchPoint:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"id", "u", "v"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        SketchPoint,
        path,
        policy,
        _field_string(data["id"], f"{path}.id", policy.decode_error),
        _field_number(data["u"], f"{path}.u", policy.decode_error, policy=policy),
        _field_number(data["v"], f"{path}.v", policy.decode_error, policy=policy),
    )


def _decode_sketch_curve_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchLine | SketchArc | SketchCircle:
    data = _field_mapping(value, path, policy.decode_error)
    kind = _field_string(data.get("type"), f"{path}.type", policy.decode_error)
    if kind == "line":
        _field_keys(
            data,
            path,
            required={"type", "id", "start_point_id", "end_point_id"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            SketchLine,
            path,
            policy,
            _field_string(data["id"], f"{path}.id", policy.decode_error),
            _field_string(
                data["start_point_id"],
                f"{path}.start_point_id",
                policy.decode_error,
            ),
            _field_string(
                data["end_point_id"],
                f"{path}.end_point_id",
                policy.decode_error,
            ),
        )
    if kind == "arc":
        _field_keys(
            data,
            path,
            required={
                "type",
                "id",
                "start_point_id",
                "center_point_id",
                "end_point_id",
                "orientation",
            },
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            SketchArc,
            path,
            policy,
            _field_string(data["id"], f"{path}.id", policy.decode_error),
            _field_string(
                data["start_point_id"],
                f"{path}.start_point_id",
                policy.decode_error,
            ),
            _field_string(
                data["center_point_id"],
                f"{path}.center_point_id",
                policy.decode_error,
            ),
            _field_string(
                data["end_point_id"],
                f"{path}.end_point_id",
                policy.decode_error,
            ),
            _field_string(
                data["orientation"],
                f"{path}.orientation",
                policy.decode_error,
            ),
        )
    if kind == "circle":
        _field_keys(
            data,
            path,
            required={"type", "id", "center_point_id", "radius"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            SketchCircle,
            path,
            policy,
            _field_string(data["id"], f"{path}.id", policy.decode_error),
            _field_string(
                data["center_point_id"],
                f"{path}.center_point_id",
                policy.decode_error,
            ),
            _field_number(
                data["radius"],
                f"{path}.radius",
                policy.decode_error,
                policy=policy,
            ),
        )
    raise policy.decode_error(f"{path}.type 不支持草图曲线：{kind!r}")


def _decode_wire_point_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> WirePoint:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"name", "x", "y", "z"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        WirePoint,
        path,
        policy,
        _field_string(data["name"], f"{path}.name", policy.decode_error),
        _field_number(data["x"], f"{path}.x", policy.decode_error, policy=policy),
        _field_number(data["y"], f"{path}.y", policy.decode_error, policy=policy),
        _field_number(data["z"], f"{path}.z", policy.decode_error, policy=policy),
    )


def _decode_wire_member_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> WireMember:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required={"name", "start", "end"},
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        WireMember,
        path,
        policy,
        _field_string(data["name"], f"{path}.name", policy.decode_error),
        _field_string(data["start"], f"{path}.start", policy.decode_error),
        _field_string(data["end"], f"{path}.end", policy.decode_error),
    )


def _encode_wire_point_field(
    point: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        point,
        WirePoint,
        {"name", "x", "y", "z"},
        path,
        policy,
    )
    return {
        "name": _field_string(point.name, f"{path}.name", policy.encode_error),
        "x": _field_number(point.x, f"{path}.x", policy.encode_error, policy=policy),
        "y": _field_number(point.y, f"{path}.y", policy.encode_error, policy=policy),
        "z": _field_number(point.z, f"{path}.z", policy.encode_error, policy=policy),
    }


def _encode_wire_member_field(
    member: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        member,
        WireMember,
        {"name", "start", "end"},
        path,
        policy,
    )
    return {
        "name": _field_string(member.name, f"{path}.name", policy.encode_error),
        "start": _field_string(member.start, f"{path}.start", policy.encode_error),
        "end": _field_string(member.end, f"{path}.end", policy.encode_error),
    }


_SKETCH_CONSTRAINT_WIRE_TYPES = {
    "coincident": (SketchCoincidentConstraint, ("first_point_id", "second_point_id"), ()),
    "point_on_curve": (SketchPointOnCurveConstraint, ("point_id", "curve_id"), ()),
    "horizontal": (SketchHorizontalConstraint, ("line_id",), ()),
    "vertical": (SketchVerticalConstraint, ("line_id",), ()),
    "parallel": (SketchParallelConstraint, ("first_line_id", "second_line_id"), ()),
    "perpendicular": (
        SketchPerpendicularConstraint,
        ("first_line_id", "second_line_id"),
        (),
    ),
    "tangent": (
        SketchTangentConstraint,
        ("first_curve_id", "second_curve_id"),
        (),
    ),
    "equal_length": (
        SketchEqualLengthConstraint,
        ("first_line_id", "second_line_id"),
        (),
    ),
    "equal_radius": (
        SketchEqualRadiusConstraint,
        ("first_curve_id", "second_curve_id"),
        (),
    ),
    "concentric": (
        SketchConcentricConstraint,
        ("first_curve_id", "second_curve_id"),
        (),
    ),
    "fixed": (SketchFixedConstraint, ("point_id",), ("u", "v")),
    "distance": (
        SketchDistanceDimension,
        ("first_point_id", "second_point_id"),
        ("value",),
    ),
    "radius": (SketchRadiusDimension, ("curve_id",), ("value",)),
    "angle": (
        SketchAngleDimension,
        ("first_line_id", "second_line_id"),
        ("value",),
    ),
}
_SKETCH_CONSTRAINT_TYPE_NAMES = {
    value[0]: key for key, value in _SKETCH_CONSTRAINT_WIRE_TYPES.items()
}


def _field_strict_bool(value: Any, path: str, policy: ProjectFieldCodecPolicy) -> bool:
    if type(value) is not bool:
        raise policy.decode_error(f"{path} 必须是 bool")
    return value


def _decode_sketch_constraint_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SketchConstraint:
    data = _field_mapping(value, path, policy.decode_error)
    kind = _field_string(data.get("type"), f"{path}.type", policy.decode_error)
    contract = _SKETCH_CONSTRAINT_WIRE_TYPES.get(kind)
    if contract is None:
        raise policy.decode_error(f"{path}.type 包含未知草图约束类型：{kind!r}")
    constraint_type, string_fields, number_fields = contract
    dimensions = constraint_type in {
        SketchDistanceDimension, SketchRadiusDimension, SketchAngleDimension
    }
    required = {"type", "id", "source", "enabled", *string_fields, *number_fields}
    if dimensions:
        required.add("driving")
    if constraint_type is SketchTangentConstraint:
        required.add("branch_hint")
    _field_keys(
        data,
        path,
        required=required,
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    arguments = {
        "id": _field_string(data["id"], f"{path}.id", policy.decode_error),
        "source": _field_string(data["source"], f"{path}.source", policy.decode_error),
        "enabled": _field_strict_bool(data["enabled"], f"{path}.enabled", policy),
    }
    arguments.update(
        {
            field: _field_string(data[field], f"{path}.{field}", policy.decode_error)
            for field in string_fields
        }
    )
    arguments.update(
        {
            field: _field_number(
                data[field], f"{path}.{field}", policy.decode_error, policy=policy
            )
            for field in number_fields
        }
    )
    if dimensions:
        arguments["driving"] = _field_strict_bool(
            data["driving"], f"{path}.driving", policy
        )
    if constraint_type is SketchTangentConstraint:
        arguments["branch_hint"] = _field_integer(
            data["branch_hint"], f"{path}.branch_hint", policy.decode_error,
            policy=policy,
        )
    return _field_construct(constraint_type, path, policy, **arguments)


def _encode_sketch_constraint_field(
    constraint: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    kind = _SKETCH_CONSTRAINT_TYPE_NAMES.get(type(constraint))
    if kind is None:
        raise policy.encode_error(f"{path} 包含不支持的草图约束类型")
    constraint_type, string_fields, number_fields = _SKETCH_CONSTRAINT_WIRE_TYPES[kind]
    dimensions = constraint_type in {
        SketchDistanceDimension, SketchRadiusDimension, SketchAngleDimension
    }
    expected_fields = {"id", "source", "enabled", *string_fields, *number_fields}
    if dimensions:
        expected_fields.add("driving")
    if constraint_type is SketchTangentConstraint:
        expected_fields.add("branch_hint")
    _field_exact_dataclass(
        constraint, constraint_type, expected_fields, path, policy
    )
    result: dict[str, Any] = {
        "type": kind,
        "id": _field_string(constraint.id, f"{path}.id", policy.encode_error),
        "source": _field_string(
            constraint.source, f"{path}.source", policy.encode_error
        ),
        "enabled": constraint.enabled,
    }
    if type(constraint.enabled) is not bool:
        raise policy.encode_error(f"{path}.enabled 必须是 bool")
    for field in string_fields:
        result[field] = _field_string(
            getattr(constraint, field), f"{path}.{field}", policy.encode_error
        )
    for field in number_fields:
        result[field] = _field_number(
            getattr(constraint, field),
            f"{path}.{field}",
            policy.encode_error,
            policy=policy,
        )
    if dimensions:
        if type(constraint.driving) is not bool:
            raise policy.encode_error(f"{path}.driving 必须是 bool")
        result["driving"] = constraint.driving
    if constraint_type is SketchTangentConstraint:
        if isinstance(constraint.branch_hint, bool) or not isinstance(
            constraint.branch_hint, int
        ):
            raise policy.encode_error(f"{path}.branch_hint 必须是整数")
        result["branch_hint"] = constraint.branch_hint
    return result


def encode_geometry_field(
    recipe: Any,
    path: str,
    ancestors: set[int],
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    """Encode one geometry recipe with shared exact dataclass guards."""

    identity = id(recipe)
    if identity in ancestors:
        raise policy.encode_error(f"{path} 包含循环几何引用")
    ancestors.add(identity)
    try:
        primitive = _PRIMITIVE_GEOMETRY_FIELDS.get(type(recipe))
        if primitive is not None:
            kind, string_fields, number_fields = primitive
            _field_exact_dataclass(
                recipe,
                type(recipe),
                {*string_fields, *number_fields},
                path,
                policy,
            )
            result: dict[str, Any] = {"type": kind}
            result.update(
                {
                    field_name: _field_string(
                        getattr(recipe, field_name),
                        f"{path}.{field_name}",
                        policy.encode_error,
                    )
                    for field_name in string_fields
                }
            )
            result.update(
                {
                    field_name: _field_number(
                        getattr(recipe, field_name),
                        f"{path}.{field_name}",
                        policy.encode_error,
                        policy=policy,
                    )
                    for field_name in number_fields
                }
            )
            return result
        if type(recipe) is WireGeometry:
            if not policy.allow_wire_geometry:
                raise policy.encode_error(
                    f"{path} 的几何类型无法由 {policy.version_label} "
                    "无损编码：WireGeometry"
                )
            _field_exact_dataclass(
                recipe,
                WireGeometry,
                {"name", "points", "members"},
                path,
                policy,
            )
            points = _field_runtime_sequence(
                recipe.points,
                f"{path}.points",
                policy,
            )
            members = _field_runtime_sequence(
                recipe.members,
                f"{path}.members",
                policy,
            )
            return {
                "type": "WireGeometry",
                "name": _field_string(
                    recipe.name,
                    f"{path}.name",
                    policy.encode_error,
                ),
                "points": [
                    _encode_wire_point_field(
                        point,
                        f"{path}.points[{index}]",
                        policy=policy,
                    )
                    for index, point in enumerate(points)
                ],
                "members": [
                    _encode_wire_member_field(
                        member,
                        f"{path}.members[{index}]",
                        policy=policy,
                    )
                    for index, member in enumerate(members)
                ],
            }
        if type(recipe) is SketchGeometry:
            name = _field_string(recipe.name, f"{path}.name", policy.encode_error)
            if recipe.is_legacy:
                if policy.allow_strict_sketch:
                    # v3 is the first schema that can represent the strict
                    # curve graph.  Upgrade compatibility contours at every
                    # current-schema encoding boundary.
                    recipe = legacy_sketch_to_strict(recipe)
                else:
                    return {
                        "type": "SketchGeometry",
                        "name": name,
                        "contours": [
                            encode_contour_field(
                                item,
                                f"{path}.contours[{index}]",
                                policy=policy,
                            )
                            for index, item in enumerate(
                                _field_runtime_sequence(
                                    recipe.contours,
                                    f"{path}.contours",
                                    policy,
                                )
                            )
                        ],
                    }
            if not policy.allow_strict_sketch:
                raise policy.encode_error(
                    f"{path} 的 curve-based sketch 无法由 "
                    f"{policy.version_label} 编码"
                )
            if recipe.constraints and not _SKETCH_CONSTRAINT_CODEC.get():
                raise policy.encode_error(
                    f"{path}.constraints 无法由 {policy.version_label} 无损编码"
                )
            result = {
                "type": "SketchGeometry",
                "name": name,
                "plane": _encode_sketch_plane_field(
                    recipe.plane,
                    f"{path}.plane",
                    policy=policy,
                ),
                "points": [
                    _encode_sketch_point_field(
                        item,
                        f"{path}.points[{index}]",
                        policy=policy,
                    )
                    for index, item in enumerate(
                        _field_runtime_sequence(recipe.points, f"{path}.points", policy)
                    )
                ],
                "curves": [
                    _encode_sketch_curve_field(
                        item,
                        f"{path}.curves[{index}]",
                        policy=policy,
                    )
                    for index, item in enumerate(
                        _field_runtime_sequence(recipe.curves, f"{path}.curves", policy)
                    )
                ],
            }
            if _SKETCH_CONSTRAINT_CODEC.get():
                result["constraints"] = [
                    _encode_sketch_constraint_field(
                        item,
                        f"{path}.constraints[{index}]",
                        policy=policy,
                    )
                    for index, item in enumerate(
                        _field_runtime_sequence(
                            recipe.constraints,
                            f"{path}.constraints",
                            policy,
                        )
                    )
                ]
            return result
        if type(recipe) is FaceSketchBooleanGeometry:
            if not policy.allow_face_sketch_boolean:
                raise policy.encode_error(
                    f"{path} 的几何类型无法由 {policy.version_label} "
                    "无损编码：FaceSketchBooleanGeometry"
                )
            _field_exact_dataclass(
                recipe,
                FaceSketchBooleanGeometry,
                {
                    "base",
                    "feature_id",
                    "name",
                    "support_face_id",
                    "workplane_strategy",
                    "sketch",
                    "operation",
                    "direction",
                    "distance",
                    "participating_profile_ids",
                    "external_references",
                    "external_coincidences",
                    "step_proofs",
                },
                path,
                policy,
            )
            if tuple(item.profile_id for item in recipe.step_proofs) != recipe.participating_profile_ids:
                raise policy.encode_error(
                    f"{path}.step_proofs 必须完整覆盖参与轮廓并保持稳定顺序"
                )
            if any(not item.result_entities or not item.topology_mappings for item in recipe.step_proofs):
                raise policy.encode_error(f"{path}.step_proofs 包含不完整的布尔证明")
            return {
                "type": "FaceSketchBooleanGeometry",
                "base": encode_geometry_field(
                    recipe.base, f"{path}.base", ancestors, policy=policy
                ),
                "feature_id": recipe.feature_id,
                "name": recipe.name,
                "support_face_id": recipe.support_face_id,
                "workplane_strategy": _encode_face_sketch_workplane_strategy(
                    recipe.workplane_strategy,
                    f"{path}.workplane_strategy",
                    policy=policy,
                ),
                "sketch": encode_geometry_field(
                    recipe.sketch, f"{path}.sketch", ancestors, policy=policy
                ),
                "operation": recipe.operation.value,
                "direction": recipe.direction.value,
                "distance": _field_number(
                    recipe.distance, f"{path}.distance", policy.encode_error, policy=policy
                ),
                "participating_profile_ids": list(recipe.participating_profile_ids),
                "external_references": [
                    _encode_face_sketch_external_reference(
                        item, f"{path}.external_references[{index}]", policy=policy
                    )
                    for index, item in enumerate(recipe.external_references)
                ],
                "external_coincidences": [
                    _encode_face_sketch_external_coincidence(
                        item, f"{path}.external_coincidences[{index}]", policy=policy
                    )
                    for index, item in enumerate(recipe.external_coincidences)
                ],
                "step_proofs": [
                    _encode_face_sketch_step_proof(
                        item, f"{path}.step_proofs[{index}]", policy=policy
                    )
                    for index, item in enumerate(recipe.step_proofs)
                ],
            }
        if type(recipe) is MovedGeometry:
            _field_exact_dataclass(
                recipe,
                MovedGeometry,
                {"base", "dx", "dy", "dz"},
                path,
                policy,
            )
            return {
                "type": "MovedGeometry",
                "base": encode_geometry_field(
                    recipe.base,
                    f"{path}.base",
                    ancestors,
                    policy=policy,
                ),
                "dx": _field_number(
                    recipe.dx,
                    f"{path}.dx",
                    policy.encode_error,
                    policy=policy,
                ),
                "dy": _field_number(
                    recipe.dy,
                    f"{path}.dy",
                    policy.encode_error,
                    policy=policy,
                ),
                "dz": _field_number(
                    recipe.dz,
                    f"{path}.dz",
                    policy.encode_error,
                    policy=policy,
                ),
            }
        if type(recipe) is RotatedGeometry:
            _field_exact_dataclass(
                recipe,
                RotatedGeometry,
                {"base", "axis", "angle_degrees"},
                path,
                policy,
            )
            return {
                "type": "RotatedGeometry",
                "base": encode_geometry_field(
                    recipe.base,
                    f"{path}.base",
                    ancestors,
                    policy=policy,
                ),
                "axis": _field_string(
                    recipe.axis,
                    f"{path}.axis",
                    policy.encode_error,
                ),
                "angle_degrees": _field_number(
                    recipe.angle_degrees,
                    f"{path}.angle_degrees",
                    policy.encode_error,
                    policy=policy,
                ),
            }
        if type(recipe) is ExtrudedGeometry:
            _field_exact_dataclass(
                recipe,
                ExtrudedGeometry,
                {"base", "height", "source_face_ids"},
                path,
                policy,
            )
            if recipe.source_face_ids and not policy.extrusion_source_faces:
                raise policy.encode_error(
                    f"{path}.source_face_ids 无法由 "
                    f"{policy.version_label} 无损表示"
                )
            encoded = {
                "type": "ExtrudedGeometry",
                "base": encode_geometry_field(
                    recipe.base,
                    f"{path}.base",
                    ancestors,
                    policy=policy,
                ),
                "height": _field_number(
                    recipe.height,
                    f"{path}.height",
                    policy.encode_error,
                    policy=policy,
                ),
            }
            if policy.extrusion_source_faces:
                encoded["source_face_ids"] = [
                    _field_string(
                        logical_id,
                        f"{path}.source_face_ids[{index}]",
                        policy.encode_error,
                    )
                    for index, logical_id in enumerate(recipe.source_face_ids)
                ]
            return encoded
        if type(recipe) is RevolvedGeometry:
            if not policy.allow_revolved_geometry:
                raise policy.encode_error(
                    f"{path} 的几何类型无法由 {policy.version_label} 无损编码"
                )
            _field_exact_dataclass(
                recipe,
                RevolvedGeometry,
                {"base", "axis", "angle_degrees", "source_face_ids"},
                path,
                policy,
            )
            return {
                "type": "RevolvedGeometry",
                "base": encode_geometry_field(
                    recipe.base, f"{path}.base", ancestors, policy=policy,
                ),
                "axis": _field_string(
                    recipe.axis, f"{path}.axis", policy.encode_error,
                ),
                "angle_degrees": _field_number(
                    recipe.angle_degrees, f"{path}.angle_degrees",
                    policy.encode_error, policy=policy,
                ),
                "source_face_ids": [
                    _field_string(
                        item, f"{path}.source_face_ids[{index}]", policy.encode_error,
                    )
                    for index, item in enumerate(recipe.source_face_ids)
                ],
            }
        if type(recipe) is PathSweptGeometry:
            if not policy.allow_path_swept_geometry:
                raise policy.encode_error(
                    f"{path} 的几何类型无法由 {policy.version_label} 无损编码"
                )
            _field_exact_dataclass(
                recipe,
                PathSweptGeometry,
                {"base", "path", "source_face_ids", "frame_strategy"},
                path,
                policy,
            )
            return {
                "type": "PathSweptGeometry",
                "base": encode_geometry_field(
                    recipe.base, f"{path}.base", ancestors, policy=policy,
                ),
                "path": encode_geometry_field(
                    recipe.path, f"{path}.path", ancestors, policy=policy,
                ),
                "source_face_ids": [
                    _field_string(
                        item, f"{path}.source_face_ids[{index}]", policy.encode_error,
                    )
                    for index, item in enumerate(recipe.source_face_ids)
                ],
                "frame_strategy": _field_string(
                    recipe.frame_strategy, f"{path}.frame_strategy", policy.encode_error,
                ),
            }
        if type(recipe) is BooleanGeometry:
            _field_exact_dataclass(
                recipe,
                BooleanGeometry,
                {
                    "name",
                    "operation",
                    "object_geometry",
                    "tool_geometry",
                    "body_context",
                    "planar_context",
                    "part_context",
                },
                path,
                policy,
            )
            if recipe.body_context is not None and not policy.allow_multi_body:
                raise policy.encode_error(
                    f"{path}.body_context 无法由 "
                    f"{policy.version_label} 无损表示"
                )
            if (
                recipe.planar_context is not None
                and not policy.allow_planar_boolean
            ):
                raise policy.encode_error(
                    f"{path}.planar_context 无法由 "
                    f"{policy.version_label} 无损表示"
                )
            if (
                recipe.part_context is not None
                and not policy.allow_part_boolean
            ):
                raise policy.encode_error(
                    f"{path}.part_context 无法由 "
                    f"{policy.version_label} 无损表示"
                )
            encoded = {
                "type": "BooleanGeometry",
                "name": _field_string(
                    recipe.name,
                    f"{path}.name",
                    policy.encode_error,
                ),
                "operation": _field_string(
                    recipe.operation,
                    f"{path}.operation",
                    policy.encode_error,
                ),
                "object": encode_geometry_field(
                    recipe.object_geometry,
                    f"{path}.object_geometry",
                    ancestors,
                    policy=policy,
                ),
                "tool": encode_geometry_field(
                    recipe.tool_geometry,
                    f"{path}.tool_geometry",
                    ancestors,
                    policy=policy,
                ),
            }
            if policy.allow_multi_body:
                encoded["body_context"] = _encode_boolean_body_context(
                    recipe.body_context,
                    f"{path}.body_context",
                    policy=policy,
                )
            if policy.allow_planar_boolean:
                encoded["planar_context"] = _encode_planar_boolean_context(
                    recipe.planar_context,
                    f"{path}.planar_context",
                    policy=policy,
                )
            if policy.allow_part_boolean:
                encoded["part_context"] = _encode_part_boolean_context(
                    recipe.part_context,
                    f"{path}.part_context",
                    policy=policy,
                )
            return encoded
        if type(recipe) is MultiBodyGeometry:
            if not policy.allow_multi_body:
                raise policy.encode_error(
                    f"{path} 的几何类型无法由 {policy.version_label} "
                    "无损编码：MultiBodyGeometry"
                )
            _field_exact_dataclass(
                recipe,
                MultiBodyGeometry,
                {
                    "name",
                    "bodies",
                    "retired_body_ids",
                    "retired_boolean_feature_ids",
                },
                path,
                policy,
            )
            return {
                "type": "MultiBodyGeometry",
                "name": _field_string(
                    recipe.name,
                    f"{path}.name",
                    policy.encode_error,
                ),
                "bodies": [
                    {
                        "id": _field_string(
                            body.id,
                            f"{path}.bodies[{index}].id",
                            policy.encode_error,
                        ),
                        "name": _field_string(
                            body.name,
                            f"{path}.bodies[{index}].name",
                            policy.encode_error,
                        ),
                        "recipe": encode_geometry_field(
                            body.recipe,
                            f"{path}.bodies[{index}].recipe",
                            ancestors,
                            policy=policy,
                        ),
                    }
                    for index, body in enumerate(recipe.bodies)
                ],
                "retired_body_ids": list(recipe.retired_body_ids),
                "retired_boolean_feature_ids": list(
                    recipe.retired_boolean_feature_ids
                ),
            }
        raise policy.encode_error(
            f"{path} 的几何类型无法由 {policy.version_label} "
            f"无损编码：{type(recipe).__name__}"
        )
    finally:
        ancestors.remove(identity)


def _encode_sketch_plane_field(
    plane: SketchPlane | None,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    if type(plane) is not SketchPlane:
        raise policy.encode_error(f"{path} 必须是 SketchPlane")
    return {
        "origin": _field_encode_number_array(plane.origin, f"{path}.origin", policy),
        "x_direction": _field_encode_number_array(
            plane.x_direction,
            f"{path}.x_direction",
            policy,
        ),
        "y_direction": _field_encode_number_array(
            plane.y_direction,
            f"{path}.y_direction",
            policy,
        ),
    }


def _encode_face_sketch_workplane_strategy(
    strategy: FaceSketchWorkplaneStrategy,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        strategy,
        FaceSketchWorkplaneStrategy,
        {"seed_axis", "sign", "origin_rule"},
        path,
        policy,
    )
    return {
        "seed_axis": strategy.seed_axis,
        "sign": strategy.sign,
        "origin_rule": strategy.origin_rule,
    }


def _encode_face_sketch_external_reference(
    reference: SketchExternalReference,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        reference,
        SketchExternalReference,
        {"id", "source", "derived_type"},
        path,
        policy,
    )
    return {
        "id": reference.id,
        "source_logical_id": reference.source.logical_id,
        "derived_type": reference.derived_type.value,
    }


def _encode_face_sketch_external_coincidence(
    coincidence: SketchExternalCoincidence,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        coincidence,
        SketchExternalCoincidence,
        {"point_id", "reference_id"},
        path,
        policy,
    )
    return {
        "point_id": coincidence.point_id,
        "reference_id": coincidence.reference_id,
    }


def _encode_face_sketch_step_proof(
    proof: FaceSketchBooleanStepProof,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        proof,
        FaceSketchBooleanStepProof,
        {"profile_id", "result_entities", "topology_mappings", "connection_proof"},
        path,
        policy,
    )
    lineage = _encode_boolean_body_context(
        BooleanBodyContext(
            "BF1",
            "B1",
            "B2",
            "Face sketch proof",
            proof.result_entities,
            proof.topology_mappings,
        ),
        path,
        policy=policy,
    )
    if lineage is None:
        raise policy.encode_error(f"{path} 的布尔谱系不能为空")
    encoded_entities_by_id = {
        item["logical_id"]: item for item in lineage["result_entities"]
    }
    encoded_entities = [
        encoded_entities_by_id[item.logical_id] for item in proof.result_entities
    ]
    encoded_mappings_by_key = {
        (
            item["source"],
            item["source_logical_id"],
            item["target_logical_id"],
            item["relation"],
        ): item
        for item in lineage["topology_mappings"]
    }
    encoded_mappings = [
        encoded_mappings_by_key[
            (
                item.source,
                item.source_logical_id,
                item.target_logical_id,
                item.relation,
            )
        ]
        for item in proof.topology_mappings
    ]
    connection = proof.connection_proof
    if connection is None:
        encoded_connection = None
    else:
        _field_exact_dataclass(
            connection,
            FaceSeedConnectionProof,
            {"support_face_id", "tool_start_face_id", "overlap_area"},
            f"{path}.connection_proof",
            policy,
        )
        encoded_connection = {
            "support_face_id": connection.support_face_id,
            "tool_start_face_id": connection.tool_start_face_id,
            "overlap_area": _field_number(
                connection.overlap_area,
                f"{path}.connection_proof.overlap_area",
                policy.encode_error,
                policy=policy,
            ),
        }
    return {
        "profile_id": proof.profile_id,
        "result_entities": encoded_entities,
        "topology_mappings": encoded_mappings,
        "connection_proof": encoded_connection,
    }


def _encode_boolean_body_context(
    context: BooleanBodyContext | None,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any] | None:
    if context is None:
        return None
    if type(context) is not BooleanBodyContext:
        raise policy.encode_error(
            f"{path} must be BooleanBodyContext or null"
        )
    return {
        "feature_id": _field_string(
            context.feature_id,
            f"{path}.feature_id",
            policy.encode_error,
        ),
        "target_body_id": _field_string(
            context.target_body_id,
            f"{path}.target_body_id",
            policy.encode_error,
        ),
        "tool_body_id": _field_string(
            context.tool_body_id,
            f"{path}.tool_body_id",
            policy.encode_error,
        ),
        "tool_body_name": _field_string(
            context.tool_body_name,
            f"{path}.tool_body_name",
            policy.encode_error,
        ),
        "result_entities": [
            {
                "kind": item.kind,
                "logical_id": item.logical_id,
                "semantic_role": item.semantic_role,
                "topology_links": list(item.topology_links),
            }
            for item in context.result_entities
        ],
        "topology_mappings": [
            {
                "source": item.source,
                "source_logical_id": item.source_logical_id,
                "target_logical_id": item.target_logical_id,
                "relation": item.relation,
            }
            for item in context.topology_mappings
        ],
    }


def _encode_planar_boolean_context(
    context: PlanarBooleanContext | None,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any] | None:
    if context is None:
        return None
    if type(context) is not PlanarBooleanContext:
        raise policy.encode_error(
            f"{path} must be PlanarBooleanContext or null"
        )
    synthetic = BooleanBodyContext(
        "BF1",
        "B1",
        "B2",
        "Planar Tool",
        context.result_entities,
        context.topology_mappings,
    )
    lineage = _encode_boolean_body_context(
        synthetic,
        path,
        policy=policy,
    )
    if lineage is None:
        raise policy.encode_error(f"{path} must not be null")
    return {
        "feature_id": context.feature_id,
        "target_face_id": context.target_face_id,
        "tool_face_ids": list(context.tool_face_ids),
        "result_entities": lineage["result_entities"],
        "topology_mappings": lineage["topology_mappings"],
    }


def _encode_part_boolean_context(
    context: PartBooleanContext | None,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any] | None:
    if context is None:
        return None
    if type(context) is not PartBooleanContext:
        raise policy.encode_error(
            f"{path} must be PartBooleanContext or null"
        )
    from fem.geometry.part_boolean import localize_part_boolean_context

    lineage = _encode_boolean_body_context(
        localize_part_boolean_context(context),
        path,
        policy=policy,
    )
    if lineage is None:
        raise policy.encode_error(f"{path} must not be null")
    # Persist the canonical Part namespace, not the localized replay view.
    return {
        "feature_id": context.feature_id,
        "target_part_id": context.target_part_id,
        "tool_part_id": context.tool_part_id,
        "result_part_id": context.result_part_id,
        "result_entities": [
            {
                "kind": item.kind,
                "logical_id": item.logical_id,
                "semantic_role": item.semantic_role,
                "topology_links": list(item.topology_links),
            }
            for item in context.result_entities
        ],
        "topology_mappings": [
            {
                "source": item.source,
                "source_logical_id": item.source_logical_id,
                "target_logical_id": item.target_logical_id,
                "relation": item.relation,
            }
            for item in context.topology_mappings
        ],
    }


def _encode_sketch_point_field(
    point: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    if type(point) is not SketchPoint:
        raise policy.encode_error(f"{path} 必须是 SketchPoint")
    return {
        "id": _field_string(point.id, f"{path}.id", policy.encode_error),
        "u": _field_number(point.u, f"{path}.u", policy.encode_error, policy=policy),
        "v": _field_number(point.v, f"{path}.v", policy.encode_error, policy=policy),
    }


def _encode_sketch_curve_field(
    curve: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    if type(curve) is SketchLine:
        return {
            "type": "line",
            "id": _field_string(curve.id, f"{path}.id", policy.encode_error),
            "start_point_id": _field_string(
                curve.start_point_id,
                f"{path}.start_point_id",
                policy.encode_error,
            ),
            "end_point_id": _field_string(
                curve.end_point_id,
                f"{path}.end_point_id",
                policy.encode_error,
            ),
        }
    if type(curve) is SketchArc:
        return {
            "type": "arc",
            "id": _field_string(curve.id, f"{path}.id", policy.encode_error),
            "start_point_id": _field_string(
                curve.start_point_id,
                f"{path}.start_point_id",
                policy.encode_error,
            ),
            "center_point_id": _field_string(
                curve.center_point_id,
                f"{path}.center_point_id",
                policy.encode_error,
            ),
            "end_point_id": _field_string(
                curve.end_point_id,
                f"{path}.end_point_id",
                policy.encode_error,
            ),
            "orientation": _field_string(
                curve.orientation,
                f"{path}.orientation",
                policy.encode_error,
            ),
        }
    if type(curve) is SketchCircle and curve.is_curve:
        return {
            "type": "circle",
            "id": _field_string(curve.id, f"{path}.id", policy.encode_error),
            "center_point_id": _field_string(
                curve.center_point_id,
                f"{path}.center_point_id",
                policy.encode_error,
            ),
            "radius": _field_number(
                curve.radius,
                f"{path}.radius",
                policy.encode_error,
                policy=policy,
            ),
        }
    raise policy.encode_error(f"{path} 不是可编码的严格草图曲线")


def encode_contour_field(
    contour: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    if type(contour) is SketchRectangle:
        _field_exact_dataclass(
            contour,
            SketchRectangle,
            {"operation", "x", "y", "width", "height"},
            path,
            policy,
        )
        return {
            "type": "rectangle",
            "operation": _field_string(
                contour.operation,
                f"{path}.operation",
                policy.encode_error,
            ),
            "x": _field_number(
                contour.x,
                f"{path}.x",
                policy.encode_error,
                policy=policy,
            ),
            "y": _field_number(
                contour.y,
                f"{path}.y",
                policy.encode_error,
                policy=policy,
            ),
            "width": _field_number(
                contour.width,
                f"{path}.width",
                policy.encode_error,
                policy=policy,
            ),
            "height": _field_number(
                contour.height,
                f"{path}.height",
                policy.encode_error,
                policy=policy,
            ),
        }
    if type(contour) is SketchCircle:
        if not contour.is_legacy:
            raise policy.encode_error(
                f"{path} 的严格 circle curve 不能作为旧 contour 编码"
            )
        return {
            "type": "circle",
            "operation": _field_string(
                contour.operation,
                f"{path}.operation",
                policy.encode_error,
            ),
            "x": _field_number(
                contour.x,
                f"{path}.x",
                policy.encode_error,
                policy=policy,
            ),
            "y": _field_number(
                contour.y,
                f"{path}.y",
                policy.encode_error,
                policy=policy,
            ),
            "radius": _field_number(
                contour.radius,
                f"{path}.radius",
                policy.encode_error,
                policy=policy,
            ),
        }
    raise policy.encode_error(
        f"{path} 的草图轮廓类型无法由 {policy.version_label} "
        f"无损编码：{type(contour).__name__}"
    )


def decode_material_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> MaterialDefinition:
    data = _field_mapping(value, path, policy.decode_error)
    required = {"name"}
    optional = {"properties"}
    if policy.require_current_fields:
        required.add("properties")
        optional.clear()
    _field_keys(
        data,
        path,
        required=required,
        optional=optional,
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        MaterialDefinition,
        path,
        policy,
        name=_field_string(
            data["name"],
            f"{path}.name",
            policy.decode_error,
        ),
        properties=_field_json_object(
            data["properties"] if "properties" in data else {},
            f"{path}.properties",
            policy.decode_error,
        ),
    )


def encode_material_field(
    material: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        material,
        MaterialDefinition,
        {"name", "properties"},
        path,
        policy,
    )
    return {
        "name": _field_string(
            material.name,
            f"{path}.name",
            policy.encode_error,
        ),
        "properties": _field_json_object(
            material.properties,
            f"{path}.properties",
            policy.encode_error,
        ),
    }


def decode_section_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SectionDefinition:
    data = _field_mapping(value, path, policy.decode_error)
    required = {"name", "material"}
    optional = {"section_type", "properties"}
    if policy.require_current_fields:
        required.update(optional)
        optional.clear()
    _field_keys(
        data,
        path,
        required=required,
        optional=optional,
        policy=policy,
        error_type=policy.decode_error,
    )
    return _field_construct(
        SectionDefinition,
        path,
        policy,
        name=_field_string(
            data["name"],
            f"{path}.name",
            policy.decode_error,
        ),
        material=_field_string(
            data["material"],
            f"{path}.material",
            policy.decode_error,
        ),
        section_type=_field_string(
            data["section_type"] if "section_type" in data else "solid",
            f"{path}.section_type",
            policy.decode_error,
        ),
        properties=_field_json_object(
            data["properties"] if "properties" in data else {},
            f"{path}.properties",
            policy.decode_error,
        ),
    )


def encode_section_field(
    section: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        section,
        SectionDefinition,
        {"name", "material", "section_type", "properties"},
        path,
        policy,
    )
    return {
        "name": _field_string(
            section.name,
            f"{path}.name",
            policy.encode_error,
        ),
        "material": _field_string(
            section.material,
            f"{path}.material",
            policy.encode_error,
        ),
        "section_type": _field_string(
            section.section_type,
            f"{path}.section_type",
            policy.encode_error,
        ),
        "properties": _field_json_object(
            section.properties,
            f"{path}.properties",
            policy.encode_error,
        ),
    }


def decode_assignment_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> RegionAssignment:
    data = _field_mapping(value, path, policy.decode_error)
    required = {"section_name", "region_name"}
    if policy.assignment_orientation:
        required.add("beam_orientation")
    _field_keys(
        data,
        path,
        required=required,
        optional=set(),
        policy=policy,
        error_type=policy.decode_error,
    )
    orientation = None
    if policy.assignment_orientation:
        orientation_value = data["beam_orientation"]
        if orientation_value is not None:
            orientation_path = f"{path}.beam_orientation"
            orientation_data = _field_mapping(
                orientation_value,
                orientation_path,
                policy.decode_error,
            )
            _field_keys(
                orientation_data,
                orientation_path,
                required={"type", "vector"},
                optional=set(),
                policy=policy,
                error_type=policy.decode_error,
            )
            if _field_string(
                orientation_data["type"],
                f"{orientation_path}.type",
                policy.decode_error,
            ) != "local_y_reference":
                raise policy.decode_error(
                    f"{orientation_path}.type 只接受 "
                    "'local_y_reference'"
                )
            vector = _field_array(
                orientation_data["vector"],
                f"{orientation_path}.vector",
                policy.decode_error,
            )
            if len(vector) != 3:
                raise policy.decode_error(
                    f"{orientation_path}.vector 必须恰有三个分量"
                )
            orientation = _field_construct(
                BeamOrientation,
                orientation_path,
                policy,
                tuple(
                    _field_number(
                        item,
                        f"{orientation_path}.vector[{index}]",
                        policy.decode_error,
                        policy=policy,
                    )
                    for index, item in enumerate(vector)
                ),
            )
    return _field_construct(
        RegionAssignment,
        path,
        policy,
        _field_string(
            data["section_name"],
            f"{path}.section_name",
            policy.decode_error,
        ),
        _field_string(
            data["region_name"],
            f"{path}.region_name",
            policy.decode_error,
        ),
        orientation,
    )


def encode_assignment_field(
    assignment: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        assignment,
        RegionAssignment,
        {"section_name", "region_name", "beam_orientation"},
        path,
        policy,
    )
    orientation = assignment.beam_orientation
    if not policy.assignment_orientation and orientation is not None:
        raise policy.encode_error(
            f"{path}.beam_orientation 无法由 .femproj v1 无损表示；"
            "v1 不支持 Beam orientation"
        )
    if (
        policy.assignment_orientation
        and orientation is not None
        and type(orientation) is not BeamOrientation
    ):
        raise policy.encode_error(
            f"{path}.beam_orientation 必须是 BeamOrientation 或 null"
        )
    result = {
        "section_name": _field_string(
            assignment.section_name,
            f"{path}.section_name",
            policy.encode_error,
        ),
        "region_name": _field_string(
            assignment.region_name,
            f"{path}.region_name",
            policy.encode_error,
        ),
    }
    if policy.assignment_orientation:
        result["beam_orientation"] = (
            None
            if orientation is None
            else {
                "type": "local_y_reference",
                "vector": [
                    _field_number(
                        item,
                        f"{path}.beam_orientation.vector[{index}]",
                        policy.encode_error,
                        policy=policy,
                    )
                    for index, item in enumerate(
                        orientation.local_y_reference
                    )
                ],
            }
        )
    return result


def decode_boundary_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> DisplacementConstraint:
    required = {"target", "first_component", "last_component"}
    optional = {
        "value",
        *(
            {"target_kind"}
            if policy.displacement_region_targets
            else set()
        ),
    }
    if policy.require_current_fields:
        required.add("value")
        optional.discard("value")
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    return _field_construct(
        DisplacementConstraint,
        path,
        policy,
        _field_target(data["target"], f"{path}.target", policy, encode=False),
        _field_integer(
            data["first_component"],
            f"{path}.first_component",
            policy.decode_error,
            policy=policy,
        ),
        _field_integer(
            data["last_component"],
            f"{path}.last_component",
            policy.decode_error,
            policy=policy,
        ),
        _field_number(
            data["value"] if "value" in data else 0.0,
            f"{path}.value",
            policy.decode_error,
            policy=policy,
        ),
        _field_string(
            data.get("target_kind", "node_set"),
            f"{path}.target_kind",
            policy.decode_error,
        ),
    )


def decode_cload_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> NodalLoad:
    data = _field_required_object(
        value,
        path,
        {"target", "component", "value"},
        policy=policy,
    )
    return _field_construct(
        NodalLoad,
        path,
        policy,
        _field_target(data["target"], f"{path}.target", policy, encode=False),
        _field_integer(
            data["component"],
            f"{path}.component",
            policy.decode_error,
            policy=policy,
        ),
        _field_number(
            data["value"],
            f"{path}.value",
            policy.decode_error,
            policy=policy,
        ),
    )


def decode_edge_load_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> EdgeLoad:
    required = {"edge"}
    optional = {"vector", "magnitude", "load_type"}
    if policy.require_current_fields:
        required.update(optional)
        optional.clear()
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    magnitude = data["magnitude"] if "magnitude" in data else None
    return _field_construct(
        EdgeLoad,
        path,
        policy,
        _field_string(data["edge"], f"{path}.edge", policy.decode_error),
        _field_decode_number_array(
            data["vector"] if "vector" in data else (),
            f"{path}.vector",
            policy,
        ),
        None
        if magnitude is None
        else _field_number(
            magnitude,
            f"{path}.magnitude",
            policy.decode_error,
            policy=policy,
        ),
        _field_string(
            data["load_type"] if "load_type" in data else "traction",
            f"{path}.load_type",
            policy.decode_error,
        ),
    )


def decode_surface_load_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> SurfaceLoad:
    required = {"surface"}
    optional = {"vector", "magnitude", "load_type"}
    if policy.require_current_fields:
        required.update(optional)
        optional.clear()
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    magnitude = data["magnitude"] if "magnitude" in data else None
    return _field_construct(
        SurfaceLoad,
        path,
        policy,
        _field_string(
            data["surface"],
            f"{path}.surface",
            policy.decode_error,
        ),
        _field_decode_number_array(
            data["vector"] if "vector" in data else (),
            f"{path}.vector",
            policy,
        ),
        None
        if magnitude is None
        else _field_number(
            magnitude,
            f"{path}.magnitude",
            policy.decode_error,
            policy=policy,
        ),
        _field_string(
            data["load_type"] if "load_type" in data else "traction",
            f"{path}.load_type",
            policy.decode_error,
        ),
    )


def decode_line_load_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> LineLoad:
    required = {"target", "vector"}
    optional = {"coordinate_system"}
    if policy.require_current_fields:
        required.add("coordinate_system")
        optional.clear()
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    coordinate_system = _field_string(
        (
            data["coordinate_system"]
            if "coordinate_system" in data
            else "global"
        ),
        f"{path}.coordinate_system",
        policy.decode_error,
    )
    if coordinate_system not in {"global", "local"}:
        raise policy.decode_error(
            f"{path}.coordinate_system 必须是 'global' 或 'local'"
        )
    return _field_construct(
        LineLoad,
        path,
        policy,
        _field_target(data["target"], f"{path}.target", policy, encode=False),
        _field_decode_number_array(data["vector"], f"{path}.vector", policy),
        coordinate_system,
    )


def decode_body_load_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> BodyForce:
    data = _field_required_object(
        value,
        path,
        {"target", "vector"},
        policy=policy,
    )
    return _field_construct(
        BodyForce,
        path,
        policy,
        _field_target(
            data["target"],
            f"{path}.target",
            policy,
            encode=False,
        ),
        _field_decode_number_array(data["vector"], f"{path}.vector", policy),
    )


def decode_gravity_load_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> GravityLoad:
    required = {"acceleration"}
    optional = {"target"}
    if policy.require_current_fields:
        required.add("target")
        optional.clear()
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    target = data["target"] if "target" in data else None
    return _field_construct(
        GravityLoad,
        path,
        policy,
        _field_decode_number_array(
            data["acceleration"],
            f"{path}.acceleration",
            policy,
        ),
        (
            None
            if target is None
            else _field_target(
                target,
                f"{path}.target",
                policy,
                encode=False,
            )
        ),
    )


def decode_output_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> OutputRequest:
    required = {"kind", "target"}
    optional = {"variables", "metadata"}
    if policy.require_current_fields:
        required.update(optional)
        optional.clear()
    data = _field_required_object(
        value,
        path,
        required,
        optional=optional,
        policy=policy,
    )
    variables_value = data["variables"] if "variables" in data else ()
    variables = tuple(
        _field_string(
            item,
            f"{path}.variables[{index}]",
            policy.decode_error,
        )
        for index, item in enumerate(
            _field_array(
                variables_value,
                f"{path}.variables",
                policy.decode_error,
            )
        )
    )
    return _field_construct(
        OutputRequest,
        path,
        policy,
        _field_string(data["kind"], f"{path}.kind", policy.decode_error),
        _field_string(
            data["target"],
            f"{path}.target",
            policy.decode_error,
        ),
        variables,
        _field_json_object(
            data["metadata"] if "metadata" in data else {},
            f"{path}.metadata",
            policy.decode_error,
        ),
        None,
    )


def encode_boundary_field(
    boundary: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        boundary,
        DisplacementConstraint,
        {
            "target",
            "first_component",
            "last_component",
            "value",
            "target_kind",
            "name",
        },
        path,
        policy,
    )
    _field_reject_named_analysis_object(boundary, path, policy)
    target_kind = _field_string(
        boundary.target_kind,
        f"{path}.target_kind",
        policy.encode_error,
    )
    if (
        target_kind != "node_set"
        and not policy.displacement_region_targets
    ):
        raise policy.encode_error(
            f"{path}.target_kind 无法由 {policy.version_label} 无损表示"
        )
    result = {
        "target": _field_target(
            boundary.target,
            f"{path}.target",
            policy,
            encode=True,
        ),
        "first_component": _field_integer(
            boundary.first_component,
            f"{path}.first_component",
            policy.encode_error,
            policy=policy,
        ),
        "last_component": _field_integer(
            boundary.last_component,
            f"{path}.last_component",
            policy.encode_error,
            policy=policy,
        ),
        "value": _field_number(
            boundary.value,
            f"{path}.value",
            policy.encode_error,
            policy=policy,
        ),
    }
    if target_kind != "node_set":
        result["target_kind"] = target_kind
    return result


def encode_cload_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        NodalLoad,
        {"target", "component", "value", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    return {
        "target": _field_target(
            load.target,
            f"{path}.target",
            policy,
            encode=True,
        ),
        "component": _field_integer(
            load.component,
            f"{path}.component",
            policy.encode_error,
            policy=policy,
        ),
        "value": _field_number(
            load.value,
            f"{path}.value",
            policy.encode_error,
            policy=policy,
        ),
    }


def encode_edge_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        EdgeLoad,
        {"edge", "vector", "magnitude", "load_type", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    return {
        "edge": _field_string(
            load.edge,
            f"{path}.edge",
            policy.encode_error,
        ),
        "vector": _field_encode_number_array(
            load.vector,
            f"{path}.vector",
            policy,
        ),
        "magnitude": (
            None
            if load.magnitude is None
            else _field_number(
                load.magnitude,
                f"{path}.magnitude",
                policy.encode_error,
                policy=policy,
            )
        ),
        "load_type": _field_string(
            load.load_type,
            f"{path}.load_type",
            policy.encode_error,
        ),
    }


def encode_surface_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        SurfaceLoad,
        {"surface", "vector", "magnitude", "load_type", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    return {
        "surface": _field_string(
            load.surface,
            f"{path}.surface",
            policy.encode_error,
        ),
        "vector": _field_encode_number_array(
            load.vector,
            f"{path}.vector",
            policy,
        ),
        "magnitude": (
            None
            if load.magnitude is None
            else _field_number(
                load.magnitude,
                f"{path}.magnitude",
                policy.encode_error,
                policy=policy,
            )
        ),
        "load_type": _field_string(
            load.load_type,
            f"{path}.load_type",
            policy.encode_error,
        ),
    }


def encode_line_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        LineLoad,
        {"target", "vector", "coordinate_system", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    coordinate_system = _field_string(
        load.coordinate_system,
        f"{path}.coordinate_system",
        policy.encode_error,
    )
    if coordinate_system not in {"global", "local"}:
        raise policy.encode_error(
            f"{path}.coordinate_system 必须是 'global' 或 'local'"
        )
    return {
        "target": _field_target(
            load.target,
            f"{path}.target",
            policy,
            encode=True,
        ),
        "vector": _field_encode_number_array(
            load.vector,
            f"{path}.vector",
            policy,
        ),
        "coordinate_system": coordinate_system,
    }


def encode_body_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        BodyForce,
        {"target", "vector", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    return {
        "target": _field_target(
            load.target,
            f"{path}.target",
            policy,
            encode=True,
        ),
        "vector": _field_encode_number_array(
            load.vector,
            f"{path}.vector",
            policy,
        ),
    }


def encode_gravity_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        GravityLoad,
        {"acceleration", "target", "name"},
        path,
        policy,
    )
    _field_reject_named_analysis_object(load, path, policy)
    return {
        "acceleration": _field_encode_number_array(
            load.acceleration,
            f"{path}.acceleration",
            policy,
        ),
        "target": (
            None
            if load.target is None
            else _field_target(
                load.target,
                f"{path}.target",
                policy,
                encode=True,
            )
        ),
    }


def encode_output_field(
    output: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        output,
        OutputRequest,
        {
            "kind",
            "target",
            "variables",
            "metadata",
            "source_evidence",
            "name",
        },
        path,
        policy,
    )
    _field_reject_named_analysis_object(output, path, policy)
    if output.source_evidence is not None:
        raise policy.encode_error(
            f"{path}.source_evidence 不能由 {policy.version_label} 无损表示"
        )
    try:
        metadata = thaw_json_mapping(
            output.metadata,
            name=f"{path}.metadata",
        )
    except (TypeError, ValueError) as error:
        raise policy.encode_error(
            f"{path}.metadata 不是有效的 immutable JSON mapping：{error}"
        ) from error
    return {
        "kind": _field_string(
            output.kind,
            f"{path}.kind",
            policy.encode_error,
        ),
        "target": _field_string(
            output.target,
            f"{path}.target",
            policy.encode_error,
        ),
        "variables": [
            _field_string(
                item,
                f"{path}.variables[{index}]",
                policy.encode_error,
            )
            for index, item in enumerate(
                _field_runtime_sequence(
                    output.variables,
                    f"{path}.variables",
                    policy,
                )
            )
        ],
        "metadata": _field_json_object(
            metadata,
            f"{path}.metadata",
            policy.encode_error,
        ),
    }


_STEP_COLLECTION_CODECS = {
    "boundaries": (decode_boundary_field, encode_boundary_field),
    "cloads": (decode_cload_field, encode_cload_field),
    "edge_loads": (decode_edge_load_field, encode_edge_load_field),
    "surface_loads": (
        decode_surface_load_field,
        encode_surface_load_field,
    ),
    "line_loads": (decode_line_load_field, encode_line_load_field),
    "body_loads": (decode_body_load_field, encode_body_load_field),
    "gravity_loads": (
        decode_gravity_load_field,
        encode_gravity_load_field,
    ),
    "outputs": (decode_output_field, encode_output_field),
}


def decode_step_field(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> AnalysisStep:
    data = _field_mapping(value, path, policy.decode_error)
    collection_codecs = {
        name: codecs
        for name, codecs in _STEP_COLLECTION_CODECS.items()
        if name != "body_loads" or policy.body_force_loads
    }
    required = {"name"}
    optional = {"procedure", "metadata", *collection_codecs}
    if policy.require_current_fields:
        required.update(optional - {"body_loads"})
        optional.intersection_update({"body_loads"})
    _field_keys(
        data,
        path,
        required=required,
        optional=optional,
        policy=policy,
        error_type=policy.decode_error,
    )
    collections: dict[str, tuple[Any, ...]] = {}
    for collection_name, (decoder, _encoder) in (
        collection_codecs.items()
    ):
        collection_path = f"{path}.{collection_name}"
        raw_items = data[collection_name] if collection_name in data else ()
        collections[collection_name] = tuple(
            decoder(
                item,
                f"{collection_path}[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_array(
                    raw_items,
                    collection_path,
                    policy.decode_error,
                )
            )
        )
    return _field_construct(
        AnalysisStep,
        path,
        policy,
        name=_field_string(
            data["name"],
            f"{path}.name",
            policy.decode_error,
        ),
        procedure=_field_string(
            data["procedure"] if "procedure" in data else "static",
            f"{path}.procedure",
            policy.decode_error,
        ),
        metadata=_field_json_object(
            data["metadata"] if "metadata" in data else {},
            f"{path}.metadata",
            policy.decode_error,
        ),
        **collections,
    )


def encode_step_field(
    step: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    expected_fields = {
        "name",
        "procedure",
        "metadata",
        *_STEP_COLLECTION_CODECS,
    }
    _field_exact_dataclass(
        step,
        AnalysisStep,
        expected_fields,
        path,
        policy,
    )
    if not policy.body_force_loads and tuple(step.body_loads):
        raise policy.encode_error(
            f"{path}.body_loads 无法由 {policy.version_label} 无损表示"
        )
    collection_codecs = {
        name: codecs
        for name, codecs in _STEP_COLLECTION_CODECS.items()
        if name != "body_loads" or policy.body_force_loads
    }
    result: dict[str, Any] = {
        "name": _field_string(
            step.name,
            f"{path}.name",
            policy.encode_error,
        ),
        "procedure": _field_string(
            step.procedure,
            f"{path}.procedure",
            policy.encode_error,
        ),
        "metadata": _field_json_object(
            step.metadata,
            f"{path}.metadata",
            policy.encode_error,
        ),
    }
    for collection_name, (_decoder, encoder) in (
        collection_codecs.items()
    ):
        collection_path = f"{path}.{collection_name}"
        result[collection_name] = [
            encoder(
                item,
                f"{collection_path}[{index}]",
                policy=policy,
            )
            for index, item in enumerate(
                _field_runtime_sequence(
                    getattr(step, collection_name),
                    collection_path,
                    policy,
                )
            )
        ]
    return result


def _field_required_object(
    value: Any,
    path: str,
    required: set[str],
    *,
    optional: set[str] | None = None,
    policy: ProjectFieldCodecPolicy,
) -> Mapping[str, Any]:
    data = _field_mapping(value, path, policy.decode_error)
    _field_keys(
        data,
        path,
        required=required,
        optional=set() if optional is None else optional,
        policy=policy,
        error_type=policy.decode_error,
    )
    return data


def _field_construct(
    constructor: type[Any],
    path: str,
    policy: ProjectFieldCodecPolicy,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return constructor(*args, **kwargs)
    except (TypeError, ValueError, OverflowError) as error:
        raise policy.decode_error(f"{path} 无效：{error}") from error


def _field_reject_named_analysis_object(
    value: Any,
    path: str,
    policy: ProjectFieldCodecPolicy,
) -> None:
    if getattr(value, "name", None) is not None:
        raise policy.encode_error(
            f"{path}.name 无法由 {policy.version_label} 无损表示"
        )


def _field_exact_dataclass(
    value: Any,
    expected_type: type[Any],
    expected_fields: set[str],
    path: str,
    policy: ProjectFieldCodecPolicy,
) -> None:
    if type(value) is not expected_type:
        raise policy.encode_error(
            f"{path} 必须是 {expected_type.__name__}，"
            f"收到 {type(value).__name__}"
        )
    actual_fields = {item.name for item in fields(expected_type)}
    if actual_fields != expected_fields:
        unsupported = sorted(actual_fields ^ expected_fields)
        raise policy.encode_error(
            f"{expected_type.__name__} 与 {policy.version_label} "
            f"字段契约不一致，拒绝静默丢失：{unsupported}"
        )


def _field_keys(
    data: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str],
    policy: ProjectFieldCodecPolicy,
    error_type: type[Exception],
) -> None:
    actual = set(data)
    missing = sorted(required - actual)
    if missing:
        raise error_type(f"{path} 缺少必需字段：{', '.join(missing)}")
    unknown = sorted(actual - required - optional)
    if unknown:
        raise error_type(
            f"{path} 包含 {policy.version_label} 未知字段："
            f"{', '.join(unknown)}"
        )


def _field_mapping(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} 必须是 JSON object")
    if any(type(key) is not str for key in value):
        raise error_type(f"{path} 的所有键必须是字符串")
    return value


def _field_array(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise error_type(f"{path} 必须是 JSON array")
    return tuple(value)


def _field_runtime_sequence(
    value: Any,
    path: str,
    policy: ProjectFieldCodecPolicy,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        Sequence,
    ):
        raise policy.encode_error(f"{path} 必须是有序序列")
    return tuple(value)


def _field_string(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{path} 必须是字符串")
    if not value.strip():
        raise error_type(f"{path} 不能为空")
    return value


def _field_enum(
    value: Any,
    path: str,
    enum_type: Any,
    policy: ProjectFieldCodecPolicy,
) -> Any:
    raw = _field_string(value, path, policy.decode_error)
    try:
        return enum_type(raw)
    except ValueError as error:
        raise policy.decode_error(f"{path} 包含不支持的枚举值：{raw!r}") from error


def _field_integer(
    value: Any,
    path: str,
    error_type: type[Exception],
    *,
    policy: ProjectFieldCodecPolicy,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "严格整数" if policy.require_current_fields else "整数"
        raise error_type(f"{path} 必须是{qualifier}")
    return value


def _field_number(
    value: Any,
    path: str,
    error_type: type[Exception],
    *,
    policy: ProjectFieldCodecPolicy,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = "有限实数" if policy.require_current_fields else "数值"
        raise error_type(f"{path} 必须是{message}")
    if policy.require_current_fields:
        try:
            finite = math.isfinite(float(value))
        except OverflowError as error:
            raise error_type(f"{path} 必须是有限实数") from error
        if not finite:
            raise error_type(f"{path} 必须是有限实数")
    elif isinstance(value, float) and not math.isfinite(value):
        raise error_type(f"{path} 必须是有限数值")
    return value


def _field_target(
    value: Any,
    path: str,
    policy: ProjectFieldCodecPolicy,
    *,
    encode: bool,
) -> str | int:
    error_type = policy.encode_error if encode else policy.decode_error
    if encode or policy.require_current_fields:
        if type(value) is str and value.strip():
            return value
        if (
            encode
            and not policy.require_current_fields
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            raise error_type(
                f"{path} 不能使用 mesh integer target；"
                "v1 writer 只接受 non-empty stable region name"
            )
        if encode and not policy.require_current_fields:
            raise error_type(
                f"{path} 必须是 non-empty stable region name"
            )
        raise error_type(f"{path} 必须是 non-empty stable region name")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise error_type(f"{path} 必须是名称或整数编号")
    if isinstance(value, str) and not value.strip():
        raise error_type(f"{path} 不能为空")
    return value


def _field_decode_number_array(
    value: Any,
    path: str,
    policy: ProjectFieldCodecPolicy,
) -> tuple[int | float, ...]:
    return tuple(
        _field_number(
            item,
            f"{path}[{index}]",
            policy.decode_error,
            policy=policy,
        )
        for index, item in enumerate(
            _field_array(value, path, policy.decode_error)
        )
    )


def _field_encode_number_array(
    value: Any,
    path: str,
    policy: ProjectFieldCodecPolicy,
) -> list[int | float]:
    return [
        _field_number(
            item,
            f"{path}[{index}]",
            policy.encode_error,
            policy=policy,
        )
        for index, item in enumerate(
            _field_runtime_sequence(value, path, policy)
        )
    ]


def _field_json_object(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise error_type(f"{path} 必须是普通 JSON object")
    return _field_json_value(value, path, error_type, set())


def _field_json_value(
    value: Any,
    path: str,
    error_type: type[Exception],
    ancestors: set[int],
) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} 必须是有限数值")
        return value
    if type(value) is list:
        identity = id(value)
        if identity in ancestors:
            raise error_type(f"{path} 包含循环 JSON 引用")
        ancestors.add(identity)
        try:
            return [
                _field_json_value(
                    item,
                    f"{path}[{index}]",
                    error_type,
                    ancestors,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in ancestors:
            raise error_type(f"{path} 包含循环 JSON 引用")
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise error_type(
                        f"{path} 的 JSON object 键必须是字符串"
                    )
                result[key] = _field_json_value(
                    item,
                    f"{path}.{key}",
                    error_type,
                    ancestors,
                )
            return result
        finally:
            ancestors.remove(identity)
    raise error_type(
        f"{path} 的 {type(value).__name__} 值无法由 JSON 无损表示"
    )


__all__ = [
    "ProjectFieldCodecPolicy",
    "atomic_write_project",
    "decode_assignment_field",
    "decode_boundary_field",
    "decode_cload_field",
    "decode_contour_field",
    "decode_edge_load_field",
    "decode_geometry_field",
    "decode_gravity_load_field",
    "decode_line_load_field",
    "decode_material_field",
    "decode_output_field",
    "decode_section_field",
    "decode_step_field",
    "decode_surface_load_field",
    "dumps_canonical_json",
    "dumps_json",
    "encode_assignment_field",
    "encode_boundary_field",
    "encode_cload_field",
    "encode_contour_field",
    "encode_edge_load_field",
    "encode_geometry_field",
    "encode_gravity_load_field",
    "encode_line_load_field",
    "encode_material_field",
    "encode_output_field",
    "encode_section_field",
    "encode_step_field",
    "encode_surface_load_field",
    "loads_json_strict",
    "unwrap_project_snapshot",
]
