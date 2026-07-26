"""Shared strict JSON and crash-safe file helpers for project codecs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from fem.core.model import (
    AnalysisStep,
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

from ._project_errors import (
    ProjectDecodeError,
    ProjectEncodeError,
)


_VerifiedT = TypeVar("_VerifiedT")


@dataclass(frozen=True, slots=True)
class ProjectFieldCodecPolicy:
    """Version policy for shared current-authoring field codecs."""

    version_label: str
    decode_error: type[ProjectDecodeError]
    encode_error: type[ProjectEncodeError]
    require_current_fields: bool
    assignment_orientation: bool


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


def atomic_write_project(
    path: str | Path,
    serialized: str,
    *,
    verifier: Callable[[Path], _VerifiedT],
    semantic_encoder: Callable[[_VerifiedT], Any],
    expected_semantic: Any,
    error_type: type[ProjectEncodeError] = ProjectEncodeError,
    mismatch_message: str = "临时项目文件校验后与保存 snapshot 不一致",
    replace_func: Callable[[str | Path, str | Path], Any] | None = None,
    unlink_func: Callable[[Path], Any] | None = None,
) -> Path:
    """Install serialized project text only after durable write and verification.

    Cleanup failures are attached to an in-flight primary exception with
    ``BaseException.add_note`` so they cannot hide the operation that failed.
    """

    if not isinstance(serialized, str):
        raise TypeError("serialized 必须是 str")
    target = Path(path)
    expected = deepcopy(expected_semantic)
    replace = os.replace if replace_func is None else replace_func
    unlink = (
        (lambda temporary: temporary.unlink())
        if unlink_func is None
        else unlink_func
    )

    descriptor = -1
    temporary: Path | None = None
    installed = False
    primary_error: BaseException | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        stream = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        descriptor = -1
        stream_error: BaseException | None = None
        try:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            try:
                stream.close()
            except BaseException as cleanup_error:
                if stream_error is None:
                    raise
                _add_cleanup_note(
                    stream_error,
                    cleanup_error,
                    action="关闭临时项目文件",
                    temporary=temporary,
                )

        verified = verifier(temporary)
        actual_semantic = semantic_encoder(verified)
        if actual_semantic != expected:
            raise error_type(mismatch_message)

        replace(temporary, target)
        installed = True
        return target
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                _add_cleanup_note(
                    primary_error,
                    cleanup_error,
                    action="关闭临时项目文件描述符",
                    temporary=temporary,
                )
        if temporary is not None and not installed:
            try:
                unlink(temporary)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                _add_cleanup_note(
                    primary_error,
                    cleanup_error,
                    action="删除临时项目文件",
                    temporary=temporary,
                )


def _add_cleanup_note(
    primary_error: BaseException,
    cleanup_error: BaseException,
    *,
    action: str,
    temporary: Path | None,
) -> None:
    location = "<尚未创建>" if temporary is None else str(temporary)
    primary_error.add_note(
        f"{action}失败；临时路径 {location}；"
        f"{type(cleanup_error).__name__}: {cleanup_error}"
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
    if kind == "SketchGeometry":
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
        _field_keys(
            data,
            path,
            required={"type", "base", "height"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
            ExtrudedGeometry,
            path,
            policy,
            decode_geometry_field(
                data["base"],
                f"{path}.base",
                policy=policy,
            ),
            _field_number(
                data["height"],
                f"{path}.height",
                policy.decode_error,
                policy=policy,
            ),
        )
    if kind == "BooleanGeometry":
        _field_keys(
            data,
            path,
            required={"type", "name", "operation", "object", "tool"},
            optional=set(),
            policy=policy,
            error_type=policy.decode_error,
        )
        return _field_construct(
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
        if type(recipe) is SketchGeometry:
            _field_exact_dataclass(
                recipe,
                SketchGeometry,
                {"name", "contours"},
                path,
                policy,
            )
            return {
                "type": "SketchGeometry",
                "name": _field_string(
                    recipe.name,
                    f"{path}.name",
                    policy.encode_error,
                ),
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
                {"base", "height"},
                path,
                policy,
            )
            return {
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
        if type(recipe) is BooleanGeometry:
            _field_exact_dataclass(
                recipe,
                BooleanGeometry,
                {
                    "name",
                    "operation",
                    "object_geometry",
                    "tool_geometry",
                },
                path,
                policy,
            )
            return {
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
        raise policy.encode_error(
            f"{path} 的几何类型无法由 {policy.version_label} "
            f"无损编码：{type(recipe).__name__}"
        )
    finally:
        ancestors.remove(identity)


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
        _field_exact_dataclass(
            contour,
            SketchCircle,
            {"operation", "x", "y", "radius"},
            path,
            policy,
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
    optional = {"value"}
    if policy.require_current_fields:
        required.add("value")
        optional.clear()
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
        {"target", "first_component", "last_component", "value"},
        path,
        policy,
    )
    return {
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


def encode_cload_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        NodalLoad,
        {"target", "component", "value"},
        path,
        policy,
    )
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
        {"edge", "vector", "magnitude", "load_type"},
        path,
        policy,
    )
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
        {"surface", "vector", "magnitude", "load_type"},
        path,
        policy,
    )
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
        {"target", "vector", "coordinate_system"},
        path,
        policy,
    )
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


def encode_gravity_load_field(
    load: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy,
) -> dict[str, Any]:
    _field_exact_dataclass(
        load,
        GravityLoad,
        {"acceleration", "target"},
        path,
        policy,
    )
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
        {"kind", "target", "variables", "metadata"},
        path,
        policy,
    )
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
            output.metadata,
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
    required = {"name"}
    optional = {"procedure", "metadata", *_STEP_COLLECTION_CODECS}
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
    collections: dict[str, tuple[Any, ...]] = {}
    for collection_name, (decoder, _encoder) in (
        _STEP_COLLECTION_CODECS.items()
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
        _STEP_COLLECTION_CODECS.items()
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
