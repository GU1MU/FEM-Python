"""Human-readable geometry proposal details and local recovery messages."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .schemas import ToolResult


def geometry_proposal_preview_message(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: ToolResult,
    user_request: str | None,
) -> str | None:
    """Expose geometry parameters not covered by the proposal card summary."""

    if not result.ok or not isinstance(result.data, Mapping):
        return None
    chinese = _uses_chinese(user_request)
    if tool_name == "prepare_planar_construction_proposal":
        details = _planar_construction_preview(arguments, result.data, chinese)
    elif tool_name == "prepare_geometry_edit":
        details = _geometry_edit_preview(arguments, chinese)
    else:
        return None
    title = "**方案预览**" if chinese else "**Plan preview**"
    return "\n".join([title, *details])[:4_000]


def _planar_construction_preview(
    arguments: Mapping[str, Any],
    data: Mapping[str, Any],
    chinese: bool,
) -> list[str]:
    lines: list[str] = []
    part_function = arguments.get("part_function")
    if isinstance(part_function, str) and part_function.strip():
        lines.append(
            f"- 建模目标：{part_function.strip()}"
            if chinese
            else f"- Modeling goal: {part_function.strip()}"
        )
    output = arguments.get("output")
    output_description = _planar_output_description(output, chinese)
    if output_description:
        lines.append(
            f"- 输出形式：{output_description}"
            if chinese
            else f"- Output: {output_description}"
        )
    construction = arguments.get("construction")
    nodes = construction.get("nodes") if isinstance(construction, Mapping) else None
    if isinstance(nodes, list):
        lines.extend(_planar_geometry_description(nodes, chinese))
    proof = data.get("proof_summary")
    if isinstance(proof, Mapping):
        material_count = proof.get("material_profile_count")
        hole_count = proof.get("hole_count")
        component_count = proof.get("component_count")
        if chinese:
            lines.append(
                "- 本地几何校验：形成 "
                f"{material_count} 个材料区域和 {hole_count} 个切除区域；"
                f"材料由 {component_count} 个连通部分组成"
            )
        else:
            lines.append(
                "- Local geometry check: "
                f"{material_count} material region(s), {hole_count} cutout(s), "
                f"and {component_count} connected component(s)"
            )
        bounding_box = proof.get("bounding_box")
        if isinstance(bounding_box, list) and len(bounding_box) == 4:
            width = _numeric_difference(bounding_box[2], bounding_box[0])
            height = _numeric_difference(bounding_box[3], bounding_box[1])
            if width is not None and height is not None:
                lines.append(
                    f"- 外包尺寸：{width} × {height}"
                    if chinese
                    else f"- Overall bounds: {width} × {height}"
                )
    return lines


def _planar_geometry_description(
    nodes: Sequence[object],
    chinese: bool,
) -> list[str]:
    usable = [node for node in nodes if isinstance(node, Mapping)]
    rectangles = [node for node in usable if node.get("kind") == "rectangle"]
    circles = [node for node in usable if node.get("kind") == "circle"]
    polygons = [node for node in usable if node.get("kind") == "polygon"]
    paths = [node for node in usable if node.get("kind") == "path_stroke"]
    patterns = [
        node
        for node in usable
        if node.get("kind")
        in {"linear_pattern", "rectangular_pattern", "circular_pattern"}
    ]
    lines = ["- 几何组成：" if chinese else "- Geometry:"]
    if rectangles:
        descriptions = [
            (
                f"左下角 ({_preview_number(node.get('x'))}, "
                f"{_preview_number(node.get('y'))})，尺寸 "
                f"{_preview_number(node.get('width'))} × "
                f"{_preview_number(node.get('height'))}"
            )
            for node in rectangles[:8]
        ]
        lines.append(
            "  - 矩形轮廓：" + "；".join(descriptions)
            if chinese
            else "  - Rectangles: " + "; ".join(descriptions)
        )
    if circles:
        descriptions = [
            (
                f"圆心 ({_preview_number(node.get('center_x'))}, "
                f"{_preview_number(node.get('center_y'))})，半径 "
                f"{_preview_number(node.get('radius'))}"
            )
            for node in circles[:8]
        ]
        lines.append(
            "  - 圆形轮廓：" + "；".join(descriptions)
            if chinese
            else "  - Circles: " + "; ".join(descriptions)
        )
    if polygons:
        counts = [
            len(node.get("vertices", ()))
            if isinstance(node.get("vertices"), list)
            else "?"
            for node in polygons
        ]
        text = "、".join(str(count) for count in counts[:8])
        lines.append(
            f"  - 闭合多边形：{len(polygons)} 个，顶点数依次为 {text}"
            if chinese
            else (
                f"  - Closed polygons: {len(polygons)}, with vertex counts "
                f"{', '.join(str(count) for count in counts[:8])}"
            )
        )
    for path in paths[:6]:
        points = _point_chain(path.get("points"))
        width = _preview_number(path.get("width"))
        if chinese:
            lines.append(f"  - 连续定宽槽：中心路径 {points}，槽宽 {width}")
        else:
            lines.append(f"  - Continuous slot: centerline {points}, width {width}")
    for pattern in patterns[:6]:
        lines.append("  - " + _pattern_description(pattern, chinese))
    represented = len(rectangles) + len(circles) + len(polygons) + len(paths)
    if represented == 0 and not patterns:
        lines.append(
            "  - 由基础轮廓和几何变换构成"
            if chinese
            else "  - Built from basic profiles and geometric transforms"
        )
    boolean_kinds = {
        str(node.get("kind"))
        for node in usable
        if node.get("kind") in {"union", "difference", "intersection"}
    }
    if boolean_kinds:
        if chinese:
            relation = (
                "合并相接轮廓，并从主体中切除槽或孔"
                if "difference" in boolean_kinds
                else "合并相接轮廓"
            )
            lines.append(f"- 组合关系：{relation}")
        else:
            relation = (
                "join connected profiles and subtract slots or holes from the body"
                if "difference" in boolean_kinds
                else "join connected profiles"
            )
            lines.append(f"- Combination: {relation}")
    return lines


def _planar_output_description(output: object, chinese: bool) -> str:
    if output == "planar":
        return "二维平面草图" if chinese else "2D planar sketch"
    if not isinstance(output, Mapping):
        return ""
    kind = output.get("kind")
    if kind == "extrusion":
        height = _preview_number(output.get("height"))
        return (
            f"沿法向拉伸 {height} 的三维实体"
            if chinese
            else f"3D extrusion, height {height}"
        )
    if kind == "revolution":
        axis = str(output.get("axis", "?"))
        angle = _preview_number(output.get("angle_degrees"))
        return (
            f"绕 {axis.upper()} 轴旋转 {angle}° 的三维实体"
            if chinese
            else f"3D revolution about the {axis.upper()} axis through {angle}°"
        )
    if kind == "path_sweep":
        path = output.get("path")
        points = path.get("points") if isinstance(path, Mapping) else None
        count = len(points) if isinstance(points, list) else "?"
        return (
            f"沿含 {count} 个控制点的路径扫掠为三维实体"
            if chinese
            else f"3D path sweep along a path with {count} control points"
        )
    return ""


def _pattern_description(pattern: Mapping[str, Any], chinese: bool) -> str:
    kind = pattern.get("kind")
    if kind == "linear_pattern":
        count = pattern.get("count")
        dx = _preview_number(pattern.get("step_x"))
        dy = _preview_number(pattern.get("step_y"))
        return (
            f"线性阵列 {count} 个，间距向量 ({dx}, {dy})"
            if chinese
            else f"Linear pattern of {count}, spacing vector ({dx}, {dy})"
        )
    if kind == "rectangular_pattern":
        count_x = pattern.get("count_x")
        count_y = pattern.get("count_y")
        dx = _preview_number(pattern.get("spacing_x"))
        dy = _preview_number(pattern.get("spacing_y"))
        return (
            f"矩形阵列 {count_x} × {count_y}，横向间距 {dx}，纵向间距 {dy}"
            if chinese
            else f"Rectangular pattern {count_x} × {count_y}, spacing {dx} × {dy}"
        )
    count = pattern.get("count")
    center_x = _preview_number(pattern.get("center_x"))
    center_y = _preview_number(pattern.get("center_y"))
    angle = _preview_number(pattern.get("total_angle_degrees"))
    return (
        f"环形阵列 {count} 个，中心 ({center_x}, {center_y})，总角度 {angle}°"
        if chinese
        else f"Circular pattern of {count}, center ({center_x}, {center_y}), total angle {angle}°"
    )


def _geometry_edit_preview(
    arguments: Mapping[str, Any],
    chinese: bool,
) -> list[str]:
    lines = ["- 修改对象：当前部件" if chinese else "- Target: current part"]
    edit = arguments.get("edit")
    if isinstance(edit, Mapping):
        lines.extend(_geometry_edit_description(edit, chinese))
    spatial_relation = arguments.get("spatial_relation")
    if isinstance(spatial_relation, Mapping):
        reference = str(spatial_relation.get("reference_feature_id", "?"))
        relation = str(spatial_relation.get("relation", "?"))
        clearance = _preview_number(spatial_relation.get("clearance"))
        labels = {
            "above": ("上方", "above"),
            "below": ("下方", "below"),
            "left_of": ("左侧", "to the left of"),
            "right_of": ("右侧", "to the right of"),
        }
        label = labels.get(relation, (relation, relation))[0 if chinese else 1]
        lines.append(
            (
                f"- 空间约束：位于特征 {reference} {label}，"
                f"要求净间距 {clearance}（提交时由本地校验）"
                if chinese
                else (
                    f"- Spatial constraint: {label} feature {reference}, "
                    f"required clearance {clearance} "
                    "(validated locally on submission)"
                )
            )
        )
    return lines


def _geometry_edit_description(
    edit: Mapping[str, Any],
    chinese: bool,
) -> list[str]:
    operation = str(edit.get("operation", ""))
    if operation == "batch":
        edits = edit.get("edits")
        if not isinstance(edits, list):
            return []
        counts = Counter(
            str(item.get("operation")) for item in edits if isinstance(item, Mapping)
        )
        labels = []
        for item_operation, count in counts.items():
            label = _edit_operation_label(item_operation, chinese)
            labels.append(f"{label} {count} 项" if chinese else f"{count} {label}")
        lines = [
            f"- 修改内容：一次完成 {len(edits)} 项草图修改（{'、'.join(labels)}）"
            if chinese
            else f"- Changes: apply {len(edits)} sketch edits ({', '.join(labels)})"
        ]
        for index, item in enumerate(edits[:8], start=1):
            if isinstance(item, Mapping):
                detail = _single_edit_description(item, chinese)
                if detail:
                    lines.append(f"  {index}. {detail}")
        return lines
    detail = _single_edit_description(edit, chinese)
    return (
        [f"- 修改内容：{detail}" if chinese else f"- Change: {detail}"]
        if detail
        else []
    )


def _single_edit_description(
    edit: Mapping[str, Any],
    chinese: bool,
) -> str:
    operation = str(edit.get("operation", ""))
    if operation == "add_circle":
        center = (
            f"({_preview_number(edit.get('center_x'))}, "
            f"{_preview_number(edit.get('center_y'))})"
        )
        radius = _preview_number(edit.get("radius"))
        return (
            f"增加圆形轮廓，圆心 {center}，半径 {radius}"
            if chinese
            else f"Add a circular profile centered at {center} with radius {radius}"
        )
    if operation == "add_rectangle":
        origin = f"({_preview_number(edit.get('x'))}, {_preview_number(edit.get('y'))})"
        size = f"{_preview_number(edit.get('width'))} × {_preview_number(edit.get('height'))}"
        return (
            f"增加矩形轮廓，左下角 {origin}，尺寸 {size}"
            if chinese
            else f"Add a rectangle at lower-left {origin}, size {size}"
        )
    if operation == "add_polygon":
        vertices = edit.get("vertices")
        count = len(vertices) if isinstance(vertices, list) else "?"
        return (
            f"增加一个由 {count} 个顶点定义的闭合轮廓"
            if chinese
            else f"Add one closed profile defined by {count} vertices"
        )
    if operation == "add_path_slot":
        width = _preview_number(edit.get("width"))
        points = _point_chain(edit.get("points"))
        return (
            f"切除一条连续定宽槽，中心路径 {points}，槽宽 {width}"
            if chinese
            else f"Cut one continuous constant-width slot, centerline {points}, width {width}"
        )
    if operation == "planar_boolean":
        boolean_operation = str(edit.get("boolean_operation", ""))
        tool = edit.get("tool")
        tool_kind = (
            str(tool.get("kind", "轮廓")) if isinstance(tool, Mapping) else "轮廓"
        )
        if chinese:
            action = "切除" if boolean_operation == "cut" else "合并"
            return f"追加二维{action}特征，工具轮廓为 {tool_kind}"
        action = "cut" if boolean_operation == "cut" else "fuse"
        return f"Append one planar {action} feature using a {tool_kind} profile"
    if operation == "replace_planar_boolean_feature":
        feature_id = str(edit.get("feature_id", "?"))
        tool = edit.get("tool")
        tool_kind = (
            str(tool.get("kind", "轮廓")) if isinstance(tool, Mapping) else "轮廓"
        )
        return (
            f"替换二维布尔特征 {feature_id} 的工具轮廓为 {tool_kind}，并重放后续特征"
            if chinese
            else (
                f"Replace the tool profile of planar Boolean feature {feature_id} "
                f"with {tool_kind} and replay later features"
            )
        )
    if operation == "translate":
        dx = _preview_number(edit.get("dx"))
        dy = _preview_number(edit.get("dy"))
        dz = edit.get("dz")
        suffix = f"，Z 方向 {_preview_number(dz)}" if chinese and dz is not None else ""
        return (
            f"整体平移：X 方向 {dx}，Y 方向 {dy}{suffix}"
            if chinese
            else f"Translate by ({dx}, {dy}{', ' + _preview_number(dz) if dz is not None else ''})"
        )
    if operation == "rotate":
        axis = str(edit.get("axis", "?")).upper()
        angle = _preview_number(edit.get("angle_degrees"))
        return (
            f"绕 {axis} 轴旋转 {angle}°"
            if chinese
            else f"Rotate {angle}° about the {axis} axis"
        )
    if operation == "delete_circles":
        circle_ids = edit.get("circle_ids")
        count = len(circle_ids) if isinstance(circle_ids, list) else "?"
        return (
            f"删除 {count} 个圆形轮廓"
            if chinese
            else f"Delete {count} circular profiles"
        )
    if operation == "replace_circle_pattern":
        count = edit.get("count")
        start = (
            f"({_preview_number(edit.get('start_center_x'))}, "
            f"{_preview_number(edit.get('start_center_y'))})"
        )
        spacing = (
            f"({_preview_number(edit.get('spacing_x'))}, "
            f"{_preview_number(edit.get('spacing_y'))})"
        )
        radius = _preview_number(edit.get("radius"))
        return (
            f"重排为 {count} 个圆孔，从 {start} 开始，间距向量 {spacing}，半径 {radius}"
            if chinese
            else f"Replace with {count} circles from {start}, spacing {spacing}, radius {radius}"
        )
    label = _edit_operation_label(operation, chinese)
    return label


def _edit_operation_label(operation: str, chinese: bool) -> str:
    labels = {
        "add_circle": ("增加圆形轮廓", "circle addition"),
        "add_rectangle": ("增加矩形轮廓", "rectangle addition"),
        "add_polygon": ("增加闭合轮廓", "closed-profile addition"),
        "add_path_slot": ("增加连续定宽槽", "continuous-slot addition"),
        "planar_boolean": ("二维布尔特征", "planar Boolean feature"),
        "replace_planar_boolean_feature": (
            "替换二维布尔特征",
            "planar Boolean feature replacement",
        ),
        "update_point": ("调整草图点", "point update"),
        "update_circle": ("调整圆形轮廓", "circle update"),
        "delete_circles": ("删除圆形轮廓", "circle deletion"),
        "replace_circle_pattern": ("重排圆孔", "circle-pattern replacement"),
        "add_line": ("增加直线", "line addition"),
        "add_arc": ("增加圆弧", "arc addition"),
        "update_line": ("调整直线", "line update"),
        "update_arc": ("调整圆弧", "arc update"),
        "delete_curves": ("删除曲线", "curve deletion"),
        "add_constraint": ("增加约束", "constraint addition"),
        "replace_constraint": ("调整约束", "constraint update"),
        "delete_constraints": ("删除约束", "constraint deletion"),
        "extrude_profiles": ("拉伸草图", "profile extrusion"),
        "revolve_profile": ("旋转草图", "profile revolution"),
        "path_sweep_profile": ("扫掠草图", "profile path sweep"),
        "part_boolean": ("部件布尔运算", "part Boolean operation"),
        "body_boolean": ("实体布尔运算", "body Boolean operation"),
        "translate": ("整体平移", "translation"),
        "rotate": ("整体旋转", "rotation"),
    }
    pair = labels.get(operation, ("修改草图", "sketch edit"))
    return pair[0] if chinese else pair[1]


def _point_chain(value: object) -> str:
    if not isinstance(value, list):
        return "?"
    rendered: list[str] = []
    for point in value[:12]:
        if isinstance(point, Mapping):
            x = point.get("x")
            y = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
        else:
            continue
        rendered.append(f"({_preview_number(x)}, {_preview_number(y)})")
    suffix = " → …" if len(value) > 12 else ""
    return " → ".join(rendered) + suffix


def _uses_chinese(value: str | None) -> bool:
    return isinstance(value, str) and any(
        "\u4e00" <= character <= "\u9fff" for character in value
    )


def _numeric_difference(end: object, start: object) -> str | None:
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    return _preview_number(end - start)


def _preview_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def planar_retry_limit_message(user_request: str | None) -> str:
    return (
        "二维构造连续三次未通过本地验证，本轮已停止继续提交。"
        "请在下一条消息中调整参数或重新发起建模。"
        if _uses_chinese(user_request)
        else (
            "The planar construction failed local validation three "
            "times, so further submissions have stopped for this turn. "
            "Adjust the parameters or retry in a new message."
        )
    )
