"""Pure-Python contract for bounded planar construction graphs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import ClassVar, Literal, Mapping, Sequence, TypeAlias


SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 120
MAX_NODE_ID_LENGTH = 64
MAX_NODES = 64
MAX_BOOLEAN_OPERANDS = 16
MAX_POLYGON_VERTICES = 128
MAX_PATH_POINTS = 64
MAX_PATTERN_INSTANCES = 256
MAX_DAG_DEPTH = 16
MAX_CANONICAL_PAYLOAD_BYTES = 32_768
MAX_DIAGNOSTIC_MESSAGE_LENGTH = 240

_NODE_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
_PRIMITIVE_KINDS = frozenset({"rectangle", "circle", "polygon", "path_stroke"})
_BOOLEAN_KINDS = frozenset({"union", "difference", "intersection"})
_TRANSFORM_KINDS = frozenset({"translate", "rotate", "mirror"})
_PATTERN_KINDS = frozenset(
    {"linear_pattern", "rectangular_pattern", "circular_pattern"}
)

Point2D: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PlanarIRDiagnostic:
    """Stable, bounded validation failure returned to callers."""

    code: str
    message: str
    node_id: str | None
    retryable: bool
    allowed_fields: tuple[str, ...]
    model_unchanged: Literal[True] = True


class PlanarIRValidationError(ValueError):
    """Raised when a planar construction payload violates the v1 contract."""

    def __init__(self, diagnostic: PlanarIRDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass(frozen=True, slots=True)
class RectangleNode:
    id: str
    x: float
    y: float
    width: float
    height: float
    kind: ClassVar[Literal["rectangle"]] = "rectangle"


@dataclass(frozen=True, slots=True)
class CircleNode:
    id: str
    center_x: float
    center_y: float
    radius: float
    kind: ClassVar[Literal["circle"]] = "circle"


@dataclass(frozen=True, slots=True)
class PolygonNode:
    id: str
    vertices: tuple[Point2D, ...]
    kind: ClassVar[Literal["polygon"]] = "polygon"


@dataclass(frozen=True, slots=True)
class PathStrokeNode:
    id: str
    points: tuple[Point2D, ...]
    width: float
    cap: Literal["butt", "square", "round"]
    join: Literal["miter", "bevel", "round"]
    kind: ClassVar[Literal["path_stroke"]] = "path_stroke"


@dataclass(frozen=True, slots=True)
class UnionNode:
    id: str
    operands: tuple[str, ...]
    kind: ClassVar[Literal["union"]] = "union"


@dataclass(frozen=True, slots=True)
class DifferenceNode:
    id: str
    base: str
    subtract: tuple[str, ...]
    kind: ClassVar[Literal["difference"]] = "difference"


@dataclass(frozen=True, slots=True)
class IntersectionNode:
    id: str
    operands: tuple[str, ...]
    kind: ClassVar[Literal["intersection"]] = "intersection"


@dataclass(frozen=True, slots=True)
class TranslateNode:
    id: str
    source: str
    dx: float
    dy: float
    kind: ClassVar[Literal["translate"]] = "translate"


@dataclass(frozen=True, slots=True)
class RotateNode:
    id: str
    source: str
    center_x: float
    center_y: float
    angle_degrees: float
    kind: ClassVar[Literal["rotate"]] = "rotate"


@dataclass(frozen=True, slots=True)
class MirrorNode:
    id: str
    source: str
    line_point_x: float
    line_point_y: float
    line_direction_x: float
    line_direction_y: float
    kind: ClassVar[Literal["mirror"]] = "mirror"


@dataclass(frozen=True, slots=True)
class LinearPatternNode:
    id: str
    seed: str
    count: int
    step_x: float
    step_y: float
    kind: ClassVar[Literal["linear_pattern"]] = "linear_pattern"


@dataclass(frozen=True, slots=True)
class RectangularPatternNode:
    id: str
    seed: str
    count_x: int
    count_y: int
    spacing_x: float
    spacing_y: float
    kind: ClassVar[Literal["rectangular_pattern"]] = "rectangular_pattern"


@dataclass(frozen=True, slots=True)
class CircularPatternNode:
    id: str
    seed: str
    count: int
    center_x: float
    center_y: float
    total_angle_degrees: float
    kind: ClassVar[Literal["circular_pattern"]] = "circular_pattern"


PlanarConstructionNode: TypeAlias = (
    RectangleNode
    | CircleNode
    | PolygonNode
    | PathStrokeNode
    | UnionNode
    | DifferenceNode
    | IntersectionNode
    | TranslateNode
    | RotateNode
    | MirrorNode
    | LinearPatternNode
    | RectangularPatternNode
    | CircularPatternNode
)


@dataclass(frozen=True, slots=True)
class PlanarConstructionSummary:
    schema_version: int
    node_count: int
    primitive_count: int
    boolean_count: int
    transform_count: int
    pattern_count: int
    expanded_pattern_instances: int
    dag_depth: int
    canonical_payload_bytes: int
    canonical_digest_short: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "node_count": self.node_count,
            "primitive_count": self.primitive_count,
            "boolean_count": self.boolean_count,
            "transform_count": self.transform_count,
            "pattern_count": self.pattern_count,
            "expanded_pattern_instances": self.expanded_pattern_instances,
            "dag_depth": self.dag_depth,
            "canonical_payload_bytes": self.canonical_payload_bytes,
            "canonical_digest_short": self.canonical_digest_short,
        }


@dataclass(frozen=True, slots=True)
class PlanarConstructionIR:
    schema_version: Literal[1]
    name: str
    plane: Literal["XY"]
    nodes: tuple[PlanarConstructionNode, ...]
    result_node_id: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PlanarConstructionIR:
        return parse_planar_construction_ir(value)

    def canonical_json(self) -> str:
        return canonical_planar_construction_json(self)

    def digest(self) -> str:
        return planar_construction_digest(self)

    def provider_safe_summary(self) -> PlanarConstructionSummary:
        return summarize_planar_construction(self)


def _fail(
    code: str,
    message: str,
    *,
    node_id: str | None = None,
    allowed_fields: Sequence[str] = (),
) -> None:
    bounded_node_id = node_id[:MAX_NODE_ID_LENGTH] if node_id is not None else None
    raise PlanarIRValidationError(
        PlanarIRDiagnostic(
            code=code,
            message=message[:MAX_DIAGNOSTIC_MESSAGE_LENGTH],
            node_id=bounded_node_id,
            retryable=True,
            allowed_fields=tuple(allowed_fields),
        )
    )


def _fields(
    value: Mapping[str, object],
    required: set[str],
    *,
    node_id: str | None = None,
) -> None:
    unknown = sorted(set(value) - required)
    if unknown:
        _fail(
            "planar-ir.schema-invalid",
            f"Unknown field: {unknown[0]}.",
            node_id=node_id,
            allowed_fields=tuple(sorted(required)),
        )
    missing = sorted(required - set(value))
    if missing:
        _fail(
            "planar-ir.schema-invalid",
            f"Missing field: {missing[0]}.",
            node_id=node_id,
            allowed_fields=tuple(sorted(required)),
        )


def _string(value: object, field: str, *, node_id: str | None = None) -> str:
    if not isinstance(value, str):
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must be a string.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    return value


def _node_id(value: object, field: str = "id", *, node_id: str | None = None) -> str:
    result = _string(value, field, node_id=node_id)
    if len(result) > MAX_NODE_ID_LENGTH or _NODE_ID_PATTERN.fullmatch(result) is None:
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must be a valid node ID of at most {MAX_NODE_ID_LENGTH} characters.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    return result


def _number(
    value: object,
    field: str,
    *,
    node_id: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        code = "planar-ir.invalid-primitive" if positive else "planar-ir.schema-invalid"
        _fail(
            code,
            f"{field} must be a finite number.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must be finite.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    if positive and result <= 0.0:
        _fail(
            "planar-ir.invalid-primitive",
            f"{field} must be greater than zero.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    return 0.0 if result == 0.0 else result


def _count(value: object, field: str, *, node_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must be a positive integer.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    return value


def _id_list(
    value: object,
    field: str,
    *,
    node_id: str,
    minimum: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must be an array of node IDs.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    if len(value) < minimum:
        _fail(
            "planar-ir.schema-invalid",
            f"{field} must contain at least {minimum} node ID(s).",
            node_id=node_id,
            allowed_fields=(field,),
        )
    if len(value) > MAX_BOOLEAN_OPERANDS:
        _fail(
            "planar-ir.budget-exceeded",
            f"{field} exceeds the {MAX_BOOLEAN_OPERANDS} operand budget.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    result = tuple(_node_id(item, field, node_id=node_id) for item in value)
    if len(set(result)) != len(result):
        _fail(
            "planar-ir.schema-invalid",
            f"{field} contains a duplicate reference.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    return result


def _points(
    value: object,
    field: str,
    *,
    node_id: str,
    minimum: int,
    maximum: int,
    code: str,
) -> tuple[Point2D, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(
            code,
            f"{field} must be an array of points.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    if len(value) < minimum:
        _fail(
            code,
            f"{field} must contain at least {minimum} points.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    if len(value) > maximum:
        _fail(
            "planar-ir.budget-exceeded",
            f"{field} exceeds the {maximum} point budget.",
            node_id=node_id,
            allowed_fields=(field,),
        )
    result: list[Point2D] = []
    for index, point in enumerate(value):
        if (
            isinstance(point, (str, bytes))
            or not isinstance(point, Sequence)
            or len(point) != 2
        ):
            _fail(
                code,
                f"{field}[{index}] must contain two coordinates.",
                node_id=node_id,
                allowed_fields=(field,),
            )
        result.append(
            (
                _number(point[0], f"{field}[{index}][0]", node_id=node_id),
                _number(point[1], f"{field}[{index}][1]", node_id=node_id),
            )
        )
    return tuple(result)


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if ((ab_c > 0.0) != (ab_d > 0.0)) and ((cd_a > 0.0) != (cd_b > 0.0)):
        return True

    def between(start: Point2D, end: Point2D, point: Point2D) -> bool:
        return min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(
            start[1], end[1]
        ) <= point[1] <= max(start[1], end[1])

    return (
        (ab_c == 0.0 and between(a, b, c))
        or (ab_d == 0.0 and between(a, b, d))
        or (cd_a == 0.0 and between(c, d, a))
        or (cd_b == 0.0 and between(c, d, b))
    )


def _validate_polygon(
    vertices: tuple[Point2D, ...], node_id: str
) -> tuple[Point2D, ...]:
    if vertices[0] == vertices[-1]:
        _fail(
            "planar-ir.invalid-primitive",
            "Polygon must not repeat its first vertex at the end.",
            node_id=node_id,
            allowed_fields=("vertices",),
        )
    if len(set(vertices)) != len(vertices):
        _fail(
            "planar-ir.invalid-primitive",
            "Polygon vertices must be unique.",
            node_id=node_id,
            allowed_fields=("vertices",),
        )
    area2 = sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True)
    )
    if area2 == 0.0:
        _fail(
            "planar-ir.invalid-primitive",
            "Polygon must have non-zero area.",
            node_id=node_id,
            allowed_fields=("vertices",),
        )
    count = len(vertices)
    for first in range(count):
        a, b = vertices[first], vertices[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count} or first == (second + 1) % count:
                continue
            c, d = vertices[second], vertices[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                _fail(
                    "planar-ir.invalid-primitive",
                    f"Polygon edges {first} and {second} intersect.",
                    node_id=node_id,
                    allowed_fields=("vertices",),
                )
    rotations = []
    for ring in (vertices, tuple(reversed(vertices))):
        rotations.extend(ring[index:] + ring[:index] for index in range(count))
    return min(rotations)


def _validate_path(points: tuple[Point2D, ...], node_id: str) -> None:
    for index in range(len(points) - 1):
        if points[index] == points[index + 1]:
            _fail(
                "planar-ir.invalid-path-stroke",
                f"Path segment {index} has zero length.",
                node_id=node_id,
                allowed_fields=("points",),
            )
    for index in range(len(points) - 2):
        start, turn, end = points[index : index + 3]
        incoming = (turn[0] - start[0], turn[1] - start[1])
        outgoing = (end[0] - turn[0], end[1] - turn[1])
        if _orientation(start, turn, end) == 0.0 and (
            incoming[0] * outgoing[0] + incoming[1] * outgoing[1] < 0.0
        ):
            _fail(
                "planar-ir.invalid-path-stroke",
                f"Path segments {index} and {index + 1} overlap at a reversal.",
                node_id=node_id,
                allowed_fields=("points",),
            )
    if points[0] == points[-1]:
        _fail(
            "planar-ir.invalid-path-stroke",
            "Path stroke must be open.",
            node_id=node_id,
            allowed_fields=("points",),
        )
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if _segments_intersect(
                points[first], points[first + 1], points[second], points[second + 1]
            ):
                _fail(
                    "planar-ir.invalid-path-stroke",
                    f"Path segments {first} and {second} intersect.",
                    node_id=node_id,
                    allowed_fields=("points",),
                )


def _normalize_angle(value: float) -> float:
    result = (value + 180.0) % 360.0 - 180.0
    return 0.0 if result == 0.0 else result


def _parse_node(value: object) -> PlanarConstructionNode:
    if not isinstance(value, Mapping):
        _fail(
            "planar-ir.schema-invalid",
            "node must be an object.",
            allowed_fields=("nodes",),
        )
    node = value
    raw_id = node.get("id")
    node_id = _node_id(raw_id) if raw_id is not None else None
    if node_id is None:
        _fail(
            "planar-ir.schema-invalid",
            "Missing field: id.",
            allowed_fields=("id", "kind"),
        )
    kind = _string(node.get("kind"), "kind", node_id=node_id)

    if kind == "rectangle":
        _fields(node, {"id", "kind", "x", "y", "width", "height"}, node_id=node_id)
        return RectangleNode(
            node_id,
            _number(node["x"], "x", node_id=node_id),
            _number(node["y"], "y", node_id=node_id),
            _number(node["width"], "width", node_id=node_id, positive=True),
            _number(node["height"], "height", node_id=node_id, positive=True),
        )
    if kind == "circle":
        _fields(node, {"id", "kind", "center_x", "center_y", "radius"}, node_id=node_id)
        return CircleNode(
            node_id,
            _number(node["center_x"], "center_x", node_id=node_id),
            _number(node["center_y"], "center_y", node_id=node_id),
            _number(node["radius"], "radius", node_id=node_id, positive=True),
        )
    if kind == "polygon":
        _fields(node, {"id", "kind", "vertices"}, node_id=node_id)
        vertices = _points(
            node["vertices"],
            "vertices",
            node_id=node_id,
            minimum=3,
            maximum=MAX_POLYGON_VERTICES,
            code="planar-ir.invalid-primitive",
        )
        return PolygonNode(node_id, _validate_polygon(vertices, node_id))
    if kind == "path_stroke":
        _fields(node, {"id", "kind", "points", "width", "cap", "join"}, node_id=node_id)
        points = _points(
            node["points"],
            "points",
            node_id=node_id,
            minimum=2,
            maximum=MAX_PATH_POINTS,
            code="planar-ir.invalid-path-stroke",
        )
        _validate_path(points, node_id)
        cap = _string(node["cap"], "cap", node_id=node_id)
        join = _string(node["join"], "join", node_id=node_id)
        if cap not in {"butt", "square", "round"}:
            _fail(
                "planar-ir.invalid-path-stroke",
                "cap must be butt, square, or round.",
                node_id=node_id,
                allowed_fields=("cap",),
            )
        if join not in {"miter", "bevel", "round"}:
            _fail(
                "planar-ir.invalid-path-stroke",
                "join must be miter, bevel, or round.",
                node_id=node_id,
                allowed_fields=("join",),
            )
        return PathStrokeNode(
            node_id,
            points,
            _number(node["width"], "width", node_id=node_id, positive=True),
            cap,
            join,
        )
    if kind in {"union", "intersection"}:
        _fields(node, {"id", "kind", "operands"}, node_id=node_id)
        operands = _id_list(
            node["operands"],
            "operands",
            node_id=node_id,
            minimum=1 if kind == "union" else 2,
        )
        return (
            UnionNode(node_id, operands)
            if kind == "union"
            else IntersectionNode(node_id, operands)
        )
    if kind == "difference":
        _fields(node, {"id", "kind", "base", "subtract"}, node_id=node_id)
        return DifferenceNode(
            node_id,
            _node_id(node["base"], "base", node_id=node_id),
            _id_list(node["subtract"], "subtract", node_id=node_id, minimum=1),
        )
    if kind == "translate":
        _fields(node, {"id", "kind", "source", "dx", "dy"}, node_id=node_id)
        return TranslateNode(
            node_id,
            _node_id(node["source"], "source", node_id=node_id),
            _number(node["dx"], "dx", node_id=node_id),
            _number(node["dy"], "dy", node_id=node_id),
        )
    if kind == "rotate":
        _fields(
            node,
            {"id", "kind", "source", "center_x", "center_y", "angle_degrees"},
            node_id=node_id,
        )
        return RotateNode(
            node_id,
            _node_id(node["source"], "source", node_id=node_id),
            _number(node["center_x"], "center_x", node_id=node_id),
            _number(node["center_y"], "center_y", node_id=node_id),
            _normalize_angle(
                _number(node["angle_degrees"], "angle_degrees", node_id=node_id)
            ),
        )
    if kind == "mirror":
        _fields(
            node,
            {
                "id",
                "kind",
                "source",
                "line_point_x",
                "line_point_y",
                "line_direction_x",
                "line_direction_y",
            },
            node_id=node_id,
        )
        direction_x = _number(
            node["line_direction_x"], "line_direction_x", node_id=node_id
        )
        direction_y = _number(
            node["line_direction_y"], "line_direction_y", node_id=node_id
        )
        if direction_x == 0.0 and direction_y == 0.0:
            _fail(
                "planar-ir.schema-invalid",
                "Mirror line direction must be non-zero.",
                node_id=node_id,
                allowed_fields=("line_direction_x", "line_direction_y"),
            )
        return MirrorNode(
            node_id,
            _node_id(node["source"], "source", node_id=node_id),
            _number(node["line_point_x"], "line_point_x", node_id=node_id),
            _number(node["line_point_y"], "line_point_y", node_id=node_id),
            direction_x,
            direction_y,
        )
    if kind == "linear_pattern":
        _fields(
            node, {"id", "kind", "seed", "count", "step_x", "step_y"}, node_id=node_id
        )
        count = _count(node["count"], "count", node_id=node_id)
        step_x = _number(node["step_x"], "step_x", node_id=node_id)
        step_y = _number(node["step_y"], "step_y", node_id=node_id)
        if count > 1 and step_x == 0.0 and step_y == 0.0:
            _fail(
                "planar-ir.schema-invalid",
                "Linear pattern step must be non-zero when count is greater than one.",
                node_id=node_id,
                allowed_fields=("step_x", "step_y"),
            )
        return LinearPatternNode(
            node_id,
            _node_id(node["seed"], "seed", node_id=node_id),
            count,
            step_x,
            step_y,
        )
    if kind == "rectangular_pattern":
        _fields(
            node,
            {"id", "kind", "seed", "count_x", "count_y", "spacing_x", "spacing_y"},
            node_id=node_id,
        )
        return RectangularPatternNode(
            node_id,
            _node_id(node["seed"], "seed", node_id=node_id),
            _count(node["count_x"], "count_x", node_id=node_id),
            _count(node["count_y"], "count_y", node_id=node_id),
            _number(node["spacing_x"], "spacing_x", node_id=node_id, positive=True),
            _number(node["spacing_y"], "spacing_y", node_id=node_id, positive=True),
        )
    if kind == "circular_pattern":
        _fields(
            node,
            {
                "id",
                "kind",
                "seed",
                "count",
                "center_x",
                "center_y",
                "total_angle_degrees",
            },
            node_id=node_id,
        )
        count = _count(node["count"], "count", node_id=node_id)
        total_angle = _number(
            node["total_angle_degrees"], "total_angle_degrees", node_id=node_id
        )
        if count > 1 and total_angle == 0.0:
            _fail(
                "planar-ir.schema-invalid",
                "Circular pattern angle must be non-zero when count is greater than one.",
                node_id=node_id,
                allowed_fields=("total_angle_degrees",),
            )
        return CircularPatternNode(
            node_id,
            _node_id(node["seed"], "seed", node_id=node_id),
            count,
            _number(node["center_x"], "center_x", node_id=node_id),
            _number(node["center_y"], "center_y", node_id=node_id),
            total_angle,
        )
    _fail(
        "planar-ir.schema-invalid",
        f"Unsupported node kind: {kind}.",
        node_id=node_id,
        allowed_fields=("kind",),
    )


def _dependencies(node: PlanarConstructionNode) -> tuple[str, ...]:
    if isinstance(node, (UnionNode, IntersectionNode)):
        return node.operands
    if isinstance(node, DifferenceNode):
        return (node.base, *node.subtract)
    if isinstance(node, (TranslateNode, RotateNode, MirrorNode)):
        return (node.source,)
    if isinstance(
        node, (LinearPatternNode, RectangularPatternNode, CircularPatternNode)
    ):
        return (node.seed,)
    return ()


def _shortest_cycle(
    table: Mapping[str, PlanarConstructionNode],
) -> tuple[str, ...] | None:
    candidates: list[tuple[str, ...]] = []
    for start in sorted(table):
        queue: deque[tuple[str, ...]] = deque([(start,)])
        best_length: dict[str, int] = {start: 0}
        while queue:
            path = queue.popleft()
            for dependency in sorted(_dependencies(table[path[-1]])):
                if dependency == start:
                    candidates.append((*path, start))
                    queue.clear()
                    break
                next_length = len(path)
                if next_length < best_length.get(dependency, len(table) + 1):
                    best_length[dependency] = next_length
                    queue.append((*path, dependency))
    return (
        min(candidates, key=lambda cycle: (len(cycle), cycle)) if candidates else None
    )


def _graph_metrics(
    ir: PlanarConstructionIR,
) -> tuple[dict[str, PlanarConstructionNode], dict[str, int], int]:
    table = {node.id: node for node in ir.nodes}
    for node in sorted(ir.nodes, key=lambda item: item.id):
        for dependency in _dependencies(node):
            if dependency not in table:
                _fail(
                    "planar-ir.reference-missing",
                    f"Node {node.id} references missing node {dependency}.",
                    node_id=node.id,
                    allowed_fields=("base", "operands", "seed", "source", "subtract"),
                )
    if ir.result_node_id not in table:
        _fail(
            "planar-ir.reference-missing",
            f"Result references missing node {ir.result_node_id}.",
            allowed_fields=("result_node_id",),
        )
    cycle = _shortest_cycle(table)
    if cycle is not None:
        _fail(
            "planar-ir.cycle-detected",
            f"Cycle detected: {' -> '.join(cycle)}.",
            node_id=cycle[0],
            allowed_fields=("base", "operands", "seed", "source", "subtract"),
        )
    reachable: set[str] = set()
    stack = [ir.result_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(_dependencies(table[node_id]))
    unreachable = sorted(set(table) - reachable)
    if unreachable:
        _fail(
            "planar-ir.unreachable-node",
            f"Node {unreachable[0]} is not reachable from the result.",
            node_id=unreachable[0],
            allowed_fields=("nodes", "result_node_id"),
        )
    depths: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id not in depths:
            dependencies = _dependencies(table[node_id])
            depths[node_id] = 1 + max((depth(item) for item in dependencies), default=0)
        return depths[node_id]

    dag_depth = depth(ir.result_node_id)
    if dag_depth > MAX_DAG_DEPTH:
        _fail(
            "planar-ir.budget-exceeded",
            f"Graph depth {dag_depth} exceeds the {MAX_DAG_DEPTH} level budget.",
            node_id=ir.result_node_id,
            allowed_fields=("nodes", "result_node_id"),
        )
    multiplicity: dict[str, int] = {}
    expanded_total = 0
    for node_id in sorted(table, key=lambda item: depths[item]):
        node = table[node_id]
        if isinstance(node, LinearPatternNode):
            multiplicity[node_id] = multiplicity[node.seed] * node.count
        elif isinstance(node, RectangularPatternNode):
            multiplicity[node_id] = (
                multiplicity[node.seed] * node.count_x * node.count_y
            )
        elif isinstance(node, CircularPatternNode):
            multiplicity[node_id] = multiplicity[node.seed] * node.count
        elif isinstance(node, (TranslateNode, RotateNode, MirrorNode)):
            multiplicity[node_id] = multiplicity[node.source]
        elif isinstance(node, (UnionNode, DifferenceNode, IntersectionNode)):
            multiplicity[node_id] = sum(
                multiplicity[dependency] for dependency in _dependencies(node)
            )
        else:
            multiplicity[node_id] = 1
        if node.kind in _PATTERN_KINDS:
            expanded_total += multiplicity[node_id]
            if expanded_total > MAX_PATTERN_INSTANCES:
                _fail(
                    "planar-ir.budget-exceeded",
                    f"Pattern expansion exceeds the {MAX_PATTERN_INSTANCES} instance budget.",
                    node_id=node_id,
                    allowed_fields=("count", "count_x", "count_y"),
                )
    return table, depths, expanded_total


def parse_planar_construction_ir(value: Mapping[str, object]) -> PlanarConstructionIR:
    if not isinstance(value, Mapping):
        _fail(
            "planar-ir.schema-invalid",
            "construction must be an object.",
            allowed_fields=("construction",),
        )
    top = value
    _fields(top, {"schema_version", "name", "plane", "nodes", "result_node_id"})
    version = top["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        _fail(
            "planar-ir.schema-invalid",
            f"schema_version must be {SCHEMA_VERSION}.",
            allowed_fields=("schema_version",),
        )
    name = _string(top["name"], "name")
    if not name.strip() or len(name) > MAX_NAME_LENGTH:
        _fail(
            "planar-ir.schema-invalid",
            f"name must be non-empty and at most {MAX_NAME_LENGTH} characters.",
            allowed_fields=("name",),
        )
    if top["plane"] != "XY":
        _fail(
            "planar-ir.schema-invalid", "plane must be XY.", allowed_fields=("plane",)
        )
    raw_nodes = top["nodes"]
    if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
        _fail(
            "planar-ir.schema-invalid",
            "nodes must be an array.",
            allowed_fields=("nodes",),
        )
    if not raw_nodes:
        _fail(
            "planar-ir.schema-invalid",
            "nodes must not be empty.",
            allowed_fields=("nodes",),
        )
    if len(raw_nodes) > MAX_NODES:
        _fail(
            "planar-ir.budget-exceeded",
            f"Node count exceeds the {MAX_NODES} node budget.",
            allowed_fields=("nodes",),
        )
    nodes = tuple(_parse_node(item) for item in raw_nodes)
    counts = Counter(node.id for node in nodes)
    duplicate = min(
        (node_id for node_id, count in counts.items() if count > 1), default=None
    )
    if duplicate is not None:
        _fail(
            "planar-ir.duplicate-node-id",
            f"Node ID {duplicate} is defined more than once.",
            node_id=duplicate,
            allowed_fields=("id",),
        )
    ir = PlanarConstructionIR(
        schema_version=SCHEMA_VERSION,
        name=name,
        plane="XY",
        nodes=nodes,
        result_node_id=_node_id(top["result_node_id"], "result_node_id"),
    )
    _graph_metrics(ir)
    encoded = canonical_planar_construction_json(ir).encode("utf-8")
    if len(encoded) > MAX_CANONICAL_PAYLOAD_BYTES:
        _fail(
            "planar-ir.budget-exceeded",
            f"Canonical payload exceeds the {MAX_CANONICAL_PAYLOAD_BYTES} byte budget.",
            allowed_fields=("name", "nodes"),
        )
    return ir


def _node_payload(
    node: PlanarConstructionNode,
    dependency_digests: Mapping[str, str],
    *,
    digest_references: bool,
) -> dict[str, object]:
    result: dict[str, object] = {"id": node.id, "kind": node.kind}

    def reference(node_id: str) -> str:
        return dependency_digests[node_id] if digest_references else node_id

    def commutative(node_ids: Sequence[str]) -> list[str]:
        return [
            reference(node_id)
            for node_id in sorted(
                node_ids, key=lambda item: (dependency_digests[item], item)
            )
        ]

    if isinstance(node, RectangleNode):
        result.update(x=node.x, y=node.y, width=node.width, height=node.height)
    elif isinstance(node, CircleNode):
        result.update(
            center_x=node.center_x, center_y=node.center_y, radius=node.radius
        )
    elif isinstance(node, PolygonNode):
        result["vertices"] = [list(point) for point in node.vertices]
    elif isinstance(node, PathStrokeNode):
        result.update(
            points=[list(point) for point in node.points],
            width=node.width,
            cap=node.cap,
            join=node.join,
        )
    elif isinstance(node, (UnionNode, IntersectionNode)):
        result["operands"] = commutative(node.operands)
    elif isinstance(node, DifferenceNode):
        result.update(base=reference(node.base), subtract=commutative(node.subtract))
    elif isinstance(node, TranslateNode):
        result.update(source=reference(node.source), dx=node.dx, dy=node.dy)
    elif isinstance(node, RotateNode):
        result.update(
            source=reference(node.source),
            center_x=node.center_x,
            center_y=node.center_y,
            angle_degrees=node.angle_degrees,
        )
    elif isinstance(node, MirrorNode):
        result.update(
            source=reference(node.source),
            line_point_x=node.line_point_x,
            line_point_y=node.line_point_y,
            line_direction_x=node.line_direction_x,
            line_direction_y=node.line_direction_y,
        )
    elif isinstance(node, LinearPatternNode):
        result.update(
            seed=reference(node.seed),
            count=node.count,
            step_x=node.step_x,
            step_y=node.step_y,
        )
    elif isinstance(node, RectangularPatternNode):
        result.update(
            seed=reference(node.seed),
            count_x=node.count_x,
            count_y=node.count_y,
            spacing_x=node.spacing_x,
            spacing_y=node.spacing_y,
        )
    elif isinstance(node, CircularPatternNode):
        result.update(
            seed=reference(node.seed),
            count=node.count,
            center_x=node.center_x,
            center_y=node.center_y,
            total_angle_degrees=node.total_angle_degrees,
        )
    return result


def _canonical_graph(
    ir: PlanarConstructionIR,
) -> tuple[list[PlanarConstructionNode], dict[str, str]]:
    table, _, _ = _graph_metrics(ir)
    digests: dict[str, str] = {}

    def node_digest(node_id: str) -> str:
        if node_id not in digests:
            node = table[node_id]
            for dependency in _dependencies(node):
                node_digest(dependency)
            payload = _node_payload(node, digests, digest_references=True)
            payload.pop("id")
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digests[node_id] = hashlib.sha256(encoded).hexdigest()
        return digests[node_id]

    node_digest(ir.result_node_id)
    ordered: list[PlanarConstructionNode] = []
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        node = table[node_id]
        for dependency in sorted(
            _dependencies(node), key=lambda item: (digests[item], item)
        ):
            visit(dependency)
        visited.add(node_id)
        ordered.append(node)

    visit(ir.result_node_id)
    return ordered, digests


def canonical_planar_construction_json(ir: PlanarConstructionIR) -> str:
    ordered, digests = _canonical_graph(ir)
    payload = {
        "schema_version": ir.schema_version,
        "name": ir.name,
        "plane": ir.plane,
        "nodes": [
            _node_payload(node, digests, digest_references=False) for node in ordered
        ],
        "result_node_id": ir.result_node_id,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def planar_construction_digest(ir: PlanarConstructionIR) -> str:
    return hashlib.sha256(
        canonical_planar_construction_json(ir).encode("utf-8")
    ).hexdigest()


def summarize_planar_construction(
    ir: PlanarConstructionIR,
) -> PlanarConstructionSummary:
    _, depths, expanded_total = _graph_metrics(ir)
    counts = Counter(node.kind for node in ir.nodes)
    canonical = canonical_planar_construction_json(ir)
    return PlanarConstructionSummary(
        schema_version=ir.schema_version,
        node_count=len(ir.nodes),
        primitive_count=sum(counts[kind] for kind in _PRIMITIVE_KINDS),
        boolean_count=sum(counts[kind] for kind in _BOOLEAN_KINDS),
        transform_count=sum(counts[kind] for kind in _TRANSFORM_KINDS),
        pattern_count=sum(counts[kind] for kind in _PATTERN_KINDS),
        expanded_pattern_instances=expanded_total,
        dag_depth=depths[ir.result_node_id],
        canonical_payload_bytes=len(canonical.encode("utf-8")),
        canonical_digest_short=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :12
        ],
    )


__all__ = [
    "MAX_BOOLEAN_OPERANDS",
    "MAX_CANONICAL_PAYLOAD_BYTES",
    "MAX_DAG_DEPTH",
    "MAX_NODES",
    "MAX_PATH_POINTS",
    "MAX_PATTERN_INSTANCES",
    "MAX_POLYGON_VERTICES",
    "SCHEMA_VERSION",
    "CircleNode",
    "CircularPatternNode",
    "DifferenceNode",
    "IntersectionNode",
    "LinearPatternNode",
    "MirrorNode",
    "PathStrokeNode",
    "PlanarConstructionIR",
    "PlanarConstructionNode",
    "PlanarConstructionSummary",
    "PlanarIRDiagnostic",
    "PlanarIRValidationError",
    "PolygonNode",
    "RectangleNode",
    "RectangularPatternNode",
    "RotateNode",
    "TranslateNode",
    "UnionNode",
    "canonical_planar_construction_json",
    "parse_planar_construction_ir",
    "planar_construction_digest",
    "summarize_planar_construction",
]
