"""Detached exact compilation of Planar Construction IR v1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Callable, Literal

from fem.geometry import model as geometry_model
from fem.geometry.construction_ir import (
    CircleNode,
    CircularPatternNode,
    DifferenceNode,
    IntersectionNode,
    LinearPatternNode,
    MirrorNode,
    PathStrokeNode,
    PlanarConstructionIR,
    PolygonNode,
    RectangleNode,
    RectangularPatternNode,
    RotateNode,
    TranslateNode,
    UnionNode,
)
from fem.geometry.recipe_analysis import SketchProfileAnalysis, analyze_sketch_profiles
from fem.geometry.extrusion_selection import resolve_extrusion_source_faces
from fem.geometry.recipes import (
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)

from .recipe_compiler import compile_recipe


MODELING_TOLERANCE = 1.0e-8
TOLERANCE_VERSION = "planar-construction-v1/1e-8"
PATH_STROKE_MITER_LIMIT = 4.0


@dataclass(frozen=True, slots=True)
class PlanarConstructionDiagnostic:
    code: str
    message: str
    node_id: str | None
    retryable: bool
    allowed_fields: tuple[str, ...]
    model_unchanged: Literal[True] = True
    evidence: tuple[tuple[str, object], ...] = ()


class PlanarConstructionCompileError(ValueError):
    def __init__(self, diagnostic: PlanarConstructionDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class PlanarConstructionCancelled(RuntimeError):
    """Raised at safe checkpoints when one planar compile is cancelled.

    The broad ``except Exception`` safety nets below re-raise this unchanged
    so cancellation is never masked by a compile diagnostic.
    """


@dataclass(frozen=True, slots=True)
class PlanarCurveLineage:
    curve_id: str
    source_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanarConstructionPreview:
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    face_logical_ids: tuple[str, ...]
    edge_logical_ids: tuple[str, ...]
    point_logical_ids: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class PlanarConstructionProof:
    tolerance_version: str
    area: float
    bounding_box: tuple[float, float, float, float]
    component_count: int
    profile_count: int
    material_profile_count: int
    hole_count: int
    curve_type_counts: tuple[tuple[str, int], ...]
    recipe_digest: str
    equivalent: Literal[True] = True


@dataclass(frozen=True, slots=True)
class CompiledPlanarConstruction:
    construction_digest: str
    recipe: SketchGeometry
    profile_analysis: SketchProfileAnalysis
    curve_lineage: tuple[PlanarCurveLineage, ...]
    proof: PlanarConstructionProof
    preview: PlanarConstructionPreview


@dataclass(frozen=True, slots=True)
class _NativeFacts:
    area: float
    bounding_box: tuple[float, float, float, float]
    component_count: int
    hole_count: int
    curve_types: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _FeatureTargetAmbiguity:
    source_node_id: str
    material_profile_count: int
    boundary_contact: str
    tool_bounding_box: tuple[float, float, float, float]

    def diagnostic_evidence(self) -> tuple[tuple[str, object], ...]:
        return (
            ("material_profile_count_before", 1),
            ("material_profile_count_after", self.material_profile_count),
            ("boundary_contact", self.boundary_contact),
            ("positive_clearance", False),
            ("tool_bounding_box", self.tool_bounding_box),
        )


def _fail(
    code: str,
    message: str,
    *,
    node_id: str | None = None,
    allowed_fields: tuple[str, ...] = (),
    evidence: tuple[tuple[str, object], ...] = (),
) -> None:
    raise PlanarConstructionCompileError(
        PlanarConstructionDiagnostic(
            code,
            message[:240],
            node_id,
            True,
            allowed_fields,
            evidence=evidence,
        )
    )


def compile_planar_construction(
    construction: PlanarConstructionIR,
    *,
    model_factory: Callable[..., Any] = geometry_model,
) -> CompiledPlanarConstruction:
    """Compile one validated IR with two fully owned detached CAD runtimes."""

    if type(construction) is not PlanarConstructionIR:
        raise TypeError("construction must be a PlanarConstructionIR")
    digest = construction.digest()
    try:
        with model_factory(f"planar-construction-{digest[:12]}", dimension=2) as cad:
            surfaces, source_nodes = _evaluate(cad, construction)
            loops = cad.planar_boundary_loops(surfaces)
            recipe, lineage = _materialize(construction, loops, source_nodes)
            analysis = analyze_sketch_profiles(recipe, tolerance=MODELING_TOLERANCE)
            if not analysis.valid:
                diagnostic = analysis.blocking_diagnostics[0]
                _fail(
                    "planar-ir.profile-invalid",
                    f"{diagnostic.code}: {diagnostic.message}",
                    node_id=construction.result_node_id,
                    allowed_fields=("nodes", "result_node_id"),
                )
            source_facts = _native_facts(cad, surfaces, loops)
    except PlanarConstructionCompileError:
        raise
    except PlanarConstructionCancelled:
        raise
    except Exception as error:
        code = (
            "planar-ir.unsupported-boundary"
            if "unsupported boundary" in str(error).casefold()
            else "planar-ir.materialization-failed"
        )
        _fail(
            code,
            f"Planar construction materialization failed: {error}",
            node_id=construction.result_node_id,
            allowed_fields=("nodes", "result_node_id"),
        )

    try:
        with model_factory(f"planar-recipe-proof-{digest[:12]}", dimension=2) as cad:
            compiled = compile_recipe(cad, recipe)
            recipe_loops = cad.planar_boundary_loops(compiled.domain)
            recipe_facts = _native_facts(cad, compiled.domain, recipe_loops)
            recipe_facts = replace(
                recipe_facts,
                curve_types=_prove_compiled_curve_types(
                    cad, recipe, compiled.logical_entities
                ),
            )
            _require_equivalent(source_facts, recipe_facts, construction.result_node_id)
            preview = _preview(cad, recipe, compiled)
    except PlanarConstructionCompileError:
        raise
    except PlanarConstructionCancelled:
        raise
    except Exception as error:
        _fail(
            "planar-ir.equivalence-failed",
            f"Materialized recipe proof failed: {error}",
            node_id=construction.result_node_id,
            allowed_fields=("nodes", "result_node_id"),
        )

    material_count = sum(profile.is_material for profile in analysis.profiles)
    hole_count = sum(profile.is_hole for profile in analysis.profiles)
    proof = PlanarConstructionProof(
        TOLERANCE_VERSION,
        recipe_facts.area,
        recipe_facts.bounding_box,
        recipe_facts.component_count,
        len(analysis.profiles),
        material_count,
        hole_count,
        source_facts.curve_types,
        _recipe_digest(recipe),
    )
    if material_count != proof.component_count or hole_count != proof.hole_count:
        _fail(
            "planar-ir.equivalence-failed",
            "Profile topology does not match the recompiled OCC domain.",
            node_id=construction.result_node_id,
            allowed_fields=("nodes", "result_node_id"),
        )
    return CompiledPlanarConstruction(
        digest,
        recipe,
        analysis,
        lineage,
        proof,
        preview,
    )


def compile_planar_feature_recipe(
    construction: PlanarConstructionIR,
    *,
    compiled: CompiledPlanarConstruction | None = None,
    model_factory: Callable[..., Any] = geometry_model,
) -> object:
    """Compile top-level planar Booleans into the native feature recipe chain.

    ``compile_planar_construction`` deliberately materializes the final exact
    boundary as one strict sketch.  That representation is useful for profile
    proof, but it erases the authoring operations that the GUI model tree is
    derived from.  This companion compiler keeps operand sketches detached and
    commits each object-side union/difference as the same proven planar Boolean
    feature used by the interactive GUI workflow.

    The Boolean chain is built incrementally in one CAD model whose lifetime
    is this single call: the base sketch is compiled once and every step
    performs exactly one cut plus its lineage proof.  The final equivalence
    proof reuses the last-step live compiled carrier on that same model
    instead of replaying the whole recipe in a fresh model.
    """

    if type(construction) is not PlanarConstructionIR:
        raise TypeError("construction must be a PlanarConstructionIR")
    if compiled is None:
        compiled = compile_planar_construction(
            construction,
            model_factory=model_factory,
        )
    elif type(compiled) is not CompiledPlanarConstruction:
        raise TypeError("compiled must be CompiledPlanarConstruction or None")
    if compiled.construction_digest != construction.digest():
        raise ValueError("compiled construction digest does not match the IR")

    nodes = {node.id: node for node in construction.nodes}
    flattened: dict[str, SketchGeometry] = {
        construction.result_node_id: compiled.recipe,
    }
    flattened_compilations: dict[str, CompiledPlanarConstruction] = {
        construction.result_node_id: compiled,
    }

    def dependencies(node: object) -> tuple[str, ...]:
        if isinstance(node, (UnionNode, IntersectionNode)):
            return node.operands
        if isinstance(node, DifferenceNode):
            return (node.base, *node.subtract)
        if isinstance(node, (TranslateNode, RotateNode, MirrorNode)):
            return (node.source,)
        if isinstance(
            node,
            (LinearPatternNode, RectangularPatternNode, CircularPatternNode),
        ):
            return (node.seed,)
        return ()

    def subconstruction(node_id: str) -> PlanarConstructionIR:
        required: set[str] = set()

        def collect(current_id: str) -> None:
            if current_id in required:
                return
            required.add(current_id)
            for dependency in dependencies(nodes[current_id]):
                collect(dependency)

        collect(node_id)
        return PlanarConstructionIR(
            construction.schema_version,
            f"{construction.name}-{node_id}",
            construction.plane,
            tuple(node for node in construction.nodes if node.id in required),
            node_id,
        )

    def flatten(node_id: str) -> SketchGeometry:
        recipe = flattened.get(node_id)
        if recipe is None:
            node_compilation = compile_planar_construction(
                subconstruction(node_id),
                model_factory=model_factory,
            )
            recipe = node_compilation.recipe
            flattened[node_id] = recipe
            flattened_compilations[node_id] = node_compilation
        return recipe

    def feature_target_ambiguity(
        result: object,
        source_node_id: str,
        tool_recipe: SketchGeometry,
    ) -> _FeatureTargetAmbiguity | None:
        material_profile_count = len(
            resolve_extrusion_source_faces(result).face_ids
        )
        if material_profile_count <= 1:
            return None
        tool_analysis = analyze_sketch_profiles(
            tool_recipe,
            tolerance=MODELING_TOLERANCE,
        )
        simply_connected_tool = (
            tool_analysis.valid
            and sum(profile.is_material for profile in tool_analysis.profiles) == 1
            and sum(profile.is_hole for profile in tool_analysis.profiles) == 0
        )
        tool_bounds = flattened_compilations[source_node_id].proof.bounding_box
        return _FeatureTargetAmbiguity(
            source_node_id,
            material_profile_count,
            (
                "detected_or_crossed"
                if simply_connected_tool
                else "indeterminate"
            ),
            tool_bounds,
        )

    def boolean_feature(
        cad: Any,
        object_recipe: object,
        object_compiled: object,
        tool_recipe: SketchGeometry,
        operation: Literal["fuse", "cut"],
        node_id: str,
        prior_ambiguity: _FeatureTargetAmbiguity | None,
    ) -> tuple[object, object]:
        from .planar_boolean import prepare_planar_boolean_incremental

        target_faces = resolve_extrusion_source_faces(object_recipe).face_ids
        tool_faces = resolve_extrusion_source_faces(tool_recipe).face_ids
        if len(target_faces) != 1:
            if prior_ambiguity is not None:
                contact = (
                    " Exterior-boundary contact or crossing was detected."
                    if prior_ambiguity.boundary_contact == "detected_or_crossed"
                    else " Boundary contact could not be ruled out."
                )
                _fail(
                    "planar-ir.feature-splits-material",
                    (
                        f"Subtraction node {prior_ambiguity.source_node_id!r} "
                        f"split one material Profile into "
                        f"{prior_ambiguity.material_profile_count}."
                        f"{contact} Keep an internal cutout strictly inside "
                        "its target with positive clearance."
                    ),
                    node_id=prior_ambiguity.source_node_id,
                    allowed_fields=("nodes", "result_node_id"),
                    evidence=prior_ambiguity.diagnostic_evidence(),
                )
            _fail(
                "planar-ir.feature-target-ambiguous",
                (
                    "A planar Boolean feature requires one material target "
                    f"Profile; {len(target_faces)} are currently selectable."
                ),
                node_id=node_id,
                allowed_fields=("nodes", "result_node_id"),
                evidence=(
                    ("material_profile_count", len(target_faces)),
                    ("boundary_contact", "indeterminate"),
                ),
            )
        try:
            if object_compiled is None:
                # The base sketch (flatten result) is compiled exactly once,
                # lazily, on the shared chain model.
                object_compiled = compile_recipe(cad, object_recipe)
            feature, compiled = prepare_planar_boolean_incremental(
                cad,
                object_recipe,
                object_compiled,
                target_faces[0],
                tool_recipe,
                tool_faces,
                operation,
            )
            return feature.geometry, compiled
        except PlanarConstructionCompileError:
            raise
        except PlanarConstructionCancelled:
            raise
        except Exception as error:
            _fail(
                "planar-ir.feature-materialization-failed",
                f"Planar Boolean feature {node_id} could not be proven: {error}",
                node_id=node_id,
                allowed_fields=("nodes", "result_node_id"),
            )

    def build(
        node_id: str,
        cad: Any,
    ) -> tuple[object, object, _FeatureTargetAmbiguity | None]:
        node = nodes[node_id]
        if isinstance(node, DifferenceNode):
            result, result_compiled, ambiguity = build(node.base, cad)
            for tool_id in node.subtract:
                tool_node = nodes[tool_id]
                tool_ids = (tool_id,)
                if (
                    isinstance(tool_node, UnionNode)
                    and len(resolve_extrusion_source_faces(flatten(tool_id)).face_ids)
                    > 1
                ):
                    tool_ids = tool_node.operands
                for separated_tool_id in tool_ids:
                    tool_recipe = flatten(separated_tool_id)
                    result, result_compiled = boolean_feature(
                        cad,
                        result,
                        result_compiled,
                        tool_recipe,
                        "cut",
                        node.id,
                        ambiguity,
                    )
                    ambiguity = feature_target_ambiguity(
                        result,
                        separated_tool_id,
                        tool_recipe,
                    )
            return result, result_compiled, ambiguity
        if isinstance(node, UnionNode):
            if len(resolve_extrusion_source_faces(flatten(node_id)).face_ids) > 1:
                return flatten(node_id), None, None
            result, result_compiled, ambiguity = build(node.operands[0], cad)
            for tool_id in node.operands[1:]:
                result, result_compiled = boolean_feature(
                    cad,
                    result,
                    result_compiled,
                    flatten(tool_id),
                    "fuse",
                    node.id,
                    ambiguity,
                )
                ambiguity = None
            return result, result_compiled, ambiguity
        return flatten(node_id), None, None

    # The whole Boolean chain shares one CAD model for this single compile
    # call; each step performs exactly one incremental cut plus its proof.
    # The final equivalence proof reuses the last-step live compiled carrier
    # on that same model instead of replaying the recipe in a fresh model, so
    # it must run before the shared model closes (the carrier dies with it).
    with model_factory(
        f"planar-feature-chain-{construction.digest()[:12]}",
        dimension=2,
    ) as chain_cad:
        feature_recipe, chain_compiled, _ambiguity = build(
            construction.result_node_id,
            chain_cad,
        )
        if type(feature_recipe) is SketchGeometry:
            # No top-level Boolean produced a feature chain (plain leaf or
            # union-multi-face degrade), so there is no live carrier to prove.
            return feature_recipe
        try:
            feature_loops = chain_cad.planar_boundary_loops(chain_compiled.domain)
            feature_facts = _native_facts(
                chain_cad, chain_compiled.domain, feature_loops
            )
            expected_facts = _NativeFacts(
                compiled.proof.area,
                compiled.proof.bounding_box,
                compiled.proof.component_count,
                compiled.proof.hole_count,
                compiled.proof.curve_type_counts,
            )
            # Native boolean features legitimately split analytic curves at
            # intersection points (for example round path-stroke joins), so the
            # final proof verifies geometry and topology but not exact curve
            # counts.
            _require_equivalent(
                expected_facts,
                feature_facts,
                construction.result_node_id,
                strict_curves=False,
            )
        except PlanarConstructionCompileError:
            raise
        except PlanarConstructionCancelled:
            raise
        except Exception as error:
            _fail(
                "planar-ir.feature-equivalence-failed",
                f"Feature recipe proof failed: {error}",
                node_id=construction.result_node_id,
                allowed_fields=("nodes", "result_node_id"),
            )
    return feature_recipe


def _canonical_nodes(construction: PlanarConstructionIR) -> tuple[Any, ...]:
    order = tuple(
        node["id"] for node in json.loads(construction.canonical_json())["nodes"]
    )
    table = {node.id: node for node in construction.nodes}
    return tuple(table[node_id] for node_id in order)


def _evaluate(
    cad: Any,
    construction: PlanarConstructionIR,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    values: dict[str, tuple[Any, ...]] = {}
    lineage: dict[str, frozenset[str]] = {}
    for node in _canonical_nodes(construction):
        try:
            if isinstance(node, RectangleNode):
                output = (cad.rectangle(node.x, node.y, node.width, node.height),)
                sources = frozenset((node.id,))
            elif isinstance(node, CircleNode):
                output = (cad.disk(node.center_x, node.center_y, node.radius),)
                sources = frozenset((node.id,))
            elif isinstance(node, PolygonNode):
                output = (_polygon(cad, node.vertices),)
                sources = frozenset((node.id,))
            elif isinstance(node, PathStrokeNode):
                output = _path_stroke(cad, node)
                sources = frozenset((node.id,))
            elif isinstance(node, UnionNode):
                output = _union(cad, tuple(values[item] for item in node.operands))
                sources = frozenset().union(*(lineage[item] for item in node.operands))
            elif isinstance(node, DifferenceNode):
                output = _difference(
                    cad,
                    values[node.base],
                    tuple((item, values[item]) for item in node.subtract),
                )
                sources = lineage[node.base].union(
                    *(lineage[item] for item in node.subtract)
                )
            elif isinstance(node, IntersectionNode):
                output = _intersection(
                    cad, tuple(values[item] for item in node.operands)
                )
                sources = frozenset().union(*(lineage[item] for item in node.operands))
            elif isinstance(node, TranslateNode):
                output = cad.translate(
                    _copies(cad, values[node.source]), node.dx, node.dy, 0.0
                )
                sources = lineage[node.source]
            elif isinstance(node, RotateNode):
                output = cad.rotate(
                    _copies(cad, values[node.source]),
                    node.center_x,
                    node.center_y,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    math.radians(node.angle_degrees),
                )
                sources = lineage[node.source]
            elif isinstance(node, MirrorNode):
                direction_length = math.hypot(
                    node.line_direction_x, node.line_direction_y
                )
                a = -node.line_direction_y / direction_length
                b = node.line_direction_x / direction_length
                d = -(a * node.line_point_x + b * node.line_point_y)
                mirrored = cad.mirror(_copies(cad, values[node.source]), a, b, 0.0, d)
                output = _rebuild_exact(cad, construction, mirrored)
                sources = lineage[node.source]
            elif isinstance(node, LinearPatternNode):
                output = _pattern(
                    cad,
                    values[node.seed],
                    tuple(
                        (index * node.step_x, index * node.step_y, 0.0)
                        for index in range(node.count)
                    ),
                )
                sources = lineage[node.seed]
            elif isinstance(node, RectangularPatternNode):
                output = _pattern(
                    cad,
                    values[node.seed],
                    tuple(
                        (x * node.spacing_x, y * node.spacing_y, 0.0)
                        for y in range(node.count_y)
                        for x in range(node.count_x)
                    ),
                )
                sources = lineage[node.seed]
            elif isinstance(node, CircularPatternNode):
                instances = []
                for index in range(node.count):
                    copied = _copies(cad, values[node.seed])
                    instances.extend(
                        cad.rotate(
                            copied,
                            node.center_x,
                            node.center_y,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            math.radians(node.total_angle_degrees * index / node.count),
                        )
                    )
                output = tuple(instances)
                sources = lineage[node.seed]
            else:  # pragma: no cover - the v1 union is exhaustive
                raise TypeError(f"Unsupported node: {type(node).__name__}")
            _require_non_degenerate(cad, output, node.id)
        except PlanarConstructionCompileError:
            raise
        except PlanarConstructionCancelled:
            raise
        except Exception as error:
            code = (
                "planar-ir.invalid-path-stroke"
                if isinstance(node, PathStrokeNode)
                else "planar-ir.invalid-primitive"
                if isinstance(node, (RectangleNode, CircleNode, PolygonNode))
                else "planar-ir.materialization-failed"
            )
            _fail(
                code,
                f"Node {node.id} could not be compiled: {error}",
                node_id=node.id,
                allowed_fields=("nodes",),
            )
        values[node.id] = tuple(output)
        lineage[node.id] = sources
    return values[construction.result_node_id], tuple(
        sorted(lineage[construction.result_node_id])
    )


def _copies(cad: Any, surfaces: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(cad.copy(surfaces))


def _difference(
    cad: Any,
    base: tuple[Any, ...],
    subtract_groups: tuple[tuple[str, tuple[Any, ...]], ...],
) -> tuple[Any, ...]:
    """Cut every declared tool and reject instances that remove no material."""

    result = _copies(cad, base)
    for operand_id, tools in subtract_groups:
        for instance_index, tool in enumerate(tools):
            before_area = sum(cad.area(surface) for surface in result)
            result = tuple(cad.cut(result, _copies(cad, (tool,))).of_dimension(2))
            after_area = sum(cad.area(surface) for surface in result)
            tolerance = max(
                MODELING_TOLERANCE * MODELING_TOLERANCE,
                1.0e-12 * max(before_area, 1.0),
            )
            if before_area - after_area <= tolerance:
                _fail(
                    "planar-ir.subtract-no-effect",
                    (
                        f"Subtraction operand {operand_id} instance "
                        f"{instance_index + 1} removed no material. Check the "
                        "primitive coordinates and rectangle lower-left anchor."
                    ),
                    node_id=operand_id,
                    allowed_fields=("nodes",),
                )
    return result


def _union(cad: Any, groups: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    entities = tuple(entity for group in groups for entity in group)
    result = _copies(cad, (entities[0],))
    for entity in entities[1:]:
        result = tuple(cad.fuse(result, _copies(cad, (entity,))).of_dimension(2))
    return result


def _intersection(cad: Any, groups: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    result = _copies(cad, groups[0])
    for group in groups[1:]:
        result = tuple(cad.intersect(result, _copies(cad, group)).of_dimension(2))
        if not result:
            break
    return result


def _pattern(
    cad: Any,
    seed: tuple[Any, ...],
    offsets: tuple[tuple[float, float, float], ...],
) -> tuple[Any, ...]:
    result = []
    for dx, dy, dz in offsets:
        result.extend(cad.translate(_copies(cad, seed), dx, dy, dz))
    return tuple(result)


def _rebuild_exact(
    cad: Any,
    construction: PlanarConstructionIR,
    surfaces: tuple[Any, ...],
) -> tuple[Any, ...]:
    loops = cad.planar_boundary_loops(surfaces)
    recipe, _lineage = _materialize(construction, loops, ())
    return tuple(compile_recipe(cad, recipe).domain)


def _polygon(cad: Any, vertices: tuple[tuple[float, float], ...]) -> Any:
    points = tuple(cad.point(x, y) for x, y in vertices)
    curves = tuple(
        cad.line(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    loop = cad.curve_loop(tuple(cad.orient(curve) for curve in curves))
    return cad.plane_surface(loop)


def _path_stroke(cad: Any, node: PathStrokeNode) -> tuple[Any, ...]:
    half = 0.5 * node.width
    points = list(node.points)
    directions = tuple(
        (
            (right[0] - left[0]) / math.dist(left, right),
            (right[1] - left[1]) / math.dist(left, right),
        )
        for left, right in zip(points, points[1:])
    )
    original_start, original_end = points[0], points[-1]
    if node.cap == "square":
        points[0] = (
            points[0][0] - directions[0][0] * half,
            points[0][1] - directions[0][1] * half,
        )
        points[-1] = (
            points[-1][0] + directions[-1][0] * half,
            points[-1][1] + directions[-1][1] * half,
        )

    left = _stroke_side(tuple(points), directions, half, 1.0, node.join)
    right = _stroke_side(tuple(points), directions, half, -1.0, node.join)
    perimeter = list(left)
    if node.cap == "round":
        perimeter.append(("arc", left[-1][2], original_end, right[-1][2], "ccw"))
    else:
        perimeter.append(("line", left[-1][2], right[-1][2]))
    perimeter.extend(_reverse_stroke_segment(segment) for segment in reversed(right))
    if node.cap == "round":
        perimeter.append(("arc", right[0][1], original_start, left[0][1], "ccw"))
    else:
        perimeter.append(("line", right[0][1], left[0][1]))

    point_refs: dict[tuple[float, float], Any] = {}

    def point_ref(point: tuple[float, float]) -> Any:
        if point not in point_refs:
            point_refs[point] = cad.point(*point)
        return point_refs[point]

    curves = []
    for segment in perimeter:
        if segment[0] == "line":
            curves.append(cad.line(point_ref(segment[1]), point_ref(segment[2])))
        else:
            start, center, end, orientation = segment[1:]
            if orientation == "ccw":
                curves.append(
                    cad.circular_arc(
                        point_ref(start), point_ref(center), point_ref(end)
                    )
                )
            else:
                curves.append(
                    cad.circular_arc(
                        point_ref(end), point_ref(center), point_ref(start)
                    )
                )
                curves[-1] = cad.orient(curves[-1], reversed=True)
    oriented = tuple(
        curve if hasattr(curve, "curve") else cad.orient(curve) for curve in curves
    )
    return (cad.plane_surface(cad.curve_loop(oriented)),)


def _stroke_side(
    points: tuple[tuple[float, float], ...],
    directions: tuple[tuple[float, float], ...],
    half: float,
    sign: float,
    join: str,
) -> tuple[tuple[Any, ...], ...]:
    def normal(direction: tuple[float, float]) -> tuple[float, float]:
        return -direction[1] * sign * half, direction[0] * sign * half

    start = (
        points[0][0] + normal(directions[0])[0],
        points[0][1] + normal(directions[0])[1],
    )
    current = start
    segments: list[tuple[Any, ...]] = []
    for index, center in enumerate(points[1:-1], start=1):
        previous = directions[index - 1]
        following = directions[index]
        previous_normal = normal(previous)
        following_normal = normal(following)
        first = (center[0] + previous_normal[0], center[1] + previous_normal[1])
        second = (center[0] + following_normal[0], center[1] + following_normal[1])
        cross = previous[0] * following[1] - previous[1] * following[0]
        if abs(cross) <= MODELING_TOLERANCE:
            continue
        outer = cross * sign < 0.0
        intersection = _line_intersection(first, previous, second, following)
        if not outer:
            segments.append(("line", current, intersection))
            current = intersection
            continue
        if current != first:
            segments.append(("line", current, first))
        if join == "round":
            orientation = "ccw" if cross > 0.0 else "cw"
            segments.append(("arc", first, center, second, orientation))
            current = second
        elif (
            join == "miter"
            and math.dist(center, intersection) <= PATH_STROKE_MITER_LIMIT * half
        ):
            segments.append(("line", first, intersection))
            current = intersection
        else:
            segments.append(("line", first, second))
            current = second
    end_normal = normal(directions[-1])
    end = (points[-1][0] + end_normal[0], points[-1][1] + end_normal[1])
    segments.append(("line", current, end))
    return tuple(segments)


def _line_intersection(
    first: tuple[float, float],
    first_direction: tuple[float, float],
    second: tuple[float, float],
    second_direction: tuple[float, float],
) -> tuple[float, float]:
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    delta = (second[0] - first[0], second[1] - first[1])
    distance = (
        delta[0] * second_direction[1] - delta[1] * second_direction[0]
    ) / denominator
    return (
        first[0] + first_direction[0] * distance,
        first[1] + first_direction[1] * distance,
    )


def _reverse_stroke_segment(segment: tuple[Any, ...]) -> tuple[Any, ...]:
    if segment[0] == "line":
        return ("line", segment[2], segment[1])
    orientation = "cw" if segment[4] == "ccw" else "ccw"
    return ("arc", segment[3], segment[2], segment[1], orientation)


def _require_non_degenerate(cad: Any, surfaces: tuple[Any, ...], node_id: str) -> None:
    if not surfaces:
        _fail(
            "planar-ir.boolean-empty",
            f"Node {node_id} produced an empty region.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    area = sum(cad.area(surface) for surface in surfaces)
    if area <= MODELING_TOLERANCE * MODELING_TOLERANCE:
        _fail(
            "planar-ir.degenerate-result",
            f"Node {node_id} produced a zero-area region.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    scale = max(1.0, math.sqrt(area))
    if any(
        cad.distance(left, right) <= MODELING_TOLERANCE * scale
        for index, left in enumerate(surfaces)
        for right in surfaces[index + 1 :]
    ):
        _fail(
            "planar-ir.degenerate-result",
            f"Node {node_id} produced touching material components.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    if any(
        cad.length(curve) <= MODELING_TOLERANCE * scale
        for curve in cad.boundary(surfaces)
    ):
        _fail(
            "planar-ir.degenerate-result",
            f"Node {node_id} produced a boundary below modeling tolerance.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )


def _materialize(
    construction: PlanarConstructionIR,
    loops: tuple[Any, ...],
    source_nodes: tuple[str, ...],
) -> tuple[SketchGeometry, tuple[PlanarCurveLineage, ...]]:
    points: list[SketchPoint] = []
    curves: list[Any] = []
    lineage: list[PlanarCurveLineage] = []

    def point_id(coordinate: tuple[float, float]) -> str:
        for point in points:
            scale = max(1.0, abs(point.u), abs(point.v), *map(abs, coordinate))
            if math.dist((point.u, point.v), coordinate) <= MODELING_TOLERANCE * scale:
                return point.id
        logical_id = f"pc-p{len(points) + 1:04d}"
        points.append(SketchPoint(logical_id, *coordinate))
        return logical_id

    for loop in loops:
        for native in loop.curves:
            curve_id = f"pc-c{len(curves) + 1:04d}"
            if native.kind == "line":
                curve = SketchLine(
                    curve_id,
                    point_id(native.start),
                    point_id(native.end),
                )
            elif native.kind == "circle":
                curve = SketchCircle(
                    curve_id,
                    point_id(native.center),
                    native.radius,
                )
            elif native.kind == "arc":
                curve = SketchArc(
                    curve_id,
                    point_id(native.start),
                    point_id(native.center),
                    point_id(native.end),
                    native.orientation,
                )
            else:  # pragma: no cover - facade contract is exhaustive
                _fail(
                    "planar-ir.unsupported-boundary",
                    f"Unsupported boundary type: {native.kind}.",
                    node_id=construction.result_node_id,
                    allowed_fields=("nodes",),
                )
            curves.append(curve)
            lineage.append(PlanarCurveLineage(curve_id, source_nodes))
    try:
        recipe = SketchGeometry(
            construction.name,
            SketchPlane.xy(),
            tuple(points),
            tuple(curves),
        )
    except (TypeError, ValueError) as error:
        _fail(
            "planar-ir.materialization-failed",
            f"Strict sketch materialization failed: {error}",
            node_id=construction.result_node_id,
            allowed_fields=("nodes", "result_node_id"),
        )
    return recipe, tuple(lineage)


def _native_facts(
    cad: Any, surfaces: tuple[Any, ...], loops: tuple[Any, ...]
) -> _NativeFacts:
    area = sum(cad.area(surface) for surface in surfaces)
    boxes = tuple(cad.bounding_box(surface) for surface in surfaces)
    curve_types = Counter(curve.kind for loop in loops for curve in loop.curves)
    return _NativeFacts(
        area,
        (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[3] for box in boxes),
            max(box[4] for box in boxes),
        ),
        len(surfaces),
        len(loops) - len(surfaces),
        tuple(sorted(curve_types.items())),
    )


def _require_equivalent(
    source: _NativeFacts,
    recipe: _NativeFacts,
    node_id: str,
    *,
    strict_curves: bool = True,
) -> None:
    scale = max(1.0, abs(source.area), *(abs(value) for value in source.bounding_box))
    if not math.isclose(
        source.area, recipe.area, rel_tol=1.0e-9, abs_tol=MODELING_TOLERANCE * scale
    ):
        _fail(
            "planar-ir.equivalence-failed",
            f"Area changed during strict sketch recompilation ({source.area:.17g} -> {recipe.area:.17g}).",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    if any(
        not math.isclose(
            left, right, rel_tol=1.0e-9, abs_tol=MODELING_TOLERANCE * scale
        )
        for left, right in zip(source.bounding_box, recipe.bounding_box)
    ):
        _fail(
            "planar-ir.equivalence-failed",
            "Bounding box changed during strict sketch recompilation.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    if (source.component_count, source.hole_count) != (
        recipe.component_count,
        recipe.hole_count,
    ):
        _fail(
            "planar-ir.equivalence-failed",
            "Topology changed during strict sketch recompilation.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )
    if strict_curves and source.curve_types != recipe.curve_types:
        _fail(
            "planar-ir.equivalence-failed",
            "Boundary curve types changed during strict sketch recompilation.",
            node_id=node_id,
            allowed_fields=("nodes",),
        )


def _prove_compiled_curve_types(
    cad: Any,
    recipe: SketchGeometry,
    logical: Any,
) -> tuple[tuple[str, int], ...]:
    points = {point.id: (point.u, point.v) for point in recipe.points}
    counts: Counter[str] = Counter()
    for curve in recipe.curves:
        entities = tuple(logical[f"edge:{curve.id}"])
        length = sum(cad.length(entity) for entity in entities)
        if isinstance(curve, SketchLine):
            counts["line"] += 1
            expected = math.dist(
                points[curve.start_point_id], points[curve.end_point_id]
            )
            if len(entities) != 1:
                raise ValueError(
                    f"line {curve.id} did not compile to one analytic edge"
                )
            endpoints = cad.boundary(entities, combined=False)
            coordinates = tuple(
                (
                    0.5 * (box[0] + box[3]),
                    0.5 * (box[1] + box[4]),
                )
                for box in (cad.bounding_box(point) for point in endpoints)
            )
            if len(coordinates) != 2 or not math.isclose(
                math.dist(*coordinates),
                length,
                rel_tol=1.0e-8,
                abs_tol=MODELING_TOLERANCE,
            ):
                raise ValueError(f"line {curve.id} lost analytic straightness")
        elif isinstance(curve, SketchCircle):
            counts["circle"] += 1
            expected = 2.0 * math.pi * curve.radius
            center = points[curve.center_point_id]
            if any(
                not all(
                    math.isclose(
                        actual, wanted, rel_tol=1.0e-8, abs_tol=MODELING_TOLERANCE
                    )
                    for actual, wanted in zip(cad.circle_center(entity)[:2], center)
                )
                for entity in entities
            ):
                raise ValueError(f"circle {curve.id} changed analytic center")
        else:
            counts["arc"] += 1
            center = points[curve.center_point_id]
            start = points[curve.start_point_id]
            end = points[curve.end_point_id]
            start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
            end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
            sweep = (
                (end_angle - start_angle) % (2.0 * math.pi)
                if curve.orientation == "ccw"
                else (start_angle - end_angle) % (2.0 * math.pi)
            )
            expected = math.dist(start, center) * sweep
            if any(
                not all(
                    math.isclose(
                        actual, wanted, rel_tol=1.0e-8, abs_tol=MODELING_TOLERANCE
                    )
                    for actual, wanted in zip(cad.circle_center(entity)[:2], center)
                )
                for entity in entities
            ):
                raise ValueError(f"arc {curve.id} changed analytic center")
        if not math.isclose(
            length, expected, rel_tol=1.0e-8, abs_tol=MODELING_TOLERANCE
        ):
            raise ValueError(f"curve {curve.id} changed analytic length")
    return tuple(sorted(counts.items()))


def _preview(
    cad: Any, recipe: SketchGeometry, compiled: Any
) -> PlanarConstructionPreview:
    face_ids: dict[Any, str] = {}
    edge_ids: dict[Any, str] = {}
    point_ids: dict[Any, str] = {}
    for logical_id, entities in compiled.logical_entities.items():
        target = (
            face_ids
            if logical_id.startswith("face:")
            else edge_ids
            if logical_id.startswith("edge:")
            else point_ids
            if logical_id.startswith("point:")
            else None
        )
        if target is not None:
            for entity in entities:
                target[entity] = logical_id
    faces = tuple(compiled.domain)
    edges = tuple(compiled.boundary)
    points = tuple(cad.boundary(edges, combined=False))
    tessellation = cad.tessellate_surfaces(faces, edges, points)
    return PlanarConstructionPreview(
        tessellation.points,
        tessellation.faces,
        tessellation.edges,
        tuple(face_ids[entity] for entity in tessellation.face_entities),
        tuple(edge_ids[entity] for entity in tessellation.edge_entities),
        tuple(
            None if entity is None else point_ids.get(entity)
            for entity in tessellation.point_entities
        ),
    )


def _recipe_digest(recipe: SketchGeometry) -> str:
    payload = {
        "name": recipe.name,
        "plane": {
            "origin": recipe.plane.origin,
            "x_direction": recipe.plane.x_direction,
            "y_direction": recipe.plane.y_direction,
        },
        "points": [(point.id, point.u, point.v) for point in recipe.points],
        "curves": [
            ("line", curve.id, curve.start_point_id, curve.end_point_id)
            if isinstance(curve, SketchLine)
            else ("circle", curve.id, curve.center_point_id, curve.radius)
            if isinstance(curve, SketchCircle)
            else (
                "arc",
                curve.id,
                curve.start_point_id,
                curve.center_point_id,
                curve.end_point_id,
                curve.orientation,
            )
            for curve in recipe.curves
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MODELING_TOLERANCE",
    "PATH_STROKE_MITER_LIMIT",
    "TOLERANCE_VERSION",
    "CompiledPlanarConstruction",
    "PlanarConstructionCompileError",
    "PlanarConstructionDiagnostic",
    "PlanarConstructionPreview",
    "PlanarConstructionProof",
    "PlanarCurveLineage",
    "compile_planar_construction",
    "compile_planar_feature_recipe",
]
