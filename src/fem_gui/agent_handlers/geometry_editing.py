"""GUI authoring handlers for geometry editing."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import replace
from fem import geometry as geometry_runtime
from fem.application import (
    NativePart,
    SessionSnapshot,
    compile_planar_construction,
    prepare_planar_boolean,
)
from fem_agent.authoring import AuthoringContractError
from fem_agent.boolean_authoring import (
    BODY_BOOLEAN_TOOL_HANDLING,
    PART_BOOLEAN_TOOL_HANDLING,
    create_body_boolean_proposal,
    create_part_boolean_proposal,
)
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome, AuthoringToolPreparation
from fem_agent.geometry_authoring import (
    GeometryDraft,
    apply_planar_edit_batch,
    add_planar_arc,
    add_planar_circle,
    add_planar_constraint,
    add_planar_line,
    add_planar_path_slot,
    add_planar_polygon,
    add_planar_rectangle,
    _as_strict_planar_sketch,
    create_geometry_edit_proposal,
    create_profile_extrusion_proposal,
    create_profile_path_sweep_proposal,
    create_profile_revolution_proposal,
    delete_planar_constraints,
    delete_planar_curves,
    delete_planar_circles,
    geometry_draft,
    feature_topology_catalog,
    planar_feature_geometry_catalog,
    planar_geometry_catalog,
    PlanarEditValidationError,
    rotate_geometry,
    replace_planar_constraint,
    replace_planar_circle_pattern,
    translate_geometry,
    update_planar_circle,
    update_planar_arc,
    update_planar_line,
    update_planar_point,
)
from fem.geometry import (
    BooleanGeometry,
    BooleanLineageResolutionError,
    ExtrudedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlanarConstructionIR,
    PlanarBooleanLineageResolutionError,
    PlanarIRValidationError,
    PathSweptGeometry,
    RevolvedGeometry,
    SketchGeometry,
    WireMember,
    WireGeometry,
    WirePoint,
    describe_recipe_topology,
    geometry_dimension,
    resolve_extrusion_source_faces,
)
from .binding import AuthoringHandlerBinding, envelope
from .geometry_support import (
    _display_number,
    _provider_recipe_kind,
    _preflight_body_boolean,
    _preflight_derived_geometry,
    _preflight_part_boolean,
    _preflight_profile_extrusions,
    _run_profile_transform_preflight,
)


_SKETCH_ENTITY_ADDITION_OPERATIONS = (
    "add_line",
    "add_arc",
    "add_circle",
    "add_rectangle",
    "add_path_slot",
    "add_polygon",
)
_SKETCH_ENTITY_MODIFICATION_OPERATIONS = (
    "update_point",
    "update_line",
    "update_arc",
    "update_circle",
    "delete_curves",
    "delete_circles",
    "replace_circle_pattern",
)
_SKETCH_CONSTRAINT_EDIT_OPERATIONS = (
    "add_constraint",
    "replace_constraint",
    "delete_constraints",
)
_RIGID_GEOMETRY_TRANSFORM_OPERATIONS = (
    "translate",
    "rotate",
)
_PLANAR_SKETCH_EDIT_OPERATIONS = (
    *_SKETCH_ENTITY_ADDITION_OPERATIONS,
    *_SKETCH_ENTITY_MODIFICATION_OPERATIONS,
    *_SKETCH_CONSTRAINT_EDIT_OPERATIONS,
    "batch",
)


def geometry_edit_impact(edit_mode: str, in_place: str) -> str:
    return (
        "Confirm to create an iteration model and migrate retained mesh settings and model definitions; "
        "mesh data, validation, runs, and results are not migrated"
        if edit_mode == "branch"
        else in_place
    )


def compile_planar_boolean_tool(raw_tool: object) -> tuple[object, str]:
    if not isinstance(raw_tool, Mapping):
        raise TypeError("planar Boolean tool must be an object")
    tool = dict(raw_tool)
    kind = str(tool.get("kind", ""))
    if kind in {"polygon", "path_stroke"}:
        coordinate_field = "vertices" if kind == "polygon" else "points"
        raw_coordinates = tool.get(coordinate_field)
        if not isinstance(raw_coordinates, list) or any(
            not isinstance(item, Mapping) or set(item) != {"x", "y"}
            for item in raw_coordinates
        ):
            raise ValueError(
                f"planar Boolean {coordinate_field} must contain x/y objects"
            )
        tool[coordinate_field] = [
            [item["x"], item["y"]] for item in raw_coordinates
        ]
    tool_node = {"id": "tool-profile", **tool}
    construction = PlanarConstructionIR.from_dict(
        {
            "schema_version": 1,
            "name": "agent-planar-boolean-tool",
            "plane": "XY",
            "nodes": [tool_node],
            "result_node_id": "tool-profile",
        }
    )
    return compile_planar_construction(construction).recipe, kind

def replace_planar_boolean_feature_tool(
    recipe: object,
    feature_id: str,
    replacement_tool: object,
) -> tuple[object, int]:
    matched = 0
    replayed = 0

    def rebuild(item: object) -> object:
        nonlocal matched, replayed
        if type(item) is not BooleanGeometry:
            return item
        context = item.planar_context
        if context is None:
            raise AuthoringContractError(
                "geometry.feature-replacement.unsupported-history: "
                "the selected feature is not in a pure planar Boolean chain"
            )
        rebuilt_object = rebuild(item.object_geometry)
        tool = item.tool_geometry
        if context.feature_id == feature_id:
            matched += 1
            tool = replacement_tool
        tool_face_ids = resolve_extrusion_source_faces(tool).face_ids
        with geometry_runtime.model(
            f"replace-{context.feature_id}",
            dimension=2,
        ) as cad:
            prepared = prepare_planar_boolean(
                cad,
                rebuilt_object,
                context.target_face_id,
                tool,
                tool_face_ids,
                item.operation,
            )
        replayed += 1
        return replace(prepared.geometry, name=item.name)

    rebuilt = rebuild(recipe)
    if matched != 1:
        raise AuthoringContractError(
            "geometry.feature-replacement.feature-missing: feature_id must "
            "identify exactly one existing planar Boolean feature"
        )
    return rebuilt, replayed

def verify_spatial_relation(
    before_recipe: object,
    after_recipe: object,
    raw_relation: object,
) -> dict[str, object] | None:
    if raw_relation is None:
        return None
    if not isinstance(raw_relation, Mapping) or not {
        "reference_feature_id",
        "relation",
        "clearance",
    } <= set(raw_relation) <= {
        "reference_feature_id",
        "relation",
        "clearance",
        "tolerance",
    }:
        raise ValueError("spatial_relation fields do not match")
    reference_feature_id = str(raw_relation["reference_feature_id"]).strip()
    relation = str(raw_relation["relation"])
    if not reference_feature_id:
        raise ValueError("reference_feature_id must be non-empty")
    if relation not in {"above", "below", "left_of", "right_of"}:
        raise ValueError("spatial relation is unsupported")
    raw_clearance = raw_relation["clearance"]
    if isinstance(raw_clearance, bool) or not isinstance(
        raw_clearance,
        (int, float),
    ):
        raise TypeError("spatial clearance must be a number")
    clearance = float(raw_clearance)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("spatial clearance must be finite and non-negative")
    raw_tolerance = raw_relation.get(
        "tolerance",
        max(1.0e-9, abs(clearance) * 1.0e-9),
    )
    if isinstance(raw_tolerance, bool) or not isinstance(
        raw_tolerance,
        (int, float),
    ):
        raise TypeError("spatial tolerance must be a number")
    tolerance = float(raw_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("spatial tolerance must be finite and positive")

    before_features = planar_feature_geometry_catalog(before_recipe)["features"]
    after_features = planar_feature_geometry_catalog(after_recipe)["features"]
    before_by_id = {
        str(item["feature_id"]): item
        for item in before_features
        if isinstance(item, Mapping) and "feature_id" in item
    }
    reference = before_by_id.get(reference_feature_id)
    if reference is None:
        raise AuthoringContractError(
            "geometry.spatial-reference-missing: reference_feature_id "
            "does not identify an existing planar feature"
        )
    added = [
        item
        for item in after_features
        if isinstance(item, Mapping)
        and str(item.get("feature_id", "")) not in before_by_id
    ]
    if len(added) != 1:
        raise AuthoringContractError(
            "geometry.spatial-target-ambiguous: relation validation "
            "requires exactly one new planar feature"
        )
    target = added[0]
    reference_box = tuple(float(value) for value in reference["bounding_box"])
    target_box = tuple(float(value) for value in target["bounding_box"])
    measured = {
        "above": target_box[1] - reference_box[3],
        "below": reference_box[1] - target_box[3],
        "left_of": reference_box[0] - target_box[2],
        "right_of": target_box[0] - reference_box[2],
    }[relation]
    if abs(measured - clearance) > tolerance:
        raise AuthoringContractError(
            "geometry.spatial-relation-mismatch: "
            f"requested {relation} clearance={clearance:g}, "
            f"measured={measured:g}"
        )
    return {
        "reference_feature_id": reference_feature_id,
        "target_feature_id": str(target["feature_id"]),
        "relation": relation,
        "requested_clearance": clearance,
        "measured_clearance": measured,
        "tolerance": tolerance,
        "verified": True,
    }


def _prepare_sketch_entity_addition(
    part: NativePart, operation: str, edit: Mapping[str, object]
) -> tuple[GeometryDraft, str]:
    if operation == "add_line":
        if set(edit) != {"start", "end"}:
            raise ValueError("add_line fields do not match")
        draft = add_planar_line(part.geometry_recipe, **edit)
        summary = f"Add a line to the planar sketch in part {part.name}"
    elif operation == "add_arc":
        if set(edit) != {"start", "center", "end", "orientation"}:
            raise ValueError("add_arc fields do not match")
        draft = add_planar_arc(part.geometry_recipe, **edit)
        summary = f"Add an arc to the planar sketch in part {part.name}"
    elif operation == "add_circle":
        if set(edit) != {"center_x", "center_y", "radius"}:
            raise ValueError("add_circle fields do not match")
        draft = add_planar_circle(part.geometry_recipe, **edit)
        summary = (
            f"Add a circle to the planar sketch in part {part.name}: "
            f"center ({edit['center_x']}, {edit['center_y']}), "
            f"radius {edit['radius']}"
        )
    elif operation == "add_rectangle":
        if set(edit) != {"x", "y", "width", "height"}:
            raise ValueError("add_rectangle fields do not match")
        draft = add_planar_rectangle(part.geometry_recipe, **edit)
        summary = (
            f"Add a rectangular profile to the planar sketch in part {part.name}: "
            f"origin ({edit['x']}, {edit['y']}), "
            f"size {edit['width']} × {edit['height']}"
        )
    elif operation == "add_path_slot":
        if set(edit) != {"points", "width", "cap", "join"}:
            raise ValueError("add_path_slot fields do not match")
        raw_points = edit["points"]
        if not isinstance(raw_points, list) or any(
            not isinstance(item, Mapping) or set(item) != {"x", "y"}
            for item in raw_points
        ):
            raise ValueError("path slot points must contain x/y objects")
        points = tuple((item["x"], item["y"]) for item in raw_points)
        draft = add_planar_path_slot(
            part.geometry_recipe,
            points=points,
            width=edit["width"],
            cap=str(edit["cap"]),
            join=str(edit["join"]),
        )
        summary = (
            f"Add a connected constant-width path slot to the planar sketch in part {part.name}: "
            f"{len(points)} centerline points, slot width {edit['width']}"
        )
    elif operation == "add_polygon":
        if set(edit) != {"vertices"}:
            raise ValueError("add_polygon fields do not match")
        raw_vertices = edit["vertices"]
        if not isinstance(raw_vertices, list) or any(
            not isinstance(item, Mapping) or set(item) != {"x", "y"}
            for item in raw_vertices
        ):
            raise ValueError("vertices must contain x/y objects")
        vertices = tuple((item["x"], item["y"]) for item in raw_vertices)
        draft = add_planar_polygon(
            part.geometry_recipe,
            vertices=vertices,
        )
        summary = f"Add a closed {len(vertices)}-sided profile to the planar sketch in part {part.name}"
    else:
        raise ValueError("unsupported incremental geometry edit")
    return draft, summary


def _prepare_sketch_entity_modification(
    part: NativePart, operation: str, edit: Mapping[str, object]
) -> tuple[GeometryDraft, str]:
    if operation == "update_point":
        allowed = {"point_id", "x", "y"}
        if not {"point_id"} <= set(edit) <= allowed or len(edit) == 1:
            raise ValueError("update_point fields do not match")
        draft = update_planar_point(part.geometry_recipe, **edit)
        summary = f"Update sketch point {edit['point_id']} in part {part.name}"
    elif operation == "update_line":
        allowed = {"line_id", "start", "end"}
        if not {"line_id"} <= set(edit) <= allowed or len(edit) == 1:
            raise ValueError("update_line fields do not match")
        draft = update_planar_line(part.geometry_recipe, **edit)
        summary = f"Update line {edit['line_id']} in part {part.name}"
    elif operation == "update_arc":
        allowed = {"arc_id", "start", "center", "end", "orientation"}
        if not {"arc_id"} <= set(edit) <= allowed or len(edit) == 1:
            raise ValueError("update_arc fields do not match")
        draft = update_planar_arc(part.geometry_recipe, **edit)
        summary = f"Update arc {edit['arc_id']} in part {part.name}"
    elif operation == "update_circle":
        allowed = {"circle_id", "center_x", "center_y", "radius"}
        if not {"circle_id"} <= set(edit) <= allowed or len(edit) == 1:
            raise ValueError("update_circle fields do not match")
        draft = update_planar_circle(part.geometry_recipe, **edit)
        summary = f"Update circle {edit['circle_id']} in part {part.name}"
    elif operation == "delete_curves":
        if set(edit) != {"curve_ids"}:
            raise ValueError("delete_curves fields do not match")
        draft = delete_planar_curves(part.geometry_recipe, **edit)
        summary = f"Delete lines or arcs from the planar sketch in part {part.name}"
    elif operation == "delete_circles":
        if set(edit) != {"circle_ids"}:
            raise ValueError("delete_circles fields do not match")
        draft = delete_planar_circles(part.geometry_recipe, **edit)
        summary = (
            f"Delete {len(edit['circle_ids'])} circles from the planar sketch in part {part.name}"
        )
    elif operation == "replace_circle_pattern":
        if set(edit) != {
            "target_circle_ids",
            "count",
            "start_center_x",
            "start_center_y",
            "spacing_x",
            "spacing_y",
            "radius",
        }:
            raise ValueError("replace_circle_pattern fields do not match")
        draft = replace_planar_circle_pattern(part.geometry_recipe, **edit)
        summary = f"Replace the circle in part {part.name} with a linear array of {edit['count']} holes"
    else:
        raise ValueError("unsupported incremental geometry edit")
    return draft, summary


def _prepare_sketch_constraint_edit(
    part: NativePart, operation: str, edit: Mapping[str, object]
) -> tuple[GeometryDraft, str]:
    if operation == "add_constraint":
        if set(edit) != {"constraint"}:
            raise ValueError("add_constraint fields do not match")
        draft = add_planar_constraint(part.geometry_recipe, **edit)
        summary = f"Add a constraint to the planar sketch in part {part.name}"
    elif operation == "replace_constraint":
        if set(edit) != {"constraint_id", "constraint"}:
            raise ValueError("replace_constraint fields do not match")
        draft = replace_planar_constraint(part.geometry_recipe, **edit)
        summary = f"Replace sketch constraint {edit['constraint_id']} in part {part.name}"
    elif operation == "delete_constraints":
        if set(edit) != {"constraint_ids"}:
            raise ValueError("delete_constraints fields do not match")
        draft = delete_planar_constraints(part.geometry_recipe, **edit)
        summary = f"Delete a constraint from the planar sketch in part {part.name}"
    else:
        raise ValueError("unsupported incremental geometry edit")
    return draft, summary


def _prepare_rigid_geometry_transform(
    part: NativePart, operation: str, edit: Mapping[str, object]
) -> tuple[GeometryDraft, str]:
    if operation == "translate":
        if not {"dx", "dy"} <= set(edit) <= {"dx", "dy", "dz"}:
            raise ValueError("translate fields do not match")
        base_draft = geometry_draft(part.geometry_recipe)
        draft = translate_geometry(base_draft, **edit)
        summary = f"Translate part {part.name}"
    elif operation == "rotate":
        if set(edit) != {"axis", "angle_degrees"}:
            raise ValueError("rotate fields do not match")
        base_draft = geometry_draft(part.geometry_recipe)
        draft = rotate_geometry(base_draft, **edit)
        summary = f"Rotate part {part.name}"
    else:
        raise ValueError("unsupported incremental geometry edit")
    return draft, summary


def _prepare_planar_sketch_batch(
    part: NativePart, edit: Mapping[str, object]
) -> tuple[GeometryDraft, str]:
    if set(edit) != {"edits"}:
        raise ValueError("batch fields do not match")
    raw_edits = edit["edits"]
    if not isinstance(raw_edits, list):
        raise TypeError("batch edits must be an array")
    draft = apply_planar_edit_batch(
        part.geometry_recipe,
        edits=raw_edits,
    )
    summary = (
        f"Batch edit the planar sketch in part {part.name}: {len(raw_edits)} atomic steps"
    )
    return draft, summary


class GeometryEditingHandlers:
    def __init__(self, binding: AuthoringHandlerBinding) -> None:
        self.binding = binding

    def read_geometry_edit_context(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part_id = str(arguments["part_id"])
        snapshot = self.binding.current_session().projection_snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "part_id does not identify one editable native Part"
            )
        try:
            catalog = planar_geometry_catalog(part.geometry_recipe)
        except TypeError:
            if part.dimension == 2:
                topology = describe_recipe_topology(part.geometry_recipe)
                try:
                    selection = resolve_extrusion_source_faces(
                        part.geometry_recipe
                    )
                except ValueError:
                    selected_face_ids: tuple[str, ...] = ()
                else:
                    selected_face_ids = selection.face_ids
                catalog = {
                    "kind": "planar_feature_recipe",
                    "recipe_kind": _provider_recipe_kind(part.geometry_recipe),
                    "supported_edits": [
                        "planar_boolean",
                        "replace_planar_boolean_feature",
                        "add_path_slot",
                        *_RIGID_GEOMETRY_TRANSFORM_OPERATIONS,
                    ],
                    "profile_summary": {
                        "topology_exact": topology.exact,
                        "material_profile_count": len(selected_face_ids),
                        "profiles": [
                            {
                                "profile_id": face_id,
                                "role": "outer",
                            }
                            for face_id in selected_face_ids
                        ],
                    },
                    "planar_feature_geometry": planar_feature_geometry_catalog(
                        part.geometry_recipe
                    ),
                }
            else:
                catalog = {
                    "kind": _provider_recipe_kind(part.geometry_recipe),
                    "supported_edits": list(_RIGID_GEOMETRY_TRANSFORM_OPERATIONS),
                }
        else:
            catalog["supported_edits"] = [
                "planar_boolean",
                *_PLANAR_SKETCH_EDIT_OPERATIONS,
                *_RIGID_GEOMETRY_TRANSFORM_OPERATIONS,
            ]
            sketch = _as_strict_planar_sketch(part.geometry_recipe)
            profile_analysis = geometry_runtime.analyze_sketch_profiles(sketch)
            profiles = profile_analysis.profiles
            catalog["profile_summary"] = {
                "topology_exact": profile_analysis.valid,
                "profile_count": len(profiles),
                "material_profile_count": sum(
                    item.role == "outer" for item in profiles
                ),
                "hole_count": sum(item.role == "hole" for item in profiles),
                "profiles": [
                    {
                        "profile_id": item.id,
                        "role": item.role,
                        "curve_count": len(item.curve_ids),
                        "area": abs(float(item.signed_area)),
                        "bounding_box": list(item.bounding_box),
                        "nesting_depth": item.nesting_depth,
                    }
                    for item in profiles[:16]
                ],
                "truncated": len(profiles) > 16,
                "diagnostics": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "affected_logical_ids": list(item.affected_ids),
                        "severity": item.severity,
                    }
                    for item in profile_analysis.diagnostics[:16]
                ],
            }
        if part.dimension == 3:
            if type(part.geometry_recipe) is MultiBodyGeometry:
                catalog["supported_edits"][:0] = ["body_boolean"]
                body_candidates = [
                    {"body_id": body.id, "body_name": body.name}
                    for body in part.geometry_recipe.bodies
                ]
            else:
                catalog["supported_edits"][:0] = ["part_boolean"]
                body_candidates = []
            catalog["exact_boolean"] = {
                "supported_operations": ["fuse", "cut"],
                "disabled_operations": [
                    {
                        "operation": "intersect",
                        "code": "boolean.agent.operation-disabled",
                        "message": (
                            "intersect is disabled until stable result Body IDs, "
                            "lineage replay, and edit semantics are proven"
                        ),
                    },
                    {
                        "operation": "fragment",
                        "code": "boolean.agent.operation-disabled",
                        "message": (
                            "fragment is disabled until stable multi-result Body "
                            "IDs, lineage replay, and edit semantics are proven"
                        ),
                    },
                ],
                "part_tool_handling": PART_BOOLEAN_TOOL_HANDLING,
                "body_tool_handling": BODY_BOOLEAN_TOOL_HANDLING,
                "body_candidates": body_candidates,
                "part_tool_candidates": [
                    {"part_id": candidate.id, "part_name": candidate.name}
                    for candidate in snapshot.parts
                    if (
                        candidate.id != part.id
                        and not candidate.suppressed
                        and candidate.dimension == 3
                        and type(candidate.geometry_recipe) is not MultiBodyGeometry
                    )
                ],
            }
        return AuthoringToolOutcome(
            "Editable geometry context read locally.",
            {
                "part_id": part_id,
                "part_name": str(part.name),
                **catalog,
            },
        )

    def prepare_geometry_edit(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        part_id = str(arguments["part_id"])
        raw_edit = arguments["edit"]
        if not isinstance(raw_edit, Mapping):
            raise TypeError("edit must be an object")
        edit = dict(raw_edit)
        operation = str(edit.pop("operation"))
        raw_spatial_relation = arguments.get("spatial_relation")
        if raw_spatial_relation is not None and operation in {
            "part_boolean",
            "body_boolean",
            "extrude_profiles",
            "revolve_profile",
            "path_sweep_profile",
        }:
            raise AuthoringContractError(
                "geometry.spatial-relation-operation: spatial_relation "
                "applies only to one new planar Boolean feature"
            )
        snapshot = self.binding.current_session().projection_snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "part_id does not identify one editable native Part"
            )
        base_recipe_override = arguments.get("_canonical_base_recipe")
        if base_recipe_override is not None and (
            not isinstance(base_recipe_override, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(base_recipe_override) != 2
        ):
            raise AuthoringContractError(
                "profile-transform.source-not-strict: canonical base recipe "
                "must be a strict planar sketch"
            )
        if operation == "planar_boolean":
            return self._prepare_planar_boolean(
                part_id, part, edit, raw_spatial_relation, controller
            )
        if operation == "replace_planar_boolean_feature":
            return self._prepare_planar_boolean_feature_replacement(
                part_id, part, edit, raw_spatial_relation, controller
            )
        if operation == "part_boolean":
            return self._prepare_part_boolean(
                part, edit, snapshot, controller
            )
        if operation == "body_boolean":
            return self._prepare_body_boolean(
                part, edit, controller
            )
        if operation == "extrude_profiles":
            return self._prepare_profile_extrusion(
                part_id, part, edit, base_recipe_override, controller
            )
        if operation == "revolve_profile":
            return self._prepare_profile_revolution(
                part_id, part, edit, base_recipe_override, controller
            )
        if operation == "path_sweep_profile":
            return self._prepare_profile_path_sweep(
                part_id, part, edit, base_recipe_override, controller
            )
        return self._prepare_incremental_geometry_edit(
            part_id, part, operation, edit, raw_spatial_relation, controller
        )

    def _prepare_planar_boolean(
        self,
        part_id: str,
        part: NativePart,
        edit: Mapping[str, object],
        raw_spatial_relation: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {"boolean_operation", "tool"}:
            raise ValueError("planar_boolean fields do not match")
        if part.dimension != 2:
            raise AuthoringContractError(
                "planar_boolean requires a two-dimensional Part"
            )
        boolean_operation = str(edit["boolean_operation"])
        if boolean_operation not in {"fuse", "cut"}:
            raise AuthoringContractError(
                "planar_boolean operation must be fuse or cut"
            )
        def prepare_detached():
            compiled_tool, kind = compile_planar_boolean_tool(edit["tool"])
            target_faces = resolve_extrusion_source_faces(part.geometry_recipe).face_ids
            tool_faces = resolve_extrusion_source_faces(compiled_tool).face_ids
            if len(target_faces) != 1:
                raise AuthoringContractError(
                    "planar_boolean requires one material target Profile"
                )
            with geometry_runtime.model(
                f"{part.name}-{boolean_operation}-agent-feature",
                dimension=2,
            ) as cad:
                prepared = prepare_planar_boolean(
                    cad,
                    part.geometry_recipe,
                    target_faces[0],
                    compiled_tool,
                    tool_faces,
                    boolean_operation,
                )
            return prepared, kind

        def finalize(result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            prepared, kind = result
            draft = geometry_draft(prepared.geometry)
            spatial_proof = verify_spatial_relation(
                part.geometry_recipe,
                draft.recipe,
                raw_spatial_relation,
            )
            action = "cut" if boolean_operation == "cut" else "union"
            summary = f"Append a 2D {action} feature to part {part.name}: tool profile is {kind}"
            metadata = envelope(controller, "geometry-planar-boolean")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_geometry_edit_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                draft=draft,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    f"Confirm to append a 2D {action} feature and invalidate the old mesh and results",
                ),
                confirm_label=f"Append {action}",
                extra_data={
                    "geometry_edit_mode": edit_mode,
                    **(
                        {"spatial_relation_proof": spatial_proof}
                        if spatial_proof is not None
                        else {}
                    ),
                },
            )

        return AuthoringToolPreparation(
            prepare_detached, finalize, threading.Event(),
            "fem-agent-planar-boolean-preflight", 30.0,
        )


    def _prepare_planar_boolean_feature_replacement(
        self,
        part_id: str,
        part: NativePart,
        edit: Mapping[str, object],
        raw_spatial_relation: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {"feature_id", "tool"}:
            raise ValueError(
                "replace_planar_boolean_feature fields do not match"
            )
        if part.dimension != 2:
            raise AuthoringContractError(
                "replace_planar_boolean_feature requires a two-dimensional Part"
            )
        if raw_spatial_relation is not None:
            raise AuthoringContractError(
                "geometry.spatial-relation-operation: spatial_relation applies "
                "only when appending one new planar Boolean feature"
            )
        feature_id = str(edit["feature_id"])
        def prepare_detached():
            compiled_tool, kind = compile_planar_boolean_tool(edit["tool"])
            rebuilt_recipe, replayed_feature_count = (
                replace_planar_boolean_feature_tool(
                    part.geometry_recipe,
                    feature_id,
                    compiled_tool,
                )
            )
            return rebuilt_recipe, replayed_feature_count, kind

        def finalize(result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            rebuilt_recipe, replayed_feature_count, kind = result
            draft = geometry_draft(rebuilt_recipe)
            summary = (
                f"Replace 2D Boolean feature {feature_id} in part {part.name}: "
                f"tool profile is {kind}; replay later features"
            )
            metadata = envelope(controller, "geometry-feature-replacement")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_geometry_edit_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                draft=draft,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to replace the selected 2D Boolean feature, replay later features, and invalidate the old mesh and results",
                ),
                confirm_label="Replace feature",
                extra_data={
                    "geometry_edit_mode": edit_mode,
                    "replaced_feature_id": feature_id,
                    "replayed_feature_count": replayed_feature_count,
                },
            )

        return AuthoringToolPreparation(
            prepare_detached, finalize, threading.Event(),
            "fem-agent-planar-boolean-preflight", 30.0,
        )


    def _prepare_part_boolean(
        self,
        part: NativePart,
        edit: Mapping[str, object],
        snapshot: SessionSnapshot,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {
            "boolean_operation",
            "tool_part_id",
            "result_name",
            "tool_handling",
        }:
            raise ValueError("part_boolean fields do not match")
        if type(part.geometry_recipe) is MultiBodyGeometry or part.dimension != 3:
            raise AuthoringContractError(
                "Part Boolean operands must each own one exact solid"
            )
        boolean_operation = str(edit["boolean_operation"])
        if boolean_operation not in {"fuse", "cut"}:
            raise AuthoringContractError(
                "intersect/fragment are disabled until stable result Body "
                "IDs and lineage replay are proven"
            )
        tool_part_id = str(edit["tool_part_id"])
        tool = next(
            (
                candidate
                for candidate in snapshot.parts
                if (
                    candidate.id == tool_part_id
                    and candidate.id != part.id
                    and not candidate.suppressed
                    and candidate.dimension == 3
                    and type(candidate.geometry_recipe) is not MultiBodyGeometry
                )
            ),
            None,
        )
        if tool is None:
            raise AuthoringContractError(
                "tool_part_id must identify another active single-solid Part"
            )
        result_name = str(edit["result_name"]).strip()
        tool_handling = str(edit["tool_handling"])
        if tool_handling != PART_BOOLEAN_TOOL_HANDLING:
            raise AuthoringContractError(
                f"tool_handling must be {PART_BOOLEAN_TOOL_HANDLING!r}"
            )
        session = self.binding.current_session()
        result_part_id = session.next_native_part_id
        feature_id = session.next_part_boolean_feature_id

        def prepare_detached():
            return _preflight_part_boolean(
                part, tool, boolean_operation,
                result_part_id=result_part_id,
                feature_id=feature_id,
                result_name=result_name,
            )

        def finalize(prepared: object, error: BaseException | None) -> AuthoringToolOutcome:
            if isinstance(error, BooleanLineageResolutionError):
                return AuthoringToolOutcome(str(error), {}, ok=False)
            if error is not None:
                raise error
            summary = (
                f"Exact {boolean_operation}: target Part {part.name} [{part.id}], "
                f"tool Part {tool.name} [{tool.id}]; result {result_name} "
                f"[{prepared.context.result_part_id}]; both source Parts are suppressed and can be restored by undo, "
                f"tool policy={tool_handling}"
            )
            metadata = envelope(controller, "geometry-part-boolean")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_part_boolean_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                target_part_id=part.id,
                tool_part_id=tool.id,
                operation=boolean_operation,
                result_name=result_name,
                tool_handling=tool_handling,
                prepared=prepared,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to create a proven result Part, suppress target/tool source Parts, "
                    "and invalidate the old mesh, definitions, and results",
                ),
                confirm_label="Exact Part Boolean",
                extra_data={
                    "result_part_id": prepared.context.result_part_id,
                    "feature_id": prepared.context.feature_id,
                    "lineage_proven": True,
                    "geometry_edit_mode": edit_mode,
                },
            )

        return AuthoringToolPreparation(
            prepare_detached, finalize, threading.Event(),
            "fem-agent-boolean-preflight", 30.0,
        )


    def _prepare_body_boolean(
        self,
        part: NativePart,
        edit: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {
            "boolean_operation",
            "target_body_id",
            "tool_body_id",
            "result_name",
            "tool_handling",
        }:
            raise ValueError("body_boolean fields do not match")
        geometry = part.geometry_recipe
        if type(geometry) is not MultiBodyGeometry:
            raise AuthoringContractError(
                "Body Boolean requires one canonical same-Part MultiBodyGeometry"
            )
        boolean_operation = str(edit["boolean_operation"])
        if boolean_operation not in {"fuse", "cut"}:
            raise AuthoringContractError(
                "intersect/fragment are disabled until stable result Body "
                "IDs and lineage replay are proven"
            )
        target_body_id = str(edit["target_body_id"])
        tool_body_id = str(edit["tool_body_id"])
        if target_body_id == tool_body_id:
            raise AuthoringContractError("target and tool Bodies must differ")
        geometry.body(target_body_id)
        geometry.body(tool_body_id)
        result_name = str(edit["result_name"]).strip()
        tool_handling = str(edit["tool_handling"])
        if tool_handling != BODY_BOOLEAN_TOOL_HANDLING:
            raise AuthoringContractError(
                f"tool_handling must be {BODY_BOOLEAN_TOOL_HANDLING!r}"
            )
        def prepare_detached():
            return _preflight_body_boolean(
                geometry, target_body_id, tool_body_id, boolean_operation,
                result_name=result_name,
            )

        def finalize(prepared: object, error: BaseException | None) -> AuthoringToolOutcome:
            if isinstance(error, BooleanLineageResolutionError):
                return AuthoringToolOutcome(str(error), {}, ok=False)
            if error is not None:
                raise error
            summary = (
                f"Exact {boolean_operation}: within Part {part.name} [{part.id}] "
                f"target Body {target_body_id}, tool Body {tool_body_id}; "
                f"result feature {result_name}, preserving target ID and consuming tool, "
                f"policy={tool_handling}"
            )
            metadata = envelope(controller, "geometry-body-boolean")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_body_boolean_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part.id,
                target_body_id=target_body_id,
                tool_body_id=tool_body_id,
                operation=boolean_operation,
                result_name=result_name,
                tool_handling=tool_handling,
                prepared=prepared,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to preserve the target Body ID and consume the tool Body within the same Part, "
                    "and invalidate the old mesh, definitions, and results",
                ),
                confirm_label="Exact Body Boolean",
                extra_data={
                    "target_body_id": target_body_id,
                    "consumed_tool_body_id": tool_body_id,
                    "lineage_proven": True,
                    "geometry_edit_mode": edit_mode,
                },
            )

        return AuthoringToolPreparation(
            prepare_detached, finalize, threading.Event(),
            "fem-agent-boolean-preflight", 30.0,
        )


    def _prepare_profile_extrusion(
        self,
        part_id: str,
        part: NativePart,
        edit: Mapping[str, object],
        base_recipe_override: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {"source_face_ids", "height"}:
            raise ValueError("extrude_profiles fields do not match")
        base_recipe = (
            base_recipe_override
            if base_recipe_override is not None
            else part.geometry_recipe
        )
        if (
            base_recipe_override is None
            and type(base_recipe) is not SketchGeometry
            and type(base_recipe) is not BooleanGeometry
        ):
            raise AuthoringContractError(
                "Agent Profile extrusion requires a canonical planar feature recipe"
            )
        if (
            not isinstance(base_recipe, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(base_recipe) != 2
        ):
            raise AuthoringContractError(
                "Agent Profile extrusion requires an exact planar feature Part"
            )
        raw_source_ids = edit["source_face_ids"]
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise ValueError(
                "source_face_ids must explicitly select material Profiles"
            )
        if any(not isinstance(item, str) for item in raw_source_ids):
            raise TypeError("source_face_ids must contain strings")
        selection = resolve_extrusion_source_faces(
            base_recipe,
            tuple(raw_source_ids),
        )
        if len(selection.face_ids) != len(raw_source_ids):
            raise ValueError("source_face_ids contain duplicate Profile aliases")
        height = float(edit["height"])
        recipes = tuple(
            ExtrudedGeometry(base_recipe, height, (face_id,))
            for face_id in selection.face_ids
        )
        def finalize(_result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            summary = (
                f"Extrude {len(selection.face_ids)} selected "
                f"Profiles in part {part.name}: height {_display_number(height)}, along the positive sketch normal, "
                f"creating {len(selection.face_ids)} separate Parts"
            )
            metadata = envelope(controller, "geometry-edit")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_profile_extrusion_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_ids=selection.face_ids,
                height=height,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to atomically convert selected Profiles to separate solid Parts, "
                    "and invalidate the old mesh, definitions, and results",
                ),
                confirm_label="Extrude Profiles",
                extra_data={"geometry_edit_mode": edit_mode},
            )

        return AuthoringToolPreparation(
            lambda: _run_profile_transform_preflight(_preflight_profile_extrusions, recipes),
            finalize,
            threading.Event(),
            "fem-agent-profile-preflight",
            30.0,
        )


    def _prepare_profile_revolution(
        self,
        part_id: str,
        part: NativePart,
        edit: Mapping[str, object],
        base_recipe_override: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {"source_face_id", "axis", "angle_degrees"}:
            raise ValueError("revolve_profile fields do not match")
        base_recipe = (
            base_recipe_override
            if base_recipe_override is not None
            else part.geometry_recipe
        )
        if (
            not isinstance(base_recipe, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(base_recipe) != 2
        ):
            raise AuthoringContractError(
                "Agent Profile revolution requires an exact planar feature Part"
            )
        source_face_id = str(edit["source_face_id"])
        selection = resolve_extrusion_source_faces(
            base_recipe,
            (source_face_id,),
        )
        if selection.face_ids != (source_face_id,):
            raise AuthoringContractError(
                "revolve_profile requires one canonical material Profile"
            )
        recipe = RevolvedGeometry(
            base_recipe,
            str(edit["axis"]),
            edit["angle_degrees"],
            (source_face_id,),
        )
        def finalize(_result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            summary = (
                f"Revolve part {part.name} around the {recipe.axis.upper()} axis, "
                f"Profile {source_face_id}: angle {recipe.angle_degrees:g}°, "
                "creating 1 solid Part"
            )
            metadata = envelope(controller, "geometry-revolve")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_profile_revolution_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                axis=recipe.axis,
                angle_degrees=recipe.angle_degrees,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to atomically replace the Profile with a revolved solid and invalidate the old mesh, definitions, and results",
                ),
                confirm_label="Revolve Profile",
                extra_data={"geometry_edit_mode": edit_mode},
            )

        return AuthoringToolPreparation(
            lambda: _run_profile_transform_preflight(_preflight_derived_geometry, recipe),
            finalize,
            threading.Event(),
            "fem-agent-profile-preflight",
            30.0,
        )


    def _prepare_profile_path_sweep(
        self,
        part_id: str,
        part: NativePart,
        edit: Mapping[str, object],
        base_recipe_override: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        if set(edit) != {"source_face_id", "path", "frame_strategy"}:
            raise ValueError("path_sweep_profile fields do not match")
        base_recipe = (
            base_recipe_override
            if base_recipe_override is not None
            else part.geometry_recipe
        )
        if (
            not isinstance(base_recipe, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(base_recipe) != 2
        ):
            raise AuthoringContractError(
                "Agent path sweep requires an exact planar feature Part"
            )
        source_face_id = str(edit["source_face_id"])
        selection = resolve_extrusion_source_faces(
            base_recipe,
            (source_face_id,),
        )
        if selection.face_ids != (source_face_id,):
            raise AuthoringContractError(
                "path_sweep_profile requires one canonical material Profile"
            )
        raw_path = edit["path"]
        if not isinstance(raw_path, Mapping) or set(raw_path) != {
            "points",
            "members",
        }:
            raise ValueError("path fields do not match")
        raw_points = raw_path["points"]
        raw_members = raw_path["members"]
        if not isinstance(raw_points, list) or not isinstance(raw_members, list):
            raise TypeError("path points and members must be arrays")
        if any(
            not isinstance(item, Mapping) or set(item) != {"name", "x", "y", "z"}
            for item in raw_points
        ):
            raise ValueError("path point fields do not match")
        if any(
            not isinstance(item, Mapping) or set(item) != {"name", "start", "end"}
            for item in raw_members
        ):
            raise ValueError("path member fields do not match")
        path = WireGeometry(
            f"SweepPath-{part.id}",
            tuple(
                WirePoint(item["name"], item["x"], item["y"], item["z"])
                for item in raw_points
            ),
            tuple(
                WireMember(item["name"], item["start"], item["end"])
                for item in raw_members
            ),
        )
        recipe = PathSweptGeometry(
            base_recipe,
            path,
            (source_face_id,),
            str(edit["frame_strategy"]),
        )
        def finalize(_result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            summary = (
                f"Sweep along an explicit open polyline with {len(path.members)} segments: part "
                f"{part.name}, Profile {source_face_id}; "
                f"frame={recipe.frame_strategy}, creating 1 solid Part"
            )
            metadata = envelope(controller, "geometry-path-sweep")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_profile_path_sweep_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                base_recipe=base_recipe,
                source_face_id=source_face_id,
                path=path,
                frame_strategy=recipe.frame_strategy,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to atomically replace the Profile with a path-swept solid, "
                    "and invalidate the old mesh, definitions, and results",
                ),
                confirm_label="Sweep Profile",
                extra_data={"geometry_edit_mode": edit_mode},
            )

        return AuthoringToolPreparation(
            lambda: _run_profile_transform_preflight(_preflight_derived_geometry, recipe),
            finalize,
            threading.Event(),
            "fem-agent-profile-preflight",
            30.0,
        )


    def _prepare_incremental_geometry_edit(
        self,
        part_id: str,
        part: NativePart,
        operation: str,
        edit: Mapping[str, object],
        raw_spatial_relation: object,
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolPreparation:
        def prepare_detached():
            if operation in _SKETCH_ENTITY_ADDITION_OPERATIONS:
                draft, summary = _prepare_sketch_entity_addition(part, operation, edit)
            elif operation in _SKETCH_ENTITY_MODIFICATION_OPERATIONS:
                draft, summary = _prepare_sketch_entity_modification(part, operation, edit)
            elif operation in _SKETCH_CONSTRAINT_EDIT_OPERATIONS:
                draft, summary = _prepare_sketch_constraint_edit(part, operation, edit)
            elif operation in _RIGID_GEOMETRY_TRANSFORM_OPERATIONS:
                draft, summary = _prepare_rigid_geometry_transform(part, operation, edit)
            elif operation == "batch":
                draft, summary = _prepare_planar_sketch_batch(part, edit)
            else:
                raise ValueError("unsupported incremental geometry edit")
            spatial_proof = verify_spatial_relation(
                part.geometry_recipe,
                draft.recipe,
                raw_spatial_relation,
            )
            return draft, summary, spatial_proof

        def finalize(result: object, error: BaseException | None) -> AuthoringToolOutcome:
            if error is not None:
                raise error
            draft, summary, spatial_proof = result
            metadata = envelope(controller, "geometry-edit")
            suffix = str(metadata.pop("identity_suffix"))
            edit_mode = self.binding.planned_geometry_edit_mode()
            proposal = create_geometry_edit_proposal(
                proposal_id=f"proposal-{suffix}",
                context=self.binding.current_context(),
                part_id=part_id,
                draft=draft,
                summary=summary,
                edit_mode=edit_mode,
                **metadata,
            )
            return self.binding.proposal_outcome(
                proposal,
                summary=summary,
                impact=geometry_edit_impact(
                    edit_mode,
                    "Confirm to update this part in the current model and invalidate the old mesh and results",
                ),
                confirm_label="Apply changes",
                extra_data={
                    "geometry_edit_mode": edit_mode,
                    **(
                        {"spatial_relation_proof": spatial_proof}
                        if spatial_proof is not None
                        else {}
                    ),
                },
            )

        return AuthoringToolPreparation(
            prepare_detached, finalize, threading.Event(),
            "fem-agent-sketch-edit-preflight", 30.0,
        )

    def prepare_geometry_edit_with_diagnostics(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        edit = arguments.get("edit")
        operation = (
            str(edit.get("operation", "")) if isinstance(edit, Mapping) else ""
        )
        tool = edit.get("tool") if isinstance(edit, Mapping) else None
        submitted_path_stroke = operation == "add_path_slot" or (
            operation in {"planar_boolean", "replace_planar_boolean_feature"}
            and isinstance(tool, Mapping)
            and tool.get("kind") == "path_stroke"
        )

        def path_slot_validation_outcome(
            *,
            code: str,
            reason: str,
            error_type: str,
            evidence: Mapping[str, object],
        ) -> AuthoringToolOutcome | None:
            if code != "planar-ir.invalid-path-stroke" or not submitted_path_stroke:
                return None
            bounded_evidence = dict(evidence)
            raw_points = (
                edit.get("points")
                if operation == "add_path_slot" and isinstance(edit, Mapping)
                else tool.get("points") if isinstance(tool, Mapping) else None
            )
            if isinstance(raw_points, list):
                bounded_evidence.setdefault("points_role", "centerline")
                bounded_evidence.setdefault("required_topology", "open")
                bounded_evidence.setdefault("submitted_point_count", len(raw_points))
                bounded_evidence.setdefault(
                    "first_equals_last",
                    bool(raw_points) and raw_points[0] == raw_points[-1],
                )
            return AuthoringToolOutcome(
                "Path-slot validation rejected the centerline.",
                {
                    "error": {
                        "code": code,
                        "failed_operation": operation,
                        "reason": reason,
                        "error_type": error_type,
                        "representation": "path_stroke",
                        "evidence": bounded_evidence,
                        "allowed_operations": list(
                            controller.geometry_edit_supported_operations
                        ),
                    },
                },
                ok=False,
            )

        def guarded(callback):
            try:
                return callback()
            except PlanarEditValidationError as error:
                data = error.to_provider_dict()
                first = error.diagnostics[0]
                code = str(first["code"])
                path_slot_outcome = path_slot_validation_outcome(
                    code=code,
                    reason=str(first["message"]),
                    error_type=type(error).__name__,
                    evidence={},
                )
                if path_slot_outcome is not None:
                    return path_slot_outcome
                affected = ", ".join(str(item) for item in first["affected_logical_ids"])
                location = f" ({affected})" if affected else ""
                return AuthoringToolOutcome(
                    "Exact planar Profile validation rejected the edit: "
                    f"{code}{location}.",
                    data,
                    ok=False,
                )
            except PlanarIRValidationError as error:
                diagnostic = error.diagnostic
                raw_evidence = getattr(diagnostic, "evidence", ())
                evidence = (
                    dict(raw_evidence)
                    if isinstance(raw_evidence, (Mapping, tuple, list))
                    else {}
                )
                path_slot_outcome = path_slot_validation_outcome(
                    code=diagnostic.code,
                    reason=diagnostic.message,
                    error_type=type(error).__name__,
                    evidence=evidence,
                )
                if path_slot_outcome is not None:
                    return path_slot_outcome
                return AuthoringToolOutcome(
                    "Planar validation rejected the geometry edit.",
                    {
                        "error": {
                            "code": diagnostic.code,
                            "failed_operation": operation,
                            "reason": diagnostic.message,
                            "error_type": type(error).__name__,
                            "evidence": evidence,
                            "allowed_operations": list(
                                controller.geometry_edit_supported_operations
                            ),
                        }
                    },
                    ok=False,
                )
            except PlanarBooleanLineageResolutionError as error:
                operation = ""
                edit = arguments.get("edit")
                if isinstance(edit, Mapping):
                    operation = str(edit.get("operation", ""))
                reason = str(error).strip()[:1_000]
                return AuthoringToolOutcome(
                    "Planar Boolean validation rejected the geometry edit.",
                    {
                        "error": {
                            "code": "geometry-edit.planar-boolean-lineage",
                            "failed_operation": operation,
                            "reason": reason,
                            "allowed_operations": list(
                                controller.geometry_edit_supported_operations
                            ),
                        }
                    },
                    ok=False,
                )
            except (AuthoringContractError, TypeError, ValueError, KeyError) as error:
                operation = ""
                edit = arguments.get("edit")
                if isinstance(edit, Mapping):
                    operation = str(edit.get("operation", ""))
                reason = str(error).strip()[:1_000]
                return AuthoringToolOutcome(
                    "Geometry edit execution was rejected.",
                    {
                        "error": {
                            "code": "geometry-edit.execution-rejected",
                            "failed_operation": operation,
                            "reason": reason,
                            "error_type": type(error).__name__,
                            "allowed_operations": list(
                                controller.geometry_edit_supported_operations
                            ),
                        }
                    },
                    ok=False,
                )

        outcome = guarded(lambda: self.prepare_geometry_edit(arguments, controller))
        if type(outcome) is AuthoringToolPreparation:
            return replace(outcome, finalize=lambda result, error: guarded(
                lambda: outcome.finalize(result, error)
            ))
        return outcome


    def read_geometry_feature_catalog(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"part_id"}:
            raise AuthoringContractError(
                "read_geometry_feature_catalog accepts only optional part_id"
            )
        requested_part_id = arguments.get("part_id")
        if requested_part_id is not None and (
            type(requested_part_id) is not str
            or not requested_part_id.strip()
        ):
            raise AuthoringContractError("part_id must be a non-empty string")
        snapshot = self.binding.current_session().projection_snapshot()
        active_parts = tuple(
            part
            for part in snapshot.parts
            if (
                not part.suppressed
                and part.geometry_recipe is not None
                and (
                    requested_part_id is None
                    or str(part.id) == requested_part_id
                )
            )
        )
        if requested_part_id is not None and not active_parts:
            raise AuthoringContractError(
                "part_id does not identify one active native Part"
            )
        source_parts = active_parts[:128]
        catalogs: list[dict[str, object]] = []
        for part in source_parts:
            item = feature_topology_catalog(
                part.geometry_recipe,
                part_id=str(part.id),
            )
            candidate = {
                "kind": "native_geometry_feature_catalog",
                "schema_version": 1,
                "session_revision": snapshot.session_revision,
                "parts": [*catalogs, item],
                "truncated": False,
                "omitted_part_count": 0,
            }
            try:
                provider_safe_authoring_payload(candidate)
            except ValueError:
                break
            catalogs.append(item)
        omitted = len(active_parts) - len(catalogs)
        data = {
            "kind": "native_geometry_feature_catalog",
            "schema_version": 1,
            "session_revision": snapshot.session_revision,
            "parts": catalogs,
            "truncated": omitted > 0,
            "omitted_part_count": omitted,
        }
        provider_safe_authoring_payload(data)
        return AuthoringToolOutcome(
            "Native geometry feature catalog read locally.",
            data,
        )
