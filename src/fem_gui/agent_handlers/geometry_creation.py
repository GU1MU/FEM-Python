"""GUI authoring handlers for geometry creation."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from fem import geometry as geometry_runtime
from fem.application import (
    PlanarConstructionCompileError,
    compile_planar_feature_recipe,
    compile_planar_construction,
)
from fem_agent.authoring import AuthoringContractError, UnitContextSummary
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome
from fem_agent.geometry_authoring import (
    box_geometry,
    create_geometry_proposal,
    cylinder_geometry,
    geometry_draft,
    planar_construction_proposal_evidence,
    wire_geometry,
)
from fem.geometry import (
    ExtrudedGeometry,
    MultiBodyGeometry,
    PlanarConstructionIR,
    PlanarIRValidationError,
    PathSweptGeometry,
    RevolvedGeometry,
    SketchGeometry,
    SolidBody,
    WireMember,
    WirePoint,
    resolve_extrusion_source_faces,
)
from ..workers import OwnedWorkerTimeout, run_owned_worker
from .binding import AuthoringHandlerBinding, envelope
from .geometry_support import (
    PlanarCompileCancelled,
    _DetachedRecipeMesh,
    _PLANAR_COMPILE_WORKER_TIMEOUT_SECONDS,
    _cancellable_planar_model_factory,
    _composite_path,
    _display_number,
    _geometry_unit_summary,
    _preflight_composite_geometry,
)
from ..agent_authoring_values import AgentProposalPreview


_PLANAR_DIAGNOSTIC_MESSAGES = {
    "planar-ir.schema-invalid": "Invalid 2D construction fields or version; correct the marked fields.",
    "planar-ir.budget-exceeded": "2D construction exceeds the local budget; simplify the construction graph.",
    "planar-ir.duplicate-node-id": "2D construction has duplicate node IDs; rename conflicting nodes.",
    "planar-ir.reference-missing": "2D construction references missing nodes; correct the failed references.",
    "planar-ir.cycle-detected": "2D construction nodes form a cycle; break the reported dependency.",
    "planar-ir.unreachable-node": "2D construction has an unused node; remove or connect it.",
    "planar-ir.invalid-primitive": "Invalid 2D primitive region parameters; correct the reported node.",
    "planar-ir.invalid-path-stroke": "Invalid constant-width path; correct the failed segment, width, or join.",
    "planar-ir.subtract-no-effect": (
        "Subtraction node removed no material; check coordinates. Rectangle x/y denotes the lower-left corner."
    ),
    "planar-ir.boolean-empty": "2D Boolean result is empty; adjust operands or dimensions.",
    "planar-ir.degenerate-result": "2D construction produces degenerate edges or regions; adjust dimension relationships.",
    "planar-ir.unsupported-boundary": "Result contains boundary curves unsupported by this version.",
    "planar-ir.materialization-failed": "Cannot materialize 2D boundaries as a strict sketch; model unchanged.",
    "planar-ir.profile-invalid": "Materialized sketch failed Profile topology proof.",
    "planar-ir.equivalence-failed": "Materialized sketch differs from the Boolean result; proposal blocked.",
    "planar-ir.feature-splits-material": (
        "Internal cuts disconnect the material region; adjust positions or dimensions using the reported nodes, boundary contacts, "
        "and bounding boxes."
    ),
    "planar-ir.transform-invalid": (
        "Invalid output parameters: for 2D use output: \"planar\" or "
        "output: {\"kind\": \"planar\"}; for 3D check transform parameters and source Profile."
    ),
    "planar-ir.preflight-failed": "Final 3D Recipe failed local exact preflight.",
    "planar-ir.stale-context": "Document, Session, or revision changed; reread the context.",
}


def _planar_construction_failure(
    diagnostic: object,
    request: Mapping[str, object],
    controller: AuthoringWorkflowController,
) -> AuthoringToolOutcome:
    code = str(getattr(diagnostic, "code"))
    allowed_fields = tuple(getattr(diagnostic, "allowed_fields"))
    raw_evidence = getattr(diagnostic, "evidence", ())
    evidence = (
        dict(raw_evidence)
        if isinstance(raw_evidence, (Mapping, tuple, list))
        else {}
    )
    retry = controller.assess_planar_construction_failure(
        request,
        code=code,
        node_id=getattr(diagnostic, "node_id"),
        retryable=bool(getattr(diagnostic, "retryable")),
        allowed_fields=allowed_fields,
        failure_evidence=evidence,
    )
    payload = {
        "code": code,
        "message": str(getattr(diagnostic, "message"))[:240],
        "node_id": getattr(diagnostic, "node_id"),
        "retryable": retry["retryable"],
        "allowed_fields": list(allowed_fields),
        "model_unchanged": True,
    }
    if evidence:
        payload["evidence"] = evidence
    user_message = _PLANAR_DIAGNOSTIC_MESSAGES.get(
        code,
        "2D construction failed local validation; model unchanged.",
    )
    if retry["blocker"] is not None:
        user_message = f"{user_message} {retry['blocker']}"
    return AuthoringToolOutcome(
        user_message,
        {
            "diagnostic": payload,
            "retry": retry,
            "authoring_path": "planar_construction_ir_v1",
            "required_action": "revise_same_planar_construction_ir",
        },
        ok=False,
    )

def _planar_construction_profile_selection(
    recipe: object,
    raw_selection: object,
) -> tuple[str, ...]:
    try:
        canonical = resolve_extrusion_source_faces(recipe).face_ids
    except (TypeError, ValueError) as error:
        raise AuthoringContractError(
            "profile-transform.topology-unproven: IR-derived Profile "
            "topology is not exact"
        ) from error
    if raw_selection == "unique_material_profile":
        if len(canonical) != 1:
            candidates = ", ".join(canonical) or "none"
            raise AuthoringContractError(
                "profile-transform.ambiguous-material-profiles: "
                "unique_material_profile requires exactly one material "
                f"Profile; candidates={candidates}"
            )
        return canonical
    if isinstance(raw_selection, Mapping):
        if set(raw_selection) != {"source_face_ids"}:
            raise ValueError("profile_selection fields do not match")
        raw_selection = raw_selection["source_face_ids"]
    if not isinstance(raw_selection, list) or not raw_selection:
        raise ValueError(
            "profile_selection must be unique_material_profile or an "
            "explicit source_face_ids array"
        )
    if any(not isinstance(item, str) for item in raw_selection):
        raise TypeError("source_face_ids must contain strings")
    requested = tuple(raw_selection)
    if len(requested) != len(set(requested)):
        raise AuthoringContractError(
            "profile-transform.invalid-source-id: source_face_ids must be unique"
        )
    unknown = tuple(item for item in requested if item not in canonical)
    if unknown:
        raise AuthoringContractError(
            "profile-transform.invalid-source-id: explicit IDs must be "
            f"canonical material Profile IDs; invalid={unknown[0]}"
        )
    return requested

def _planar_construction_output_recipe(
    planar_recipe: object,
    output: object,
    *,
    part_function: str,
) -> tuple[str, object, _DetachedRecipeMesh | None]:
    output = _normalize_planar_construction_output(output)
    if output == "planar":
        return "planar", planar_recipe, None
    assert isinstance(output, Mapping)
    kind = str(output["kind"])
    selected = _planar_construction_profile_selection(
        planar_recipe,
        output["profile_selection"],
    )
    if kind == "extrusion":
        extrusions = tuple(
            ExtrudedGeometry(planar_recipe, output["height"], (face_id,))
            for face_id in selected
        )
        recipe = (
            extrusions[0]
            if len(extrusions) == 1
            else MultiBodyGeometry(
                f"{part_function} Geometry",
                tuple(
                    SolidBody(f"B{index}", f"Body-{index}", extrusion)
                    for index, extrusion in enumerate(extrusions, start=1)
                ),
            )
        )
    elif kind == "revolution":
        if len(selected) != 1:
            raise AuthoringContractError(
                "profile-transform.ambiguous-material-profiles: revolution "
                "requires exactly one canonical material Profile"
            )
        recipe = RevolvedGeometry(
            planar_recipe,
            str(output["axis"]),
            output["angle_degrees"],
            selected,
        )
    else:
        if len(selected) != 1:
            raise AuthoringContractError(
                "profile-transform.ambiguous-material-profiles: path sweep "
                "requires exactly one canonical material Profile"
            )
        path = _composite_path(
            output["path"],
            name=f"SweepPath-{part_function}",
        )
        recipe = PathSweptGeometry(
            planar_recipe,
            path,
            selected,
            str(output["frame_strategy"]),
        )
    preview = _preflight_composite_geometry(recipe, include_preview=True)
    if preview is None:
        raise AuthoringContractError(
            "planar-ir.preflight-failed: detached preview was not produced"
        )
    return kind, recipe, preview

def _normalize_planar_construction_output(
    output: object,
) -> str | dict[str, object]:
    """Normalize the two accepted planar spellings before any CAD work."""

    if output == "planar":
        return "planar"
    if not isinstance(output, Mapping) or not isinstance(output.get("kind"), str):
        raise ValueError(
            "output must be 'planar', {'kind': 'planar'}, or a valid "
            "derived 3D output object"
        )
    kind = str(output["kind"])
    if kind == "planar":
        if set(output) != {"kind"}:
            raise ValueError(
                "planar output object must contain only kind='planar'"
            )
        return "planar"
    required = {
        "extrusion": {"kind", "profile_selection", "height"},
        "revolution": {
            "kind",
            "profile_selection",
            "axis",
            "angle_degrees",
        },
        "path_sweep": {
            "kind",
            "profile_selection",
            "path",
            "frame_strategy",
        },
    }.get(kind)
    if required is None:
        raise ValueError(
            "output.kind must be planar, extrusion, revolution, or path_sweep"
        )
    if set(output) != required:
        fields = ", ".join(sorted(required))
        raise ValueError(
            f"output kind {kind!r} requires exactly these fields: {fields}"
        )
    normalized = dict(output)
    selection = normalized["profile_selection"]
    if selection != "unique_material_profile":
        if isinstance(selection, list):
            source_ids = selection
        elif isinstance(selection, Mapping) and set(selection) == {
            "source_face_ids"
        }:
            source_ids = selection["source_face_ids"]
        else:
            raise ValueError(
                "profile_selection must be unique_material_profile, a "
                "source_face_ids array, or an object containing only "
                "source_face_ids"
            )
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(item, str) or not item for item in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError(
                "profile_selection.source_face_ids must be a non-empty "
                "array of unique strings"
            )
    if kind == "extrusion":
        height = normalized["height"]
        if (
            isinstance(height, bool)
            or not isinstance(height, (int, float))
            or height <= 0
        ):
            raise ValueError("extrusion height must be greater than zero")
    elif kind == "revolution":
        if normalized["axis"] not in {"x", "y", "z"}:
            raise ValueError("revolution axis must be x, y, or z")
        angle = normalized["angle_degrees"]
        if (
            isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not 0 < angle <= 360
        ):
            raise ValueError(
                "revolution angle_degrees must be greater than zero and at most 360"
            )
    else:
        if normalized["frame_strategy"] not in {"fixed", "transport"}:
            raise ValueError(
                "path_sweep frame_strategy must be fixed or transport"
            )
        _composite_path(normalized["path"], name="SweepPath-output-validation")
    return normalized


class GeometryCreationHandlers:
    def __init__(self, binding: AuthoringHandlerBinding) -> None:
        self.binding = binding

    def prepare_geometry(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"part_function", "geometry"}:
            raise AuthoringContractError(
                "prepare_geometry_proposal requires part_function and geometry"
            )
        part_function = str(arguments["part_function"]).strip()
        raw_geometry = arguments["geometry"]
        if not part_function or not isinstance(raw_geometry, Mapping):
            raise AuthoringContractError("part_function and geometry must be non-empty")
        geometry = dict(raw_geometry)
        kind = str(geometry.get("kind", ""))
        recipe_name = f"Wire-{part_function}" if kind == "wire" else f"Body-{part_function}"
        if kind == "wire":
            if set(geometry) != {"kind", "points", "members"}:
                raise ValueError("wire geometry fields do not match")
            raw_points = geometry["points"]
            raw_members = geometry["members"]
            if not isinstance(raw_points, list):
                raise ValueError("wire points must be an array")
            if not isinstance(raw_members, list):
                raise ValueError("wire members must be an array")
            points = []
            for raw_point in raw_points:
                if not isinstance(raw_point, Mapping) or set(raw_point) != {
                    "name",
                    "x",
                    "y",
                    "z",
                }:
                    raise ValueError("wire point fields do not match")
                points.append(
                    WirePoint(
                        raw_point["name"],
                        raw_point["x"],
                        raw_point["y"],
                        raw_point["z"],
                    )
                )
            members = []
            for raw_member in raw_members:
                if not isinstance(raw_member, Mapping) or set(raw_member) != {
                    "name",
                    "start",
                    "end",
                }:
                    raise ValueError("wire member fields do not match")
                members.append(
                    WireMember(
                        raw_member["name"],
                        raw_member["start"],
                        raw_member["end"],
                    )
                )
            draft = wire_geometry(
                recipe_name,
                points=points,
                members=members,
            )
            geometry_summary = f"1D spatial wire geometry(points={len(points)}, members={len(members)})"
        elif kind == "box":
            if set(geometry) != {"kind", "width", "depth", "height"}:
                raise ValueError("box geometry fields do not match")
            draft = box_geometry(
                recipe_name,
                width=geometry["width"],
                depth=geometry["depth"],
                height=geometry["height"],
            )
            geometry_summary = (
                f"3D box(width={_display_number(geometry['width'])}, "
                f"depth={_display_number(geometry['depth'])}, "
                f"height={_display_number(geometry['height'])})"
            )
        elif kind == "cylinder":
            if set(geometry) != {"kind", "radius", "height"}:
                raise ValueError("cylinder geometry fields do not match")
            draft = cylinder_geometry(
                recipe_name,
                radius=geometry["radius"],
                height=geometry["height"],
            )
            geometry_summary = (
                f"3D cylinder(radius={_display_number(geometry['radius'])}, "
                f"height={_display_number(geometry['height'])})"
            )
        else:
            raise ValueError("unsupported geometry kind")
        requirements = controller.collected_requirements("geometry")
        defaulted_keys = controller.defaulted_requirement_keys("geometry")
        proposal_summary = (
            f"Design proposal: {geometry_summary}; units "
            f"{_geometry_unit_summary(requirements, defaulted_keys)}"
        )
        metadata = envelope(controller, "geometry")
        context = self.binding.current_context()
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_geometry_proposal(
            proposal_id=f"proposal-{suffix}",
            context=context,
            draft=draft,
            part_function=part_function,
            project_function=(
                part_function if context.binding.source_kind == "blank" else None
            ),
            summary=proposal_summary,
            unit_context=UnitContextSummary(
                str(requirements["length_unit"]),
                str(requirements["force_unit"]),
                str(requirements["stress_unit"]),
                convention=(
                    f"{requirements['force_unit']}-"
                    f"{requirements['length_unit']}-"
                    f"{requirements['stress_unit']}"
                ),
            ),
            **metadata,
        )
        return self.binding.proposal_outcome(
            proposal,
            summary=proposal_summary,
            impact="Confirm to create this geometry and refresh the GUI",
            confirm_label="Add part",
        )

    def prepare_planar_construction(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"part_function", "construction", "output"}:
            raise AuthoringContractError(
                "prepare_planar_construction_proposal fields do not match"
            )
        part_function = arguments["part_function"]
        raw_construction = arguments["construction"]
        if (
            type(part_function) is not str
            or not part_function.strip()
            or part_function != part_function.strip()
            or len(part_function) > 96
        ):
            raise AuthoringContractError(
                "part_function must be a bounded nonblank string"
            )
        if not isinstance(raw_construction, Mapping):
            raise AuthoringContractError("construction must be an object")

        retry_blocker = controller.planar_construction_retry_blocker()
        if retry_blocker is not None:
            return AuthoringToolOutcome(
                "2D construction reached the three-retry limit for this turn; resubmit next turn.",
                {
                    "retry": retry_blocker,
                    "authoring_path": "planar_construction_ir_v1",
                    "required_action": "restart_planar_construction_in_new_turn",
                },
                ok=False,
            )

        try:
            normalized_output = _normalize_planar_construction_output(
                arguments["output"]
            )
        except (TypeError, ValueError) as error:
            return _planar_construction_failure(
                geometry_runtime.PlanarIRDiagnostic(
                    "planar-ir.transform-invalid",
                    str(error),
                    None,
                    True,
                    ("output",),
                ),
                arguments,
                controller,
            )

        initial_context = self.binding.current_context()
        try:
            construction = PlanarConstructionIR.from_dict(raw_construction)
        except PlanarIRValidationError as error:
            return _planar_construction_failure(
                error.diagnostic,
                arguments,
                controller,
            )

        compile_cancel = threading.Event()

        def compile_detached() -> tuple[
            object, object, str, object, object
        ]:
            # Every CAD model of one compile opens on this single worker
            # thread (gmsh affinity); only plain data returns to the owner.
            factory = _cancellable_planar_model_factory(compile_cancel)
            local_compiled = compile_planar_construction(
                construction,
                model_factory=factory,
            )
            if compile_cancel.is_set():
                raise PlanarCompileCancelled(
                    "planar construction compile cancelled after sketch proof"
                )
            if (
                type(local_compiled.recipe) is not SketchGeometry
                or not local_compiled.recipe.is_strict
            ):
                raise AuthoringContractError(
                    "planar construction compiler returned no strict sketch"
                )
            local_feature_recipe = compile_planar_feature_recipe(
                construction,
                compiled=local_compiled,
                model_factory=factory,
            )
            if compile_cancel.is_set():
                raise PlanarCompileCancelled(
                    "planar construction compile cancelled after feature chain"
                )
            local_kind, local_recipe, local_mesh = (
                _planar_construction_output_recipe(
                    local_feature_recipe,
                    normalized_output,
                    part_function=part_function,
                )
            )
            return (
                local_compiled,
                local_feature_recipe,
                local_kind,
                local_recipe,
                local_mesh,
            )

        try:
            (
                compiled,
                feature_recipe,
                output_kind,
                output_recipe,
                output_mesh,
            ) = run_owned_worker(
                compile_detached,
                name="fem-agent-planar-compile",
                timeout_seconds=_PLANAR_COMPILE_WORKER_TIMEOUT_SECONDS,
                cancel_event=compile_cancel,
                # cancel_requested is lock-free: this dispatch holds the
                # controller lock, so cancel_turn cannot flip the stage yet.
                should_cancel=lambda: controller.cancel_requested,
            )
        except PlanarCompileCancelled:
            return AuthoringToolOutcome(
                "2D construction compilation cancelled; model unchanged.",
                {
                    "diagnostic": {
                        "code": "planar-ir.cancelled",
                        "message": (
                            "The planar construction compile stopped at a "
                            "cancellation checkpoint."
                        ),
                        "model_unchanged": True,
                    },
                    "authoring_path": "planar_construction_ir_v1",
                    "required_action": "wait_for_next_request",
                },
                ok=False,
            )
        except OwnedWorkerTimeout:
            return AuthoringToolOutcome(
                "2D construction compilation exceeded the local time limit; model unchanged.",
                {
                    "diagnostic": {
                        "code": "planar-ir.compile-timeout",
                        "message": (
                            "The planar construction compile exceeded its "
                            f"{_PLANAR_COMPILE_WORKER_TIMEOUT_SECONDS:.0f}s "
                            "worker budget."
                        ),
                        "model_unchanged": True,
                    },
                    "authoring_path": "planar_construction_ir_v1",
                    "required_action": "simplify_construction_and_retry",
                },
                ok=False,
            )
        except PlanarConstructionCompileError as error:
            return _planar_construction_failure(
                error.diagnostic,
                arguments,
                controller,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            code = (
                "planar-ir.preflight-failed"
                if "preflight" in str(error).casefold()
                else "planar-ir.transform-invalid"
            )
            return _planar_construction_failure(
                geometry_runtime.PlanarIRDiagnostic(
                    code,
                    str(error),
                    None,
                    True,
                    ("output",),
                ),
                arguments,
                controller,
            )

        current = self.binding.current_context()
        if (
            initial_context.binding.document_id,
            initial_context.binding.session_id,
            initial_context.binding.session_revision,
        ) != (
            current.binding.document_id,
            current.binding.session_id,
            current.binding.session_revision,
        ):
            return _planar_construction_failure(
                geometry_runtime.PlanarIRDiagnostic(
                    "planar-ir.stale-context",
                    "The document, session, or revision changed during compilation.",
                    None,
                    True,
                    ("construction",),
                ),
                arguments,
                controller,
            )

        evidence = planar_construction_proposal_evidence(
            construction,
            compiled,
            output_kind=output_kind,
            output_recipe=output_recipe,
        )
        source_preview = compiled.preview if output_mesh is None else output_mesh
        detached_preview = AgentProposalPreview(
            dimension=2 if output_kind == "planar" else 3,
            points=source_preview.points,
            faces=source_preview.faces,
            edges=source_preview.edges,
            recipe_digest=str(evidence["output_recipe_digest"]),
            proof_digest=str(evidence["output_proof_digest"]),
        )
        draft = geometry_draft(output_recipe)
        requirements = controller.collected_requirements("geometry")
        defaulted_keys = controller.defaulted_requirement_keys("geometry")
        construction_summary = evidence["construction_summary"]
        proof_summary = evidence["proof_summary"]
        output_label = {
            "planar": "2D planar construction",
            "extrusion": "3D Profile extrusion",
            "revolution": "3D Profile revolution",
            "path_sweep": "3D Profile path sweep",
        }[output_kind]
        proposal_summary = (
            f"Design proposal: {output_label}"
            f"(nodes={construction_summary['node_count']}, "
            f"material regions={proof_summary['material_profile_count']}, "
            f"holes={proof_summary['hole_count']}); units "
            f"{_geometry_unit_summary(requirements, defaulted_keys)}"
        )
        metadata = envelope(controller, "planar-construction")
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_geometry_proposal(
            proposal_id=f"proposal-{suffix}",
            context=current,
            draft=draft,
            part_function=part_function,
            project_function=(
                part_function if current.binding.source_kind == "blank" else None
            ),
            summary=proposal_summary,
            unit_context=UnitContextSummary(
                str(requirements["length_unit"]),
                str(requirements["force_unit"]),
                str(requirements["stress_unit"]),
                convention=(
                    f"{requirements['force_unit']}-"
                    f"{requirements['length_unit']}-"
                    f"{requirements['stress_unit']}"
                ),
            ),
            local_evidence=evidence,
            include_static_preview=False,
            **metadata,
        )
        outcome = self.binding.proposal_outcome(
            proposal,
            summary=proposal_summary,
            impact=(
                "Confirm to create this 2D geometry and refresh the GUI"
                if output_kind == "planar"
                else "Confirm to create the final 3D geometry directly and refresh the GUI"
            ),
            confirm_label="Add part",
            detached_preview=detached_preview,
            extra_data={
                "authoring_path": "planar_construction_ir_v1",
                "output_kind": output_kind,
                "construction_summary": construction_summary,
                "proof_summary": proof_summary,
                "construction_digest_short": str(evidence["construction_digest"])[:12],
                "recipe_proof_digest_short": str(evidence["recipe_proof_digest"])[:12],
                "output_recipe_digest_short": str(evidence["output_recipe_digest"])[
                    :12
                ],
                "output_proof_digest_short": str(evidence["output_proof_digest"])[:12],
            },
        )
        controller.record_planar_construction_proposal(
            str(evidence["construction_digest"]),
            proposal.proposal_id,
        )
        return outcome
