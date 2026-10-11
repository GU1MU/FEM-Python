"""GUI authoring handlers for profile transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Callable
from fem_agent.authoring import AuthoringContractError
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome, AuthoringToolPreparation
from fem_agent.diagnostics import profile_transform_diagnostic
from fem_agent.geometry_authoring import _as_strict_planar_sketch, profile_transform_context
from fem.geometry import (
    NATIVE_GEOMETRY_TYPES,
    geometry_dimension,
    resolve_extrusion_source_faces,
)
from .binding import AuthoringHandlerBinding


def _profile_preflight_outcome(
    outcome: AuthoringToolOutcome | AuthoringToolPreparation,
    *, operation: str, first_failed_member: str | None = None,
) -> AuthoringToolOutcome | AuthoringToolPreparation:
    if type(outcome) is not AuthoringToolPreparation:
        return outcome

    def finalize(result: object, error: BaseException | None) -> AuthoringToolOutcome:
        try:
            return outcome.finalize(result, error)
        except (AuthoringContractError, TypeError, ValueError) as failure:
            return _profile_transform_error(
                failure, operation=operation, first_failed_member=first_failed_member,
            )

    return replace(outcome, finalize=finalize)


def _strict_profile_recipe(part):
    try:
        try:
            return _as_strict_planar_sketch(part.geometry_recipe)
        except TypeError:
            pass
        recipe = part.geometry_recipe
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError("recipe is not native geometry")
        if geometry_dimension(recipe) != 2:
            raise ValueError("recipe is not planar")
        resolve_extrusion_source_faces(recipe)
        return recipe
    except (TypeError, ValueError) as error:
        raise AuthoringContractError(
            "profile-transform.source-not-planar: Profile transforms "
            "require a canonical planar sketch source"
        ) from error

def _profile_selection(
    arguments: Mapping[str, object],
    *,
    snapshot,
    part,
) -> tuple[str, ...]:
    raw_selection = arguments["profile_selection"]
    unique_selection = (
        raw_selection == "unique_material_profile"
        or (
            isinstance(raw_selection, Mapping)
            and set(raw_selection) == {"mode"}
            and raw_selection.get("mode") == "unique_material_profile"
        )
        or (
            isinstance(raw_selection, Mapping)
            and set(raw_selection) == {"kind"}
            and raw_selection.get("kind") == "unique_material_profile"
        )
    )
    if not unique_selection and "context_revision" not in arguments:
        raise AuthoringContractError(
            "profile-transform.stale-context: explicit source_face_ids "
            "require context_revision from read_profile_transform_context"
        )
    if "context_revision" in arguments:
        revision = arguments["context_revision"]
        if type(revision) is not int or revision != snapshot.session_revision:
            raise AuthoringContractError(
                "profile-transform.stale-context: transform context revision "
                "does not match the current Session"
            )
    context = profile_transform_context(
        part.geometry_recipe,
        part_id=str(part.id),
        session_revision=snapshot.session_revision,
    )
    operation_context = context.get("extrusion")
    if not isinstance(operation_context, Mapping) or not operation_context.get(
        "available", False
    ):
        blocking_code = (
            str(operation_context.get("blocking_code"))
            if isinstance(operation_context, Mapping)
            and operation_context.get("blocking_code")
            else "profile-transform.topology-unproven"
        )
        blocking_reason = (
            str(operation_context.get("blocking_reason"))
            if isinstance(operation_context, Mapping)
            and operation_context.get("blocking_reason")
            else "Profile topology is not available"
        )
        raise AuthoringContractError(f"{blocking_code}: {blocking_reason}")
    if not context["topology_exact"]:
        reason = str(
            context.get("extrusion", {}).get(
                "blocking_reason",
                "Profile topology is not exact",
            )
        )
        raise AuthoringContractError(
            f"profile-transform.topology-unproven: {reason}"
        )
    profiles = context.get("profiles")
    if not isinstance(profiles, list):
        raise AuthoringContractError(
            "profile-transform.no-material-profile: no canonical Profiles"
        )
    canonical = tuple(
        str(item["face_id"])
        for item in profiles
        if isinstance(item, Mapping) and isinstance(item.get("face_id"), str)
    )
    if unique_selection:
        if len(canonical) != 1 or int(context["material_profile_count"]) != 1:
            candidates = ", ".join(canonical[:32]) or "none"
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
    unknown_ids = tuple(item for item in requested if item not in canonical)
    if unknown_ids or int(context["material_profile_count"]) > len(canonical):
        if unknown_ids:
            raise AuthoringContractError(
                "profile-transform.invalid-source-id: explicit IDs must be "
                "canonical material Profile IDs from the current context; "
                f"invalid={unknown_ids[0]}"
            )
        raise AuthoringContractError(
            "profile-transform.topology-unproven: Profile context was truncated; "
            "reread the transform context before selecting IDs"
        )
    return requested

def _profile_transform_error(
    error: BaseException,
    *,
    operation: str = "Profile transform",
    required_fields: tuple[str, ...] | None = None,
    first_failed_member: str | None = None,
) -> AuthoringToolOutcome:
    """Normalize one transform failure without changing the Session."""

    raw_message = str(error).strip() or type(error).__name__
    code = "profile-transform.preflight-failed"
    detail = raw_message
    if raw_message.startswith("profile-transform."):
        prefix, separator, remainder = raw_message.partition(":")
        if prefix in {
            "profile-transform.part-not-found",
            "profile-transform.source-not-planar",
            "profile-transform.source-not-strict",
            "profile-transform.no-material-profile",
            "profile-transform.ambiguous-material-profiles",
            "profile-transform.invalid-profile",
            "profile-transform.invalid-source-id",
            "profile-transform.nonpositive-height",
            "profile-transform.invalid-path",
            "profile-transform.unsupported-frame",
            "profile-transform.topology-unproven",
            "profile-transform.unexpected-body-count",
            "profile-transform.stale-context",
            "profile-transform.preflight-failed",
        }:
            code = prefix
            detail = remainder.strip() if separator else prefix
    else:
        lowered = raw_message.casefold()
        if "preflight" in lowered or "occ" in lowered or "gmsh" in lowered:
            code = "profile-transform.preflight-failed"
        elif "frame_strategy" in lowered or "frame" in lowered:
            code = "profile-transform.unsupported-frame"
        elif any(
            marker in lowered
            for marker in (
                "path",
                "路径",
                "branch",
                "self-intersect",
                "open",
                "wire",
                "member",
            )
        ):
            code = "profile-transform.invalid-path"
        elif "height" in lowered or "高度" in lowered:
            code = "profile-transform.nonpositive-height"
        elif "body" in lowered or "volume" in lowered:
            code = "profile-transform.unexpected-body-count"
        elif "topolog" in lowered or "strict" in lowered:
            code = "profile-transform.topology-unproven"
        elif any(
            marker in lowered
            for marker in (
                "composite profile",
                "polygon vertices",
                "rectangle profile",
                "circle profile",
                "slot path",
                "slot width",
            )
        ):
            code = "profile-transform.invalid-profile"
        elif "source" in lowered or "profile" in lowered:
            code = "profile-transform.invalid-source-id"

    candidates: tuple[str, ...] = ()
    marker = "candidates="
    if marker in detail:
        candidate_text = detail.split(marker, 1)[1].split(";", 1)[0]
        candidates = tuple(
            item.strip().strip("()[]{}'\"")[:192]
            for item in candidate_text.split(",")
            if item.strip().strip("()[]{}'\"")
            and item.strip().strip("()[]{}'\"") != "none"
        )[:32]
    if first_failed_member is None:
        member_marker = "member "
        if member_marker in detail.casefold():
            fragment = detail.casefold().split(member_marker, 1)[1]
            first_failed_member = fragment.split()[0].strip("'\"`()[]{}:,")
    diagnostic = profile_transform_diagnostic(
        code,
        operation=operation,
        detail=detail,
        required_fields=required_fields,
        candidates=candidates,
        first_failed_member=first_failed_member,
    )
    payload = {"diagnostic": diagnostic.to_dict()}
    return AuthoringToolOutcome(diagnostic.message, payload, ok=False)

def _first_failed_path_member(value: object) -> str | None:
    """Return the first member that breaks the caller-provided order."""

    if not isinstance(value, Mapping):
        return None
    raw_members = value.get("members")
    if not isinstance(raw_members, list):
        return None
    members = [item for item in raw_members if isinstance(item, Mapping)]
    for previous, current in zip(members, members[1:]):
        if previous.get("end") != current.get("start"):
            name = current.get("name")
            return str(name) if name is not None else None
    return None


class ProfileTransformHandlers:
    def __init__(
        self,
        binding: AuthoringHandlerBinding,
        prepare_geometry_edit: Callable[[Mapping[str, object], AuthoringWorkflowController], AuthoringToolOutcome],
    ) -> None:
        self.binding = binding
        self.prepare_geometry_edit = prepare_geometry_edit

    def _profile_transform_part(self, part_id: object):
        normalized_part_id = str(part_id).strip()
        snapshot = self.binding.current_session().snapshot()
        if snapshot.source_kind != "native":
            raise AuthoringContractError(
                "profile-transform.part-not-found: current document is not a "
                "native authoring Session"
            )
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if str(candidate.id) == normalized_part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "profile-transform.part-not-found: part_id does not identify "
                "one active native Part"
            )
        return snapshot, part

    def read_profile_transform_context(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        try:
            if set(arguments) != {"part_id"}:
                raise ValueError("read_profile_transform_context fields do not match")
            snapshot, part = self._profile_transform_part(arguments["part_id"])
            data = profile_transform_context(
                part.geometry_recipe,
                part_id=str(part.id),
                session_revision=snapshot.session_revision,
            )
            data["part_name"] = str(part.name)
            data["active"] = str(part.id) == str(snapshot.active_part_id)
            provider_safe_authoring_payload(data)
            return AuthoringToolOutcome(
                "Profile transform context read locally.",
                data,
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="Read Profile transform context",
                required_fields=("part_id",),
            )

    def prepare_profile_extrusion(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        try:
            required = {"part_id", "profile_selection", "height"}
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing:
                missing = tuple(
                    field
                    for field in ("part_id", "profile_selection", "height")
                    if field not in arguments
                )
                missing_code = (
                    "profile-transform.nonpositive-height"
                    if "height" in missing
                    else (
                        "profile-transform.part-not-found"
                        if "part_id" in missing
                        else "profile-transform.ambiguous-material-profiles"
                    )
                )
                raise AuthoringContractError(
                    missing_code
                    + ": "
                    + (
                        "missing height"
                        if "height" in missing
                        else "missing required fields"
                    )
                )
            if extra:
                raise ValueError("prepare_profile_extrusion fields do not match")
            snapshot, part = self._profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            height = arguments["height"]
            base_recipe = _strict_profile_recipe(part)
            outcome = self.prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "extrude_profiles",
                        "source_face_ids": list(selected),
                        "height": height,
                    },
                },
                controller,
            )
            return _profile_preflight_outcome(outcome, operation="Profile extrusion")
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="Profile extrusion",
            )

    def prepare_profile_revolution(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        try:
            required = {"part_id", "profile_selection", "axis", "angle_degrees"}
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing or extra:
                raise AuthoringContractError(
                    "profile-transform.preflight-failed: missing required fields"
                )
            snapshot, part = self._profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: revolution "
                    "requires exactly one canonical material Profile"
                )
            base_recipe = _strict_profile_recipe(part)
            outcome = self.prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "revolve_profile",
                        "source_face_id": selected[0],
                        "axis": arguments["axis"],
                        "angle_degrees": arguments["angle_degrees"],
                    },
                },
                controller,
            )
            return _profile_preflight_outcome(outcome, operation="Profile revolution")
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(error, operation="Profile revolution")

    def prepare_profile_path_sweep(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome | AuthoringToolPreparation:
        try:
            required = {
                "part_id",
                "profile_selection",
                "path",
                "frame_strategy",
            }
            missing = required - set(arguments)
            extra = set(arguments) - required - {"context_revision"}
            if missing:
                if "frame_strategy" in missing:
                    raise AuthoringContractError(
                        "profile-transform.unsupported-frame: missing frame_strategy"
                    )
                raise AuthoringContractError(
                    "profile-transform.invalid-path: missing path"
                )
            if extra:
                raise ValueError("prepare_profile_path_sweep fields do not match")
            snapshot, part = self._profile_transform_part(arguments["part_id"])
            selected = _profile_selection(arguments, snapshot=snapshot, part=part)
            if len(selected) != 1:
                raise AuthoringContractError(
                    "profile-transform.ambiguous-material-profiles: path sweep "
                    "requires exactly one canonical material Profile"
                )
            base_recipe = _strict_profile_recipe(part)
            outcome = self.prepare_geometry_edit(
                {
                    "part_id": str(part.id),
                    "_canonical_base_recipe": base_recipe,
                    "edit": {
                        "operation": "path_sweep_profile",
                        "source_face_id": selected[0],
                        "path": arguments["path"],
                        "frame_strategy": arguments["frame_strategy"],
                    },
                },
                controller,
            )
            return _profile_preflight_outcome(
                outcome,
                operation="Profile path sweep",
                first_failed_member=_first_failed_path_member(arguments.get("path")),
            )
        except (AuthoringContractError, TypeError, ValueError) as error:
            return _profile_transform_error(
                error,
                operation="Profile path sweep",
                first_failed_member=_first_failed_path_member(arguments.get("path")),
            )
