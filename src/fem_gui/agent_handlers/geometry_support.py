"""Geometry preflight and detached recipe helpers for GUI authoring."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from fem import geometry as geometry_runtime
from fem.application import (
    PlanarConstructionCancelled,
    prepare_part_boolean,
    prepare_solid_body_boolean,
)
from fem.application.recipe_compiler import compile_recipe
from fem_agent.authoring import AuthoringContractError
from fem_agent.diagnostics import PROFILE_TRANSFORM_DIAGNOSTIC_CODES
from fem.geometry import (
    ExtrudedGeometry,
    MultiBodyGeometry,
    PathSweptGeometry,
    RevolvedGeometry,
    WireMember,
    WireGeometry,
    WirePoint,
    describe_recipe_topology,
)


# One planar construction compile runs on one dedicated worker thread.
_PLANAR_COMPILE_WORKER_TIMEOUT_SECONDS = 110.0


@dataclass(frozen=True, slots=True)
class _DetachedRecipeMesh:
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]

class PlanarCompileCancelled(PlanarConstructionCancelled):
    """Raised at safe checkpoints when one planar compile is cancelled."""

def _cancellable_planar_model_factory(
    cancel_event: threading.Event,
) -> Callable[..., Any]:
    """Checkpoint every CAD model open so cancellation unwinds context managers.

    The compilers accept a ``model_factory``; wrapping it turns each model
    open into a cancellation safe point.  The raise happens before a new
    model is entered, and every previously opened model is already closed by
    its ``with`` block, so no live gmsh entity survives a cancellation.
    """

    def factory(*args: object, **kwargs: object) -> Any:
        if cancel_event.is_set():
            raise PlanarCompileCancelled(
                "planar construction compile cancelled before opening a CAD model"
            )
        return geometry_runtime.model(*args, **kwargs)

    return factory

def _normalize_profile_preflight_error(error: BaseException) -> AuthoringContractError:
    message = str(error).strip()
    prefix = message.split(":", 1)[0].strip()
    if (
        isinstance(error, AuthoringContractError)
        and prefix in PROFILE_TRANSFORM_DIAGNOSTIC_CODES
    ):
        return error
    return AuthoringContractError(
        "profile-transform.preflight-failed: native geometry preflight failed"
    )

def _run_profile_transform_preflight(
    callback: Callable[..., None], *args: object
) -> None:
    """Normalize backend failures at the detached preflight boundary."""

    try:
        callback(*args)
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error

def _preflight_profile_extrusions(
    recipes: tuple[ExtrudedGeometry, ...],
) -> None:
    """Compile selected Profiles in detached OCC models before proposal display."""

    def compile_all() -> None:
        for index, recipe in enumerate(recipes, start=1):
            with geometry_runtime.model(
                f"agent-extrusion-preflight-{index}",
                dimension=3,
            ) as cad:
                compile_recipe(cad, recipe)

    try:
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-extrusion-preflight",
        ) as executor:
            executor.submit(compile_all).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error

def _preflight_derived_geometry(
    recipe: RevolvedGeometry | PathSweptGeometry,
) -> None:
    """Compile OCC evidence off the GUI owner thread and finalize there."""

    def compile_one() -> None:
        with geometry_runtime.model(
            f"agent-{type(recipe).__name__}-preflight",
            dimension=3,
        ) as cad:
            compiled = compile_recipe(cad, recipe)
            if len(compiled.domain) != 1 or cad.volume(compiled.domain[0]) <= 0.0:
                raise AuthoringContractError(
                    "derived Profile preflight did not prove one positive volume"
                )

    try:
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-derived-preflight",
        ) as executor:
            executor.submit(compile_one).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error

def _preflight_composite_geometry(
    recipe: object,
    *,
    include_preview: bool = False,
) -> _DetachedRecipeMesh | None:
    """Prove exact, positive-volume Bodies before a blank proposal is shown."""

    try:
        topology = describe_recipe_topology(recipe)
        if not topology.exact:
            raise AuthoringContractError(
                "profile-transform.topology-unproven: composite recipe topology "
                "could not be proven exactly"
            )
        expected_bodies = len(topology.entities_of("body", selectable_only=True))
        if expected_bodies < 1:
            raise AuthoringContractError(
                "profile-transform.unexpected-body-count: composite recipe must "
                "describe at least one selectable Body"
            )

        def compile_one() -> _DetachedRecipeMesh | None:
            with geometry_runtime.model(
                f"agent-composite-{type(recipe).__name__}-preflight",
                dimension=3,
            ) as cad:
                compiled = compile_recipe(cad, recipe)
                if len(compiled.domain) != expected_bodies:
                    raise AuthoringContractError(
                        "profile-transform.unexpected-body-count: composite "
                        "preflight domain count does not match the exact topology"
                    )
                if any(cad.volume(item) <= 0.0 for item in compiled.domain):
                    raise AuthoringContractError(
                        "profile-transform.topology-unproven: composite preflight "
                        "did not prove positive volume"
                    )
                if not include_preview:
                    return None
                faces = tuple(compiled.boundary)
                edges = tuple(cad.boundary(faces, combined=False))
                points = tuple(cad.boundary(edges, combined=False))
                tessellation = cad.tessellate_surfaces(faces, edges, points)
                return _DetachedRecipeMesh(
                    tessellation.points,
                    tessellation.faces,
                    tessellation.edges,
                )

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-composite-preflight",
        ) as executor:
            return executor.submit(compile_one).result()
    except Exception as error:
        raise _normalize_profile_preflight_error(error) from error

def _preflight_part_boolean(
    target,
    tool,
    operation: str,
    *,
    result_part_id: str,
    feature_id: str,
    result_name: str,
):
    """Prepare a detached exact Part proof on the Phase-3 worker seam."""

    def compile_one():
        with geometry_runtime.model(
            f"agent-part-boolean-{operation}-preflight",
            dimension=3,
        ) as cad:
            return prepare_part_boolean(
                cad,
                target,
                tool,
                operation,
                result_part_id=result_part_id,
                feature_id=feature_id,
                result_name=result_name,
            )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-part-boolean-preflight",
    ) as executor:
        return executor.submit(compile_one).result()

def _preflight_body_boolean(
    geometry: MultiBodyGeometry,
    target_body_id: str,
    tool_body_id: str,
    operation: str,
    *,
    result_name: str,
):
    """Prepare a detached exact same-Part Body proof off the GUI thread."""

    def compile_one():
        with geometry_runtime.model(
            f"agent-body-boolean-{operation}-preflight",
            dimension=3,
        ) as cad:
            return prepare_solid_body_boolean(
                cad,
                geometry,
                target_body_id,
                tool_body_id,
                operation,
                result_name=result_name,
            )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-body-boolean-preflight",
    ) as executor:
        return executor.submit(compile_one).result()

def _display_number(value: object) -> str:
    normalized = float(value)
    if normalized == 0.0:
        normalized = 0.0
    return format(normalized, ".12g")

def _geometry_unit_summary(
    requirements: Mapping[str, object],
    defaulted_keys: tuple[str, ...],
) -> str:
    label = (
        f"{requirements['length_unit']}-"
        f"{requirements['force_unit']}-"
        f"{requirements['stress_unit']}"
    )
    if len(defaulted_keys) == 3:
        return f"{label} (default)"
    if defaulted_keys:
        names = {
            "length_unit": "length",
            "force_unit": "force",
            "stress_unit": "stress",
        }
        defaulted = ", ".join(names[key] for key in defaulted_keys)
        return f"{label} ({defaulted} defaulted)"
    return label

def _provider_recipe_kind(recipe: object | None) -> str | None:
    if recipe is None:
        return None
    return {
        "SketchGeometry": "planar_sketch",
        "PlateWithHoleGeometry": "planar_sketch",
        "RectangleGeometry": "planar_profile",
        "DiskGeometry": "planar_profile",
        "WireGeometry": "wire",
        "BoxGeometry": "solid_primitive",
        "CylinderGeometry": "solid_primitive",
        "MovedGeometry": "transformed_geometry",
        "RotatedGeometry": "transformed_geometry",
    }.get(type(recipe).__name__, "native_geometry")

def _composite_path(value: object, *, name: str) -> WireGeometry:
    if not isinstance(value, Mapping) or set(value) != {"points", "members"}:
        raise ValueError("composite path fields do not match")
    raw_points = value["points"]
    raw_members = value["members"]
    if not isinstance(raw_points, list) or not isinstance(raw_members, list):
        raise TypeError("composite path points and members must be arrays")
    if any(
        not isinstance(item, Mapping) or set(item) != {"name", "x", "y", "z"}
        for item in raw_points
    ):
        raise ValueError("composite path point fields do not match")
    if any(
        not isinstance(item, Mapping) or set(item) != {"name", "start", "end"}
        for item in raw_members
    ):
        raise ValueError("composite path member fields do not match")
    return WireGeometry(
        name,
        tuple(
            WirePoint(item["name"], item["x"], item["y"], item["z"])
            for item in raw_points
        ),
        tuple(
            WireMember(item["name"], item["start"], item["end"]) for item in raw_members
        ),
    )
