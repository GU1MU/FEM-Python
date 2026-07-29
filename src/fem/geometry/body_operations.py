"""Pure Body ownership operations for top-level multi-solid geometry."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .extrusion_selection import resolve_extrusion_source_faces
from .recipes import (
    BooleanBodyContext,
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    RevolvedGeometry,
    RotatedGeometry,
    SolidBody,
    geometry_dimension,
)


def next_body_id(recipe: object | None) -> str:
    """Allocate the first never-observed stable Body ID."""

    used, _features = _historical_ids(recipe)
    number = max((int(value[1:]) for value in used), default=0) + 1
    return f"B{number}"


def next_boolean_feature_id(recipe: object | None) -> str:
    """Allocate the first never-observed strict Boolean feature ID."""

    _bodies, used = _historical_ids(recipe)
    number = max((int(value[2:]) for value in used), default=0) + 1
    return f"BF{number}"


def materialize_multi_body(
    recipe: object,
    *,
    geometry_name: str | None = None,
    first_body_name: str = "Body-1",
    retired_body_ids: tuple[str, ...] = (),
    retired_boolean_feature_ids: tuple[str, ...] = (),
) -> MultiBodyGeometry:
    """Promote one legacy 3D recipe into canonical independent Bodies."""

    if isinstance(recipe, MultiBodyGeometry):
        return recipe
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("recipe must be a native geometry recipe")
    if geometry_dimension(recipe) != 3:
        raise ValueError("only 3D geometry can be promoted to MultiBodyGeometry")
    recipes = _singleton_solid_recipes(recipe)
    multi_name_prefix = (
        first_body_name[:-1]
        if first_body_name.endswith("1")
        else "Body-"
    )
    observed_numbers = tuple(int(value[1:]) for value in retired_body_ids)
    first_number = max(observed_numbers, default=0) + 1
    bodies = tuple(
        SolidBody(
            f"B{first_number + index - 1}",
            (
                first_body_name
                if len(recipes) == 1
                else f"{multi_name_prefix}{first_number + index - 1}"
            ),
            body_recipe,
        )
        for index, body_recipe in enumerate(recipes, start=1)
    )
    return MultiBodyGeometry(
        geometry_name or f"{getattr(recipe, 'name', 'Part-1')} Geometry",
        bodies,
        retired_body_ids,
        retired_boolean_feature_ids,
    )


def add_solid_body(
    geometry: MultiBodyGeometry,
    recipe: object,
    *,
    name: str | None = None,
) -> MultiBodyGeometry:
    """Append one or more singleton solid recipes with stable identities."""

    _require_multi_body(geometry)
    if isinstance(recipe, MultiBodyGeometry):
        raise ValueError("a MultiBodyGeometry cannot be nested as one Body")
    recipes = _singleton_solid_recipes(recipe)
    used_names = {body.name for body in geometry.bodies}
    bodies = list(geometry.bodies)
    next_number = int(next_body_id(geometry)[1:])
    for offset, body_recipe in enumerate(recipes):
        body_id = f"B{next_number + offset}"
        requested = name if len(recipes) == 1 else None
        body_name = (
            requested
            if requested is not None
            else _available_body_name(
                "Body",
                frozenset(used_names),
                suffix=next_number + offset,
            )
        )
        body = SolidBody(body_id, body_name, body_recipe)
        if body.name in used_names:
            raise ValueError(f"Body name {body.name!r} already exists")
        used_names.add(body.name)
        bodies.append(body)
    return replace(geometry, bodies=tuple(bodies))


def rename_solid_body(
    geometry: MultiBodyGeometry,
    body_id: str,
    name: str,
) -> MultiBodyGeometry:
    """Rename one Body without changing topology or stable identity."""

    body = geometry.body(body_id)
    replacement = replace(body, name=name)
    return replace(
        geometry,
        bodies=tuple(
            replacement if candidate.id == body.id else candidate
            for candidate in geometry.bodies
        ),
    )


def delete_solid_body(
    geometry: MultiBodyGeometry,
    body_id: str,
) -> MultiBodyGeometry | None:
    """Delete one Body; deleting the final Body clears native geometry."""

    body = geometry.body(body_id)
    remaining = tuple(
        candidate for candidate in geometry.bodies if candidate.id != body.id
    )
    historical_body_ids, historical_feature_ids = _historical_ids(body.recipe)
    retired_body_ids = tuple(
        sorted(
            {
                *geometry.retired_body_ids,
                body.id,
                *historical_body_ids,
            },
            key=lambda value: int(value[1:]),
        )
    )
    retired_feature_ids = tuple(
        sorted(
            {
                *geometry.retired_boolean_feature_ids,
                *historical_feature_ids,
            },
            key=lambda value: int(value[2:]),
        )
    )
    return (
        None
        if not remaining
        else replace(
            geometry,
            bodies=remaining,
            retired_body_ids=retired_body_ids,
            retired_boolean_feature_ids=retired_feature_ids,
        )
    )


def transform_solid_body(
    geometry: MultiBodyGeometry,
    body_id: str,
    *,
    move: tuple[float, float, float] | None = None,
    rotate: tuple[Literal["x", "y", "z"], float] | None = None,
) -> MultiBodyGeometry:
    """Apply exactly one rigid feature to one selected Body."""

    if (move is None) == (rotate is None):
        raise ValueError("provide exactly one of move or rotate")
    body = geometry.body(body_id)
    if move is not None:
        transformed = MovedGeometry(body.recipe, *move)
    else:
        axis, angle = rotate  # type: ignore[misc]
        transformed = RotatedGeometry(body.recipe, axis, angle)
    replacement = replace(body, recipe=transformed)
    return replace(
        geometry,
        bodies=tuple(
            replacement if candidate.id == body.id else candidate
            for candidate in geometry.bodies
        ),
    )


def provisional_body_boolean(
    geometry: MultiBodyGeometry,
    target_body_id: str,
    tool_body_id: str,
    operation: Literal["fuse", "cut"],
) -> BooleanGeometry:
    """Build an uncommitted strict Boolean recipe awaiting CAD proof."""

    if operation not in {"fuse", "cut"}:
        raise ValueError("Body Boolean operation must be fuse or cut")
    target = geometry.body(target_body_id)
    tool = geometry.body(tool_body_id)
    context = BooleanBodyContext(
        next_boolean_feature_id(geometry),
        target.id,
        tool.id,
        tool.name,
    )
    return BooleanGeometry(
        target.name,
        operation,
        target.recipe,
        tool.recipe,
        context,
    )


def install_proven_body_boolean(
    geometry: MultiBodyGeometry,
    recipe: BooleanGeometry,
) -> MultiBodyGeometry:
    """Consume the tool Body and install one proven target recipe."""

    context = recipe.body_context
    if context is None or not context.proven:
        raise ValueError("strict Body Boolean requires proven lineage")
    target = geometry.body(context.target_body_id)
    geometry.body(context.tool_body_id)
    replacement = SolidBody(target.id, target.name, recipe)
    return replace(
        geometry,
        bodies=tuple(
            replacement if body.id == target.id else body
            for body in geometry.bodies
            if body.id != context.tool_body_id
        ),
        retired_body_ids=tuple(
            sorted(
                {*geometry.retired_body_ids, context.tool_body_id},
                key=lambda value: int(value[1:]),
            )
        ),
    )


def undo_solid_body_feature(
    geometry: MultiBodyGeometry,
    body_id: str,
) -> MultiBodyGeometry:
    """Undo the selected Body's top feature, restoring a consumed tool."""

    body = geometry.body(body_id)
    recipe = body.recipe
    if isinstance(recipe, BooleanGeometry) and recipe.body_context is not None:
        context = recipe.body_context
        if any(candidate.id == context.tool_body_id for candidate in geometry.bodies):
            raise ValueError("Boolean undo cannot restore a duplicate tool Body ID")
        if any(candidate.name == context.tool_body_name for candidate in geometry.bodies):
            raise ValueError(
                "Boolean undo cannot restore a duplicate tool Body name"
            )
        restored_target = SolidBody(body.id, body.name, recipe.object_geometry)
        restored_tool = SolidBody(
            context.tool_body_id,
            context.tool_body_name,
            recipe.tool_geometry,
        )
        return replace(
            geometry,
            bodies=tuple(
                restored_target if candidate.id == body.id else candidate
                for candidate in geometry.bodies
            )
            + (restored_tool,),
            retired_body_ids=tuple(
                value
                for value in geometry.retired_body_ids
                if value != context.tool_body_id
            ),
            retired_boolean_feature_ids=tuple(
                sorted(
                    {
                        *geometry.retired_boolean_feature_ids,
                        context.feature_id,
                    },
                    key=lambda value: int(value[2:]),
                )
            ),
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        replacement = replace(body, recipe=recipe.base)
        return replace(
            geometry,
            bodies=tuple(
                replacement if candidate.id == body.id else candidate
                for candidate in geometry.bodies
            ),
        )
    raise ValueError("selected Body has no undoable top feature")


def _singleton_solid_recipes(recipe: object) -> tuple[object, ...]:
    if isinstance(recipe, (ExtrudedGeometry, RevolvedGeometry)):
        source_ids = resolve_extrusion_source_faces(
            recipe.base,
            recipe.source_face_ids,
        ).face_ids
        return tuple(
            replace(recipe, source_face_ids=(source_id,))
            for source_id in source_ids
        )
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("Body recipe must be a native geometry recipe")
    if geometry_dimension(recipe) != 3:
        raise ValueError("Body recipe must be three-dimensional")
    return (recipe,)


def _available_body_name(
    base: str,
    used: frozenset[str],
    *,
    suffix: int,
) -> str:
    candidate = f"{base}-{suffix}" if base == "Body" else (
        base if suffix == 1 else f"{base}-{suffix}"
    )
    if candidate not in used:
        return candidate
    number = max(1, suffix)
    while True:
        number += 1
        candidate = f"Body-{number}"
        if candidate not in used:
            return candidate


def _historical_ids(recipe: object | None) -> tuple[set[str], set[str]]:
    body_ids: set[str] = set()
    feature_ids: set[str] = set()

    def visit(item: object | None) -> None:
        if item is None:
            return
        if isinstance(item, MultiBodyGeometry):
            body_ids.update(item.retired_body_ids)
            feature_ids.update(item.retired_boolean_feature_ids)
            for body in item.bodies:
                body_ids.add(body.id)
                visit(body.recipe)
            return
        if isinstance(item, BooleanGeometry):
            if item.body_context is not None:
                body_ids.update(
                    (
                        item.body_context.target_body_id,
                        item.body_context.tool_body_id,
                    )
                )
                feature_ids.add(item.body_context.feature_id)
            visit(item.object_geometry)
            visit(item.tool_geometry)
            return
        if isinstance(
            item,
            (
                MovedGeometry,
                RotatedGeometry,
                ExtrudedGeometry,
                RevolvedGeometry,
            ),
        ):
            visit(item.base)

    visit(recipe)
    return body_ids, feature_ids


def historical_recipe_ids(
    recipe: object | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return every Body and strict-Boolean ID carried by one recipe."""

    body_ids, feature_ids = _historical_ids(recipe)
    return frozenset(body_ids), frozenset(feature_ids)


def retired_recipe_ids(
    recipe: object | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return IDs explicitly retired by a canonical Body container."""

    if not isinstance(recipe, MultiBodyGeometry):
        return frozenset(), frozenset()
    return (
        frozenset(recipe.retired_body_ids),
        frozenset(recipe.retired_boolean_feature_ids),
    )


def _require_multi_body(geometry: object) -> None:
    if type(geometry) is not MultiBodyGeometry:
        raise TypeError("geometry must be a MultiBodyGeometry")


__all__ = [
    "add_solid_body",
    "delete_solid_body",
    "historical_recipe_ids",
    "install_proven_body_boolean",
    "materialize_multi_body",
    "next_body_id",
    "next_boolean_feature_id",
    "provisional_body_boolean",
    "rename_solid_body",
    "retired_recipe_ids",
    "transform_solid_body",
    "undo_solid_body_feature",
]
