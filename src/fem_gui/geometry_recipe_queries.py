"""Pure queries shared by coarse previews and persisted CAD replay."""

from __future__ import annotations

from typing import Literal

from fem.geometry import (
    BooleanGeometry, ExtrudedGeometry, MovedGeometry, MultiBodyGeometry,
    PathSweptGeometry, RevolvedGeometry, RotatedGeometry,
)


def contains_proven_strict_boolean(
    recipe: object, *, kind: Literal["part", "planar"] | None = None,
) -> bool:
    """Find authenticated Boolean lineage through transforms and Body recipes."""

    if isinstance(recipe, BooleanGeometry):
        contexts = (
            (recipe.part_context,)
            if kind == "part"
            else (recipe.planar_context,)
            if kind == "planar"
            else (recipe.body_context, recipe.planar_context, recipe.part_context)
        )
        return any(context is not None and context.proven for context in contexts) or (
            contains_proven_strict_boolean(recipe.object_geometry, kind=kind)
            or contains_proven_strict_boolean(recipe.tool_geometry, kind=kind)
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry)):
        return contains_proven_strict_boolean(recipe.base, kind=kind)
    if isinstance(recipe, MultiBodyGeometry):
        return any(contains_proven_strict_boolean(body.recipe, kind=kind) for body in recipe.bodies)
    return False
