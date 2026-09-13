"""Independent geometry and compile-cost expectations for a seven-cut plate.

The reconstructed 300x100 plate has S, H, and U grooves plus four corner
holes. The original stroke coordinates were unavailable, so area and curve
counts describe this reconstruction, not the lost session. Bounding box,
component count, hole count, and chained-cut count match the original shape.

Keep the numerical expectations literal: deriving them from the compiler
under test would hide regressions. No session paths or user data are stored.
"""

from __future__ import annotations

import hashlib
import json

from fem.application.planar_construction import _recipe_digest
from fem.geometry import BooleanGeometry, SketchGeometry
from fem.geometry.construction_ir import PlanarConstructionIR


PLATE_SLOT_SHU_IR_DICT: dict[str, object] = {
    "schema_version": 1,
    "name": "plate_300x100_slot_shu",
    "plane": "XY",
    "nodes": [
        {
            "id": "plate",
            "kind": "rectangle",
            "x": 0.0,
            "y": 0.0,
            "width": 300.0,
            "height": 100.0,
        },
        {
            # S-groove: serpentine polyline approximation of the lost S stroke.
            "id": "s_slot",
            "kind": "path_stroke",
            "points": [
                [40.0, 50.0],
                [48.0, 67.119017],
                [56.0, 60.580135],
                [64.0, 39.419865],
                [72.0, 32.880983],
                [80.0, 50.0],
                [88.0, 67.119017],
                [96.0, 60.580135],
                [104.0, 39.419865],
                [112.0, 32.880983],
                [120.0, 50.0],
            ],
            "width": 4.55,
            "cap": "round",
            "join": "round",
        },
        {
            "id": "h_slot",
            "kind": "polygon",
            "vertices": [
                [150.0, 36.0],
                [158.0, 36.0],
                [158.0, 46.0],
                [182.0, 46.0],
                [182.0, 36.0],
                [190.0, 36.0],
                [190.0, 64.0],
                [182.0, 64.0],
                [182.0, 54.0],
                [158.0, 54.0],
                [158.0, 64.0],
                [150.0, 64.0],
            ],
        },
        {
            "id": "u_slot",
            "kind": "path_stroke",
            "points": [
                [225.0, 58.0],
                [225.0, 34.0],
                [255.0, 34.0],
                [255.0, 58.0],
            ],
            "width": 8.0,
            "cap": "butt",
            "join": "round",
        },
        *[
            {
                "id": f"hole_{x}_{y}",
                "kind": "circle",
                "center_x": float(x),
                "center_y": float(y),
                "radius": 5.0,
            }
            for x, y in ((15, 15), (285, 15), (15, 85), (285, 85))
        ],
        {
            # Seven subtract operands -> seven chained boolean cut features.
            "id": "result",
            "kind": "difference",
            "base": "plate",
            "subtract": [
                "s_slot",
                "h_slot",
                "u_slot",
                "hole_15_15",
                "hole_285_15",
                "hole_15_85",
                "hole_285_85",
            ],
        },
    ],
    "result_node_id": "result",
}


def plate_300x100_slot_shu() -> PlanarConstructionIR:
    """Parse the baseline IR dict into a validated PlanarConstructionIR."""

    return PlanarConstructionIR.from_dict(PLATE_SLOT_SHU_IR_DICT)


# Golden geometry facts frozen from the reconstructed IR.

BASELINE_CHAINED_CUT_COUNT = 7
BASELINE_AREA = 27682.531373945072
# OCC pads planar bounding boxes by 1e-7.
BASELINE_BOUNDING_BOX = (-1.0e-7, -1.0e-7, 300.0000001, 100.0000001)
BASELINE_COMPONENT_COUNT = 1
BASELINE_HOLE_COUNT = 7
# Final flattened boundary produced by compile_planar_construction.
BASELINE_DIRECT_CURVE_TYPE_COUNTS = (("arc", 12), ("circle", 4), ("line", 42))
# Boundary of the compiled 7-feature BooleanGeometry chain (arc count grows
# because native Boolean features split analytic curves at intersections).
BASELINE_FEATURE_CURVE_TYPE_COUNTS = (("arc", 19), ("circle", 4), ("line", 42))

# Structural equivalence oracle: fingerprint of the legacy implementation's
# feature recipe, computed over operation, feature_id, target_face_id,
# tool_face_ids, tool/object recursion and sketch digests.  Verified
# deterministic across runs; compiler changes must preserve it.
BASELINE_FEATURE_RECIPE_SHA256 = (
    "8ba0059e96cecd4b2aee61ef4edd5eb657a76585531014f127015ff2d868290d"
)


def feature_recipe_fingerprint(recipe: object) -> str:
    """Hash the structural shape of a feature recipe chain.

    Only authoring structure is hashed (operations, PB identities, selected
    logical faces and operand sketches); OCC-side proof details intentionally
    stay out of the oracle.
    """

    def walk(item: object) -> object:
        if isinstance(item, BooleanGeometry):
            context = item.planar_context
            return {
                "boolean": item.operation,
                "feature_id": None if context is None else context.feature_id,
                "target_face_id": (
                    None if context is None else context.target_face_id
                ),
                "tool_face_ids": (
                    None if context is None else list(context.tool_face_ids)
                ),
                "tool": walk(item.tool_geometry),
                "object": walk(item.object_geometry),
            }
        if isinstance(item, SketchGeometry):
            return {"sketch_digest": _recipe_digest(item)}
        raise TypeError(f"unexpected recipe node: {type(item)!r}")

    payload = json.dumps(walk(recipe), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# One compile_planar_feature_recipe call with a compiled construction:
# each cut resolves lineage once and captures both operands as evidence.
# Reusing the last live carrier avoids replaying the feature chain for proof.
BASELINE_CUT_COUNT = 7
BASELINE_LINEAGE_COUNT = 7
BASELINE_EVIDENCE_COUNT = 14

# Only the shared planar-feature-* chain model is opened; flattening models
# (planar-construction-* and planar-recipe-proof-*) are counted separately.
BASELINE_FEATURE_CHAIN_MODEL_COUNT = 1
