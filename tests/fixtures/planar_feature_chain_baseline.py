"""Phase-0 baseline fixture for the planar feature-chain incremental compile plan.

Reconstruction of the failed-session IR ``plate_300x100_slot_shu``: a 300x100
plate with an S-groove (round path_stroke), an H-groove (polygon), a U-groove
(butt/round path_stroke) and four corner holes.  The top-level DifferenceNode
compiles into exactly seven chained cut features.

The original session IR is not in the repository, so this fixture is a
reconstruction that follows the textual description.  Deviations from the
plan's measured numbers (docs/2026-08-16-...-incremental-compilation-plan.md):

- area 27682.531374 vs plan 27682.983960 (-0.45 units, 0.0016%);
- direct-IR curves {line 42, circle 4, arc 12} vs plan {line 40, circle 4,
  arc 11} and feature-chain curves {line 42, circle 4, arc 19} vs plan
  {line 40, circle 4, arc 14}: exact curve counts depend on the lost stroke
  point lists; structural facts (bbox, 1 component, 7 holes, 7 chained cuts)
  match strictly;
- baseline counts cut/lineage/evidence = 35/35/70 (Phase-0 legacy): the plan
  table counted only the replay path; the legacy implementation counted every
  call inside one ``compile_planar_feature_recipe`` invocation (7 direct cuts
  + 21 step-side replays + 7 final-proof replays, evidence = 2x lineage),
  which is O(N^2) for the N=7 chain.  Phase 1 replaced the per-step replay
  with single-model incremental compilation; see the count constants below.

The fixture contains only IR and facts; no session paths or user data.
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


# --- Golden facts, frozen from the reconstructed IR (Phase-0 baseline) ---

BASELINE_CHAINED_CUT_COUNT = 7
BASELINE_AREA = 27682.531373945072
# OCC pads planar bounding boxes by 1e-7; within the plan's (0,0,300,100)+-1e-7.
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
# deterministic across runs; every Phase 1+ implementation must reproduce it.
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


# Instrumented counts over one compile_planar_feature_recipe call (compiled
# construction passed in): GeometryModel.cut invocations,
# resolve_planar_boolean_lineage calls and capture_planar_operand_evidence
# calls summed over fem.application.planar_boolean and
# fem.application.recipe_compiler.
#
# Phase-0 legacy baseline (full-chain replay per step) was 35/35/70.  Phase 1
# (single-model incremental chain build) removed the per-step replay, leaving
# only the final-proof replay.  Phase 2 removed that replay too by reusing the
# last-step live compiled carrier, so the whole-call window now IS the chain
# build alone:
#   chain build    = 7 incremental cuts / 7 lineage proofs / 14 evidence
#                    captures (the plan's O(N) targets 7/7/<=14);
#   final proof    = 0 replayed cuts / 0 lineage proofs / 0 evidence captures
#                    (reuses the live carrier; gone in Phase 2);
#   whole call     = 7/7/14.
BASELINE_CUT_COUNT = 7
BASELINE_LINEAGE_COUNT = 7
BASELINE_EVIDENCE_COUNT = 14

# New CAD models opened by compile_planar_feature_recipe's own feature-chain
# logic (models named "planar-feature-*"; the flatten sub-compilations open
# their own "planar-construction-*"/"planar-recipe-proof-*" models).  Phase 1
# opened 2 (chain build + final-proof replay); Phase 2 reuses the live carrier
# and opens exactly 1 (the chain model).
BASELINE_FEATURE_CHAIN_MODEL_COUNT = 1
