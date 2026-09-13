"""Analytical seven-cut plate for inexpensive feature-chain regression checks.

Seven disjoint 1x2 rectangular holes preserve chain length and topology
coverage without the dense sampled curves of the former S/H/U groove model.
"""

from fem.geometry.construction_ir import PlanarConstructionIR


BASELINE_CHAINED_CUT_COUNT = 7
BASELINE_AREA = 286.0
BASELINE_BOUNDING_BOX = (-1.0e-7, -1.0e-7, 30.0000001, 10.0000001)
BASELINE_COMPONENT_COUNT = 1
BASELINE_HOLE_COUNT = 7
BASELINE_DIRECT_CURVE_TYPE_COUNTS = (("line", 32),)
BASELINE_FEATURE_CURVE_TYPE_COUNTS = (("line", 32),)
BASELINE_CUT_COUNT = 7
BASELINE_LINEAGE_COUNT = 7
BASELINE_EVIDENCE_COUNT = 14
BASELINE_FEATURE_CHAIN_MODEL_COUNT = 1


def seven_hole_plate() -> PlanarConstructionIR:
    return PlanarConstructionIR.from_dict({
        "schema_version": 1,
        "name": "seven-hole-plate",
        "plane": "XY",
        "nodes": [
            {"id": "plate", "kind": "rectangle", "x": 0, "y": 0,
             "width": 30, "height": 10},
            *[
                {"id": f"hole_{index}", "kind": "rectangle",
                 "x": 2 + 3 * index, "y": 4, "width": 1, "height": 2}
                for index in range(7)
            ],
            {"id": "result", "kind": "difference", "base": "plate",
             "subtract": [f"hole_{index}" for index in range(7)]},
        ],
        "result_node_id": "result",
    })
