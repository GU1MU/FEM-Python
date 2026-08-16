"""Phase-0 regression oracle for the planar feature-chain compilation baseline.

Freezes the golden facts and the O(N^2) replay counts of the reconstructed
``plate_300x100_slot_shu`` IR (see tests/fixtures/planar_feature_chain_baseline.py)
so Phases 1-3 of the incremental-compilation plan can prove semantic
equivalence while reducing the instrumented counts.  No timing assertions:
CI wall clocks are unstable, the time budget lives in the plan document.
"""

from __future__ import annotations

import math

import pytest

from fem.application import planar_boolean as planar_boolean_module
from fem.application import planar_construction as planar_construction_module
from fem.application import recipe_compiler as recipe_compiler_module
from fem.application.planar_construction import (
    CompiledPlanarConstruction,
    compile_planar_construction,
    compile_planar_feature_recipe,
)
from fem.geometry import BooleanGeometry
from fem.geometry._gmsh.model import GeometryModel
from tests.fixtures.planar_feature_chain_baseline import (
    BASELINE_AREA,
    BASELINE_BOUNDING_BOX,
    BASELINE_CHAINED_CUT_COUNT,
    BASELINE_COMPONENT_COUNT,
    BASELINE_CUT_COUNT,
    BASELINE_DIRECT_CURVE_TYPE_COUNTS,
    BASELINE_EVIDENCE_COUNT,
    BASELINE_FEATURE_CURVE_TYPE_COUNTS,
    BASELINE_HOLE_COUNT,
    BASELINE_LINEAGE_COUNT,
    plate_300x100_slot_shu,
)

# The unoptimized baseline replays the whole chain per step, so the shared
# feature-recipe run costs ~2 minutes and stays behind the slow opt-in.
pytestmark = pytest.mark.slow

_CONSTRUCTION = plate_300x100_slot_shu()

_direct_compiled_cache: CompiledPlanarConstruction | None = None
_feature_run_cache: dict[str, object] | None = None


def _direct_compiled() -> CompiledPlanarConstruction:
    """Flatten the baseline IR once for the whole module (~2s)."""

    global _direct_compiled_cache
    if _direct_compiled_cache is None:
        _direct_compiled_cache = compile_planar_construction(_CONSTRUCTION)
    return _direct_compiled_cache


def _feature_run() -> dict[str, object]:
    """Run one fresh, instrumented ``compile_planar_feature_recipe`` call.

    The single compilation shared by the feature-fact and counting tests is
    measured with monkeypatch counters for ``GeometryModel.cut``,
    ``resolve_planar_boolean_lineage`` and ``capture_planar_operand_evidence``
    plus a spy that captures the final-proof facts.
    """

    global _feature_run_cache
    if _feature_run_cache is not None:
        return _feature_run_cache

    counts = {"cut": 0, "lineage": 0, "evidence": 0}
    feature_facts: dict[str, object] = {}

    original_cut = GeometryModel.cut
    original_lineage = planar_boolean_module.resolve_planar_boolean_lineage
    original_evidence = planar_boolean_module.capture_planar_operand_evidence
    original_require_equivalent = planar_construction_module._require_equivalent

    def counting_cut(self, *args, **kwargs):
        counts["cut"] += 1
        return original_cut(self, *args, **kwargs)

    def counting_lineage(*args, **kwargs):
        counts["lineage"] += 1
        return original_lineage(*args, **kwargs)

    def counting_evidence(*args, **kwargs):
        counts["evidence"] += 1
        return original_evidence(*args, **kwargs)

    def capturing_require_equivalent(source, recipe, node_id, *, strict_curves=True):
        if not strict_curves:
            feature_facts["facts"] = recipe
        return original_require_equivalent(
            source, recipe, node_id, strict_curves=strict_curves
        )

    compiled = _direct_compiled()
    GeometryModel.cut = counting_cut
    planar_boolean_module.resolve_planar_boolean_lineage = counting_lineage
    recipe_compiler_module.resolve_planar_boolean_lineage = counting_lineage
    planar_boolean_module.capture_planar_operand_evidence = counting_evidence
    recipe_compiler_module.capture_planar_operand_evidence = counting_evidence
    planar_construction_module._require_equivalent = capturing_require_equivalent
    try:
        recipe = compile_planar_feature_recipe(_CONSTRUCTION, compiled=compiled)
    finally:
        GeometryModel.cut = original_cut
        planar_boolean_module.resolve_planar_boolean_lineage = original_lineage
        recipe_compiler_module.resolve_planar_boolean_lineage = original_lineage
        planar_boolean_module.capture_planar_operand_evidence = original_evidence
        recipe_compiler_module.capture_planar_operand_evidence = original_evidence
        planar_construction_module._require_equivalent = original_require_equivalent

    _feature_run_cache = {
        "recipe": recipe,
        "counts": counts,
        "feature_facts": feature_facts["facts"],
    }
    return _feature_run_cache


def test_direct_construction_matches_golden_facts(real_gmsh) -> None:
    del real_gmsh
    proof = _direct_compiled().proof

    assert math.isclose(
        proof.area, BASELINE_AREA, rel_tol=1.0e-9, abs_tol=1.0e-6
    )
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-7)
        for left, right in zip(proof.bounding_box, BASELINE_BOUNDING_BOX)
    )
    assert proof.component_count == BASELINE_COMPONENT_COUNT
    assert proof.hole_count == BASELINE_HOLE_COUNT
    assert proof.curve_type_counts == BASELINE_DIRECT_CURVE_TYPE_COUNTS


def test_feature_chain_has_seven_chained_cuts_and_golden_curves(
    real_gmsh,
) -> None:
    del real_gmsh
    run = _feature_run()
    recipe = run["recipe"]

    chained_cuts = 0
    current = recipe
    while isinstance(current, BooleanGeometry):
        assert current.operation == "cut"
        chained_cuts += 1
        current = current.object_geometry
    assert chained_cuts == BASELINE_CHAINED_CUT_COUNT

    facts = run["feature_facts"]
    assert facts.curve_types == BASELINE_FEATURE_CURVE_TYPE_COUNTS
    assert math.isclose(facts.area, BASELINE_AREA, rel_tol=1.0e-9, abs_tol=1.0e-6)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-7)
        for left, right in zip(facts.bounding_box, BASELINE_BOUNDING_BOX)
    )
    assert facts.component_count == BASELINE_COMPONENT_COUNT
    assert facts.hole_count == BASELINE_HOLE_COUNT


def test_instrumented_baseline_counts_are_quadratic(real_gmsh) -> None:
    del real_gmsh
    counts = _feature_run()["counts"]

    # Phase-0 baseline for one compile_planar_feature_recipe call on the
    # N=7 chain: 7 direct cuts, 0+1+...+6 step-side replays and 7 final-proof
    # replays (see the fixture module for the deviation from the plan table).
    assert counts["cut"] == BASELINE_CUT_COUNT
    assert counts["lineage"] == BASELINE_LINEAGE_COUNT
    assert counts["evidence"] == BASELINE_EVIDENCE_COUNT
    assert min(counts.values()) > BASELINE_CHAINED_CUT_COUNT
