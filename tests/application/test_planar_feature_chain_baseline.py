"""Regression oracle for the planar feature-chain compilation baseline.

Freezes the golden facts, the structural recipe fingerprint and the
whole-call instrumented counts of the reconstructed
``plate_300x100_slot_shu`` IR (see tests/fixtures/planar_feature_chain_baseline.py)
so the incremental-compilation plan phases can prove semantic equivalence.
After Phase 2 the chain build is O(N) and IS the whole call (7 cuts / 7
lineage proofs / 14 evidence captures); the final proof reuses the last-step
live carrier instead of replaying the recipe, giving 7/7/14 for the whole
``compile_planar_feature_recipe`` call.  No timing assertions: CI wall clocks
are unstable, the time budget lives in the plan document.
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
from fem.geometry import model as geometry_model
from fem.geometry._gmsh.model import GeometryModel
from tests.fixtures.planar_feature_chain_baseline import (
    BASELINE_AREA,
    BASELINE_BOUNDING_BOX,
    BASELINE_CHAINED_CUT_COUNT,
    BASELINE_COMPONENT_COUNT,
    BASELINE_CUT_COUNT,
    BASELINE_DIRECT_CURVE_TYPE_COUNTS,
    BASELINE_EVIDENCE_COUNT,
    BASELINE_FEATURE_CHAIN_MODEL_COUNT,
    BASELINE_FEATURE_CURVE_TYPE_COUNTS,
    BASELINE_FEATURE_RECIPE_SHA256,
    BASELINE_HOLE_COUNT,
    BASELINE_LINEAGE_COUNT,
    feature_recipe_fingerprint,
    plate_300x100_slot_shu,
)

# The shared feature-recipe run still includes the final-proof full-chain
# replay, so it costs tens of seconds and stays behind the slow opt-in.
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
    ``resolve_planar_boolean_lineage`` and ``capture_planar_operand_evidence``,
    a spy that captures the final-proof facts, and a thin counting wrapper
    around ``geometry_model`` that records every CAD model the call opens.
    """

    global _feature_run_cache
    if _feature_run_cache is not None:
        return _feature_run_cache

    counts = {"cut": 0, "lineage": 0, "evidence": 0}
    feature_facts: dict[str, object] = {}
    created_models: list[str] = []

    def counting_model_factory(*args, **kwargs):
        created_models.append(args[0] if args else kwargs.get("name", ""))
        return geometry_model(*args, **kwargs)

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
        recipe = compile_planar_feature_recipe(
            _CONSTRUCTION,
            compiled=compiled,
            model_factory=counting_model_factory,
        )
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
        "created_models": created_models,
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

    # Structural equivalence with the legacy implementation: same operation,
    # feature IDs, selected logical faces and operand sketches at every node.
    assert feature_recipe_fingerprint(recipe) == BASELINE_FEATURE_RECIPE_SHA256

    facts = run["feature_facts"]
    assert facts.curve_types == BASELINE_FEATURE_CURVE_TYPE_COUNTS
    assert math.isclose(facts.area, BASELINE_AREA, rel_tol=1.0e-9, abs_tol=1.0e-6)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-7)
        for left, right in zip(facts.bounding_box, BASELINE_BOUNDING_BOX)
    )
    assert facts.component_count == BASELINE_COMPONENT_COUNT
    assert facts.hole_count == BASELINE_HOLE_COUNT


def test_instrumented_counts_match_incremental_window(real_gmsh) -> None:
    del real_gmsh
    run = _feature_run()
    counts = run["counts"]

    # Phase-2 whole-call window for the N=7 chain: the call IS the chain
    # build (7 incremental cuts / 7 lineage proofs / 14 evidence captures =
    # the plan's O(N) targets 7/7/<=14).  The final proof reuses the last-step
    # live carrier, so it adds no replayed cuts/proofs (evidence = 2x lineage).
    assert counts["cut"] == BASELINE_CUT_COUNT
    assert counts["lineage"] == BASELINE_LINEAGE_COUNT
    assert counts["evidence"] == BASELINE_EVIDENCE_COUNT
    assert counts["cut"] == BASELINE_CHAINED_CUT_COUNT

    # Phase 2 opens only the shared chain model for the feature chain; the
    # final proof reuses the live carrier instead of opening a proof model.
    # (Flatten sub-compilations open their own planar-construction models and
    # are deliberately not counted here.)
    feature_models = [
        name
        for name in run["created_models"]
        if name.startswith("planar-feature-")
    ]
    assert len(feature_models) == BASELINE_FEATURE_CHAIN_MODEL_COUNT
    assert feature_models == [
        f"planar-feature-chain-{_CONSTRUCTION.digest()[:12]}"
    ]
