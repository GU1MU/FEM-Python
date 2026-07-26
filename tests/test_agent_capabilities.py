from fem.abaqus.contracts import STANDARD_LINE_SUBSET
from fem_agent.capabilities import (
    CapabilityDisposition,
    SUPPORTED_ELEMENT_TYPES,
    keyword_capability,
    show_capabilities,
)


def test_capability_declaration_is_versioned_and_provider_safe():
    capabilities = show_capabilities()

    assert capabilities["schema_version"] == 1
    assert capabilities["raw_input_sent_to_provider"] is False
    assert capabilities["arbitrary_code_execution"] is False
    assert capabilities["post_solve_result_queries"] is True
    assert capabilities["precomputed_result_queries_required"] is False
    assert capabilities["reusable_solution_state"] is True
    assert set(capabilities["element_types"]) == set(SUPPORTED_ELEMENT_TYPES)
    assert {"B31", "BEAM2", "T3D2", "TRUSS2"} <= set(
        capabilities["element_types"]
    )
    assert {"beam", "solid", "truss"} == set(capabilities["section_types"])
    assert {"QGLOBAL", "QLOCAL"} <= set(capabilities["dload_labels"])


def test_unknown_keywords_are_blocking_by_default():
    capability = keyword_capability("Dynamic")

    assert capability.disposition == CapabilityDisposition.BLOCKING


def test_output_keywords_publish_static_postprocess_categories():
    node = keyword_capability("Node Output")
    history = keyword_capability("History Output")

    assert node.disposition == CapabilityDisposition.POSTPROCESS_CANDIDATE
    assert history.disposition == CapabilityDisposition.PRESERVED_OUTPUT
    assert all(
        keyword_capability(name).disposition
        is CapabilityDisposition.POSTPROCESS_CANDIDATE
        for name in STANDARD_LINE_SUBSET.postprocess_candidate_keywords
    )
    assert all(
        keyword_capability(name).disposition
        is CapabilityDisposition.PRESERVED_OUTPUT
        for name in STANDARD_LINE_SUBSET.preserved_output_keywords
    )
