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


def test_unknown_keywords_are_blocking_by_default():
    capability = keyword_capability("Dynamic")

    assert capability.disposition == CapabilityDisposition.BLOCKING


def test_output_metadata_is_visible_but_not_physical_input():
    capability = keyword_capability("Node Output")

    assert capability.disposition == CapabilityDisposition.IGNORED
