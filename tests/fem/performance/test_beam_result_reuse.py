from __future__ import annotations


import pytest

from fem.application.results import (
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    execute_output_requests,
)
from fem.core.model import (
    OutputRequest,
)
from fem.post.stress import beam as beam_stress
from fem.solvers import static_linear


from tests.helpers.beam_section_builders import (
    _inline_cantilever,
)


def test_sf_sm_and_s_share_one_constitutive_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = static_linear.solve(
        _inline_cantilever("rectangle", {"height": 0.4, "width": 0.2}),
        "Load",
    )
    provider = build_result_provider(
        ResultSourceKey(
            "shared-recovery-result",
            "shared-recovery-session",
            "shared-recovery-artifact",
            1,
            "Load",
            "shared-recovery-run",
        ),
        result,
    )
    original = beam_stress.recover_integration_point_stress
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        beam_stress,
        "recover_integration_point_stress",
        counted,
    )
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("SF", "SM", "S")),),
    )

    assert outcome.report.diagnostics == ()
    # SF, SM and four S fields share the same costly section recovery.
    assert calls == 1
    assert {
        field.key.request.field_id.variable
        for field in outcome.provider_draft.snapshot.fields
    }.issuperset({ResultVariable.SF, ResultVariable.SM, ResultVariable.S})
