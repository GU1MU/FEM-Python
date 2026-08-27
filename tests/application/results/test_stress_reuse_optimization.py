from __future__ import annotations

from dataclasses import replace

import numpy as np

from fem.post.averaging import NodalAveragingPolicy
from fem.results import (
    FieldPosition,
    FieldRequest,
    ModelResult,
    ResultFrame,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
)
from fem.results._materializers import materialize_derived_fields
from fem.results.fields import ResultFieldId
from fem.results.registry import registry_entry_for
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _source(suffix: str = "") -> ResultSourceKey:
    suffix = str(suffix)
    return ResultSourceKey(
        result_id=f"stress-reuse-result{suffix}",
        session_id="stress-reuse-session",
        artifact_id="stress-reuse-artifact",
        model_revision=1,
        step_name="Step-1",
        run_id=f"stress-reuse-run{suffix}",
    )


def _keys(provider):
    element_nodal = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.ELEMENT_NODAL,
            )
        )
    )
    resolved_nodal = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.RESOLVED_NODAL,
            ),
            averaging_policy=NodalAveragingPolicy(),
        )
    )
    return element_nodal, resolved_nodal


def _one_frame_result(base: ModelResult) -> ModelResult:
    frame = ResultFrame(
        model=base.model,
        step=base.step,
        U=base.U,
        reactions=base.reactions,
        frame_index=1,
        load_factor=1.0,
        outputs=base.outputs,
    )
    return replace(base, frames=(frame,))


def test_resolved_nodal_reuses_ready_element_nodal_values_exactly() -> None:
    provider = build_result_provider(_source(), make_continuum_nodal_semantics_result())
    element_nodal_key, resolved_key = _keys(provider)
    provider = provider.apply(provider.materialize((element_nodal_key,)))

    fast = provider.materialize((resolved_key,)).fields[0]
    entry = registry_entry_for(provider.profile, resolved_key.request.field_id)
    reference = materialize_derived_fields(
        source=provider.source,
        result=provider.model_result,
        topology=provider.snapshot.topology,
        profile=provider.profile,
        targets=((resolved_key, entry),),
        existing_fields=(),
    )[0]

    assert fast.locations == reference.locations
    assert np.array_equal(fast.values, reference.values)


def test_one_frame_provider_reuses_ready_derived_fields_but_history_does_not() -> None:
    base = make_continuum_nodal_semantics_result()
    provider = build_result_provider(_source(), _one_frame_result(base))
    element_nodal_key, resolved_key = _keys(provider)
    provider = provider.apply(
        provider.materialize((element_nodal_key, resolved_key))
    )

    frame_provider = provider.frame_provider(1)
    assert frame_provider.field_status(element_nodal_key).state.value == "ready"
    assert frame_provider.field_status(resolved_key).state.value == "ready"

    second_frame = ResultFrame(
        model=base.model,
        step=base.step,
        U=base.U * 0.5,
        reactions=base.reactions,
        frame_index=2,
        load_factor=0.5,
        outputs={"frame": 2},
    )
    history = replace(base, frames=(
        _one_frame_result(base).frames[0],
        second_frame,
    ))
    history_provider = build_result_provider(_source("history"), history)
    history_element_key, _history_resolved_key = _keys(history_provider)
    history_provider = history_provider.apply(
        history_provider.materialize((history_element_key,))
    )
    history_frame = history_provider.frame_provider(1)
    assert history_frame.field_status(history_element_key).state.value == "lazy"
