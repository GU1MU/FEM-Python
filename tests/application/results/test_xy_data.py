from __future__ import annotations

import numpy as np

from fem.results import (
    DynamicDiagnostics,
    DynamicFrameData,
    DynamicEnergy,
    ResultProbeKind,
    ResultProbeTarget,
    ResultSourceKey,
    ResultXYAxis,
    ResultXYRequest,
    build_result_provider,
    build_result_xy_series,
)
from fem.results import ModelResult, ResultFrame
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="xy-data",
        session_id="xy-data-session",
        artifact_id="xy-data-artifact",
        model_revision=1,
        step_name="Step-1",
        run_id="xy-data-run",
    )


def _provider():
    base = make_continuum_nodal_semantics_result()
    frames = tuple(
        ResultFrame(
            model=base.model,
            step=base.step,
            U=base.U * scale,
            reactions=np.zeros_like(base.reactions),
            frame_index=frame_index,
            load_factor=load_factor,
            outputs={"step_time": load_factor},
        )
        for frame_index, scale, load_factor in (
            (1, 0.5, 0.25),
            (3, 1.0, 0.75),
        )
    )
    result = ModelResult(
        model=base.model,
        step=base.step,
        U=base.U,
        reactions=base.reactions,
        frames=frames,
    )
    return build_result_provider(_source(), result)


def test_xy_series_reuses_exact_probe_semantics_and_frame_metadata() -> None:
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None
    request = ResultXYRequest(
        selection,
        ResultProbeTarget(ResultProbeKind.NODE, node_id=2),
        ResultXYAxis.LOAD_FACTOR,
    )

    series = build_result_xy_series(
        tuple(
            (frame_index, provider.frame_provider(frame_index))
            for frame_index in provider.frame_indices
        ),
        request,
        frame_catalog=provider.frame_catalog,
    )

    assert [point.frame_index for point in series.points] == [1, 3]
    assert [point.x_value for point in series.points] == [0.25, 0.75]
    assert [point.y_value for point in series.points] == [0.05, 0.1]


def test_xy_series_rejects_a_target_that_is_not_unique_in_a_frame() -> None:
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None
    request = ResultXYRequest(
        selection,
        ResultProbeTarget(ResultProbeKind.ELEMENT, element_id=1),
    )

    try:
        build_result_xy_series(
            tuple(
                (frame_index, provider.frame_provider(frame_index))
                for frame_index in provider.frame_indices
            ),
            request,
            frame_catalog=provider.frame_catalog,
        )
    except ValueError as error:
        assert "命中" in str(error)
    else:
        raise AssertionError("an element target must not silently choose a node")


def test_xy_series_uses_total_time_for_dynamic_frames() -> None:
    base = make_continuum_nodal_semantics_result()
    frames = tuple(
        ResultFrame(
            model=base.model,
            step=base.step,
            U=base.U * scale,
            reactions=np.zeros_like(base.reactions),
            frame_index=frame_index,
            load_factor=1.0,
            outputs={"increment_number": frame_index},
            dynamic_data=DynamicFrameData(
                solver_kind="dynamic_implicit",
                step_time=step_time,
                total_time=total_time,
                time_increment=0.1,
                energy=DynamicEnergy(total=scale),
                diagnostics=DynamicDiagnostics(
                    solver_kind="dynamic_implicit",
                    increment=frame_index,
                ),
            ),
        )
        for frame_index, scale, step_time, total_time in (
            (1, 0.5, 0.1, 1.1),
            (2, 1.0, 0.2, 1.2),
        )
    )
    result = ModelResult(
        model=base.model,
        step=base.step,
        U=base.U,
        reactions=base.reactions,
        frames=frames,
    )
    provider = build_result_provider(_source(), result)
    selection = provider.catalog().default_selection
    assert selection is not None
    series = build_result_xy_series(
        tuple(
            (frame_index, provider.frame_provider(frame_index))
            for frame_index in provider.frame_indices
        ),
        ResultXYRequest(
            selection,
            ResultProbeTarget(ResultProbeKind.NODE, node_id=2),
            ResultXYAxis.TOTAL_TIME,
        ),
        frame_catalog=provider.frame_catalog,
    )

    assert [point.x_value for point in series.points] == [1.1, 1.2]
