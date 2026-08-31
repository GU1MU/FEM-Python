from __future__ import annotations

import pytest

from fem.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultAveragingOptions,
    ResultDeformationMode,
    ResultDisplayComputation,
    ResultDisplayQuery,
    ResultDisplayScope,
    ResultDisplayScopeKind,
    ResultFieldId,
    ResultFrameCatalog,
    ResultFrameKey,
    ResultFrameMetadata,
    ResultLegendMode,
    ResultLegendPolicy,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    build_result_field_topology_template,
    build_result_provider,
    display_computation_for_position,
    frame_catalog_from_model_result,
    prepare_result_export_snapshot,
    project_scalar_field_topology,
    project_scalar_field_topology_from_template,
)
from fem.model import FEMModel
from fem.results import ModelResult, ResultFrame
from tests.helpers.mesh_builders import make_selection_quad_mesh


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=1,
        step_name="Step-1",
        run_id="run-1",
    )


def _selection(
    position: FieldPosition = FieldPosition.ELEMENT_NODAL,
) -> ScalarFieldSelection:
    key = FieldMaterializationKey(
        FieldRequest(ResultFieldId(ResultVariable.S, position)),
        recovery_contract=1,
    )
    return ScalarFieldSelection(key, "Mises")


def test_frame_key_binds_frame_to_result_source() -> None:
    key = ResultFrameKey(_source(), 0)
    assert key.frame_index == 0
    assert key.source.step_name == "Step-1"


def test_frame_catalog_rejects_duplicate_or_unsorted_frames() -> None:
    source = _source()
    first = ResultFrameMetadata(ResultFrameKey(source, 0))
    duplicate = ResultFrameMetadata(ResultFrameKey(source, 0))
    with pytest.raises(ValueError, match="unique"):
        ResultFrameCatalog(source, (first, duplicate))

    later = ResultFrameMetadata(ResultFrameKey(source, 2))
    earlier = ResultFrameMetadata(ResultFrameKey(source, 1))
    with pytest.raises(ValueError, match="sorted"):
        ResultFrameCatalog(source, (later, earlier))


def test_frame_catalog_last_is_a_real_frame_not_a_final_alias() -> None:
    source = _source()
    frames = ResultFrameCatalog(
        source,
        (
            ResultFrameMetadata(ResultFrameKey(source, 0), description="Initial"),
            ResultFrameMetadata(
                ResultFrameKey(source, 1),
                increment_number=1,
                load_factor=0.5,
            ),
            ResultFrameMetadata(
                ResultFrameKey(source, 2),
                increment_number=2,
                load_factor=1.0,
            ),
        ),
    )
    assert frames.last is not None
    assert frames.last.key.frame_index == 2
    assert frames.last.increment_number == 2


def test_current_model_result_frames_adapt_without_inventing_final_frame() -> None:
    source = _source()
    mesh = make_selection_quad_mesh()
    model = FEMModel(mesh=mesh)
    zero = [0.0] * mesh.num_dofs
    result = ModelResult(
        model=model,
        step=None,
        U=zero,
        reactions=zero,
        frames=(
            ResultFrame(
                model=model,
                step=None,
                U=zero,
                reactions=zero,
                frame_index=1,
                load_factor=0.5,
                iterations=3,
                residual_norm=1.0e-10,
            ),
            ResultFrame(
                model=model,
                step=None,
                U=zero,
                reactions=zero,
                frame_index=2,
                load_factor=1.0,
                iterations=4,
                residual_norm=2.0e-11,
            ),
        ),
    )
    catalog = frame_catalog_from_model_result(source, result)
    assert catalog.frame_keys == (
        ResultFrameKey(source, 1),
        ResultFrameKey(source, 2),
    )
    assert catalog.last is not None
    assert catalog.last.key.frame_index == 2
    assert catalog.last.load_factor == 1.0


def test_frame_provider_carries_frame_identity_into_snapshot_and_export() -> None:
    source = _source()
    mesh = make_selection_quad_mesh()
    model = FEMModel(mesh=mesh)
    zero = [0.0] * mesh.num_dofs
    result = ModelResult(
        model=model,
        step=None,
        U=zero,
        reactions=zero,
        frames=(
            ResultFrame(
                model=model,
                step=None,
                U=zero,
                reactions=zero,
                frame_index=1,
                load_factor=0.5,
            ),
        ),
    )

    provider = build_result_provider(source, result)
    assert provider.frame_key is None

    frame_provider = provider.frame_provider(1)
    expected_key = ResultFrameKey(source, 1)
    assert frame_provider.frame_key == expected_key
    assert frame_provider.snapshot.frame_key == expected_key

    selection = frame_provider.catalog().default_selection
    assert selection is not None
    export = prepare_result_export_snapshot(
        frame_provider.snapshot,
        selection,
    )
    assert export.frame_key == expected_key
    topology = project_scalar_field_topology(export)
    assert topology.frame_key == expected_key
    query = ResultDisplayQuery(
        frame=expected_key,
        selection=selection,
    )
    queried_topology = project_scalar_field_topology(
        export,
        display_query=query,
    )
    assert queried_topology.display_query == query
    template = build_result_field_topology_template(
        queried_topology,
        export.field,
    )
    assert template.frame_key == expected_key
    rebound = project_scalar_field_topology_from_template(
        export,
        template,
        0.0,
        display_query=query,
    )
    assert rebound.display_query == query


def test_display_query_separates_scope_and_sampling_computation() -> None:
    source = _source()
    query = ResultDisplayQuery(
        frame=ResultFrameKey(source, 1),
        selection=_selection(),
        computation=ResultDisplayComputation.ELEMENT_NODAL_QUILT,
        scope=ResultDisplayScope(
            ResultDisplayScopeKind.ELEMENT_SET,
            "Set-1",
        ),
        deformation=ResultDeformationMode.DEFORMED,
        deformation_scale=2.0,
        averaging=ResultAveragingOptions(threshold_percent=60.0),
        legend=ResultLegendPolicy(ResultLegendMode.GLOBAL_STEP),
    )
    assert query.field_position is FieldPosition.ELEMENT_NODAL
    assert query.scope.name == "Set-1"
    assert query.averaging.threshold_percent == 60.0


def test_native_display_computation_resolves_quad4_sampling_meaning() -> None:
    assert (
        display_computation_for_position(FieldPosition.INTEGRATION_POINT)
        is ResultDisplayComputation.INTEGRATION_POINT_MARKERS
    )
    assert (
        display_computation_for_position(FieldPosition.CENTROID)
        is ResultDisplayComputation.CENTROID_CELLS
    )
    assert (
        display_computation_for_position(FieldPosition.ELEMENT_NODAL)
        is ResultDisplayComputation.ELEMENT_NODAL_QUILT
    )
    assert (
        display_computation_for_position(FieldPosition.RESOLVED_NODAL)
        is ResultDisplayComputation.NODAL_AVERAGED
    )
    assert (
        ResultDisplayQuery(
            frame=ResultFrameKey(_source(), 1),
            selection=_selection(FieldPosition.CENTROID),
        ).effective_computation
        is ResultDisplayComputation.CENTROID_CELLS
    )


def test_display_query_rejects_centroid_mode_for_non_centroid_field() -> None:
    with pytest.raises(ValueError, match="centroid field"):
        ResultDisplayQuery(
            frame=ResultFrameKey(_source(), 1),
            selection=_selection(FieldPosition.ELEMENT_NODAL),
            computation=ResultDisplayComputation.CENTROID_CELLS,
        )


def test_display_query_rejects_incoherent_integration_point_mode() -> None:
    with pytest.raises(ValueError, match="integration-point"):
        ResultDisplayQuery(
            frame=ResultFrameKey(_source(), 1),
            selection=_selection(FieldPosition.ELEMENT_NODAL),
            computation=ResultDisplayComputation.INTEGRATION_POINT_MARKERS,
        )


def test_display_query_rejects_unnamed_set_scope() -> None:
    with pytest.raises(ValueError, match="require a name"):
        ResultDisplayScope(ResultDisplayScopeKind.ELEMENT_SET)


def test_manual_legend_requires_valid_bounds() -> None:
    with pytest.raises(ValueError, match="requires both bounds"):
        ResultLegendPolicy(ResultLegendMode.MANUAL)
    with pytest.raises(ValueError, match="below maximum"):
        ResultLegendPolicy(ResultLegendMode.MANUAL, 2.0, 1.0)
