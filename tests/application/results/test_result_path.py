from __future__ import annotations

import pytest

from fem.results import (
    ResultPathRequest,
    build_result_path_result,
)
from tests.application.results.test_xy_data import _provider


def test_result_path_samples_current_field_and_preserves_source_locations() -> None:
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None

    result = build_result_path_result(
        provider.frame_provider(provider.frame_indices[0]),
        ResultPathRequest(
            selection,
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
            sample_count=3,
        ),
    )

    assert result.source == provider.source
    assert result.materialization_generation == provider.snapshot.generation
    assert [sample.distance for sample in result.samples] == [0.0, 0.5, 1.0]
    assert [sample.source_location.node_id for sample in result.samples] == [1, 1, 2]
    assert [sample.value for sample in result.samples] == [0.0, 0.0, 0.05]


def test_result_path_can_follow_a_polyline_of_mesh_nodes() -> None:
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None

    result = build_result_path_result(
        provider.frame_provider(provider.frame_indices[0]),
        ResultPathRequest(
            selection,
            start=(0.0, 0.0, 0.0),
            end=(0.0, 0.0, 0.0),
            sample_count=5,
            edge_node_ids=(1, 2, 3),
        ),
    )

    assert result.samples[0].coordinates == (0.0, 0.0, 0.0)
    assert result.samples[-1].coordinates == (0.0, 1.0, 0.0)
    assert result.samples[-1].distance == pytest.approx(1.0 + 2.0**0.5)
    assert [sample.distance for sample in result.samples] == sorted(
        sample.distance for sample in result.samples
    )


def test_result_path_rejects_a_degenerate_straight_path() -> None:
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None

    with pytest.raises(ValueError, match="不能重合"):
        build_result_path_result(
            provider.frame_provider(provider.frame_indices[0]),
            ResultPathRequest(
                selection,
                start=(0.0, 0.0, 0.0),
                end=(0.0, 0.0, 0.0),
                sample_count=3,
            ),
        )
