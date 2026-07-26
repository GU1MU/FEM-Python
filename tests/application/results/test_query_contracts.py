from __future__ import annotations

import numpy as np
import pytest

from fem.application.results import (
    FieldAssociation,
    FieldLocation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
)
from fem.application.results.query import (
    ResultQuery,
    ResultQueryRecord,
    ResultQueryResult,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    ResultRegionKey,
    make_result_region_signature,
)


def _source(suffix: str = "1") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result-{suffix}",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=4,
        step_name="Step-1",
        run_id=f"run-{suffix}",
    )


def _key() -> FieldMaterializationKey:
    return FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.RESOLVED_NODAL,
            ),
            averaging_policy=NodalAveragingPolicy(),
        ),
        recovery_contract=1,
    )


def _region(name: str = "steel") -> ResultRegionKey:
    return ResultRegionKey(
        make_result_region_signature({"material": name}),
        make_result_region_signature({"section": "solid"}),
    )


def _location(
    *,
    node_id: int = 10,
    region: ResultRegionKey | None = None,
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.RESOLVED_NODAL,
        coordinates=(0.0, 0.0, 0.0),
        displacement=(0.1, 0.0, 0.0),
        node_id=node_id,
        region_key=region or _region(),
        averaged=True,
    )


def test_query_preserves_typed_filter_order_and_empty_means_unfiltered() -> None:
    query = ResultQuery(
        field_key=_key(),
        component="Mises",
        node_ids=(30, 10),
        element_ids=(200, 100),
        region_keys=(_region("steel"), _region("aluminium")),
    )

    assert query.node_ids == (30, 10)
    assert query.element_ids == (200, 100)
    assert tuple(
        region.material_signature.canonical_json
        for region in query.region_keys
    ) == ('{"material":"steel"}', '{"material":"aluminium"}')
    assert ResultQuery(_key(), "S11").node_ids == ()


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"field_key": object()}, TypeError),
        ({"component": 1}, TypeError),
        ({"component": " "}, ValueError),
        ({"node_ids": [1]}, TypeError),
        ({"node_ids": (True,)}, TypeError),
        ({"node_ids": (0,)}, ValueError),
        ({"node_ids": (1, 1)}, ValueError),
        ({"element_ids": (-1,)}, ValueError),
        ({"region_keys": [_region()]}, TypeError),
        ({"region_keys": (object(),)}, TypeError),
        ({"region_keys": (_region(), _region())}, ValueError),
    ),
)
def test_query_rejects_ambiguous_or_forged_identity_filters(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "field_key": _key(),
        "component": "Mises",
    }
    arguments.update(changes)

    with pytest.raises(error):
        ResultQuery(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (True, TypeError),
        ("1", TypeError),
        (np.float64(np.nan), ValueError),
        (float("inf"), ValueError),
    ),
)
def test_query_record_requires_one_finite_real_scalar(
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ResultQueryRecord(
            source=_source(),
            location=_location(),
            value=value,  # type: ignore[arg-type]
        )


def test_query_result_keeps_record_order_and_exact_source_generation() -> None:
    source = _source()
    query = ResultQuery(_key(), "Mises")
    first = ResultQueryRecord(source, _location(node_id=30), np.float64(3))
    second = ResultQueryRecord(source, _location(node_id=10), 1)

    result = ResultQueryResult(
        source=source,
        materialization_generation=2,
        query=query,
        records=(first, second),
    )

    assert tuple(record.location.node_id for record in result.records) == (
        30,
        10,
    )
    assert tuple(record.value for record in result.records) == (3.0, 1.0)
    assert result.materialization_generation == 2


def test_query_result_rejects_foreign_records_and_invalid_generation() -> None:
    source = _source()
    query = ResultQuery(_key(), "Mises")
    foreign = ResultQueryRecord(_source("foreign"), _location(), 1.0)

    with pytest.raises(ValueError, match="source"):
        ResultQueryResult(source, 0, query, (foreign,))
    with pytest.raises(TypeError, match="generation"):
        ResultQueryResult(source, True, query, ())
    with pytest.raises(ValueError, match="generation"):
        ResultQueryResult(source, -1, query, ())
