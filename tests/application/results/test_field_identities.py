from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fem.application.results import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    PhysicalQuantity,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    field_materialization_sort_key,
)
from fem.post.averaging import NodalAveragingPolicy


def _field_id(
    variable: ResultVariable = ResultVariable.S,
    position: FieldPosition = FieldPosition.CENTROID,
) -> ResultFieldId:
    return ResultFieldId(variable, position)


def _request(
    *,
    variable: ResultVariable = ResultVariable.S,
    position: FieldPosition = FieldPosition.CENTROID,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
) -> FieldRequest:
    return FieldRequest(
        _field_id(variable, position),
        averaging_policy=policy,
        gauss_order=gauss_order,
    )


def _key(
    *,
    variable: ResultVariable = ResultVariable.S,
    position: FieldPosition = FieldPosition.CENTROID,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
    recovery_contract: int = 1,
) -> FieldMaterializationKey:
    return FieldMaterializationKey(
        _request(
            variable=variable,
            position=position,
            policy=policy,
            gauss_order=gauss_order,
        ),
        recovery_contract,
    )


def test_enum_members_are_the_canonical_wire_values() -> None:
    assert tuple(FieldAssociation) == (
        FieldAssociation.NODE,
        FieldAssociation.ELEMENT,
        FieldAssociation.INTEGRATION_POINT,
        FieldAssociation.ELEMENT_NODE,
        FieldAssociation.NODE_REGION,
        FieldAssociation.RESOLVED_NODAL,
    )
    assert tuple(item.value for item in PhysicalQuantity) == (
        "displacement",
        "rotation",
        "force",
        "moment",
        "stress",
        "strain",
    )
    assert tuple(item.value for item in ResultVariable) == (
        "U",
        "UR",
        "RF",
        "RM",
        "SF",
        "SM",
        "S",
        "LE",
    )
    assert tuple(item.value for item in FieldPosition) == (
        "node",
        "integration_point",
        "centroid",
        "element_nodal",
        "node_region",
        "resolved_nodal",
        "section_point",
        "section_end",
        "section_node_envelope",
    )


@pytest.mark.parametrize(
    ("variable", "positions"),
    (
        (ResultVariable.U, (FieldPosition.NODE,)),
        (ResultVariable.UR, (FieldPosition.NODE,)),
        (ResultVariable.RF, (FieldPosition.NODE,)),
        (ResultVariable.RM, (FieldPosition.NODE,)),
        (ResultVariable.SF, (FieldPosition.INTEGRATION_POINT,)),
        (ResultVariable.SM, (FieldPosition.INTEGRATION_POINT,)),
        (ResultVariable.LE, (FieldPosition.CENTROID,)),
        (
            ResultVariable.S,
            (
                FieldPosition.INTEGRATION_POINT,
                FieldPosition.CENTROID,
                FieldPosition.ELEMENT_NODAL,
                FieldPosition.NODE_REGION,
                FieldPosition.RESOLVED_NODAL,
                FieldPosition.SECTION_END,
                FieldPosition.SECTION_NODE_ENVELOPE,
            ),
        ),
    ),
)
def test_result_field_id_accepts_every_intrinsic_combination(
    variable: ResultVariable,
    positions: tuple[FieldPosition, ...],
) -> None:
    for position in positions:
        assert _field_id(variable, position) == ResultFieldId(
            variable=variable,
            position=position,
        )


@pytest.mark.parametrize(
    ("variable", "position"),
    (
        (ResultVariable.U, FieldPosition.CENTROID),
        (ResultVariable.UR, FieldPosition.SECTION_END),
        (ResultVariable.RF, FieldPosition.INTEGRATION_POINT),
        (ResultVariable.RM, FieldPosition.RESOLVED_NODAL),
        (ResultVariable.LE, FieldPosition.NODE),
        (ResultVariable.S, FieldPosition.NODE),
    ),
)
def test_result_field_id_rejects_intrinsically_invalid_combinations(
    variable: ResultVariable,
    position: FieldPosition,
) -> None:
    with pytest.raises(ValueError):
        _field_id(variable, position)


@pytest.mark.parametrize(
    ("variable", "position"),
    (
        ("S", FieldPosition.CENTROID),
        (ResultVariable.S, "centroid"),
        (PhysicalQuantity.STRESS, FieldPosition.CENTROID),
    ),
)
def test_result_field_id_requires_exact_enum_types(
    variable: object,
    position: object,
) -> None:
    with pytest.raises(TypeError):
        ResultFieldId(variable, position)  # type: ignore[arg-type]


def test_field_request_requires_and_preserves_exact_averaging_policy() -> None:
    policy = NodalAveragingPolicy(threshold_percent=42)
    request = _request(
        position=FieldPosition.RESOLVED_NODAL,
        policy=policy,
    )

    assert request.averaging_policy is policy
    with pytest.raises(ValueError, match="require"):
        _request(position=FieldPosition.RESOLVED_NODAL)
    with pytest.raises(ValueError, match="only valid"):
        _request(position=FieldPosition.NODE_REGION, policy=policy)
    with pytest.raises(TypeError, match="NodalAveragingPolicy"):
        FieldRequest(
            _field_id(ResultVariable.S, FieldPosition.RESOLVED_NODAL),
            averaging_policy=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "position",
    (
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
        FieldPosition.NODE_REGION,
        FieldPosition.RESOLVED_NODAL,
    ),
)
def test_gauss_order_is_allowed_for_continuum_stress_recovery_positions(
    position: FieldPosition,
) -> None:
    policy = (
        NodalAveragingPolicy()
        if position is FieldPosition.RESOLVED_NODAL
        else None
    )
    request = _request(position=position, policy=policy, gauss_order=17)

    assert request.gauss_order == 17


@pytest.mark.parametrize("gauss_order", (True, 1.0, "1"))
def test_gauss_order_rejects_bool_float_and_string(
    gauss_order: object,
) -> None:
    with pytest.raises(TypeError):
        _request(gauss_order=gauss_order)  # type: ignore[arg-type]


@pytest.mark.parametrize("gauss_order", (0, -1))
def test_gauss_order_rejects_non_positive_integers(
    gauss_order: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _request(gauss_order=gauss_order)


@pytest.mark.parametrize(
    ("variable", "position"),
    (
        (ResultVariable.S, FieldPosition.SECTION_END),
        (ResultVariable.S, FieldPosition.SECTION_NODE_ENVELOPE),
        (ResultVariable.LE, FieldPosition.CENTROID),
        (ResultVariable.U, FieldPosition.NODE),
    ),
)
def test_gauss_order_rejects_non_continuum_recovery_fields(
    variable: ResultVariable,
    position: FieldPosition,
) -> None:
    with pytest.raises(ValueError, match="continuum stress"):
        _request(variable=variable, position=position, gauss_order=2)


def test_identity_dataclasses_require_exact_nested_types() -> None:
    field_id = _field_id()
    request = _request()
    key = _key()

    with pytest.raises(TypeError, match="field_id"):
        FieldRequest(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="request"):
        FieldMaterializationKey(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="field_key"):
        ScalarFieldSelection(object(), "S11")  # type: ignore[arg-type]

    assert request.field_id is field_id or request.field_id == field_id
    assert key.request == request


@pytest.mark.parametrize("recovery_contract", (True, 1.0, "1"))
def test_recovery_contract_rejects_bool_float_and_string(
    recovery_contract: object,
) -> None:
    with pytest.raises(TypeError):
        FieldMaterializationKey(
            _request(),
            recovery_contract,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("recovery_contract", (0, -1))
def test_recovery_contract_rejects_non_positive_integer(
    recovery_contract: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        FieldMaterializationKey(_request(), recovery_contract)


def test_scalar_selection_keeps_the_complete_materialization_key() -> None:
    key = _key(gauss_order=3, recovery_contract=7)
    selection = ScalarFieldSelection(key, "S11")

    assert selection.field_key is key
    assert selection.component == "S11"
    for invalid in ("", " \t"):
        with pytest.raises(ValueError, match="blank"):
            ScalarFieldSelection(key, invalid)
    with pytest.raises(TypeError, match="string"):
        ScalarFieldSelection(key, 11)  # type: ignore[arg-type]


def test_result_source_key_requires_exact_nonblank_source_identity() -> None:
    source = ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=0,
        step_name="Step-1",
        run_id="run-1",
    )

    assert source.model_revision == 0
    for field_name in (
        "result_id",
        "session_id",
        "artifact_id",
        "step_name",
        "run_id",
    ):
        values = {
            "result_id": "result-1",
            "session_id": "session-1",
            "artifact_id": "artifact-1",
            "model_revision": 2,
            "step_name": "Step-1",
            "run_id": "run-1",
        }
        values[field_name] = " "
        with pytest.raises(ValueError, match=field_name):
            ResultSourceKey(**values)  # type: ignore[arg-type]
        values[field_name] = 1
        with pytest.raises(TypeError, match=field_name):
            ResultSourceKey(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_revision", (True, 1.0, "1"))
def test_result_source_key_requires_exact_revision_type(
    model_revision: object,
) -> None:
    with pytest.raises(TypeError, match="model_revision"):
        ResultSourceKey(
            "result-1",
            "session-1",
            "artifact-1",
            model_revision,  # type: ignore[arg-type]
            "Step-1",
            "run-1",
        )


def test_result_source_key_rejects_negative_revision() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ResultSourceKey(
            "result-1",
            "session-1",
            "artifact-1",
            -1,
            "Step-1",
            "run-1",
        )


def test_identity_values_are_frozen_and_hashable() -> None:
    source = ResultSourceKey("r", "s", "a", 0, "Step-1", "run")
    request = _request()
    key = FieldMaterializationKey(request, 1)
    selection = ScalarFieldSelection(key, "S11")

    assert len({source, request, key, selection}) == 4
    with pytest.raises(FrozenInstanceError):
        source.run_id = "different"  # type: ignore[misc]


def test_materialization_sort_key_uses_the_complete_contract_identity() -> None:
    keys = (
        _key(
            position=FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(75),
            recovery_contract=2,
        ),
        _key(gauss_order=2, recovery_contract=1),
        _key(
            position=FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(25),
            recovery_contract=1,
        ),
        _key(recovery_contract=2),
        _key(recovery_contract=1),
        _key(gauss_order=1, recovery_contract=1),
        _key(
            variable=ResultVariable.U,
            position=FieldPosition.NODE,
            recovery_contract=1,
        ),
    )

    assert sorted(keys, key=field_materialization_sort_key) == [
        keys[6],
        keys[4],
        keys[3],
        keys[5],
        keys[1],
        keys[2],
        keys[0],
    ]
    assert len(
        {field_materialization_sort_key(key) for key in keys}
    ) == len(keys)
    with pytest.raises(TypeError):
        field_materialization_sort_key(object())  # type: ignore[arg-type]


def test_materialization_sort_key_follows_registry_variable_order() -> None:
    strain = _key(
        variable=ResultVariable.LE,
        position=FieldPosition.CENTROID,
        recovery_contract=1,
    )
    stress = _key(
        variable=ResultVariable.S,
        position=FieldPosition.CENTROID,
        recovery_contract=1,
    )

    assert sorted(
        (stress, strain),
        key=field_materialization_sort_key,
    ) == [strain, stress]
