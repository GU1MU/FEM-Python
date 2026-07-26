from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fem.application.results.fields import (
    FieldAssociation,
    FieldPosition,
    PhysicalQuantity,
    ResultFieldId,
    ResultVariable,
)
from fem.application.results.registry import (
    ElementResultProfile,
    FieldRecoveryKind,
    ResultModelFamily,
    catalog_diagnostics,
    catalog_entries,
    classify_result_element_types,
    classify_result_model,
    descriptor_for,
    registry_entry_for,
)
from fem.post.averaging import NodalAveragingPolicy


def _model(
    element_types: tuple[str, ...],
    *,
    dofs_per_node: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        mesh=SimpleNamespace(
            elements=tuple(
                SimpleNamespace(type=element_type)
                for element_type in element_types
            ),
            dofs_per_node=dofs_per_node,
        )
    )


def _entry_by_id(
    model: SimpleNamespace,
    variable: ResultVariable,
    position: FieldPosition,
):
    profile = classify_result_model(model)
    field_id = ResultFieldId(variable, position)
    return registry_entry_for(profile, field_id)


@pytest.mark.parametrize(
    ("element_types", "dofs_per_node", "expected_family"),
    (
        (("Quad4",), 2, ResultModelFamily.PLANE_CONTINUUM),
        (
            ("Quad4", "Tri3", "Quad8", "Tri6"),
            2,
            ResultModelFamily.PLANE_CONTINUUM,
        ),
        (("Hex8",), 3, ResultModelFamily.SOLID_CONTINUUM),
        (
            ("Hex8", "Tet4", "Hex20", "Tet10"),
            3,
            ResultModelFamily.SOLID_CONTINUUM,
        ),
        (("Truss2", "Truss2"), 3, ResultModelFamily.TRUSS),
        (("Beam2", "Beam2"), 6, ResultModelFamily.BEAM),
    ),
)
def test_classification_uses_exact_element_capability_families(
    element_types: tuple[str, ...],
    dofs_per_node: int,
    expected_family: ResultModelFamily,
) -> None:
    profile = classify_result_model(
        _model(element_types, dofs_per_node=dofs_per_node)
    )

    assert profile.family is expected_family
    assert profile.primary_compatible is True
    assert profile.stress_compatible is True
    assert profile.dofs_per_node == dofs_per_node


def test_classification_preserves_first_seen_canonical_type_order() -> None:
    profile = classify_result_model(
        _model(("Tri3", "Quad4", "Tri3"), dofs_per_node=2)
    )

    assert profile.canonical_element_types == ("Tri3", "Quad4")
    assert profile.element_families == ("plane_continuum",)


def test_expected_element_classification_matches_realized_model_profile() -> None:
    element_types = ("Tri3", "Quad4", "Tri3")

    expected = classify_result_element_types(
        element_types,
        dofs_per_node=2,
    )
    realized = classify_result_model(
        _model(element_types, dofs_per_node=2)
    )

    assert expected == realized
    assert catalog_entries(expected) == catalog_entries(realized)


@pytest.mark.parametrize(
    ("element_types", "dofs_per_node", "primary_compatible"),
    (
        (("Quad4", "Hex8"), 3, False),
        (("Truss2", "Beam2"), 6, False),
        (("Hex8", "Truss2"), 3, True),
        (("Quad4", "Truss2"), 3, False),
        (("Unknown42",), 3, False),
        ((), 3, False),
    ),
)
def test_mixed_or_unknown_families_have_no_partial_stress_registry(
    element_types: tuple[str, ...],
    dofs_per_node: int,
    primary_compatible: bool,
) -> None:
    profile = classify_result_model(
        _model(element_types, dofs_per_node=dofs_per_node)
    )

    assert profile.family is ResultModelFamily.MIXED_UNSUPPORTED
    assert profile.primary_compatible is primary_compatible
    assert profile.stress_compatible is False
    entries = catalog_entries(profile)
    assert all(
        entry.recovery_kind is FieldRecoveryKind.PRIMARY
        for entry in entries
    )
    assert {
        entry.descriptor.field_id.variable for entry in entries
    }.isdisjoint({ResultVariable.S, ResultVariable.LE})


def test_exact_common_dof_profile_can_publish_primary_for_stress_mixed_model() -> None:
    profile = classify_result_model(
        _model(("Hex8", "Truss2"), dofs_per_node=3)
    )

    assert profile.dof_labels == ("U1", "U2", "U3")
    assert [
        entry.descriptor.field_id.variable
        for entry in catalog_entries(profile)
    ] == [ResultVariable.U, ResultVariable.RF]


def test_mixed_stress_omission_has_one_stable_catalog_diagnostic() -> None:
    profile = classify_result_element_types(
        ("Hex8", "Truss2"),
        dofs_per_node=3,
    )

    diagnostics = catalog_diagnostics(profile)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    result_namespace = type(diagnostic).__module__.split(".")[-2]
    assert diagnostic.code == "result.catalog.stress_family_unsupported"
    assert diagnostic.severity == "warning"
    assert diagnostic.path == (
        result_namespace,
        "catalog",
        "variables",
        "S",
    )
    assert diagnostic.details == {
        "canonical_variable": "S",
        "model_family": "mixed_unsupported",
        "canonical_element_types": ("Hex8", "Truss2"),
        "element_families": ("solid_continuum", "truss"),
    }
    with pytest.raises(TypeError):
        diagnostic.details["model_family"] = "forged"  # type: ignore[index]


def test_supported_profile_has_no_catalog_diagnostics() -> None:
    profile = classify_result_element_types(("Quad4",), dofs_per_node=2)

    assert catalog_diagnostics(profile) == ()


@pytest.mark.parametrize(
    ("element_type", "dofs", "components"),
    (
        ("Quad4", 2, ("U1", "U2")),
        ("Hex8", 3, ("U1", "U2", "U3")),
        ("Truss2", 3, ("U1", "U2", "U3")),
    ),
)
def test_translational_primary_descriptors_publish_only_actual_dofs(
    element_type: str,
    dofs: int,
    components: tuple[str, ...],
) -> None:
    profile = classify_result_model(
        _model((element_type,), dofs_per_node=dofs)
    )
    entries = {
        entry.descriptor.field_id.variable: entry
        for entry in catalog_entries(profile)
        if entry.recovery_kind is FieldRecoveryKind.PRIMARY
    }

    assert tuple(entries) == (ResultVariable.U, ResultVariable.RF)
    assert entries[ResultVariable.U].descriptor.components == components
    assert entries[ResultVariable.U].descriptor.derived_components == (
        "Magnitude",
    )
    assert entries[ResultVariable.U].descriptor.default_component == "Magnitude"
    assert entries[ResultVariable.U].descriptor.quantity is (
        PhysicalQuantity.DISPLACEMENT
    )
    assert entries[ResultVariable.RF].descriptor.components == tuple(
        f"RF{component[-1]}" for component in components
    )
    assert entries[ResultVariable.RF].descriptor.derived_components == (
        "Magnitude",
    )
    assert entries[ResultVariable.RF].descriptor.default_component == "Magnitude"


def test_beam_primary_descriptors_use_canonical_ur_and_rm_names() -> None:
    profile = classify_result_model(
        _model(("Beam2",), dofs_per_node=6)
    )
    entries = {
        entry.descriptor.field_id.variable: entry.descriptor
        for entry in catalog_entries(profile)
        if entry.recovery_kind is FieldRecoveryKind.PRIMARY
    }

    assert tuple(entries) == (
        ResultVariable.U,
        ResultVariable.UR,
        ResultVariable.RF,
        ResultVariable.RM,
    )
    assert entries[ResultVariable.U].components == ("U1", "U2", "U3")
    assert entries[ResultVariable.UR].components == ("UR1", "UR2", "UR3")
    assert entries[ResultVariable.UR].derived_components == ()
    assert entries[ResultVariable.UR].default_component == "UR1"
    assert entries[ResultVariable.RF].components == ("RF1", "RF2", "RF3")
    assert entries[ResultVariable.RM].components == ("RM1", "RM2", "RM3")
    assert entries[ResultVariable.RM].derived_components == ()
    assert entries[ResultVariable.RM].default_component == "RM1"


def test_registry_order_is_contextual_stable_and_unique() -> None:
    expectations = (
        (("Quad4",), 2, (0, 2, 20, 21, 22, 23, 24)),
        (("Hex8",), 3, (0, 2, 20, 21, 22, 23, 24)),
        (("Truss2",), 3, (0, 2, 10, 20)),
        (("Beam2",), 6, (0, 1, 2, 3, 20, 21)),
    )

    for element_types, dofs, expected in expectations:
        profile = classify_result_model(
            _model(element_types, dofs_per_node=dofs)
        )
        entries = catalog_entries(profile)
        assert tuple(entry.descriptor.order for entry in entries) == expected
        assert len({
            entry.descriptor.field_id for entry in entries
        }) == len(entries)


def test_continuum_descriptors_use_canonical_tensor_column_order() -> None:
    plane = _entry_by_id(
        _model(("Quad4",), dofs_per_node=2),
        ResultVariable.S,
        FieldPosition.CENTROID,
    ).descriptor
    solid = _entry_by_id(
        _model(("Hex8",), dofs_per_node=3),
        ResultVariable.S,
        FieldPosition.CENTROID,
    ).descriptor

    assert plane.components == ("S11", "S22", "S33", "S12")
    assert solid.components == (
        "S11",
        "S22",
        "S33",
        "S12",
        "S23",
        "S13",
    )
    assert plane.derived_components == solid.derived_components == (
        "Mises",
        "MaxPrincipal",
        "MidPrincipal",
        "MinPrincipal",
    )
    assert plane.association is solid.association is FieldAssociation.ELEMENT
    assert plane.default_component == solid.default_component == "Mises"


def test_centroid_stress_descriptor_is_resolved_in_family_context() -> None:
    field_id = ResultFieldId(ResultVariable.S, FieldPosition.CENTROID)
    continuum_profile = classify_result_model(
        _model(("Hex8",), dofs_per_node=3)
    )
    truss_profile = classify_result_model(
        _model(("Truss2",), dofs_per_node=3)
    )

    continuum = descriptor_for(continuum_profile, field_id)
    truss = descriptor_for(truss_profile, field_id)

    assert continuum.field_id == truss.field_id == field_id
    assert continuum != truss
    assert continuum.components == (
        "S11",
        "S22",
        "S33",
        "S12",
        "S23",
        "S13",
    )
    assert continuum.derived_components == (
        "Mises",
        "MaxPrincipal",
        "MidPrincipal",
        "MinPrincipal",
    )
    assert truss.components == ("S11",)
    assert truss.derived_components == ("Mises",)
    assert truss.default_component == "Mises"


def test_truss_registry_separates_strain_and_stress_recovery() -> None:
    profile = classify_result_model(
        _model(("Truss2",), dofs_per_node=3)
    )
    strain = registry_entry_for(
        profile,
        ResultFieldId(ResultVariable.LE, FieldPosition.CENTROID),
    )
    stress = registry_entry_for(
        profile,
        ResultFieldId(ResultVariable.S, FieldPosition.CENTROID),
    )

    assert strain.recovery_kind is FieldRecoveryKind.TRUSS_STRAIN
    assert strain.descriptor.quantity is PhysicalQuantity.STRAIN
    assert strain.descriptor.components == ("LE11",)
    assert strain.descriptor.default_component == "LE11"
    assert stress.recovery_kind is FieldRecoveryKind.TRUSS_STRESS
    assert stress.descriptor.quantity is PhysicalQuantity.STRESS


def test_beam_registry_distinguishes_section_end_from_node_envelope() -> None:
    profile = classify_result_model(
        _model(("Beam2",), dofs_per_node=6)
    )
    section_end = registry_entry_for(
        profile,
        ResultFieldId(ResultVariable.S, FieldPosition.SECTION_END),
    )
    envelope = registry_entry_for(
        profile,
        ResultFieldId(
            ResultVariable.S,
            FieldPosition.SECTION_NODE_ENVELOPE,
        ),
    )

    assert section_end.recovery_kind is FieldRecoveryKind.BEAM_SECTION_END
    assert section_end.descriptor.association is FieldAssociation.ELEMENT_NODE
    assert envelope.recovery_kind is FieldRecoveryKind.BEAM_NODE_ENVELOPE
    assert envelope.descriptor.association is FieldAssociation.NODE
    for entry in (section_end, envelope):
        assert entry.descriptor.components == ("S11Max", "S11Min")
        assert entry.descriptor.derived_components == ("S11AbsMax",)
        assert entry.descriptor.default_component == "S11AbsMax"


def test_continuum_registry_maps_every_position_to_typed_association() -> None:
    profile = classify_result_model(
        _model(("Quad4",), dofs_per_node=2)
    )
    expected = {
        FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
        FieldPosition.CENTROID: FieldAssociation.ELEMENT,
        FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
        FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
        FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    }

    for position, association in expected.items():
        entry = registry_entry_for(
            profile,
            ResultFieldId(ResultVariable.S, position),
        )
        assert entry.descriptor.association is association
        assert entry.recovery_kind is FieldRecoveryKind.CONTINUUM_STRESS
        assert entry.descriptor.unit_label is None


def test_resolved_nodal_entry_owns_the_only_default_averaging_policy() -> None:
    profile = classify_result_model(
        _model(("Quad4",), dofs_per_node=2)
    )
    entries = catalog_entries(profile)
    resolved = registry_entry_for(
        profile,
        ResultFieldId(ResultVariable.S, FieldPosition.RESOLVED_NODAL),
    )

    assert resolved.default_averaging_policy == NodalAveragingPolicy()
    assert resolved.default_request().averaging_policy == NodalAveragingPolicy()
    assert resolved.default_key().recovery_contract == 1
    assert all(
        entry.default_averaging_policy is None
        for entry in entries
        if entry.descriptor.field_id != resolved.descriptor.field_id
    )


def test_registry_lookup_rejects_unknown_contextual_field() -> None:
    profile = classify_result_model(
        _model(("Truss2",), dofs_per_node=3)
    )

    with pytest.raises(KeyError):
        descriptor_for(
            profile,
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
            ),
        )


def test_classifier_rejects_mesh_dof_contract_mismatch() -> None:
    profile = classify_result_model(
        _model(("Beam2",), dofs_per_node=3)
    )

    assert profile.family is ResultModelFamily.MIXED_UNSUPPORTED
    assert profile.primary_compatible is False
    assert catalog_entries(profile) == ()


def _forged_profile(**changes: object) -> ElementResultProfile:
    values = {
        "family": ResultModelFamily.PLANE_CONTINUUM,
        "canonical_element_types": ("Quad4",),
        "element_families": ("plane_continuum",),
        "dofs_per_node": 2,
        "dof_labels": ("U1", "U2"),
        "force_labels": ("Fx", "Fy"),
        "primary_compatible": True,
        "stress_compatible": True,
    }
    values.update(changes)
    return ElementResultProfile(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"dofs_per_node": None},
        {"dof_labels": ("U1",)},
        {"force_labels": ("Fx",)},
        {
            "primary_compatible": False,
            "stress_compatible": False,
        },
    ),
)
def test_forged_profile_cannot_expose_an_inconsistent_dof_contract(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _forged_profile(**changes)


def test_forged_primary_incompatible_profile_requires_empty_dof_contract() -> None:
    profile = _forged_profile(
        family=ResultModelFamily.MIXED_UNSUPPORTED,
        canonical_element_types=("Hex8", "Truss2"),
        element_families=("solid_continuum", "truss"),
        dofs_per_node=None,
        dof_labels=(),
        force_labels=(),
        primary_compatible=False,
        stress_compatible=False,
    )

    assert profile.primary_compatible is False
    assert catalog_entries(profile) == ()


def test_forged_mixed_profile_cannot_claim_stress_compatibility() -> None:
    with pytest.raises(ValueError, match="stress-compatible"):
        _forged_profile(
            family=ResultModelFamily.MIXED_UNSUPPORTED,
            stress_compatible=True,
        )


@pytest.mark.parametrize("recovery_contract", (True, 1.0, "1"))
def test_registry_entry_recovery_contract_rejects_non_integer_types(
    recovery_contract: object,
) -> None:
    entry = catalog_entries(
        classify_result_model(_model(("Quad4",), dofs_per_node=2))
    )[0]

    with pytest.raises(TypeError, match="integer"):
        replace(entry, recovery_contract=recovery_contract)


@pytest.mark.parametrize("recovery_contract", (0, -1))
def test_registry_entry_recovery_contract_rejects_non_positive_values(
    recovery_contract: int,
) -> None:
    entry = catalog_entries(
        classify_result_model(_model(("Quad4",), dofs_per_node=2))
    )[0]

    with pytest.raises(ValueError, match="positive"):
        replace(entry, recovery_contract=recovery_contract)


@pytest.mark.parametrize("dofs_per_node", (True, 0, -1, 3.0, "3"))
def test_classifier_requires_a_strict_positive_mesh_dof_count(
    dofs_per_node: object,
) -> None:
    with pytest.raises(TypeError):
        classify_result_model(
            _model(
                ("Hex8",),
                dofs_per_node=dofs_per_node,  # type: ignore[arg-type]
            )
        )
