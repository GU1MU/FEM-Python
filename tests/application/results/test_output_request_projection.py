from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from fem.application.results.fields import (
    FieldPosition,
    ResultVariable,
)
from fem.application.results.output_requests import (
    ResultCapabilityCatalog,
    project_output_request,
)
from fem.application.results.registry import (
    ElementResultProfile,
    ResultModelFamily,
    catalog_entries,
)
from fem.core.model import OutputRequest, OutputSourceEvidence


def _profile(family: ResultModelFamily) -> ElementResultProfile:
    if family is ResultModelFamily.PLANE_CONTINUUM:
        element_types = ("Quad4",)
        element_families = ("plane_continuum",)
        dof_labels = ("U1", "U2")
        force_labels = ("Fx", "Fy")
        dofs = 2
        stress = True
    elif family is ResultModelFamily.SOLID_CONTINUUM:
        element_types = ("Hex8",)
        element_families = ("solid_continuum",)
        dof_labels = ("U1", "U2", "U3")
        force_labels = ("Fx", "Fy", "Fz")
        dofs = 3
        stress = True
    elif family is ResultModelFamily.TRUSS:
        element_types = ("Truss2",)
        element_families = ("truss",)
        dof_labels = ("U1", "U2", "U3")
        force_labels = ("Fx", "Fy", "Fz")
        dofs = 3
        stress = True
    elif family is ResultModelFamily.BEAM:
        element_types = ("Beam2",)
        element_families = ("beam",)
        dof_labels = ("U1", "U2", "U3", "UR1", "UR2", "UR3")
        force_labels = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
        dofs = 6
        stress = True
    else:
        element_types = ("Hex8", "Truss2")
        element_families = ("solid_continuum", "truss")
        dof_labels = ("U1", "U2", "U3")
        force_labels = ("Fx", "Fy", "Fz")
        dofs = 3
        stress = False
    return ElementResultProfile(
        family=family,
        canonical_element_types=element_types,
        element_families=element_families,
        dofs_per_node=dofs,
        dof_labels=dof_labels,
        force_labels=force_labels,
        primary_compatible=True,
        stress_compatible=stress,
    )


def _capabilities(
    family: ResultModelFamily = ResultModelFamily.PLANE_CONTINUUM,
) -> ResultCapabilityCatalog:
    return ResultCapabilityCatalog.from_profile(_profile(family))


def _project(
    request: OutputRequest,
    family: ResultModelFamily = ResultModelFamily.PLANE_CONTINUUM,
    *,
    request_index: int = 0,
):
    return project_output_request(
        request,
        _capabilities(family),
        request_index=request_index,
    )


def _codes(projection) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in projection.diagnostics)


def test_capability_catalog_is_exact_immutable_registry_projection() -> None:
    profile = _profile(ResultModelFamily.BEAM)
    entries = catalog_entries(profile)

    from_profile = ResultCapabilityCatalog.from_profile(profile)
    from_entries = ResultCapabilityCatalog.from_entries(profile, entries)

    assert from_profile == from_entries
    assert from_profile.entries is not entries
    assert from_profile.entries == entries
    with pytest.raises(FrozenInstanceError):
        from_profile.entries = ()
    with pytest.raises(ValueError, match="exactly match"):
        ResultCapabilityCatalog.from_entries(profile, tuple(reversed(entries)))
    with pytest.raises(ValueError, match="exactly match"):
        ResultCapabilityCatalog.from_entries(profile, entries[:-1])
    plane_profile = _profile(ResultModelFamily.PLANE_CONTINUUM)
    with pytest.raises(ValueError, match="exactly match"):
        ResultCapabilityCatalog.from_entries(
            profile,
            catalog_entries(plane_profile),
        )
    with pytest.raises(TypeError, match="ElementResultProfile"):
        ResultCapabilityCatalog.from_profile(object())


def test_capability_catalog_exposes_registry_owned_profile_diagnostics() -> None:
    catalog = _capabilities(ResultModelFamily.MIXED_UNSUPPORTED)

    assert tuple(item.code for item in catalog.diagnostics) == (
        "result.catalog.stress_family_unsupported",
    )
    assert catalog.diagnostics[0].details["canonical_variable"] == "S"


@pytest.mark.parametrize(
    ("family", "expected"),
    (
        (
            ResultModelFamily.PLANE_CONTINUUM,
            (
                ("node", "U", None),
                ("node", "RF", None),
                ("element", "S", None),
            ),
        ),
        (
            ResultModelFamily.TRUSS,
            (
                ("node", "U", None),
                ("node", "RF", None),
                ("element", "S", None),
            ),
        ),
        (
            ResultModelFamily.BEAM,
            (
                ("node", "U", None),
                ("node", "UR", None),
                ("node", "RF", None),
                ("node", "RM", None),
                ("element", "S", None),
            ),
        ),
    ),
)
def test_capability_catalog_owns_only_legal_source_independent_candidates(
    family: ResultModelFamily,
    expected: tuple[tuple[str, str, str | None], ...],
) -> None:
    catalog = _capabilities(family)

    assert type(catalog.candidates) is tuple
    assert tuple(
        (
            candidate.authoring_request.target,
            candidate.authoring_request.variables[0],
            candidate.authoring_request.metadata.get("position"),
        )
        for candidate in catalog.candidates
    ) == expected
    assert tuple(
        candidate.request_index for candidate in catalog.candidates
    ) == tuple(range(len(expected)))
    assert all(candidate.executable for candidate in catalog.candidates)
    assert all(
        candidate.authoring_request.source_evidence is None
        for candidate in catalog.candidates
    )
    assert all(
        not candidate.authoring_request.metadata
        for candidate in catalog.candidates
    )
    with pytest.raises(FrozenInstanceError):
        catalog.candidates = ()


def test_projection_rejects_a_forged_capability_catalog() -> None:
    with pytest.raises(TypeError, match="ResultCapabilityCatalog"):
        project_output_request(
            OutputRequest("field", "node", ("U",)),
            object(),
            request_index=0,
        )


def test_node_variables_are_case_insensitive_collapsed_and_canonically_ordered():
    request = OutputRequest(
        "field",
        "node",
        ("rf", "U", "u", "RM", "ur", "RF"),
        {"frequency": "1"},
    )
    before = deepcopy(request)

    projection = _project(request, ResultModelFamily.BEAM, request_index=4)

    assert projection.executable
    assert projection.authoring_request is request
    assert projection.diagnostics == ()
    executable = projection.executable_request
    assert executable is not None
    assert executable.request_index == 4
    assert executable.frequency == 1
    assert tuple(
        variable.canonical_variable for variable in executable.variables
    ) == (
        ResultVariable.U,
        ResultVariable.UR,
        ResultVariable.RF,
        ResultVariable.RM,
    )
    assert tuple(
        variable.source_variable_indices for variable in executable.variables
    ) == ((1, 2), (4,), (0, 5), (3,))
    assert tuple(
        request.field_id.variable for request in executable.field_requests
    ) == (
        ResultVariable.U,
        ResultVariable.UR,
        ResultVariable.RF,
        ResultVariable.RM,
    )
    assert request == before
    assert request.variables == ("rf", "U", "u", "RM", "ur", "RF")
    assert request.metadata == {"frequency": "1"}


@pytest.mark.parametrize(
    ("family", "expected_position"),
    (
        (
            ResultModelFamily.PLANE_CONTINUUM,
            FieldPosition.INTEGRATION_POINT,
        ),
        (
            ResultModelFamily.SOLID_CONTINUUM,
            FieldPosition.INTEGRATION_POINT,
        ),
        (ResultModelFamily.TRUSS, FieldPosition.CENTROID),
        (ResultModelFamily.BEAM, FieldPosition.SECTION_END),
    ),
)
def test_stress_uses_deterministic_family_default(
    family: ResultModelFamily,
    expected_position: FieldPosition,
) -> None:
    projection = _project(OutputRequest("field", "element", ("s",)), family)

    assert projection.executable
    executable = projection.executable_request
    assert executable is not None
    assert executable.field_requests[0].field_id.position is expected_position


@pytest.mark.parametrize(
    "position",
    (
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
    ),
)
def test_continuum_stress_accepts_explicit_canonical_positions(
    position: FieldPosition,
) -> None:
    request = OutputRequest(
        "field",
        "element",
        ("S", "s"),
        {"POSITION": position.value.upper()},
    )

    projection = _project(request)

    assert projection.executable
    executable = projection.executable_request
    assert executable is not None
    assert len(executable.field_requests) == 1
    assert executable.field_requests[0].field_id.position is position
    assert executable.variables[0].source_variable_indices == (0, 1)
    assert request.metadata == {"POSITION": position.value.upper()}


@pytest.mark.parametrize(
    ("family", "position"),
    (
        (ResultModelFamily.PLANE_CONTINUUM, "node_region"),
        (ResultModelFamily.PLANE_CONTINUUM, "resolved_nodal"),
        (ResultModelFamily.PLANE_CONTINUUM, "section_end"),
        (ResultModelFamily.TRUSS, "integration_point"),
        (ResultModelFamily.BEAM, "centroid"),
        (ResultModelFamily.BEAM, "section_node_envelope"),
        (ResultModelFamily.PLANE_CONTINUUM, "NODES"),
        (ResultModelFamily.PLANE_CONTINUUM, "AVERAGED AT NODES"),
        (ResultModelFamily.PLANE_CONTINUUM, 1),
    ),
)
def test_stress_rejects_nonexecutable_or_abaqus_positions(
    family: ResultModelFamily,
    position: object,
) -> None:
    projection = _project(
        OutputRequest("field", "element", ("S",), {"position": position}),
        family,
    )

    assert not projection.executable
    assert _codes(projection) == ("output.request.position_unsupported",)
    diagnostic = projection.diagnostics[0]
    assert diagnostic.path == ("outputs", 0, "metadata", "position")
    assert diagnostic.details["position"] == position


@pytest.mark.parametrize("variable", ("UR", "RM"))
def test_rotational_primary_requires_catalog_entry(variable: str) -> None:
    projection = _project(OutputRequest("field", "node", (variable,)))

    assert not projection.executable
    assert _codes(projection) == (
        "output.request.model_family_unsupported",
    )
    assert projection.variables[0].canonical_variable is ResultVariable[
        variable
    ]


def test_mixed_stress_family_is_typed_model_unsupported() -> None:
    projection = _project(
        OutputRequest("field", "element", ("S",)),
        ResultModelFamily.MIXED_UNSUPPORTED,
    )

    assert _codes(projection) == (
        "output.request.model_family_unsupported",
    )
    assert projection.diagnostics[0].details["model_family"] == (
        "mixed_unsupported"
    )


def test_mixed_stress_family_still_projects_exact_common_primary_fields() -> None:
    request = OutputRequest("field", "node", ("RF", "U", "rf"))

    projection = _project(
        request,
        ResultModelFamily.MIXED_UNSUPPORTED,
    )

    assert projection.executable
    executable = projection.executable_request
    assert executable is not None
    assert tuple(
        field_request.field_id.variable
        for field_request in executable.field_requests
    ) == (ResultVariable.U, ResultVariable.RF)
    assert tuple(
        variable.source_variable_indices
        for variable in executable.variables
    ) == ((1,), (0, 2))


@pytest.mark.parametrize(
    ("authoring", "expected_codes"),
    (
        (
            OutputRequest("history", "node", ("U",)),
            ("output.request.kind_unsupported",),
        ),
        (
            OutputRequest("field", "preselect", ("PRESELECT",)),
            ("output.request.target_unsupported",),
        ),
        (
            OutputRequest("field", "node", ("E",)),
            ("output.request.variable_unsupported",),
        ),
        (
            OutputRequest("field", "node", ("PRESELECT",)),
            ("output.request.variable_unsupported",),
        ),
        (
            OutputRequest("field", "node", ("S",)),
            ("output.request.variable_unsupported",),
        ),
        (
            OutputRequest("field", "element", ("U",)),
            ("output.request.variable_unsupported",),
        ),
        (
            OutputRequest("field", "node", ()),
            ("output.request.variables_empty",),
        ),
    ),
)
def test_intrinsically_valid_but_nonexecutable_combinations_are_typed(
    authoring: OutputRequest,
    expected_codes: tuple[str, ...],
) -> None:
    projection = _project(authoring, request_index=7)

    assert not projection.executable
    assert _codes(projection) == expected_codes
    assert all(
        diagnostic.severity == "warning"
        for diagnostic in projection.diagnostics
    )
    assert all(
        diagnostic.details["request_index"] == 7
        for diagnostic in projection.diagnostics
    )


def test_unknown_kind_and_target_are_reported_without_variable_cascade() -> None:
    projection = _project(
        OutputRequest("future", "future_target", ("FUTURE",)),
        request_index=2,
    )

    assert _codes(projection) == (
        "output.request.kind_unsupported",
        "output.request.target_unsupported",
    )
    assert projection.variables[0].canonical_variable is None
    assert projection.variables[0].diagnostics == ()


def test_request_level_atomicity_keeps_supported_sibling_unexecuted() -> None:
    request = OutputRequest("field", "node", ("E", "U", "u"))

    projection = _project(request)

    assert not projection.executable
    assert _codes(projection) == ("output.request.variable_unsupported",)
    supported, unsupported = projection.variables
    assert supported.canonical_variable is ResultVariable.U
    assert supported.source_variable_indices == (1, 2)
    assert len(supported.field_requests) == 1
    assert supported.diagnostics == ()
    assert unsupported.canonical_variable is None
    assert unsupported.source_variable_indices == (0,)
    assert unsupported.field_requests == ()
    assert unsupported.diagnostics == projection.diagnostics


@pytest.mark.parametrize("frequency", (None, 1, "1"))
def test_frequency_absent_integer_one_and_exact_string_one_execute(
    frequency: object,
) -> None:
    metadata = {} if frequency is None else {"frequency": frequency}

    projection = _project(OutputRequest("field", "node", ("U",), metadata))

    assert projection.executable
    assert projection.executable_request is not None
    assert projection.executable_request.frequency == 1


@pytest.mark.parametrize(
    "frequency",
    (True, 1.0, "1.0", "01", 0, 2, "LAST", [1]),
)
def test_frequency_rejects_every_noncanonical_value(frequency: object) -> None:
    request = OutputRequest(
        "field",
        "node",
        ("U",),
        {"frequency": frequency},
    )

    projection = _project(request)

    assert not projection.executable
    assert _codes(projection) == (
        "output.request.frequency_unsupported",
    )
    assert projection.diagnostics[0].details["frequency"] == (
        tuple(frequency) if type(frequency) is list else frequency
    )
    assert request.metadata["frequency"] == (
        tuple(frequency) if type(frequency) is list else frequency
    )


@pytest.mark.parametrize(
    ("target", "variables", "metadata"),
    (
        ("node", ("U",), {"position": "node"}),
        ("node", ("RF",), {"nset": "Tip"}),
        ("element", ("S",), {"elset": "Domain"}),
        ("element", ("S",), {"directions": "YES"}),
        ("element", ("S",), {"name": "Saved"}),
        ("element", ("S",), {"coordinate_system": "local"}),
    ),
)
def test_metadata_allowlist_rejects_scope_and_orientation_options(
    target: str,
    variables: tuple[str, ...],
    metadata: dict[str, object],
) -> None:
    projection = _project(
        OutputRequest("field", target, variables, metadata)
    )

    assert not projection.executable
    assert _codes(projection) == (
        "output.request.metadata_unsupported",
    )
    assert projection.diagnostics[0].details["reason"] == (
        "option_not_allowed"
    )


@pytest.mark.parametrize(
    ("target", "variables", "expected_codes"),
    (
        (
            "node",
            ("U",),
            ("output.request.metadata_unsupported",),
        ),
        (
            "element",
            ("U",),
            (
                "output.request.metadata_unsupported",
                "output.request.variable_unsupported",
            ),
        ),
        (
            "future_target",
            ("S",),
            (
                "output.request.target_unsupported",
                "output.request.metadata_unsupported",
            ),
        ),
    ),
)
def test_position_is_a_position_diagnostic_only_for_field_element_stress(
    target: str,
    variables: tuple[str, ...],
    expected_codes: tuple[str, ...],
) -> None:
    projection = _project(
        OutputRequest(
            "field",
            target,
            variables,
            {"position": "centroid"},
        )
    )

    assert _codes(projection) == expected_codes
    assert "output.request.position_unsupported" not in _codes(projection)


def test_native_metadata_casefold_collision_does_not_choose_a_value() -> None:
    request = OutputRequest(
        "field",
        "node",
        ("U",),
        {"frequency": 1, "FREQUENCY": "1"},
    )

    projection = _project(request, request_index=3)

    assert _codes(projection) == (
        "output.request.metadata_unsupported",
    )
    diagnostic = projection.diagnostics[0]
    assert diagnostic.path == ("outputs", 3, "metadata")
    assert diagnostic.details["canonical_key"] == "frequency"
    assert diagnostic.details["source_keys"] == (
        "frequency",
        "FREQUENCY",
    )
    assert diagnostic.details["reason"] == "casefold_collision"


@pytest.mark.parametrize(
    ("parent_frequency", "child_frequency", "metadata_frequency"),
    (
        ("1", None, None),
        ("2", "1", None),
        ("2", None, "1"),
        ("2", "2", "1"),
    ),
)
def test_abaqus_effective_metadata_uses_parent_default_and_child_override(
    parent_frequency: str,
    child_frequency: str | None,
    metadata_frequency: str | None,
) -> None:
    evidence = OutputSourceEvidence(
        "ABAQUS",
        (("FREQUENCY", parent_frequency),),
        ("FIELD",),
        (
            ()
            if child_frequency is None
            else (("frequency", child_frequency),)
        ),
        ("HISTORY",),
    )
    metadata = (
        {}
        if metadata_frequency is None
        else {"Frequency": metadata_frequency}
    )
    request = OutputRequest(
        "field",
        "node",
        ("u",),
        metadata,
        evidence,
    )
    before = deepcopy(request)

    projection = _project(request)

    assert projection.executable
    assert projection.executable_request is not None
    assert projection.executable_request.frequency == 1
    assert request == before
    assert request.source_evidence is evidence


def test_abaqus_unsupported_parent_default_remains_typed() -> None:
    evidence = OutputSourceEvidence(
        "abaqus",
        (("frequency", "2"),),
        ("field",),
    )

    projection = _project(
        OutputRequest("field", "node", ("U",), {}, evidence)
    )

    assert _codes(projection) == (
        "output.request.frequency_unsupported",
    )
    assert projection.diagnostics[0].details["frequency"] == "2"


@pytest.mark.parametrize("layer", ("parent", "child"))
def test_abaqus_same_layer_parameter_collision_is_unsupported(
    layer: str,
) -> None:
    collisions = (("frequency", "1"), ("FREQUENCY", "2"))
    evidence = OutputSourceEvidence(
        "abaqus",
        collisions if layer == "parent" else (),
        ("field",),
        collisions if layer == "child" else (),
        (),
    )

    projection = _project(
        OutputRequest("field", "node", ("U",), {}, evidence)
    )

    assert _codes(projection) == (
        "output.request.metadata_unsupported",
    )
    diagnostic = projection.diagnostics[0]
    assert diagnostic.details["layer"] == f"{layer}_parameters"
    assert diagnostic.details["canonical_key"] == "frequency"
    assert diagnostic.details["reason"] == "casefold_collision"


def test_abaqus_unknown_bare_flags_are_unsupported_but_structural_flags_are_not():
    evidence = OutputSourceEvidence(
        "abaqus",
        (),
        ("FIELD", "FutureParent"),
        (),
        ("history", "FutureChild"),
    )

    projection = _project(
        OutputRequest("field", "node", ("U",), {}, evidence)
    )

    assert _codes(projection) == (
        "output.request.metadata_unsupported",
        "output.request.metadata_unsupported",
    )
    assert tuple(
        diagnostic.details["reason"]
        for diagnostic in projection.diagnostics
    ) == ("unknown_bare_flag", "unknown_bare_flag")
    assert tuple(
        diagnostic.details["flags"]
        for diagnostic in projection.diagnostics
    ) == (("FutureParent",), ("FutureChild",))


def test_diagnostics_have_stable_variable_path_and_occurrence_details() -> None:
    request = OutputRequest("field", "node", ("Future", "future"))

    projection = _project(request, request_index=9)

    assert len(projection.variables) == 1
    diagnostic = projection.diagnostics[0]
    assert diagnostic.code == "output.request.variable_unsupported"
    assert diagnostic.path == ("outputs", 9, "variables", 0)
    assert diagnostic.details == {
        "request_index": 9,
        "source_indices": (0, 1),
        "source_variables": ("Future", "future"),
    }
    assert diagnostic.remediation


@pytest.mark.parametrize("request_index", (True, -1, 1.0, "1"))
def test_projection_requires_a_strict_nonnegative_request_index(
    request_index: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="request_index"):
        project_output_request(
            OutputRequest("field", "node", ("U",)),
            _capabilities(),
            request_index=request_index,
        )
