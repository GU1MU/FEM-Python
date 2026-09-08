from __future__ import annotations

from fem.application.results import (
    FieldAssociation,
    FieldAvailability,
    FieldDescriptor,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    FieldState,
    PhysicalQuantity,
    ResultCatalog,
    ResultDiagnostic,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)


def make_result_field_key(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    contract: int,
) -> FieldMaterializationKey:
    return FieldMaterializationKey(
        FieldRequest(ResultFieldId(variable, position)),
        contract,
    )


def _descriptor(
    key: FieldMaterializationKey,
    *,
    association: FieldAssociation,
    quantity: PhysicalQuantity,
    components: tuple[str, ...],
    derived_components: tuple[str, ...],
    label_key: str,
    default_component: str,
    order: int,
) -> FieldDescriptor:
    return FieldDescriptor(
        field_id=key.request.field_id,
        association=association,
        quantity=quantity,
        components=components,
        derived_components=derived_components,
        label_key=label_key,
        unit_label=None,
        default_component=default_component,
        order=order,
    )


def make_result_catalog(
    *,
    source: ResultSourceKey | None = None,
    unavailable_diagnostic: ResultDiagnostic | None = None,
) -> ResultCatalog:
    displacement = make_result_field_key(
        ResultVariable.U,
        FieldPosition.NODE,
        contract=11,
    )
    reaction = make_result_field_key(
        ResultVariable.RF,
        FieldPosition.NODE,
        contract=12,
    )
    stress = make_result_field_key(
        ResultVariable.S,
        FieldPosition.ELEMENT_NODAL,
        contract=13,
    )
    fields = (
        FieldAvailability(
            displacement,
            _descriptor(
                displacement,
                association=FieldAssociation.NODE,
                quantity=PhysicalQuantity.DISPLACEMENT,
                components=("U2", "U1"),
                derived_components=("Magnitude",),
                label_key="result.field.u.node",
                default_component="Magnitude",
                order=80,
            ),
            FieldState.READY,
        ),
        FieldAvailability(
            reaction,
            _descriptor(
                reaction,
                association=FieldAssociation.NODE,
                quantity=PhysicalQuantity.FORCE,
                components=("RF2", "RF1"),
                derived_components=("Magnitude",),
                label_key="vendor.result.reaction",
                default_component="Magnitude",
                order=2,
            ),
            FieldState.LAZY,
        ),
        FieldAvailability(
            stress,
            _descriptor(
                stress,
                association=FieldAssociation.ELEMENT_NODE,
                quantity=PhysicalQuantity.STRESS,
                components=("S22", "S11"),
                derived_components=("Mises",),
                label_key="result.field.s.element_nodal",
                default_component="Mises",
                order=1,
            ),
            FieldState.UNAVAILABLE,
            () if unavailable_diagnostic is None else (unavailable_diagnostic,),
        ),
    )
    return ResultCatalog(
        source=source or ResultSourceKey(
            result_id="result-1",
            session_id="session-1",
            artifact_id="artifact-1",
            model_revision=4,
            step_name="Static-1",
            run_id="run-1",
        ),
        fields=fields,
        default_selection=ScalarFieldSelection(displacement, "U1"),
    )
