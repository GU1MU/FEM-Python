from __future__ import annotations

from copy import deepcopy

import pytest

from fem import materials
from fem.application import RegionRef, resolve_effective_beam_frames
from fem.core import Element3D, Mesh3D, Node3D
from fem.core.model import (
    ElementSet,
    FEMModel,
    MaterialDefinition,
    SectionAssignment,
)
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY, BeamOrientation


def _beam_model(
    *,
    reference: tuple[float, float, float] | None = None,
) -> FEMModel:
    mesh = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
        ),
        elements=(
            Element3D(
                1,
                (1, 2),
                type="Beam2",
                props={},
            ),
        ),
        dofs_per_node=6,
    )
    properties: dict[str, object] = {
        "height": 0.2,
        "width": 0.1,
    }
    if reference is not None:
        properties[BEAM_LOCAL_Y_REFERENCE_KEY] = reference
    return FEMModel(
        mesh,
        element_sets={"BEAMS": ElementSet("BEAMS", (1,))},
        materials={
            "Steel": MaterialDefinition(
                "Steel",
                {"E": 210.0e9, "nu": 0.3},
            )
        },
        sections=[
            SectionAssignment(
                "BEAMS",
                "Steel",
                "rectangle",
                properties,
            )
        ],
    )


def test_effective_query_uses_assignment_properties_without_mutating_element():
    model = _beam_model(reference=(0.0, 1.0, 0.0))
    element = model.mesh.elements[0]

    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAMS"),
    )

    assert report.passed
    assert report.element_ids == (1,)
    assert report.entries[0].assignment_index == 0
    assert report.entries[0].element_set == "BEAMS"
    assert report.entries[0].section_type == "rectangle"
    assert report.entries[0].frame.source == "explicit"
    assert report.entries[0].frame.local_y == pytest.approx((0.0, 1.0, 0.0))
    assert report.suggested_orientation == BeamOrientation((0.0, 1.0, 0.0))
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in element.props

    copied = deepcopy(report)
    assert copied.entries[0].frame is report.entries[0].frame
    assert not copied.entries[0].frame.rotation.flags.writeable


def test_effective_query_preserves_direct_uncovered_automatic_frame():
    model = _beam_model()
    model.sections.clear()

    report = resolve_effective_beam_frames(model, 1)

    assert report.passed
    assert report.entries[0].assignment_index is None
    assert report.entries[0].frame.source == "automatic"
    assert report.suggested_orientation == BeamOrientation((0.0, 1.0, 0.0))


def test_effective_query_restores_uncovered_direct_orientation_read_only():
    model = _beam_model(reference=(0.0, 0.0, 1.0))
    element = model.mesh.elements[0]
    element.props[BEAM_LOCAL_Y_REFERENCE_KEY] = (0.0, 1.0, 0.0)
    materials.apply_sections(model)
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == (0.0, 0.0, 1.0)
    model.sections.clear()

    report = resolve_effective_beam_frames(model, 1)

    assert report.passed
    assert report.entries[0].frame.orientation == BeamOrientation(
        (0.0, 1.0, 0.0)
    )
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == (0.0, 0.0, 1.0)


def test_effective_query_restores_direct_orientation_after_model_deepcopy():
    model = _beam_model(reference=(0.0, 0.0, 1.0))
    element = model.mesh.elements[0]
    element.props[BEAM_LOCAL_Y_REFERENCE_KEY] = (0.0, 1.0, 0.0)
    materials.apply_sections(model)

    copied = deepcopy(model)
    copied.sections.clear()
    report = resolve_effective_beam_frames(copied, 1)

    assert report.passed
    assert report.entries[0].frame.orientation == BeamOrientation(
        (0.0, 1.0, 0.0)
    )
    assert copied.mesh.elements[0].props[
        BEAM_LOCAL_Y_REFERENCE_KEY
    ] == (0.0, 0.0, 1.0)


def test_effective_query_reports_parallel_reference_with_element_context():
    model = _beam_model(reference=(1.0, 0.0, 0.0))

    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAMS"),
    )

    assert not report.passed
    assert report.entries == ()
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "beam.orientation.parallel"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.details_dict()["element_id"] == 1
    assert diagnostic.details_dict()["assignment_index"] == 0


def test_effective_query_keeps_shadowed_orientation_errors_blocking():
    model = _beam_model(reference=(0.0, 1.0, 0.0))
    valid = model.sections[0]
    model.sections = [
        SectionAssignment(
            "BEAMS",
            "Steel",
            "rectangle",
            {
                "height": 0.2,
                "width": 0.1,
                BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 0.0),
            },
        ),
        valid,
    ]

    report = resolve_effective_beam_frames(model, 1)

    assert not report.passed
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "beam.orientation.invalid"
    assert diagnostic.details_dict()["assignment_index"] == 0
    assert diagnostic.details_dict()["reference"] == (0.0, 0.0, 0.0)


def test_effective_query_keeps_shadowed_parallel_orientation_blocking():
    model = _beam_model(reference=(0.0, 1.0, 0.0))
    valid = model.sections[0]
    model.sections = [
        SectionAssignment(
            "BEAMS",
            "Steel",
            "rectangle",
            {
                "height": 0.2,
                "width": 0.1,
                BEAM_LOCAL_Y_REFERENCE_KEY: (1.0, 0.0, 0.0),
            },
        ),
        valid,
    ]

    report = resolve_effective_beam_frames(model, 1)

    assert not report.passed
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "beam.orientation.parallel"
    assert diagnostic.details_dict()["assignment_index"] == 0
    assert diagnostic.details_dict()["reference"] == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("node_ids", "second_node", "message"),
    (
        ((1, 99), Node3D(2, 2.0, 0.0, 0.0), "missing node"),
        ((1, 2), Node3D(2, 0.0, 0.0, 0.0), "zero length"),
    ),
)
def test_effective_query_classifies_geometry_fault_as_structure(
    node_ids,
    second_node,
    message,
):
    model = _beam_model()
    model.mesh.nodes = (model.mesh.nodes[0], second_node)
    model.mesh.elements[0].node_ids = list(node_ids)

    report = resolve_effective_beam_frames(model, 1)

    assert not report.passed
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "model.structure.invalid"
    assert diagnostic.stage.value == "structure"
    assert message in diagnostic.message
    assert diagnostic.details_dict()["element_id"] == 1


def test_effective_query_rejects_non_beam_target_without_fallback():
    model = _beam_model()
    model.mesh.elements[0].type = "Truss2"
    model.sections.clear()

    report = resolve_effective_beam_frames(model, 1)

    assert not report.passed
    assert report.entries == ()
    assert report.diagnostics[0].code == (
        "beam.orientation.unsupported_target"
    )


def test_multi_element_automatic_prefill_requires_one_full_frame_candidate():
    mesh = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 0.0, 1.0, 0.0),
        ),
        elements=(
            Element3D(1, (1, 2), type="Beam2", props={}),
            Element3D(2, (1, 3), type="Beam2", props={}),
        ),
        dofs_per_node=6,
    )
    model = FEMModel(
        mesh,
        element_sets={"BEAMS": ElementSet("BEAMS", (1, 2))},
    )

    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAMS"),
    )

    assert report.passed
    assert tuple(entry.frame.source for entry in report.entries) == (
        "automatic",
        "automatic",
    )
    assert report.suggested_orientation is None


def test_effective_query_reports_missing_typed_target():
    model = _beam_model()

    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "MISSING"),
    )

    assert not report.passed
    assert report.element_ids == ()
    assert report.diagnostics[0].code == "step.reference.invalid"
