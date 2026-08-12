from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application.results import (
    ElementResultInspectionRequest,
    FieldPosition,
    FieldState,
    ResultFieldId,
    ResultQuery,
    ResultSourceKey,
    ResultValueLayout,
    ResultVariable,
    ScalarFieldSelection,
    build_result_provider,
    execute_output_requests,
    prepare_result_export_snapshot,
    project_scalar_field_topology,
)
from fem.core.model import OutputRequest
from fem.elements.beam_section import BeamSectionPoint
from fem_gui.inspection_service import InspectionService
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import TypedResultQueryDialog
from fem_gui.result_csv_export_dialog import ResultCsvExportDialog
from fem_gui.result_presentation import (
    result_provider_section_point_labels,
    section_point_relative_position_label,
)
from fem_gui.visualization.result_renderer import build_result_render_payload
from fem_gui.widgets.result_tree import ResultTree
from fem_gui.widgets.viewport import FEMViewport
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
)


_POINT_COMPONENTS = (
    "S11",
    "S22",
    "S12",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)
_POSITION_LABELS = (
    "右上",
    "左上",
    "左下",
    "右下",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _provider():
    result = make_beam_field_characterization_result()
    provider = build_result_provider(
        ResultSourceKey(
            "beam-gui-result",
            "beam-gui-session",
            "beam-gui-artifact",
            1,
            "Step-1",
            "beam-gui-run",
        ),
        result,
    )
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("SF", "SM", "S")),),
    )
    return result, outcome.provider_draft


def _beam_fields(provider):
    return tuple(
        availability
        for availability in provider.catalog().fields
        if availability.descriptor.field_id.variable is ResultVariable.S
        and availability.descriptor.field_id.position
        is FieldPosition.INTEGRATION_POINT
        and availability.descriptor.field_id.section_point_number is not None
        and availability.state is not FieldState.UNAVAILABLE
    )


def _point_field(provider, number: int):
    return next(
        availability
        for availability in _beam_fields(provider)
        if availability.descriptor.field_id
        == ResultFieldId(
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
            section_point_number=number,
        )
    )


def _combo_texts(combo) -> tuple[str, ...]:
    return tuple(combo.itemText(index) for index in range(combo.count()))


def test_circle_sections_keep_numbered_section_point_names() -> None:
    for section_type, dimensions in (
        ("solid_circle", {"radius": 1.0}),
        ("hollow_circle", {"outer_radius": 1.0, "inner_radius": 0.5}),
    ):
        result = make_beam_field_characterization_result()
        result.model.mesh.elements[0].props = {
            "E": 100.0,
            "nu": 0.25,
            "section_type": section_type,
            **dimensions,
        }
        provider = build_result_provider(
            ResultSourceKey(
                f"{section_type}-gui-result",
                f"{section_type}-gui-session",
                f"{section_type}-gui-artifact",
                1,
                "Step-1",
                f"{section_type}-gui-run",
            ),
            result,
        )

        assert result_provider_section_point_labels(provider) == {}

    assert section_point_relative_position_label(
        BeamSectionPoint(1, 1.0, 0.0)
    ) == "截面点 1"


def test_beam_result_tree_and_ribbon_publish_four_exact_ip_locations() -> None:
    _application()
    _result, provider = _provider()
    catalog = provider.catalog()
    tree = ResultTree()
    section_point_labels = result_provider_section_point_labels(provider)
    tree.set_catalog(
        "Step-1",
        catalog,
        section_point_labels=section_point_labels,
    )
    step = tree.topLevelItem(0).child(0)
    stress = next(
        step.child(index)
        for index in range(step.childCount())
        if step.child(index).text(0) == "应力 S"
    )
    position_items = tuple(
        stress.child(index)
        for index in range(stress.childCount())
    )

    assert tuple(item.text(0) for item in position_items) == _POSITION_LABELS
    assert (
        tuple(
            position_items[0].child(index).text(0)
            for index in range(position_items[0].childCount())
        )
        == _POINT_COMPONENTS
    )
    assert all("（" not in item.text(0) for item in position_items)

    stress.setExpanded(False)
    position_items[1].setExpanded(False)
    point_two_selection = ScalarFieldSelection(
        _point_field(provider, 2).key,
        "S11",
    )
    assert tree.select_selection(point_two_selection)
    assert not stress.isExpanded()
    assert not position_items[1].isExpanded()

    stress.setExpanded(True)
    position_items[0].setExpanded(True)
    position_items[1].setExpanded(False)
    tree.set_catalog(
        "Step-1",
        catalog,
        section_point_labels=section_point_labels,
    )
    refreshed_step = tree.topLevelItem(0).child(0)
    refreshed_stress = next(
        refreshed_step.child(index)
        for index in range(refreshed_step.childCount())
        if refreshed_step.child(index).text(0) == "应力 S"
    )
    assert refreshed_stress.isExpanded()
    assert refreshed_stress.child(0).isExpanded()
    assert not refreshed_stress.child(1).isExpanded()

    window = FEMMainWindow()
    window._current_result_provider = lambda: provider
    window.result_selection = ScalarFieldSelection(
        _point_field(provider, 1).key,
        "S11",
    )
    window._refresh_result_controls()
    stress_index = window.result_variable_combo.findData(ResultVariable.S)
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._populate_result_positions(window.result_selection)

    assert _combo_texts(window.result_position_combo) == _POSITION_LABELS
    field_ids = tuple(
        window.result_position_combo.itemData(index)
        for index in range(window.result_position_combo.count())
    )
    assert all(type(field_id) is ResultFieldId for field_id in field_ids)
    assert tuple(field_id.section_point_number for field_id in field_ids) == (
        1,
        2,
        3,
        4,
    )

    for index, expected in enumerate(
        (_POINT_COMPONENTS for _point in range(4))
    ):
        window.result_position_combo.setCurrentIndex(index)
        window._populate_result_components()
        assert _combo_texts(window.result_component_combo) == expected
        assert all(
            selection.field_key.request.field_id == field_ids[index]
            for selection in (
                window.result_component_combo.itemData(component_index)
                for component_index in range(
                    window.result_component_combo.count()
                )
            )
        )

    for variable, expected_components in (
        (ResultVariable.SF, ("N", "Vy", "Vz")),
        (ResultVariable.SM, ("T", "My", "Mz")),
    ):
        variable_index = window.result_variable_combo.findData(variable)
        assert variable_index >= 0
        window.result_variable_combo.setCurrentIndex(variable_index)
        window._populate_result_positions()
        window._populate_result_components()
        assert _combo_texts(window.result_position_combo) == ("积分点",)
        field_id = window.result_position_combo.currentData()
        assert field_id == ResultFieldId(
            variable,
            FieldPosition.INTEGRATION_POINT,
        )
        assert field_id.section_point_number is None
        assert _combo_texts(window.result_component_combo) == expected_components

    forbidden = {"Tresca", "Pressure", "MisesMax"}
    assert forbidden.isdisjoint(
        component
        for availability in _beam_fields(provider)
        for component in availability.descriptor.columns
    )
    window.close()
    tree.close()


def test_beam_section_selection_batches_all_four_lazy_ip_fields() -> None:
    result = make_beam_field_characterization_result()
    provider = build_result_provider(
        ResultSourceKey(
            "beam-lazy-result",
            "beam-lazy-session",
            "beam-lazy-artifact",
            1,
            "Step-1",
            "beam-lazy-run",
        ),
        result,
    )
    point_three = _point_field(provider, 3)
    selection = ScalarFieldSelection(point_three.key, "S11")

    keys = FEMMainWindow._result_materialization_keys(provider, selection)

    field_ids = tuple(key.request.field_id for key in keys)
    assert tuple(
        result_field_id.section_point_number for result_field_id in field_ids
    ) == (1, 2, 3, 4)
    assert tuple(result_field_id.position for result_field_id in field_ids) == (
        FieldPosition.INTEGRATION_POINT,
    ) * 4
    patch = provider.materialize(keys)
    assert tuple(field.key for field in patch.fields) == keys


def test_csv_dialog_keeps_section_point_field_identity_and_components() -> None:
    _application()
    _result, provider = _provider()
    selected_field = _point_field(provider, 3)
    selected = ScalarFieldSelection(selected_field.key, "S11")
    dialog = ResultCsvExportDialog(
        provider.catalog(),
        current_selection=selected,
        section_point_labels=result_provider_section_point_labels(provider),
    )
    stress_index = dialog.variable_combo.findData(ResultVariable.S)
    dialog.variable_combo.setCurrentIndex(stress_index)
    dialog._populate_positions(selected)

    assert _combo_texts(dialog.position_combo) == _POSITION_LABELS
    assert dialog.position_combo.currentData() == selected_field.descriptor.field_id
    assert tuple(
        selection.component for selection in dialog._component_selections
    ) == _POINT_COMPONENTS
    assert all(
        selection.field_key == selected_field.key
        for selection in dialog._component_selections
    )

    dialog.close()


def test_probe_and_inspection_expose_section_point_location_identity() -> None:
    _application()
    result, provider = _provider()
    selected_field = _point_field(provider, 2)
    dialog = TypedResultQueryDialog(provider)
    dialog.association_combo.setCurrentIndex(1)
    dialog._association_changed()
    dialog.field_combo.setCurrentIndex(
        dialog.field_combo.findData(selected_field.key)
    )
    dialog.component_combo.setCurrentIndex(
        dialog.component_combo.findData("S11")
    )
    query = ResultQuery(selected_field.key, "S11", element_ids=(30,))
    dialog._last_query = query
    query_result = provider.query(query)
    dialog.set_query_result(query_result)

    assert tuple(
        dialog.table.horizontalHeaderItem(index).text()
        for index in range(dialog.table.columnCount())
    )[5:8] == ("截面位置", "截面局部 Y", "截面局部 Z")
    for row, record in enumerate(query_result.records):
        section_point = record.location.section_point
        assert section_point is not None
        assert dialog.record_at(row) == record
        assert dialog.table.item(row, 5).text() == "左上"
        assert float(dialog.table.item(row, 6).text()) == section_point.local_y
        assert float(dialog.table.item(row, 7).text()) == section_point.local_z

    inspected = provider.inspect_result(
        ElementResultInspectionRequest(30)
    )
    service = InspectionService(result.model, result_provider=provider)
    report = service.inspect("element", 30)
    result_page = next(page for page in report.pages if page.title == "结果")
    table = next(
        table
        for table in result_page.tables
        if table.title == "应力 S（左上）（就绪）"
    )
    assert inspected.fields
    assert table.columns[7:10] == (
        "截面位置",
        "截面局部 Y",
        "截面局部 Z",
    )
    assert {row[7] for row in table.rows} == {"左上"}
    dialog.close()


def test_viewport_payload_and_legend_keep_selected_point_identity() -> None:
    _application()
    _result, provider = _provider()
    point_two = _point_field(provider, 2)
    point_three = _point_field(provider, 3)
    payloads = []
    for availability in (point_two, point_three):
        selection = ScalarFieldSelection(availability.key, "S11")
        export = prepare_result_export_snapshot(provider.snapshot, selection)
        topology = project_scalar_field_topology(export)
        payloads.append(
            build_result_render_payload(
                topology,
                reusable=payloads[-1] if payloads else None,
            )
        )

    assert payloads[0].topology.selection.field_key != (
        payloads[1].topology.selection.field_key
    )
    assert payloads[1].dataset is not payloads[0].dataset
    assert payloads[0].topology.value_layout is ResultValueLayout.CELL
    assert payloads[0].topology.cells == ((0, 1),)
    assert payloads[0].topology.canonical_element_types == ("Beam2",)
    assert payloads[0].dataset.celltypes.tolist() == [3]
    assert {
        location.section_point.number
        for location in payloads[0].topology.cell_locations
        if location is not None and location.section_point is not None
    } == {2}
    assert {
        location.section_point.number
        for location in payloads[1].topology.cell_locations
        if location is not None and location.section_point is not None
    } == {3}

    viewport = FEMViewport()
    assert viewport._contour_bar_args(payloads[0])["title"] == "S, S11"
    assert viewport._contour_bar_args(payloads[1])["title"] == "S, S11"
    location = next(
        location
        for location in payloads[0].topology.cell_locations
        if location is not None and location.section_point is not None
    )
    identity = viewport._result_location_identity(location)
    assert "截面位置 左上" in identity
    assert "截面坐标" in identity
    viewport.close()
