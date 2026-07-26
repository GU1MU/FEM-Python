from __future__ import annotations

import numpy as np
import pytest

from fem.abaqus import read
from fem.post.averaging import NodalAveragingPolicy, resolve_nodal_stress
from fem.post.stress import field
from fem.solvers.static_linear import solve
from fem_gui.visualization import stress_adapter
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


@pytest.mark.parametrize("threshold", (0.0, 75.0, 100.0))
@pytest.mark.parametrize("component", ("S11", "Mises"))
def test_current_continuum_contour_matches_canonical_resolver(
    gui_inp_path,
    threshold: float,
    component: str,
) -> None:
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    data = build_result_data(result, geometry)
    element_nodal = data.stress_fields[field.StressPosition.ELEMENT_NODAL]

    resolved = resolve_nodal_stress(
        element_nodal,
        NodalAveragingPolicy(threshold),
        node_ids=tuple(
            geometry.point_index_to_node_id[index]
            for index in range(len(geometry.points))
        ),
        element_ids=tuple(
            geometry.cell_index_to_element_id[index]
            for index in range(len(geometry.cells))
        ),
    )
    rendered = stress_adapter.build_stress_render_geometry(
        geometry,
        data,
        f"NODAL:{component}",
        threshold,
    )

    averaged = {
        (record.node_id, record.region_key): record
        for record in resolved.records
        if record.averaged
    }
    raw = {
        (
            record.node_id,
            record.region_key,
            record.elem_id,
            record.local_node,
        ): record
        for record in resolved.records
        if not record.averaged
    }
    expected_point_keys: set[tuple[object, ...]] = set()
    for cell_index, source_cell in enumerate(geometry.cells):
        element_id = geometry.cell_index_to_element_id[cell_index]
        for local_index, source_point in enumerate(source_cell):
            node_id = geometry.point_index_to_node_id[source_point]
            sample = next(
                item
                for item in data.nodal_stress_samples[node_id]
                if item.element_id == element_id
            )
            averaged_record = averaged.get((node_id, sample.region_key))
            if averaged_record is None:
                record = raw[
                    (
                        node_id,
                        sample.region_key,
                        element_id,
                        sample.local_node,
                    )
                ]
                expected_element_id = element_id
                expected_point_keys.add(("raw", element_id, node_id))
            else:
                record = averaged_record
                expected_element_id = None
                expected_point_keys.add(("average", node_id, sample.region_key))

            point_index = rendered.cells[cell_index][local_index]
            expected = record.values(resolved.component_names)[component]
            assert rendered.values[point_index] == pytest.approx(expected)
            assert rendered.point_index_to_node_id[point_index] == node_id
            assert (
                rendered.point_index_to_element_id[point_index]
                == expected_element_id
            )

    assert len(rendered.points) == len(expected_point_keys)
    assert np.isfinite(rendered.values).all()


def test_current_continuum_threshold_decision_delegates_to_post(
    gui_inp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    data = build_result_data(result, geometry)
    calls: list[
        tuple[
            object,
            NodalAveragingPolicy,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []
    canonical_resolver = stress_adapter.resolve_nodal_stress

    def counted_resolver(
        element_nodal_field,
        policy,
        *,
        node_ids,
        element_ids,
    ):
        calls.append(
            (
                element_nodal_field,
                policy,
                tuple(node_ids),
                tuple(element_ids),
            )
        )
        return canonical_resolver(
            element_nodal_field,
            policy,
            node_ids=node_ids,
            element_ids=element_ids,
        )

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("current continuum must not use GUI legacy averaging")

    monkeypatch.setattr(
        stress_adapter,
        "resolve_nodal_stress",
        counted_resolver,
    )
    monkeypatch.setattr(
        stress_adapter,
        "_legacy_average_decisions",
        reject_legacy,
    )

    stress_adapter.build_stress_render_geometry(
        geometry,
        data,
        "NODAL:Mises",
        37.5,
    )

    assert len(calls) == 1
    element_nodal, policy, node_ids, element_ids = calls[0]
    assert (
        element_nodal
        is data.stress_fields[field.StressPosition.ELEMENT_NODAL]
    )
    assert policy == NodalAveragingPolicy(37.5)
    assert node_ids == tuple(model.mesh.node_ids)
    assert element_ids == tuple(element.id for element in model.mesh.elements)


def test_contour_threshold_uses_canonical_policy_validation(
    gui_inp_path,
) -> None:
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    data = build_result_data(result, geometry)

    for value in (True, False, "75", None):
        with pytest.raises(TypeError, match="threshold_percent"):
            stress_adapter.build_stress_render_geometry(
                geometry,
                data,
                "NODAL:S11",
                value,  # type: ignore[arg-type]
            )
    for value in (float("nan"), float("inf"), -0.1, 100.1):
        with pytest.raises(ValueError, match="threshold_percent"):
            stress_adapter.build_stress_render_geometry(
                geometry,
                data,
                "NODAL:S11",
                value,
            )
