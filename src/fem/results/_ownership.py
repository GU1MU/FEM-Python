"""Deep-ownership helpers shared by providers and accepted result records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from fem.results import ModelResult, ResultFrame

from .data import (
    FieldData,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
)
from .fields import ResultSourceKey
from .frames import ResultFrameKey


def deep_owned_result(result: ModelResult) -> ModelResult:
    """Clone a result graph while keeping both public vectors readonly."""

    if type(result) is not ModelResult:
        raise TypeError("result must be exactly ModelResult")
    memo: dict[int, Any] = {}
    owned_model = deepcopy(result.model, memo)
    owned_step = deepcopy(result.step, memo)
    owned_compiled_model = (
        None
        if result.compiled_model is None
        else deepcopy(result.compiled_model, memo)
    )
    owned_frames = tuple(
        _owned_frame(
            frame,
            result.model,
            result.step,
            owned_model,
            owned_step,
            result.compiled_model,
            owned_compiled_model,
            memo,
        )
        for frame in result.frames
    )
    owned = ModelResult(
        model=owned_model,
        step=owned_step,
        U=np.array(result.U, dtype=float, order="C", copy=True),
        reactions=np.array(
            result.reactions,
            dtype=float,
            order="C",
            copy=True,
        ),
        name=deepcopy(result.name, memo),
        outputs=_owned_value(result.outputs, memo),
        load_factor=result.load_factor,
        iterations=result.iterations,
        residual_norm=result.residual_norm,
        frames=owned_frames,
        compiled_model=owned_compiled_model,
        dynamic_data=_owned_value(result.dynamic_data, memo),
    )
    owned.U.setflags(write=False)
    owned.reactions.setflags(write=False)
    return owned


def _owned_frame(
    frame: ResultFrame,
    original_model: Any,
    original_step: Any,
    owned_model: Any,
    owned_step: Any,
    original_compiled_model: Any | None,
    owned_compiled_model: Any | None,
    memo: dict[int, Any],
) -> ResultFrame:
    if type(frame) is not ResultFrame:
        raise TypeError("result frames must contain exact ResultFrame values")
    return ResultFrame(
        model=(
            owned_model
            if frame.model is original_model
            else _owned_value(frame.model, memo)
        ),
        step=(
            owned_step
            if frame.step is original_step
            else _owned_value(frame.step, memo)
        ),
        U=np.array(frame.U, dtype=float, order="C", copy=True),
        reactions=np.array(frame.reactions, dtype=float, order="C", copy=True),
        frame_index=frame.frame_index,
        load_factor=frame.load_factor,
        name=_owned_value(frame.name, memo),
        outputs=_owned_value(frame.outputs, memo),
        iterations=frame.iterations,
        residual_norm=frame.residual_norm,
        converged=frame.converged,
        compiled_model=(
            None
            if frame.compiled_model is None
            else (
                owned_compiled_model
                if frame.compiled_model is original_compiled_model
                else _owned_value(frame.compiled_model, memo)
            )
        ),
        dynamic_data=_owned_value(frame.dynamic_data, memo),
    )


def _owned_value(value: Any, memo: dict[int, Any]) -> Any:
    """Deep-copy result output values that may contain read-only mappings."""

    if isinstance(value, np.ndarray):
        return np.array(value, dtype=value.dtype, order="C", copy=True)
    if isinstance(value, dict):
        return {
            key: _owned_value(item, memo)
            for key, item in value.items()
        }
    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        return {
            key: _owned_value(item, memo)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_owned_value(item, memo) for item in value)
    if isinstance(value, list):
        return [_owned_value(item, memo) for item in value]
    return deepcopy(value, memo)


def deep_owned_materialization(
    materialization: ResultMaterializationSnapshot,
) -> ResultMaterializationSnapshot:
    """Rebuild an immutable snapshot through its owned-array constructors."""

    if type(materialization) is not ResultMaterializationSnapshot:
        raise TypeError(
            "materialization must be ResultMaterializationSnapshot"
        )
    source = materialization.source
    topology = materialization.topology
    fields = materialization.fields
    if type(source) is not ResultSourceKey:
        raise TypeError("materialization source must be ResultSourceKey")
    if type(topology) is not ResultTopologyProjection:
        raise TypeError(
            "materialization topology must be ResultTopologyProjection"
        )
    if topology.source != source:
        raise ValueError(
            "materialization topology source must match materialization source"
        )
    if type(fields) is not tuple:
        raise TypeError("materialization fields must be a tuple")
    for field_data in fields:
        if type(field_data) is not FieldData:
            raise TypeError(
                "materialization fields must contain only FieldData values"
            )
        if field_data.source != source:
            raise ValueError(
                "materialization field source must match materialization source"
            )

    owned_source = deepcopy(source)
    owned_topology = ResultTopologyProjection(
        source=owned_source,
        node_ids=deepcopy(topology.node_ids),
        node_coordinates=topology.node_coordinates,
        nodal_displacements=topology.nodal_displacements,
        element_ids=deepcopy(topology.element_ids),
        element_types=deepcopy(topology.element_types),
        connectivity=deepcopy(topology.connectivity),
        element_region_keys=deepcopy(topology.element_region_keys),
    )
    owned_fields = tuple(
        FieldData(
            descriptor=deepcopy(field_data.descriptor),
            source=owned_source,
            key=deepcopy(field_data.key),
            locations=deepcopy(field_data.locations),
            values=field_data.values,
        )
        for field_data in fields
    )
    owned_frame_key = None
    if materialization.frame_key is not None:
        owned_frame_key = ResultFrameKey(
            owned_source,
            materialization.frame_key.frame_index,
        )
    return ResultMaterializationSnapshot(
        source=owned_source,
        generation=materialization.generation,
        topology=owned_topology,
        fields=owned_fields,
        frame_key=owned_frame_key,
    )


__all__ = ["deep_owned_materialization", "deep_owned_result"]
