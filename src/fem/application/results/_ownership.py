"""Deep-ownership helpers shared by providers and accepted result records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from fem.core.result import ModelResult

from .data import (
    FieldData,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
)
from .fields import ResultSourceKey


def deep_owned_result(result: ModelResult) -> ModelResult:
    """Clone a result graph while keeping both public vectors readonly."""

    if type(result) is not ModelResult:
        raise TypeError("result must be exactly ModelResult")
    memo: dict[int, Any] = {}
    owned_model = deepcopy(result.model, memo)
    owned_step = deepcopy(result.step, memo)
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
    )
    owned.U.setflags(write=False)
    owned.reactions.setflags(write=False)
    return owned


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
    return ResultMaterializationSnapshot(
        source=owned_source,
        generation=materialization.generation,
        topology=owned_topology,
        fields=owned_fields,
    )


__all__ = ["deep_owned_materialization", "deep_owned_result"]
