from __future__ import annotations

from datetime import datetime, timezone

from fem.application.results import (
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    ResultSourceKey,
    build_result_provider,
    execute_output_requests,
)
from fem.application.units import UnitContext
from fem.core.model import OutputRequest


def make_result_archive(builder, name: str) -> ResultArchiveSnapshot:
    source = ResultSourceKey(
        result_id=f"result-{name}",
        session_id="session",
        artifact_id="artifact",
        model_revision=3,
        step_name="Step-1",
        run_id=f"run-{name}",
    )
    provider = build_result_provider(source, builder())
    lazy_keys = tuple(
        item.key
        for item in provider.catalog().fields
        if item.state.value == "lazy"
    )
    if lazy_keys:
        provider = provider.advance(provider.materialize(lazy_keys))
    provider = provider.publish_fields(
        tuple(field_data.key for field_data in provider.snapshot.fields)
    )
    report = execute_output_requests(
        provider,
        (OutputRequest("field", "node", ("U",)),),
    ).report
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    units = UnitContext("m", "N", "Pa")
    return ResultArchiveSnapshot(
        archive_id=f"archive-{name}",
        created_at=now,
        producer_version="test",
        origin=ResultArchiveOrigin(
            model_name=f"model-{name}",
            source_basename=f"model-{name}.fempy",
            model_fingerprint="a" * 64,
            provenance={"run_id": source.run_id},
        ),
        run=ResultArchiveRun("job", source.step_name, now, output_report=report),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=ResultArchiveModelProjection(
            provider.snapshot.topology,
            unit_context=units,
            named_region_node_ids={"all_nodes": provider.snapshot.topology.node_ids},
            named_region_element_ids={"all_elements": provider.snapshot.topology.element_ids},
            summaries={"model_family": provider.profile.family.value},
        ),
        unit_context=units,
    )
