"""Application task payloads for result-archive persistence."""

from __future__ import annotations

from dataclasses import dataclass

from fem.results import ResultArchiveSnapshot, ResultSourceKey

from .revisions import TaskToken


@dataclass(frozen=True, slots=True)
class ResultArchiveSaveSnapshot:
    """Worker-safe payload bound to one accepted result generation.

    Archive content belongs to :mod:`fem.results`; the asynchronous task
    token belongs to the application session.  Keeping their composition at
    this boundary prevents the result domain from importing application code.
    """

    token: TaskToken
    archive: ResultArchiveSnapshot
    source: ResultSourceKey
    materialization_generation: int
    run_id: str
    result_id: str

    def __post_init__(self) -> None:
        if type(self.token) is not TaskToken:
            raise TypeError("token must be exactly TaskToken")
        if type(self.archive) is not ResultArchiveSnapshot:
            raise TypeError("archive must be exactly ResultArchiveSnapshot")
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be exactly ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError("materialization_generation must be non-negative")
        for value, label in (
            (self.run_id, "run_id"),
            (self.result_id, "result_id"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a nonblank string")
        if self.archive.source != self.source:
            raise ValueError("archive source must match save source")
        if (
            self.archive.materialization.generation
            != self.materialization_generation
        ):
            raise ValueError(
                "archive generation must match save materialization generation"
            )
        if self.source.run_id != self.run_id:
            raise ValueError("save run_id must match source")
        if self.source.result_id != self.result_id:
            raise ValueError("save result_id must match source")
        if self.token.task_kind != "result_archive_save":
            raise ValueError("save token must be a result_archive_save token")
        if (
            self.token.session_id != self.source.session_id
            or self.token.artifact_id != self.source.artifact_id
            or self.token.step_name != self.source.step_name
            or self.token.run_id != self.source.run_id
            or self.token.result_id != self.source.result_id
        ):
            raise ValueError("save token identity must match result source")
        dependencies = dict(self.token.dependency_revisions)
        if dependencies.get("model_revision") != self.source.model_revision:
            raise ValueError("save token model revision must match result source")
        if (
            dependencies.get("materialization_generation")
            != self.materialization_generation
        ):
            raise ValueError(
                "save token generation must match materialization generation"
            )

    @property
    def snapshot(self) -> ResultArchiveSnapshot:
        return self.archive

    @property
    def result_archive(self) -> ResultArchiveSnapshot:
        return self.archive

    @property
    def task_token(self) -> TaskToken:
        return self.token

    @property
    def result_source(self) -> ResultSourceKey:
        return self.source

    @property
    def generation(self) -> int:
        return self.materialization_generation


__all__ = ["ResultArchiveSaveSnapshot"]
