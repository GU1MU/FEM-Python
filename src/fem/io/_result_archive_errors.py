"""Typed errors for the versioned result archive codecs."""

from __future__ import annotations


class ResultArchiveError(Exception):
    """Base class for all result archive persistence errors."""


class ResultArchiveEncodeError(ResultArchiveError, ValueError):
    """The supplied archive snapshot cannot be encoded."""


class ResultArchiveDecodeError(ResultArchiveError, ValueError):
    """The archive bytes fail strict schema or semantic validation."""


class UnsupportedResultArchiveSchemaError(ResultArchiveDecodeError):
    """The archive declares a schema which this build cannot read."""


__all__ = [
    "ResultArchiveDecodeError",
    "ResultArchiveEncodeError",
    "ResultArchiveError",
    "UnsupportedResultArchiveSchemaError",
]
