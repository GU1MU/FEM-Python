"""Version-neutral errors for native project persistence."""

from __future__ import annotations


class ProjectError(ValueError):
    """Base error for a project that cannot be decoded or encoded safely."""


class ProjectDecodeError(ProjectError):
    """A serialized project is malformed, incomplete, or unsupported."""


class ProjectEncodeError(ProjectError):
    """A detached project snapshot cannot be represented losslessly."""


class UnsupportedProjectSchemaError(ProjectDecodeError):
    """The project declares a schema that this build does not support."""


__all__ = [
    "ProjectDecodeError",
    "ProjectEncodeError",
    "ProjectError",
    "UnsupportedProjectSchemaError",
]
