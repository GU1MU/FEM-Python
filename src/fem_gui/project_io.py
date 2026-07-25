"""Compatibility exports for detached ``.femproj`` v1 persistence.

The implementation lives in :mod:`fem.io.project_v1`.  Loading returns a
detached ``ProjectSnapshot`` and saving accepts a detached project/save
snapshot; neither operation installs or mutates a live application Session.

Remove the two ``*_native_project`` aliases after remaining GUI callers
import the canonical v1 names directly.
"""

from fem.io.project_v1 import (
    SCHEMA_VERSION,
    ProjectV1DecodeError,
    ProjectV1EncodeError,
    ProjectV1Error,
    decode_project_v1,
    dumps_project_v1,
    encode_project_v1,
    load_native_project,
    load_project_v1,
    loads_project_v1,
    read_project_v1,
    save_native_project,
    save_project_v1,
    write_project_v1,
)


__all__ = [
    "SCHEMA_VERSION",
    "ProjectV1DecodeError",
    "ProjectV1EncodeError",
    "ProjectV1Error",
    "decode_project_v1",
    "dumps_project_v1",
    "encode_project_v1",
    "load_native_project",
    "load_project_v1",
    "loads_project_v1",
    "read_project_v1",
    "save_native_project",
    "save_project_v1",
    "write_project_v1",
]
