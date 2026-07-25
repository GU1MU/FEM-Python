"""Compatibility aliases for Session-owned analysis runs.

The canonical immutable types live in :mod:`fem.application`.  Remove this
module once the remaining GUI imports use those canonical names directly.
"""

from fem.application import AnalysisRun, RunStatus


AnalysisJob = AnalysisRun
JobStatus = RunStatus


__all__ = ["AnalysisJob", "JobStatus"]
