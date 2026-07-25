"""Headless application lifecycle contracts for FEM front ends."""

from .changes import ArtifactKind, ChangeKind, SessionDelta
from .definitions import (
    FeatureRecord,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from .revisions import (
    ImportTaskSnapshot,
    MeshTaskSnapshot,
    ModelArtifact,
    ResultProjectionTaskSnapshot,
    ResultTaskSnapshot,
    SolveTaskSnapshot,
    TaskToken,
    TokenStatus,
    ValidationTaskSnapshot,
)
from .runs import (
    AnalysisRun,
    ResultProvenance,
    ResultRecord,
    RunStatus,
)
from .preprocessing import (
    LogicalRecipeTopologyResolver,
    TopologyResolutionError,
    generate_fem_model,
)
from .session import (
    ModelSession,
    ProjectSaveSnapshot,
    ProjectSnapshot,
    RevisionConflictError,
    SessionSnapshot,
    SessionStateError,
)
from .validation import ValidationRecord, ValidationStamp

__all__ = [
    "AnalysisRun",
    "ArtifactKind",
    "ChangeKind",
    "FeatureRecord",
    "ImportTaskSnapshot",
    "LogicalRecipeTopologyResolver",
    "MeshTaskSnapshot",
    "ModelArtifact",
    "ModelSession",
    "NamedRegion",
    "NativePart",
    "ProjectSaveSnapshot",
    "ProjectSnapshot",
    "RegionAssignment",
    "ResultProjectionTaskSnapshot",
    "ResultProvenance",
    "ResultRecord",
    "ResultTaskSnapshot",
    "RevisionConflictError",
    "RunStatus",
    "SectionDefinition",
    "SessionDelta",
    "SessionSnapshot",
    "SessionStateError",
    "SolveTaskSnapshot",
    "TaskToken",
    "TopologyResolutionError",
    "TokenStatus",
    "ValidationRecord",
    "ValidationStamp",
    "ValidationTaskSnapshot",
    "generate_fem_model",
]
