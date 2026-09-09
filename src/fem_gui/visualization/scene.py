"""Independent geometry and coloring state for postprocessing scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ShapeMode = Literal["undeformed", "deformed"]


@dataclass(frozen=True, slots=True)
class DisplayState:
    """Describe current geometry and contour visibility."""

    shape_mode: ShapeMode = "undeformed"
    contour_enabled: bool = False
