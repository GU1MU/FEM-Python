"""后处理场景中相互独立的形状和着色状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ShapeMode = Literal["undeformed", "deformed"]


@dataclass(frozen=True, slots=True)
class DisplayState:
    """描述当前几何形状和云图开关。"""

    shape_mode: ShapeMode = "undeformed"
    contour_enabled: bool = False
