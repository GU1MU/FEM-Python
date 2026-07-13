"""视口与模型树共享的选择状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SelectionMode = Literal["node", "element"]


@dataclass(slots=True)
class SelectionState:
    """保存单一节点或单元选择。"""

    mode: SelectionMode = "node"
    node_id: int | None = None
    element_id: int | None = None

    def clear(self) -> None:
        self.node_id = None
        self.element_id = None

    def select_node(self, node_id: int) -> None:
        self.mode = "node"
        self.node_id = int(node_id)
        self.element_id = None

    def select_element(self, element_id: int) -> None:
        self.mode = "element"
        self.element_id = int(element_id)
        self.node_id = None
