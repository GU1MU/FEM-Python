from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DofMap:
    """Generic node-to-DOF map."""
    dofs_per_node: int
    node_ids: List[int]
    node_id_to_index: Dict[int, int]

    def __post_init__(self) -> None:
        self.dofs_per_node = int(self.dofs_per_node)
        if self.dofs_per_node <= 0:
            raise ValueError("dofs_per_node must be positive")
        self.node_ids = [int(node_id) for node_id in self.node_ids]
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node ids must be unique")
        self.node_id_to_index = {
            int(node_id): int(index)
            for node_id, index in self.node_id_to_index.items()
        }

    @classmethod
    def from_nodes(cls, nodes: List[Any], dofs_per_node: int):
        """Build a DOF map from mesh nodes."""
        raw_node_ids = [int(node.id) for node in nodes]
        if len(set(raw_node_ids)) != len(raw_node_ids):
            raise ValueError("node ids must be unique")
        node_ids = sorted(raw_node_ids)
        node_id_to_index = {
            node_id: node_index for node_index, node_id in enumerate(node_ids)
        }
        return cls(dofs_per_node, node_ids, node_id_to_index)

    @property
    def num_nodes(self):
        """Number of nodes."""
        return len(self.node_ids)

    @property
    def num_dofs(self):
        """Total number of DOFs."""
        return self.num_nodes * self.dofs_per_node

    def global_dof(self, node_id: int, component: int) -> int:
        """Return global DOF index for a node component."""
        component = int(component)
        if component < 0 or component >= self.dofs_per_node:
            raise IndexError(
                f"component {component} out of range for {self.dofs_per_node} DOFs per node"
            )
        node_index = self.node_id_to_index[node_id]
        return node_index * self.dofs_per_node + component

    def node_dofs(self, node_id: int) -> List[int]:
        """Return global DOF indices for a node."""
        base = self.node_id_to_index[node_id] * self.dofs_per_node
        return [base + i for i in range(self.dofs_per_node)]

    def element_dofs(self, node_ids: List[int]) -> List[int]:
        """Return global DOF indices for element nodes."""
        dofs = []
        for node_id in node_ids:
            dofs.extend(self.node_dofs(node_id))
        return dofs

    def generate_global_dof_sequence(self) -> List[Tuple[int, int, int]]:
        """Generate (node_id, component, dof_id) tuples."""
        seq = []
        for node_id in self.node_ids:
            for component in range(self.dofs_per_node):
                seq.append(
                    (node_id, component, self.global_dof(node_id, component))
                )
        return seq
