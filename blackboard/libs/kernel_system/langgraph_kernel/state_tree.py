"""
全局状态树 - 存储完整的执行历史和中间结果
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import time
import uuid


@dataclass
class StateTreeNode:
    """状态树节点"""

    # 标识信息
    node_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # 时间信息
    timestamp: float = field(default_factory=time.time)
    step_number: int = 0

    # 执行信息
    worker_name: str = ""
    worker_instruction: Optional[str] = None

    # 状态数据
    domain_state: Dict[str, Any] = field(default_factory=dict)
    patch: List[Dict[str, Any]] = field(default_factory=list)
    patch_valid: bool = True
    patch_error: str = ""

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 分支信息
    branch_name: Optional[str] = None
    is_leaf: bool = True
    is_terminal: bool = False
    terminal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "timestamp": self.timestamp,
            "step_number": self.step_number,
            "worker_name": self.worker_name,
            "worker_instruction": self.worker_instruction,
            "domain_state": self.domain_state,
            "patch": self.patch,
            "patch_valid": self.patch_valid,
            "patch_error": self.patch_error,
            "metadata": self.metadata,
            "branch_name": self.branch_name,
            "is_leaf": self.is_leaf,
            "is_terminal": self.is_terminal,
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateTreeNode:
        """从字典创建"""
        return cls(**data)


class StateTree:
    """全局状态树"""

    def __init__(self):
        self.root_id: Optional[str] = None
        self.nodes: Dict[str, StateTreeNode] = {}
        self.current_node_id: Optional[str] = None
        self.branches: Dict[str, List[str]] = {"main": []}

    def create_root(self, domain_state: Dict[str, Any]) -> str:
        """创建根节点"""
        root_node = StateTreeNode(
            node_id=self._generate_id(),
            parent_id=None,
            step_number=0,
            worker_name="init",
            domain_state=domain_state.copy(),
            branch_name="main",
        )

        self.root_id = root_node.node_id
        self.nodes[root_node.node_id] = root_node
        self.current_node_id = root_node.node_id
        self.branches["main"].append(root_node.node_id)

        return root_node.node_id

    def add_node(
        self,
        parent_id: str,
        worker_name: str,
        domain_state: Dict[str, Any],
        patch: List[Dict[str, Any]],
        patch_valid: bool,
        patch_error: str = "",
        worker_instruction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加新节点"""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")

        parent_node = self.nodes[parent_id]

        # 创建新节点
        new_node = StateTreeNode(
            node_id=self._generate_id(),
            parent_id=parent_id,
            step_number=parent_node.step_number + 1,
            worker_name=worker_name,
            worker_instruction=worker_instruction,
            domain_state=domain_state.copy(),
            patch=patch.copy() if patch else [],
            patch_valid=patch_valid,
            patch_error=patch_error,
            metadata=metadata or {},
            branch_name=parent_node.branch_name,
        )

        # 更新父节点
        parent_node.children_ids.append(new_node.node_id)
        parent_node.is_leaf = False

        # 添加到树中
        self.nodes[new_node.node_id] = new_node
        self.current_node_id = new_node.node_id

        # 添加到分支
        branch_name = new_node.branch_name or "main"
        if branch_name not in self.branches:
            self.branches[branch_name] = []
        self.branches[branch_name].append(new_node.node_id)

        return new_node.node_id

    def get_node(self, node_id: str) -> Optional[StateTreeNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_current_node(self) -> Optional[StateTreeNode]:
        """获取当前节点"""
        if self.current_node_id:
            return self.nodes.get(self.current_node_id)
        return None

    def get_path_to_root(self, node_id: str) -> List[StateTreeNode]:
        """获取从节点到根的路径"""
        path = []
        current_id = node_id

        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))

    def get_children(self, node_id: str) -> List[StateTreeNode]:
        """获取子节点"""
        node = self.nodes.get(node_id)
        if not node:
            return []

        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def find_nodes_by_worker(self, worker_name: str) -> List[StateTreeNode]:
        """查找特定 worker 执行的所有节点"""
        return [node for node in self.nodes.values() if node.worker_name == worker_name]

    def find_nodes_by_status(self, status: str) -> List[StateTreeNode]:
        """查找特定状态的所有节点"""
        return [
            node for node in self.nodes.values()
            if node.domain_state.get("status") == status
        ]

    def get_execution_timeline(self) -> List[StateTreeNode]:
        """获取执行时间线（按步骤编号排序）"""
        return sorted(self.nodes.values(), key=lambda n: n.step_number)

    def mark_terminal(self, node_id: str, reason: str) -> None:
        """标记节点为终止节点"""
        node = self.nodes.get(node_id)
        if node:
            node.is_terminal = True
            node.terminal_reason = reason

    def calculate_depth(self, node_id: str) -> int:
        """计算节点深度"""
        return len(self.get_path_to_root(node_id)) - 1

    def get_branch_statistics(self, branch_name: str) -> Dict[str, Any]:
        """获取分支统计信息"""
        node_ids = self.branches.get(branch_name, [])
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]

        if not nodes:
            return {}

        return {
            "branch_name": branch_name,
            "node_count": len(nodes),
            "worker_distribution": self._count_workers(nodes),
            "error_count": sum(1 for n in nodes if not n.patch_valid),
            "terminal_nodes": sum(1 for n in nodes if n.is_terminal),
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        data = {
            "root_id": self.root_id,
            "current_node_id": self.current_node_id,
            "branches": self.branches,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def from_json(self, json_str: str) -> None:
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        self.root_id = data["root_id"]
        self.current_node_id = data["current_node_id"]
        self.branches = data["branches"]
        self.nodes = {
            nid: StateTreeNode.from_dict(node_data)
            for nid, node_data in data["nodes"].items()
        }

    def save_to_file(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def load_from_file(self, filepath: str) -> None:
        """从文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.from_json(f.read())

    def _generate_id(self) -> str:
        """生成唯一 ID"""
        return str(uuid.uuid4())

    def _count_workers(self, nodes: List[StateTreeNode]) -> Dict[str, int]:
        """统计 worker 分布"""
        counts: Dict[str, int] = {}
        for node in nodes:
            worker = node.worker_name
            counts[worker] = counts.get(worker, 0) + 1
        return counts
