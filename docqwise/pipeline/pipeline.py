"""Pipeline DAG engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable

@dataclass
class PipelineNode:
    name: str
    node_type: str
    config: dict = field(default_factory=dict)
    processor: Callable = None

class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, PipelineNode] = {}
        self.edges: list[tuple[str, str]] = []
        self._adj: dict[str, list[str]] = defaultdict(list)
        self._deps: dict[str, list[str]] = defaultdict(list)

    def add_node(self, name: str, node_type: str = "custom", processor: Callable = None, **config) -> "Pipeline":
        self.nodes[name] = PipelineNode(name=name, node_type=node_type, config=config, processor=processor)
        return self

    def connect(self, from_node: str, to_node: str) -> "Pipeline":
        self.edges.append((from_node, to_node))
        self._adj[from_node].append(to_node)
        self._deps[to_node].append(from_node)
        return self

    def validate(self) -> bool:
        visited = set()
        rec_stack = set()
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._adj.get(node, []):
                if neighbor not in visited and has_cycle(neighbor):
                    return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False
        for node in self.nodes:
            if node not in visited and has_cycle(node):
                return False
        return True

    def topological_sort(self) -> list[list[str]]:
        in_degree = {n: 0 for n in self.nodes}
        for _, to in self.edges:
            in_degree[to] = in_degree.get(to, 0) + 1
        levels = []
        remaining = dict(in_degree)
        while remaining:
            level = [n for n, d in remaining.items() if d == 0]
            if not level:
                break
            levels.append(level)
            for n in level:
                del remaining[n]
                for neighbor in self._adj.get(n, []):
                    if neighbor in remaining:
                        remaining[neighbor] -= 1
        return levels

    def run(self, input_data: Any = None, **kwargs) -> dict:
        if not self.validate():
            raise ValueError("Pipeline contains a cycle")
        levels = self.topological_sort()
        results = {}
        for level in levels:
            for node_name in level:
                node = self.nodes[node_name]
                deps_data = {d: results.get(d) for d in self._deps.get(node_name, [])}
                node_input = deps_data if deps_data else input_data
                if node.processor:
                    results[node_name] = node.processor(node_input, **node.config)
                else:
                    results[node_name] = node_input
        return results

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        pipe = cls(data.get("name", "pipeline"))
        for name, config in data.get("nodes", {}).items():
            pipe.add_node(name, node_type=config.get("type", "custom"), **{k: v for k, v in config.items() if k != "type"})
        flow = data.get("flow", "")
        if isinstance(flow, str):
            for connection in flow.replace("→", "->").split("\n"):
                parts = [p.strip() for p in connection.split("->") if p.strip()]
                for i in range(len(parts) - 1):
                    pipe.connect(parts[i], parts[i + 1])
        return pipe

    def visualize(self) -> str:
        levels = self.topological_sort()
        lines = [f"Pipeline: {self.name}", ""]
        for i, level in enumerate(levels):
            lines.append(f"Level {i}: {' | '.join(level)}")
            if i < len(levels) - 1:
                for node in level:
                    for neighbor in self._adj.get(node, []):
                        lines.append(f"  {node} -> {neighbor}")
        return "\n".join(lines)
