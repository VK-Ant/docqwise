"""Document knowledge graph."""
from __future__ import annotations
from collections import defaultdict

class DocumentGraph:
    def __init__(self, backend=None):
        self._backend = backend
        self._nodes = {}
        self._edges = []
        self._adj = defaultdict(list)

    def add_document(self, doc_id: str, metadata: dict = None):
        self._nodes[doc_id] = {"type": "document", **(metadata or {})}

    def add_entity(self, entity_id: str, entity_type: str, text: str):
        self._nodes[entity_id] = {"type": entity_type, "text": text}

    def add_edge(self, from_id: str, to_id: str, relation: str, properties: dict = None):
        edge = {"from": from_id, "to": to_id, "relation": relation, **(properties or {})}
        self._edges.append(edge)
        self._adj[from_id].append((to_id, relation))
        self._adj[to_id].append((from_id, f"inv_{relation}"))

    def neighbors(self, node_id: str, hops: int = 1) -> list[dict]:
        visited = set()
        current = {node_id}
        for _ in range(hops):
            next_level = set()
            for nid in current:
                for neighbor, rel in self._adj.get(nid, []):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current = next_level
        return [{"id": nid, **self._nodes.get(nid, {})} for nid in visited]

    def query(self, text: str) -> list[dict]:
        results = []
        text_lower = text.lower()
        for nid, props in self._nodes.items():
            node_text = props.get("text", nid).lower()
            if text_lower in node_text or node_text in text_lower:
                results.append({"id": nid, **props, "neighbors": self.neighbors(nid)})
        return results

    def communities(self) -> list[list[str]]:
        visited = set()
        communities = []
        for node in self._nodes:
            if node not in visited:
                community = []
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    community.append(n)
                    for neighbor, _ in self._adj.get(n, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
                if community:
                    communities.append(community)
        return communities

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def to_dict(self) -> dict:
        return {"nodes": self._nodes, "edges": self._edges}
