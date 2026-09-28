"""GraphRAG engine — entity-graph-enhanced retrieval and Q&A.

Extracts entities from documents, builds a knowledge graph,
traverses the graph for context, then uses LLM for answers
with evidence chains.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("docqwise")


@dataclass
class DocumentGraph:
    """In-memory document knowledge graph."""
    nodes: dict = field(default_factory=dict)  # {entity_text: {type, sources, fields}}
    edges: list = field(default_factory=list)   # [{source, target, relation, doc_id}]
    documents: dict = field(default_factory=dict)  # {doc_id: {path, entities, fields}}

    def add_entity(self, text: str, entity_type: str, doc_id: str, source_path: str = ""):
        key = text.lower().strip()
        if key not in self.nodes:
            self.nodes[key] = {
                "text": text, "type": entity_type,
                "sources": set(), "doc_ids": set(),
            }
        self.nodes[key]["sources"].add(source_path)
        self.nodes[key]["doc_ids"].add(doc_id)

    def add_edge(self, source: str, target: str, relation: str, doc_id: str = ""):
        self.edges.append({
            "source": source.lower().strip(),
            "target": target.lower().strip(),
            "relation": relation,
            "doc_id": doc_id,
        })

    def neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        """Get neighboring entities up to depth."""
        key = entity.lower().strip()
        visited = {key}
        current = [key]
        result = []

        for _ in range(depth):
            next_level = []
            for node in current:
                for edge in self.edges:
                    neighbor = None
                    if edge["source"] == node and edge["target"] not in visited:
                        neighbor = edge["target"]
                    elif edge["target"] == node and edge["source"] not in visited:
                        neighbor = edge["source"]
                    if neighbor and neighbor in self.nodes:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        result.append({
                            **self.nodes[neighbor],
                            "relation": edge["relation"],
                            "sources": list(self.nodes[neighbor].get("sources", set())),
                            "doc_ids": list(self.nodes[neighbor].get("doc_ids", set())),
                        })
            current = next_level

        return result

    def to_dict(self) -> dict:
        """Serialize graph."""
        nodes = {}
        for k, v in self.nodes.items():
            nodes[k] = {**v, "sources": list(v.get("sources", set())),
                        "doc_ids": list(v.get("doc_ids", set()))}
        return {"nodes": nodes, "edges": self.edges, "documents": self.documents}

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "documents": len(self.documents),
        }


class GraphRAGEngine:
    """Build knowledge graphs from documents and query them with LLM."""

    def __init__(self, llm=None, embedder=None):
        self._llm = llm
        self._embedder = embedder
        self.graph = DocumentGraph()

    def _ensure_llm(self):
        if self._llm:
            return self._llm
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            from docqwise.llm.ollama import OllamaLLM
            self._llm = OllamaLLM(model="nemotron-mini")
            return self._llm
        except Exception:
            pass
        return None

    def build_from_documents(self, documents: list, reader=None) -> DocumentGraph:
        """Build graph from document objects or file paths."""
        for doc_or_path in documents:
            if isinstance(doc_or_path, str) and reader:
                doc = reader.read(doc_or_path)
            else:
                doc = doc_or_path

            doc_id = getattr(doc, "doc_id", str(id(doc)))
            source = getattr(doc, "source_path", "")
            text = getattr(doc, "text", str(doc))

            # Extract entities
            entities = self._extract_entities(text, doc_id, source)

            # Extract fields
            fields = self._extract_fields_simple(text)

            self.graph.documents[doc_id] = {
                "path": source,
                "entities": [e["text"] for e in entities],
                "fields": fields,
            }

            # Add edges between entities in same document
            entity_keys = [e["text"].lower().strip() for e in entities]
            for i, e1 in enumerate(entity_keys):
                for e2 in entity_keys[i + 1:]:
                    if e1 != e2:
                        self.graph.add_edge(e1, e2, "co_occurs", doc_id)

        return self.graph

    def build_from_ingested(self, sources: list[str], reader=None) -> DocumentGraph:
        """Build graph from file paths."""
        return self.build_from_documents(sources, reader=reader)

    def query(self, question: str, source: str = None, prompt: str = None) -> dict:
        """Query the graph with a natural language question."""
        llm = self._ensure_llm()

        # Find relevant entities from the question
        question_entities = self._find_question_entities(question)

        # Get graph context via traversal
        graph_context = []
        highlighted_nodes = set()
        for entity in question_entities:
            neighbors = self.graph.neighbors(entity, depth=2)
            for n in neighbors:
                highlighted_nodes.add(n.get("text", "").lower())
                graph_context.append(
                    f"- {n['text']} ({n['type']}): related via '{n.get('relation', 'unknown')}'"
                )
            if entity in self.graph.nodes:
                highlighted_nodes.add(entity)
                node = self.graph.nodes[entity]
                graph_context.append(
                    f"- {node['text']} ({node['type']}): found in {list(node.get('sources', set()))}"
                )

        context_text = "\n".join(graph_context) if graph_context else "No graph context found."

        # Build evidence chain
        evidence = []
        for entity in question_entities:
            if entity in self.graph.nodes:
                node = self.graph.nodes[entity]
                evidence.append({
                    "entity": node["text"],
                    "type": node["type"],
                    "sources": list(node.get("sources", set())),
                })

        if llm:
            if prompt:
                if "{context}" in prompt:
                    final_prompt = prompt.replace("{context}", context_text)
                else:
                    final_prompt = f"{prompt}\n\nGraph context:\n{context_text}"
            else:
                final_prompt = (
                    f"Answer this question using the knowledge graph context below.\n\n"
                    f"Question: {question}\n\n"
                    f"Graph context:\n{context_text}\n\n"
                    f"Answer concisely with evidence."
                )
            answer = llm.generate(final_prompt)
        else:
            answer = context_text

        source_name = ""
        if evidence:
            source_name = evidence[0].get("sources", [""])[0]

        return {
            "answer": answer,
            "evidence": evidence,
            "source": source_name,
            "source_name": os.path.basename(source_name) if source_name else "",
            "method": "graphrag",
            "confidence": min(0.95, 0.7 + 0.05 * len(evidence)),
            "highlighted_nodes": list(highlighted_nodes),
            "graph_stats": self.graph.stats(),
        }

    def _extract_entities(self, text: str, doc_id: str, source: str) -> list[dict]:
        """Extract entities using regex patterns."""
        entities = []
        patterns = {
            "MONEY": r'(?:Rs\.?|₹|\$|€|£)\s*[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*(?:USD|INR|EUR|GBP)',
            "DATE": r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}',
            "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "ORG": r'(?:(?:[A-Z][a-z]+\s+){1,3}(?:Inc|LLC|Ltd|Corp|Co|Group|Solutions|Services|Technologies|Enterprises|International)\.?)',
            "PERSON": r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
        }

        for etype, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entity_text = match.group().strip()
                if len(entity_text) > 2:
                    self.graph.add_entity(entity_text, etype, doc_id, source)
                    entities.append({"text": entity_text, "type": etype})

        return entities

    def _extract_fields_simple(self, text: str) -> dict:
        """Extract key-value pairs with simple regex."""
        fields = {}
        for line in text.split("\n"):
            match = re.match(r'^([A-Za-z][A-Za-z\s]{2,30}):\s*(.+)$', line.strip())
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                value = match.group(2).strip()
                if value and len(value) < 200:
                    fields[key] = value
        return fields

    def _find_question_entities(self, question: str) -> list[str]:
        """Find entities from the graph that appear in the question."""
        q_lower = question.lower()
        found = []
        for key, node in self.graph.nodes.items():
            if key in q_lower or node["text"].lower() in q_lower:
                found.append(key)

        # Also try partial word matching
        q_words = set(q_lower.split())
        for key, node in self.graph.nodes.items():
            if key not in found:
                node_words = set(key.split())
                if node_words & q_words:
                    found.append(key)

        return found[:10]  # Limit


class GraphVisualizer:
    """Visualize document knowledge graphs."""

    def __init__(self, graph: DocumentGraph):
        self.graph = graph
        self._highlighted = set()

    @classmethod
    def from_query(cls, graph: DocumentGraph, query_result: dict) -> "GraphVisualizer":
        """Create visualizer with query-aware highlighting."""
        viz = cls(graph)
        viz._highlighted = set(query_result.get("highlighted_nodes", []))
        return viz

    def to_image(self, output: str, highlight: set = None, format: str = "png",
                 figsize: tuple = (16, 12), dpi: int = 150) -> str:
        """Render graph to PNG/SVG/PDF using matplotlib + networkx."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("Install matplotlib and networkx: pip install matplotlib networkx")

        G = nx.Graph()
        highlights = highlight or self._highlighted

        # Color map by entity type
        type_colors = {
            "ORG": "#4A90D9", "PERSON": "#50C878", "MONEY": "#FFD700",
            "DATE": "#FF6B6B", "EMAIL": "#9B59B6", "PHONE": "#E67E22",
            "LOCATION": "#1ABC9C",
        }

        # Add nodes
        for key, node in self.graph.nodes.items():
            G.add_node(key, label=node["text"], type=node["type"])

        # Add edges
        for edge in self.graph.edges:
            if edge["source"] in G.nodes and edge["target"] in G.nodes:
                G.add_edge(edge["source"], edge["target"],
                           relation=edge["relation"])

        if len(G.nodes) == 0:
            logger.warning("Empty graph — nothing to visualize")
            return output

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node colors and sizes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            ntype = G.nodes[node].get("type", "")
            base_color = type_colors.get(ntype, "#CCCCCC")
            if highlights and node in highlights:
                node_colors.append("#FF4444")
                node_sizes.append(800)
            else:
                node_colors.append(base_color)
                node_sizes.append(400)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#888888", ax=ax)

        # Labels
        labels = {n: G.nodes[n].get("label", n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

        ax.set_title("Document Knowledge Graph", fontsize=14, fontweight="bold")
        ax.axis("off")

        # Legend
        legend_elements = []
        from matplotlib.patches import Patch
        for etype, color in type_colors.items():
            if any(G.nodes[n].get("type") == etype for n in G.nodes()):
                legend_elements.append(Patch(facecolor=color, label=etype))
        if highlights:
            legend_elements.append(Patch(facecolor="#FF4444", label="Query Match"))
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        # Determine format from extension
        ext = os.path.splitext(output)[1].lower()
        if ext in (".svg", ".pdf"):
            format = ext[1:]

        plt.tight_layout()
        plt.savefig(output, format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Graph saved: {output} ({len(G.nodes)} nodes, {len(G.edges)} edges)")
        return output

    def to_html(self, output: str) -> str:
        """Render interactive graph using vis.js."""
        highlights = self._highlighted

        nodes_js = []
        for key, node in self.graph.nodes.items():
            type_colors = {
                "ORG": "#4A90D9", "PERSON": "#50C878", "MONEY": "#FFD700",
                "DATE": "#FF6B6B", "EMAIL": "#9B59B6", "PHONE": "#E67E22",
            }
            color = "#FF4444" if key in highlights else type_colors.get(node["type"], "#CCCCCC")
            size = 30 if key in highlights else 15
            nodes_js.append({
                "id": key, "label": node["text"][:25],
                "color": color, "size": size,
                "title": f"{node['text']} ({node['type']})",
            })

        edges_js = []
        for i, edge in enumerate(self.graph.edges):
            edges_js.append({
                "from": edge["source"], "to": edge["target"],
                "title": edge["relation"], "id": str(i),
            })

        html = f"""<!DOCTYPE html>
<html><head><title>DocQWise Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/vis-network.min.js"></script>
<style>body{{margin:0;font-family:sans-serif}}#graph{{width:100%;height:100vh}}
h2{{text-align:center;padding:10px;margin:0;background:#1a1a2e;color:white}}</style></head>
<body><h2>DocQWise Knowledge Graph</h2><div id="graph"></div>
<script>
var nodes = new vis.DataSet({json.dumps(nodes_js)});
var edges = new vis.DataSet({json.dumps(edges_js)});
var container = document.getElementById('graph');
var data = {{nodes: nodes, edges: edges}};
var options = {{
  physics: {{stabilization: {{iterations: 100}}}},
  nodes: {{shape: 'dot', font: {{size: 12}}}},
  edges: {{color: '#888', arrows: 'to', smooth: true}}
}};
new vis.Network(container, data, options);
</script></body></html>"""

        with open(output, "w") as f:
            f.write(html)
        logger.info(f"Interactive graph saved: {output}")
        return output

    def save(self, output: str) -> str:
        """Save graph — auto-detect format from extension."""
        ext = os.path.splitext(output)[1].lower()
        if ext == ".html":
            return self.to_html(output)
        elif ext in (".png", ".svg", ".pdf"):
            return self.to_image(output, format=ext[1:])
        else:
            return self.to_image(output)
