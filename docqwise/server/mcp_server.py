"""MCP server mode for docqwise."""
from __future__ import annotations
import json

class DocqwiseMCPServer:
    def __init__(self, store_path: str = "./docqwise_db"):
        self.store_path = store_path
        self.tools = {
            "docqwise_extract": {"description": "Extract fields from a document"},
            "docqwise_retrieve": {"description": "Semantic search across corpus"},
            "docqwise_classify": {"description": "Classify document type"},
            "docqwise_tables": {"description": "Extract tables from document"},
            "docqwise_entities": {"description": "Extract entities from document"},
            "docqwise_compare": {"description": "Compare two documents"},
            "docqwise_schema": {"description": "Detect schema from data"},
            "docqwise_pii": {"description": "Detect PII in document"},
        }

    def get_tools(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self.tools.items()]

    def run(self, port: int = 8080):
        print(f"DocQWise MCP Server starting on port {port}")
        print(f"Available tools: {list(self.tools.keys())}")
        from docqwise.server.api import run_server
        run_server(port=port, store_path=self.store_path)
