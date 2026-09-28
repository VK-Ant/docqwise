"""MCP (Model Context Protocol) server for docqwise.

Exposes docqwise document intelligence as MCP tools that any AI agent can call.
Communicates via JSON-RPC 2.0 over stdin/stdout (MCP transport).

Usage:
    # Start as MCP server (stdio transport):
    python -m docqwise.server.mcp_server

    # Or via CLI:
    docqwise serve --mcp

    # In your MCP client config (e.g. claude_desktop_config.json):
    {
        "mcpServers": {
            "docqwise": {
                "command": "python",
                "args": ["-m", "docqwise.server.mcp_server"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from typing import Any

logger = logging.getLogger("docqwise.mcp")

# MCP Protocol version
MCP_VERSION = "2024-11-05"
SERVER_NAME = "docqwise"
SERVER_VERSION = "0.4.0"


# ══════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "docqwise_extract",
        "description": (
            "Extract structured fields from a document (PDF, DOCX, image, etc.). "
            "Supports templates: invoice, contract, resume, receipt. "
            "Methods: auto, rag, llm, vision, agentic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
                "template": {
                    "type": "string",
                    "description": "Pre-built template: invoice, contract, resume, receipt",
                    "enum": ["invoice", "contract", "resume", "receipt"],
                },
                "schema": {
                    "type": "object",
                    "description": "Custom field schema {field_name: type}",
                },
                "method": {
                    "type": "string",
                    "description": "Extraction method",
                    "enum": ["auto", "rag", "llm", "vision", "agentic"],
                    "default": "auto",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_extract_agentic",
        "description": (
            "Multi-pass, self-correcting document extraction using an AI agent. "
            "Validates, retries failed fields, and cross-checks results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
                "template": {
                    "type": "string",
                    "description": "Pre-built template name",
                },
                "schema": {
                    "type": "object",
                    "description": "Custom field schema",
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum retry passes",
                    "default": 2,
                },
                "cross_check": {
                    "type": "boolean",
                    "description": "Cross-check with second method",
                    "default": True,
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_ask",
        "description": (
            "Ask a question about ingested documents. "
            "Uses RAG (general, graphrag, or multimodal) for retrieval-augmented answers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask",
                },
                "source": {
                    "type": "string",
                    "description": "Specific document to query",
                },
                "mode": {
                    "type": "string",
                    "description": "RAG mode: general, graphrag, multimodal",
                    "enum": ["general", "graphrag", "multimodal"],
                    "default": "general",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "docqwise_ingest",
        "description": (
            "Ingest documents into the vector store for retrieval. "
            "Supports files, folders, and database connections."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "File path, folder path, or database URI",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_classify",
        "description": "Classify document type using zero-shot classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom classification labels",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_extract_tables",
        "description": "Extract tables from a document as structured data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_extract_entities",
        "description": "Extract named entities (people, organizations, dates, amounts, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity types to extract: PERSON, ORG, DATE, MONEY, etc.",
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "llm"],
                    "default": "auto",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_compare",
        "description": "Compare two documents and return differences.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_a": {
                    "type": "string",
                    "description": "Path to first document",
                },
                "source_b": {
                    "type": "string",
                    "description": "Path to second document",
                },
            },
            "required": ["source_a", "source_b"],
        },
    },
    {
        "name": "docqwise_detect_pii",
        "description": "Detect personally identifiable information (PII) in a document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the document file",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "docqwise_retrieve",
        "description": "Semantic search across ingested documents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]


# ══════════════════════════════════════════
# MCP SERVER
# ══════════════════════════════════════════

class DocqwiseMCPServer:
    """MCP server that wraps docqwise as tool provider.

    Implements the Model Context Protocol (JSON-RPC 2.0 over stdio).
    """

    def __init__(self, store_path: str = "./docqwise_db"):
        self.store_path = store_path
        self._engine = None
        self._initialized = False

    def _ensure_engine(self):
        """Lazy-init the docqwise engine."""
        if not self._initialized:
            from docqwise.engine import Docqwise
            self._engine = Docqwise(store_path=self.store_path)
            self._initialized = True
        return self._engine

    # ── JSON-RPC handlers ──

    def handle_initialize(self, params: dict) -> dict:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request."""
        return {"tools": MCP_TOOLS}

    def handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request — dispatch to the right tool."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            result = self._dispatch_tool(name, arguments)
            return {
                "content": [
                    {"type": "text", "text": json.dumps(result, default=str, indent=2)}
                ],
            }
        except FileNotFoundError as e:
            return {
                "content": [{"type": "text", "text": f"File not found: {e}"}],
                "isError": True,
            }
        except Exception as e:
            logger.error(f"Tool error: {name}: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    def _dispatch_tool(self, name: str, args: dict) -> Any:
        """Dispatch tool call to engine method."""
        engine = self._ensure_engine()

        if name == "docqwise_extract":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            result = engine.extract_fields(
                source,
                schema=args.get("schema"),
                template=args.get("template"),
                method=args.get("method", "auto"),
            )
            return {
                "fields": result.to_dict(),
                "confidence": result.confidence,
                "strategy": result.strategy_used,
            }

        elif name == "docqwise_extract_agentic":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            result = engine.extract_agentic(
                source,
                schema=args.get("schema"),
                template=args.get("template"),
                max_retries=args.get("max_retries", 2),
                cross_check=args.get("cross_check", True),
            )
            return {
                "fields": result.to_dict(),
                "confidence": result.confidence,
                "strategy": result.strategy_used,
                "processing_time_ms": result.processing_time_ms,
                "warnings": result.warnings,
            }

        elif name == "docqwise_ask":
            result = engine.ask_rag(
                args["question"],
                mode=args.get("mode", "general"),
                source=args.get("source"),
            )
            return result

        elif name == "docqwise_ingest":
            source = args["source"]
            if not source.startswith(("sqlite:///", "http://", "https://")) \
                    and not os.path.exists(source):
                raise FileNotFoundError(source)
            return engine.ingest(source)

        elif name == "docqwise_classify":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            labels = engine.classify(source, labels=args.get("labels"))
            return {"classifications": labels}

        elif name == "docqwise_extract_tables":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            tables = engine.extract_tables(source)
            return {
                "tables": [
                    {"headers": t.headers, "rows": t.rows, "page": t.page}
                    for t in tables
                ]
            }

        elif name == "docqwise_extract_entities":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            entities = engine.extract_entities(
                source,
                types=args.get("types"),
                method=args.get("method", "auto"),
            )
            return {
                "entities": [
                    {"text": e.text, "type": e.entity_type,
                     "start": e.start, "end": e.end}
                    for e in entities
                ]
            }

        elif name == "docqwise_compare":
            a, b = args["source_a"], args["source_b"]
            if not os.path.exists(a):
                raise FileNotFoundError(a)
            if not os.path.exists(b):
                raise FileNotFoundError(b)
            return engine.compare(a, b)

        elif name == "docqwise_detect_pii":
            source = args["source"]
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            pii = engine.detect_pii(source)
            return {"pii_detected": pii}

        elif name == "docqwise_retrieve":
            results = engine.retrieve(args["query"], top_k=args.get("top_k", 5))
            return {"results": results}

        else:
            raise ValueError(f"Unknown tool: {name}")

    # ── Main loop ──

    def handle_message(self, message: dict) -> dict | None:
        """Handle a single JSON-RPC message."""
        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        # Notifications (no id) — just acknowledge
        if method == "notifications/initialized":
            return None
        if method == "notifications/cancelled":
            return None

        # Requests (have id)
        result = None
        error = None

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            elif method == "ping":
                result = {}
            else:
                error = {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                }
        except Exception as e:
            error = {
                "code": -32603,
                "message": str(e),
            }

        if msg_id is None:
            return None

        response = {"jsonrpc": "2.0", "id": msg_id}
        if error:
            response["error"] = error
        else:
            response["result"] = result
        return response

    def run(self, port: int = None):
        """Run the MCP server.

        If port is given, starts HTTP server (FastAPI).
        Otherwise, runs stdio JSON-RPC transport (standard MCP).
        """
        if port:
            self._run_http(port)
        else:
            self._run_stdio()

    def _run_stdio(self):
        """Standard MCP transport: JSON-RPC over stdin/stdout."""
        logger.info("DocQWise MCP server starting (stdio transport)")

        # Read/write in line-delimited JSON mode
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_message(message)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def _run_http(self, port: int):
        """HTTP transport for testing and web integrations."""
        try:
            from docqwise.server.api import run_server
            run_server(port=port, store_path=self.store_path)
        except ImportError:
            logger.error("HTTP server requires: pip install docqwise[server]")
            raise

    def get_tools(self) -> list[dict]:
        """List available tools (for introspection)."""
        return MCP_TOOLS


def main():
    """Entry point for MCP server."""
    import argparse
    parser = argparse.ArgumentParser(description="DocQWise MCP Server")
    parser.add_argument("--port", type=int, default=None,
                        help="HTTP port (omit for stdio MCP transport)")
    parser.add_argument("--store-path", type=str, default="./docqwise_db",
                        help="Vector store path")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    server = DocqwiseMCPServer(store_path=args.store_path)
    server.run(port=args.port)


if __name__ == "__main__":
    main()
