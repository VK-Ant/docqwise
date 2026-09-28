"""Demo 06: MCP Server (v0.4.0)

Demonstrates docqwise as a Model Context Protocol (MCP) server
that any AI agent (Claude, GPT, etc.) can use as a tool provider.

Usage:
    # Start the MCP server (stdio):
    python -m docqwise.server.mcp_server

    # Start with HTTP transport:
    python -m docqwise.server.mcp_server --port 8080

    # This demo simulates MCP tool calls programmatically:
    python demo/06_mcp_server.py

Requirements:
    pip install docqwise
"""

import json
import os
from docqwise.server.mcp_server import DocqwiseMCPServer


def demo_mcp_tools():
    """List available MCP tools."""
    print("=" * 60)
    print("DEMO: MCP Server — Available Tools")
    print("=" * 60)

    server = DocqwiseMCPServer()

    # Simulate MCP initialize
    init_result = server.handle_initialize({})
    print(f"\nServer: {init_result['serverInfo']['name']} v{init_result['serverInfo']['version']}")
    print(f"Protocol: {init_result['protocolVersion']}")

    # List tools
    tools = server.handle_tools_list({})
    print(f"\nAvailable tools ({len(tools['tools'])}):")
    for tool in tools["tools"]:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")


def demo_mcp_extract():
    """Simulate MCP tool call for extraction."""
    print("\n" + "=" * 60)
    print("DEMO: MCP Tool Call — Extract Fields")
    print("=" * 60)

    server = DocqwiseMCPServer()

    sample = "demo/sample_invoice.pdf"
    if not os.path.exists(sample):
        print(f"  [skip] {sample} not found")
        return

    # Simulate tools/call
    result = server.handle_tools_call({
        "name": "docqwise_extract",
        "arguments": {
            "source": sample,
            "template": "invoice",
            "method": "auto",  # auto selects best AI method available
        },
    })

    print("\nMCP Response:")
    for content in result.get("content", []):
        if content["type"] == "text":
            data = json.loads(content["text"])
            print(json.dumps(data, indent=2, default=str)[:500])


def demo_mcp_classify():
    """Simulate MCP tool call for classification."""
    print("\n" + "=" * 60)
    print("DEMO: MCP Tool Call — Classify Document")
    print("=" * 60)

    server = DocqwiseMCPServer()

    sample = "demo/sample_invoice.pdf"
    if not os.path.exists(sample):
        print(f"  [skip] {sample} not found")
        return

    result = server.handle_tools_call({
        "name": "docqwise_classify",
        "arguments": {"source": sample},
    })

    print("\nMCP Response:")
    for content in result.get("content", []):
        if content["type"] == "text":
            print(content["text"][:300])


def demo_mcp_jsonrpc():
    """Simulate full JSON-RPC message flow."""
    print("\n" + "=" * 60)
    print("DEMO: MCP JSON-RPC Message Flow")
    print("=" * 60)

    server = DocqwiseMCPServer()

    # 1. Initialize
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = server.handle_message(msg)
    print(f"\n1. Initialize → {resp['result']['serverInfo']['name']}")

    # 2. Notification (no response expected)
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    resp = server.handle_message(msg)
    print(f"2. Notification → {resp}")  # None

    # 3. List tools
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = server.handle_message(msg)
    print(f"3. List tools → {len(resp['result']['tools'])} tools available")

    # 4. Ping
    msg = {"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}}
    resp = server.handle_message(msg)
    print(f"4. Ping → {resp['result']}")


def demo_claude_config():
    """Show Claude Desktop config for docqwise MCP."""
    print("\n" + "=" * 60)
    print("DEMO: Claude Desktop Configuration")
    print("=" * 60)

    config = {
        "mcpServers": {
            "docqwise": {
                "command": "python",
                "args": ["-m", "docqwise.server.mcp_server"],
            }
        }
    }

    print("\nAdd this to your claude_desktop_config.json:")
    print(json.dumps(config, indent=2))
    print("\nLocation:")
    print("  macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("  Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
    print("  Linux: ~/.config/Claude/claude_desktop_config.json")


if __name__ == "__main__":
    demo_mcp_tools()
    demo_mcp_extract()
    demo_mcp_classify()
    demo_mcp_jsonrpc()
    demo_claude_config()
    print("\n" + "=" * 60)
    print("All MCP demos complete!")
    print("=" * 60)
