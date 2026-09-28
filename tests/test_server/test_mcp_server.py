"""Tests for MCP server module."""

from docqwise.server.mcp_server import DocqwiseMCPServer


def test_server_creation():
    server = DocqwiseMCPServer()
    assert server is not None


def test_initialize():
    server = DocqwiseMCPServer()
    result = server.handle_initialize({})
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "docqwise"
    assert "version" in result["serverInfo"]
    assert "tools" in result["capabilities"]


def test_tools_list():
    server = DocqwiseMCPServer()
    result = server.handle_tools_list({})
    tools = result["tools"]
    assert len(tools) >= 8

    tool_names = [t["name"] for t in tools]
    assert "docqwise_extract" in tool_names
    assert "docqwise_extract_agentic" in tool_names
    assert "docqwise_ask" in tool_names
    assert "docqwise_classify" in tool_names
    assert "docqwise_detect_pii" in tool_names


def test_tool_has_schema():
    server = DocqwiseMCPServer()
    result = server.handle_tools_list({})
    for tool in result["tools"]:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"


def test_jsonrpc_initialize():
    server = DocqwiseMCPServer()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = server.handle_message(msg)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp


def test_jsonrpc_tools_list():
    server = DocqwiseMCPServer()
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = server.handle_message(msg)
    assert "result" in resp
    assert "tools" in resp["result"]


def test_jsonrpc_ping():
    server = DocqwiseMCPServer()
    msg = {"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}}
    resp = server.handle_message(msg)
    assert resp["result"] == {}


def test_jsonrpc_notification():
    server = DocqwiseMCPServer()
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    resp = server.handle_message(msg)
    assert resp is None


def test_jsonrpc_unknown_method():
    server = DocqwiseMCPServer()
    msg = {"jsonrpc": "2.0", "id": 99, "method": "unknown/method", "params": {}}
    resp = server.handle_message(msg)
    assert "error" in resp
    assert resp["error"]["code"] == -32601
