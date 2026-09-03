from __future__ import annotations

from app.services.mcp_client import (
    _extract_json,
    bind_arguments,
    content_text,
    contents_text,
    messages_text,
)
from app.services.tool_router import mcp_tool_specs
from app.domain.models import AgentType


def test_extract_json_from_plain_response():
    """普通 JSON 响应应被原样解析。"""
    assert _extract_json('{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}') == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []},
    }


def test_extract_json_from_sse_response():
    """SSE 数据行应提取出 data 后的 JSON。"""
    sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}\n\n"
    assert _extract_json(sse) == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_extract_json_returns_none_for_garbage():
    """无法解析的响应体返回 None。"""
    assert _extract_json("not-json") is None
    assert _extract_json("") is None


def test_bind_arguments_uses_first_required_property():
    """有 required 参数时，任务目标映射到第一个必填参数。"""
    schema = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}
    assert bind_arguments(schema, "hello mcp") == {"message": "hello mcp"}


def test_bind_arguments_falls_back_to_query():
    """无 required 参数或 properties 时回退到 query 键。"""
    assert bind_arguments(None, "hello") == {"query": "hello"}
    assert bind_arguments({"type": "object", "properties": {}}, "hello") == {"query": "hello"}


def test_mcp_tool_specs_converts_and_keeps_schemas():
    """MCP 工具转换为带 mcp: 前缀的 ToolSpec，并保留 inputSchema 映射。"""
    tools = [
        {
            "name": "echo_dynamic",
            "description": "回显输入",
            "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        }
    ]
    specs, schemas = mcp_tool_specs(tools)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.tool_id == "mcp:echo_dynamic"
    assert spec.agent_type == AgentType.TOOLS
    assert spec.enabled is True
    assert schemas["mcp:echo_dynamic"]["required"] == ["message"]


def test_content_text_joins_tool_result():
    """tools/call 的 content 文本应拼接返回。"""
    result = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert content_text(result) == "a\nb"


def test_messages_text_joins_prompt_result():
    """prompts/get 的 messages 文本应拼接返回。"""
    result = {"messages": [{"role": "user", "content": {"type": "text", "text": "你好"}}]}
    assert messages_text(result) == "你好"


def test_contents_text_joins_resource_result():
    """resources/read 的 contents 文本应拼接返回。"""
    result = {"contents": [{"uri": "doc://1", "mimeType": "text/markdown", "text": "正文"}]}
    assert contents_text(result) == "正文"