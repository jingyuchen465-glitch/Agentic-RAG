from __future__ import annotations

import itertools
import json
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError

# MCP 客户端声明的协议版本与自身标识，供 initialize 握手使用。
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "agentic-rag", "version": "0.1.0"}


def _extract_json(text: str | None) -> dict[str, Any] | None:
    """从 MCP Streamable HTTP 响应体中提取 JSON 对象。

    中台后端（Spring AI / 改造 Filter）可能直接返回 JSON，也可能返回
    SSE 风格的 ``data: {...}`` 文本。这里先尝试整体解析，失败后按行扫描
    首个 ``data:`` 行再解析，兼容两种格式。
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for line in stripped.splitlines():
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:"):].strip()
        if not chunk:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


def content_text(result: dict[str, Any]) -> str:
    """拼接 tools/call 返回的 result.content 里的文本片段。"""
    content = result.get("content") or []
    texts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)


def messages_text(result: dict[str, Any]) -> str:
    """拼接 prompts/get 返回的 result.messages 里的文本片段。"""
    messages = result.get("messages") or []
    texts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, dict):
            texts.append(content.get("text", ""))
    return "\n".join(t for t in texts if t) or json.dumps(result, ensure_ascii=False)


def contents_text(result: dict[str, Any]) -> str:
    """拼接 resources/read 返回的 result.contents 里的文本片段。"""
    contents = result.get("contents") or []
    texts = [
        item.get("text", "")
        for item in contents
        if isinstance(item, dict)
    ]
    return "\n".join(t for t in texts if t) or json.dumps(result, ensure_ascii=False)


def bind_arguments(input_schema: dict[str, Any] | None, goal: str) -> dict[str, Any]:
    """按工具 inputSchema 把任务目标映射为 tools/call 的 arguments。

    作为 LLM 入参绑定的确定性兜底：若 schema 声明了 required 参数，取第一个
    必填参数承载任务目标；否则回退到 ``query``。
    """
    schema = input_schema or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if required and properties:
        key = str(required[0])
        return {key: goal}
    return {"query": goal}


class McpClient:
    """面向 MCP 中台的 Streamable HTTP 客户端。

    完整对齐 bear-mcp-single 六个网关 Filter 的能力面：
    - ApiKeyAuthFilter          -> Bearer Token 鉴权
    - McpInitializeCapabilitiesFilter -> initialize 能力声明（tools/prompts/resources）
    - McpToolsListFilter        -> tools/list
    - McpToolsCallFilter        -> tools/call
    - McpPromptsFilter          -> prompts/list、prompts/get
    - McpResourcesFilter        -> resources/list、resources/read
    协议为 JSON-RPC 2.0，携带 MCP 会话 ID 保持同一会话上下文。
    """

    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 30.0):
        settings = get_settings()
        base = (base_url or settings.mcp_base_url or "").strip().rstrip("/")
        self._endpoint = base if base.endswith("/mcp") else f"{base}/mcp"
        self._token = token or settings.mcp_token
        self._timeout = timeout
        self._session_id: str | None = None
        self._capabilities: dict[str, Any] = {}
        self._id_counter = itertools.count(1)

    @property
    def configured(self) -> bool:
        """是否已配置 MCP 中台地址与令牌；未配置时工作流应跳过 MCP 集成。"""
        return bool(self._endpoint != "/mcp" and self._token)

    @property
    def capabilities(self) -> dict[str, Any]:
        """initialize 之后中台声明的能力（prompts/resources 能力位）。"""
        return self._capabilities

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送一次 JSON-RPC 请求并解析响应，错误统一转为 AppError。"""
        try:
            import httpx
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install httpx to call MCP tools.", 503) from exc

        headers = {
            "Content-Type": "application/json",
            # 同时声明 JSON 与 SSE，客户端自行兼容两种响应格式。
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._token}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        payload = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
            "params": params,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._endpoint, headers=headers, json=payload)

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        data = _extract_json(response.text)
        if data is None:
            raise AppError("MCP_PROTOCOL_ERROR", f"MCP returned unrecognized body: {response.text[:300]}", 502)
        if "error" in data:
            message = data.get("error", {}).get("message") or "Unknown MCP error"
            raise AppError("MCP_TOOL_FAILED", str(message), 502)
        return data

    async def initialize(self) -> dict[str, Any]:
        """MCP 握手：协商协议版本、捕获会话 ID 与能力声明。"""
        data = await self._rpc(
            "initialize",
            {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO},
        )
        self._capabilities = data.get("result", {}).get("capabilities", {}) or {}
        return self._capabilities

    async def list_tools(self) -> list[dict[str, Any]]:
        """发现当前 token 可见的工具列表，返回 name/description/inputSchema。"""
        data = await self._rpc("tools/list", {})
        return list(data.get("result", {}).get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用工具并把 result.content 中的文本片段拼接返回。"""
        data = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return content_text(data.get("result", {}))

    async def list_prompts(self) -> list[dict[str, Any]]:
        """列出当前 token 可见的 Prompt 模板。"""
        data = await self._rpc("prompts/list", {})
        return list(data.get("result", {}).get("prompts", []))

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """按名称渲染 Prompt 模板，返回渲染后的消息文本。"""
        data = await self._rpc("prompts/get", {"name": name, "arguments": arguments})
        return messages_text(data.get("result", {}))

    async def list_resources(self) -> list[dict[str, Any]]:
        """列出当前 token 可见的 Resource 资源。"""
        data = await self._rpc("resources/list", {})
        return list(data.get("result", {}).get("resources", []))

    async def read_resource(self, uri: str) -> str:
        """按 uri 读取 Resource 文本内容。"""
        data = await self._rpc("resources/read", {"uri": uri})
        return contents_text(data.get("result", {}))