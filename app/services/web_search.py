from __future__ import annotations

from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.models import Citation


class TavilySearch:
    """基于 Tavily API 的网络搜索：返回可引用的网络来源列表。"""

    async def search(self, query: str) -> list[Citation]:
        try:
            import httpx
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install httpx to use Tavily search.", 503) from exc
        api_key = get_settings().tavily_api_key
        if not api_key:
            raise AppError("WEB_SEARCH_NOT_CONFIGURED", "TAVILY_API_KEY is required.", 503)
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": 5, "include_answer": False},
            )
        if response.is_error:
            raise AppError("WEB_SEARCH_FAILED", "Tavily search failed.", 502, [response.text[:500]])
        # 仅保留同时包含正文与 URL 的结果，转成统一引用模型
        return [
            Citation(
                source_type="web",
                document_name=item.get("title", item["url"]),
                url=item["url"],
                content=item.get("content", ""),
            )
            for item in response.json().get("results", [])
            if item.get("content") and item.get("url")
        ]
