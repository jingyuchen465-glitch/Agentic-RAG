from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.errors import RuntimeConfigurationError


class RedisSessionStore:
    """基于 Redis 的匿名会话存储：保存/加载最近一次问答状态，带 TTL 过期。"""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or get_settings().redis_url
        self._client = None

    async def _get_client(self):
        """懒加载 Redis 异步客户端；未配置或缺少依赖时抛错。"""
        if not self.redis_url:
            raise RuntimeConfigurationError(["REDIS_URL"])
        if self._client is None:
            try:
                from redis.asyncio import Redis
            except ImportError as exc:
                raise RuntimeConfigurationError(["redis package"]) from exc
            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def save(self, session_id: str, state: dict[str, object]) -> None:
        """写入会话状态并设置过期时间（TTL）。"""
        client = await self._get_client()
        await client.setex(
            f"agentic-rag:session:{session_id}",
            get_settings().session_ttl_seconds,
            json.dumps(state, ensure_ascii=False),
        )

    async def load(self, session_id: str) -> dict[str, object] | None:
        """读取会话状态；不存在或已过期时返回 None。"""
        client = await self._get_client()
        value = await client.get(f"agentic-rag:session:{session_id}")
        return json.loads(value) if value else None
