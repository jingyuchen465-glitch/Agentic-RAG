from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """应用运行配置，全部字段由环境变量注入，实例不可变。"""

    app_env: str
    api_v1_prefix: str
    upload_dir: Path
    database_url: str | None
    redis_url: str | None
    milvus_uri: str | None
    milvus_token: str | None
    milvus_database: str
    knowledge_collection: str
    tool_collection: str
    openai_api_key: str | None
    openai_base_url: str | None
    chat_model: str
    embedding_model: str
    tavily_api_key: str | None
    max_upload_size_mb: int
    retrieval_top_k: int
    router_threshold: float
    router_ambiguity_delta: float
    max_agent_retries: int
    session_ttl_seconds: int

    def missing_runtime_settings(self) -> list[str]:
        """返回尚未配置的外部服务环境变量名，用于健康检查与启动校验。"""
        required = {
            "DATABASE_URL": self.database_url,
            "REDIS_URL": self.redis_url,
            "MILVUS_URI": self.milvus_uri,
            "OPENAI_API_KEY": self.openai_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
        }
        return [name for name, value in required.items() if not value]


def _optional(name: str) -> str | None:
    """读取环境变量，空值或纯空白返回 None（表示未配置）。"""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例获取配置；lru_cache 保证只解析一次，测试中可用 cache_clear() 重置。"""
    load_dotenv()  # 读取项目根目录 .env，不覆盖已存在的进程环境变量
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
        upload_dir=Path(os.getenv("UPLOAD_DIR", "./data/uploads")),
        database_url=_optional("DATABASE_URL"),
        redis_url=_optional("REDIS_URL"),
        milvus_uri=_optional("MILVUS_URI"),
        milvus_token=_optional("MILVUS_TOKEN"),
        milvus_database=os.getenv("MILVUS_DATABASE", "default"),
        knowledge_collection=os.getenv("MILVUS_KNOWLEDGE_COLLECTION", "knowledge_chunks"),
        tool_collection=os.getenv("MILVUS_TOOL_COLLECTION", "tool_registry"),
        openai_api_key=_optional("OPENAI_API_KEY"),
        openai_base_url=_optional("OPENAI_BASE_URL"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        tavily_api_key=_optional("TAVILY_API_KEY"),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
        router_threshold=float(os.getenv("ROUTER_THRESHOLD", "0.62")),
        router_ambiguity_delta=float(os.getenv("ROUTER_AMBIGUITY_DELTA", "0.05")),
        max_agent_retries=int(os.getenv("MAX_AGENT_RETRIES", "2")),
        session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
    )
