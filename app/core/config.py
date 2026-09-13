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
    vision_model: str
    oss_access_key_id: str | None
    oss_access_key_secret: str | None
    oss_bucket: str | None
    oss_endpoint: str | None
    oss_public_base_url: str | None
    tavily_api_key: str | None
    mcp_base_url: str | None
    mcp_token: str | None
    max_upload_size_mb: int
    retrieval_top_k: int
    router_threshold: float
    router_ambiguity_delta: float
    retrieval_grounded_threshold: float
    max_agent_retries: int
    session_ttl_seconds: int
    chunk_size: int
    chunk_overlap: int
    max_chunk_size: int
    pdf_ocr_language: str
    pdf_render_dpi: int
    pdf_concurrency: int
    harness_max_steps: int
    harness_timeout_seconds: float
    harness_max_plan_tasks: int
>>>>>>> f63cdf9 (feat: add workflow harness controls and evaluation)

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
        vision_model=os.getenv("VISION_MODEL", "qwen-vl-plus"),
        oss_access_key_id=_optional("OSS_ACCESS_KEY_ID"),
        oss_access_key_secret=_optional("OSS_ACCESS_KEY_SECRET"),
        oss_bucket=_optional("OSS_BUCKET"),
        oss_endpoint=_optional("OSS_ENDPOINT"),
        oss_public_base_url=_optional("OSS_PUBLIC_BASE_URL"),
        tavily_api_key=_optional("TAVILY_API_KEY"),
        mcp_base_url=_optional("MCP_BASE_URL"),
        mcp_token=_optional("MCP_TOKEN"),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
        router_threshold=float(os.getenv("ROUTER_THRESHOLD", "0.62")),
        router_ambiguity_delta=float(os.getenv("ROUTER_AMBIGUITY_DELTA", "0.05")),
        retrieval_grounded_threshold=float(os.getenv("RETRIEVAL_GROUNDED_THRESHOLD", "0.35")),
        max_agent_retries=int(os.getenv("MAX_AGENT_RETRIES", "2")),
        session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
<<<<<<< HEAD
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "2000")),
        pdf_ocr_language=os.getenv("PDF_OCR_LANGUAGE", "ch"),
        pdf_render_dpi=int(os.getenv("PDF_RENDER_DPI", "144")),
        harness_max_steps=int(os.getenv("HARNESS_MAX_STEPS", "32")),
        harness_timeout_seconds=float(os.getenv("HARNESS_TIMEOUT_SECONDS", "120")),
        harness_max_plan_tasks=int(os.getenv("HARNESS_MAX_PLAN_TASKS", "16")),
    )
