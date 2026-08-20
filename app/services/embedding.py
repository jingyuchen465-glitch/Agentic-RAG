from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.errors import AppError


class EmbeddingProvider:
    """向量化抽象接口：生产用 OpenAI，测试用确定性哈希实现。"""

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI Embeddings 的真实向量化实现。"""

    async def embed(self, text: str) -> list[float]:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AppError("MODEL_NOT_CONFIGURED", "OPENAI_API_KEY is required for vector routing.", 503)
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install langchain-openai to use embeddings.", 503) from exc
        client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            # 阿里云百炼兼容模式不识别 OpenAI 长度检查所用的 contents 字段，
            # 关闭后走经典 input 参数，保证 text-embedding-v4 正常返回
            check_embedding_ctx_length=False,
        )
        return await client.aembed_query(text)


class HashEmbeddingProvider(EmbeddingProvider):
    """确定性向量化实现（仅用于单元测试，生产环境不使用）。

    对每个词做 SHA-256 哈希，映射到固定维度并叠加正负号，最后做 L2 归一化，
    保证相同文本始终得到相同向量，便于测试路由相似度。
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        # 初始化全零向量，按词哈希累加得到稀疏特征
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        # L2 归一化，避免文本长度影响相似度
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个向量的余弦相似度；维度不一致时抛出 ValueError。"""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
