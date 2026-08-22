from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError

_SPARSE_VOCAB_SIZE = 25000


def _get_embedding_client() -> Any:
    """模块级单例：复用 OpenAIEmbeddings 客户端，避免每次调用重新实例化。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise AppError("MODEL_NOT_CONFIGURED", "OPENAI_API_KEY is required for vector routing.", 503)
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise AppError("DEPENDENCY_MISSING", "Install langchain-openai to use embeddings.", 503) from exc
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        check_embedding_ctx_length=False,
    )


def _tokenize(text: str) -> list[str]:
    """中文用 jieba 分词，英文/混合文本回退到空格分词。"""
    try:
        import jieba
    except ImportError:
        return text.lower().split()
    tokens: list[str] = []
    for segment in jieba.cut(text):
        segment = segment.strip()
        if segment:
            tokens.append(segment.lower())
    return tokens or text.lower().split()


def _sparse_dict(text: str) -> dict[int, float]:
    """将文本转为稀疏向量字典：{token_hash_index: normalized_tf}。"""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: dict[int, float] = {}
    for token in tokens:
        idx = int(hashlib.sha256(token.encode("utf-8")).digest()[0:4].hex(), 16) % _SPARSE_VOCAB_SIZE
        tf[idx] = tf.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {k: v / norm for k, v in tf.items()}


class EmbeddingProvider:
    """向量化抽象接口：稠密向量 + 稀疏向量。"""

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    async def sparse_embed(self, text: str) -> dict[int, float]:
        return _sparse_dict(text)

    async def sparse_embed_many(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [_sparse_dict(t) for t in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI Embeddings 的真实向量化实现，客户端复用。"""

    def __init__(self):
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = _get_embedding_client()
        return self._client

    async def embed(self, text: str) -> list[float]:
        return await self._ensure_client().aembed_query(text)

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """批量嵌入：单次 HTTP 请求提交全部文本，大幅减少往返延迟。"""
        if not texts:
            return []
        return await self._ensure_client().aembed_documents(list(texts))


class HashEmbeddingProvider(EmbeddingProvider):
    """确定性向量化实现（仅用于单元测试，生产环境不使用）。

    对每个词做 SHA-256 哈希，映射到固定维度并叠加正负号，最后做 L2 归一化，
    保证相同文本始终得到相同向量，便于测试路由相似度。
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个向量的余弦相似度；维度不一致时抛出 ValueError。"""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)