from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import RuntimeConfigurationError
from app.domain.models import Citation
from app.services.embedding import OpenAIEmbeddingProvider

_MMR_LAMBDA = 0.7
_RRF_K = 60


@dataclass(slots=True)
class RetrievedChunk:
    """检索结果：一条引用信息、融合后的得分，以及稠密余弦相关度（0~1，可空）。"""

    citation: Citation
    score: float
    relevance: float | None = None


class HybridRetriever:
    """混合检索抽象接口（Milvus 稠密 + 稀疏），生产适配器实现同一方法。"""

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


def _sparse_dot(left: dict[int, float], right: dict[int, float]) -> float:
    """两个稀疏向量（字典格式）的内积。"""
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(idx, 0.0) for idx, weight in left.items())


def _mmr(
    candidates: list[tuple[dict, float, dict[int, float] | None, list[float] | None]],
    lambda_param: float,
    top_k: int,
) -> list[int]:
    """MMR 多样性重排：从候选集中选 top_k 个兼顾相关性与多样性的结果。

    candidates: [(entity, score, sparse_vec, dense_vec), ...]
    返回入选候选在原始列表中的下标。
    """
    if not candidates:
        return []
    n = len(candidates)
    selected: list[int] = []
    remaining = list(range(n))
    # 第一轮选最高分
    best = max(remaining, key=lambda i: candidates[i][1])
    selected.append(best)
    remaining.remove(best)
    while len(selected) < min(top_k, n) and remaining:
        best_idx = -1
        best_score = -float("inf")
        for idx in remaining:
            relevance = candidates[idx][1]
            # 与已选结果的最大相似度
            max_sim = 0.0
            for s_idx in selected:
                sim = _candidate_similarity(candidates[idx], candidates[s_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _candidate_similarity(
    left: tuple[dict, float, dict[int, float] | None, list[float] | None],
    right: tuple[dict, float, dict[int, float] | None, list[float] | None],
) -> float:
    """计算两个候选之间的相似度：优先用稠密向量余弦，其次用稀疏向量内积。"""
    _, _, l_sparse, l_dense = left
    _, _, r_sparse, r_dense = right
    if l_dense is not None and r_dense is not None:
        return _cosine(l_dense, r_dense)
    if l_sparse is not None and r_sparse is not None:
        return _sparse_dot(l_sparse, r_sparse)
    return 0.0


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _rrf_fuse(
    dense_ranked: list[tuple[float, RetrievedChunk]],
    sparse_ranked: list[tuple[float, RetrievedChunk]],
    k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """RRF 融合：对两路结果按 1/(k+rank) 累加得分，返回 top_k。"""
    fused: dict[str, float] = {}
    chunk_by_key: dict[str, RetrievedChunk] = {}
    for ranked in (dense_ranked, sparse_ranked):
        for rank, (_, chunk) in enumerate(ranked):
            key = chunk.citation.chunk_id or chunk.citation.content[:80]
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_key.setdefault(key, chunk)
    ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(citation=chunk_by_key[key].citation, score=score, relevance=chunk_by_key[key].relevance)
        for key, score in ranked
    ]


class ConfiguredHybridRetriever(HybridRetriever):
    """Milvus 混合检索：稠密 ANN + 稀疏向量两路召回 → MMR 去重 → RRF 融合。

    不再依赖 MySQL 做 BM25 稀疏检索，稀疏向量存储在 Milvus 的 sparse_vector 字段中，
    每次检索时同时召回两路结果，保证新入库文档立即可被检索。
    """

    def __init__(self):
        self._embedding = OpenAIEmbeddingProvider()
        self._client = None
        self._resolved = False
        self._collection: str | None = None

    def _ensure_collection(self) -> str | None:
        """懒加载 MilvusClient 并检查知识库集合是否存在（结果缓存）。"""
        if self._resolved:
            return self._collection
        settings = get_settings()
        if not settings.milvus_uri:
            raise RuntimeConfigurationError(["MILVUS_URI"])
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeConfigurationError(["pymilvus package"]) from exc
        self._client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_database,
        )
        name = settings.knowledge_collection
        self._collection = name if self._client.has_collection(name) else None
        self._resolved = True
        return self._collection

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """双路召回 + MMR + RRF 混合检索。"""
        collection = self._ensure_collection()
        if collection is None:
            return []

        dense_vector = await self._embedding.embed(query)
        sparse_vector = await self._embedding.sparse_embed(query)

        candidate_count = top_k * 3
        output_fields = ["document_id", "chunk_id", "content", "page", "document_name"]

        dense_results: list[tuple[dict, float, dict[int, float] | None, list[float] | None]] = []
        sparse_results: list[tuple[dict, float, dict[int, float] | None, list[float] | None]] = []

        # 稠密召回
        dense_hits = self._client.search(
            collection_name=collection,
            data=[dense_vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=candidate_count,
            output_fields=output_fields + ["vector"],
        )
        for hit in dense_hits[0]:
            entity = hit.entity
            dense_vec = entity.get("vector")
            dense_results.append((entity, float(hit.distance), None, dense_vec if isinstance(dense_vec, list) else None))

        # 稀疏召回：额外取稠密向量用于 MMR 相似度，避免稀疏候选两两相似度恒为 0 导致多样性失效
        if sparse_vector:
            sparse_hits = self._client.search(
                collection_name=collection,
                data=[sparse_vector],
                anns_field="sparse_vector",
                search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=candidate_count,
                output_fields=output_fields + ["vector"],
            )
            for hit in sparse_hits[0]:
                entity = hit.entity
                dense_vec = entity.get("vector")
                sparse_results.append((entity, float(hit.distance), None, dense_vec if isinstance(dense_vec, list) else None))

        # MMR 多样性重排
        dense_mmr_indices = _mmr(dense_results, _MMR_LAMBDA, top_k)
        sparse_mmr_indices = _mmr(sparse_results, _MMR_LAMBDA, top_k)

        def _to_chunk(entity: dict, score: float) -> RetrievedChunk:
            return RetrievedChunk(
                citation=Citation(
                    source_type="knowledge_base",
                    document_id=entity.get("document_id"),
                    document_name=entity.get("document_name", "Knowledge base"),
                    chunk_id=entity.get("chunk_id"),
                    page=entity.get("page"),
                    content=entity.get("content", ""),
                ),
                score=score,
                relevance=score,
            )

        dense_ranked = [
            (dense_results[i][1], _to_chunk(dense_results[i][0], dense_results[i][1]))
            for i in dense_mmr_indices
        ]
        sparse_ranked = [
            (sparse_results[i][1], _to_chunk(sparse_results[i][0], sparse_results[i][1]))
            for i in sparse_mmr_indices
        ]

        return _rrf_fuse(dense_ranked, sparse_ranked, _RRF_K, top_k)


class FakeHybridRetriever(HybridRetriever):
    """测试用检索器：直接返回预设的分块列表。"""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks or []

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.chunks[:top_k]