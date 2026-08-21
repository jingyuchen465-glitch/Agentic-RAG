from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import RuntimeConfigurationError
from app.domain.models import Citation
from app.services.embedding import OpenAIEmbeddingProvider
from app.services.repository import MySQLKnowledgeRepository


@dataclass(slots=True)
class RetrievedChunk:
    """检索结果：一条引用信息、融合后的得分，以及稠密余弦相关度（0~1，可空）。"""

    citation: Citation
    score: float
    relevance: float | None = None


class HybridRetriever:
    """混合检索抽象接口（Milvus 稠密 + BM25 稀疏），生产适配器实现同一方法。"""

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class ConfiguredHybridRetriever(HybridRetriever):
    """生产混合检索实现：Milvus 向量检索 + MySQL 分块 BM25 检索，再按倒数排名融合。"""

    def __init__(self):
        self._embedding = OpenAIEmbeddingProvider()
        self._loaded = False
        self._collection = None
        self._repository = MySQLKnowledgeRepository(get_settings().database_url)
        self._sparse_rows: list[dict[str, object]] | None = None

    def _load_collection(self):
        """懒加载 Milvus 集合；未配置或集合不存在时返回 None。"""
        if self._loaded:
            return self._collection
        settings = get_settings()
        if not settings.milvus_uri:
            raise RuntimeConfigurationError(["MILVUS_URI"])
        try:
            from pymilvus import Collection, connections, utility
        except ImportError as exc:
            raise RuntimeConfigurationError(["pymilvus package"]) from exc
        connections.connect(alias="default", uri=settings.milvus_uri, token=settings.milvus_token or "", db_name=settings.milvus_database)
        if not utility.has_collection(settings.knowledge_collection):
            self._loaded = True
            return None
        self._collection = Collection(settings.knowledge_collection)
        self._collection.load()
        self._loaded = True
        return self._collection

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """基于 MySQL 中全部分块做 BM25 稀疏检索，仅保留得分大于 0 的结果。"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeConfigurationError(["rank-bm25 package"]) from exc
        if self._sparse_rows is None:
            self._sparse_rows = self._repository.list_chunks()
        if not self._sparse_rows:
            return []
        tokenized = [str(row["content"]).lower().split() for row in self._sparse_rows]
        scores = BM25Okapi(tokenized).get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                citation=Citation(
                    source_type="knowledge_base",
                    document_id=str(self._sparse_rows[index]["document_id"]),
                    document_name="Knowledge base",
                    chunk_id=str(self._sparse_rows[index]["chunk_id"]),
                    page=self._sparse_rows[index].get("page"),
                    content=str(self._sparse_rows[index]["content"]),
                ),
                score=float(score),
            )
            for index, score in ranked
            if score > 0
        ]

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """执行稠密+稀疏混合检索，并用倒数排名融合（RRF）合并两类结果。"""
        collection = self._load_collection()
        dense: list[RetrievedChunk] = []
        if collection is not None:
            vector = await self._embedding.embed(query)
            result = collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["document_id", "chunk_id", "content", "page", "document_name"],
            )
            for hit in result[0]:
                entity = hit.entity
                dense.append(
                    RetrievedChunk(
                        citation=Citation(
                            source_type="knowledge_base",
                            document_id=entity.get("document_id"),
                            document_name=entity.get("document_name", "Knowledge base"),
                            chunk_id=entity.get("chunk_id"),
                            page=entity.get("page"),
                            content=entity.get("content", ""),
                        ),
                        score=float(hit.distance),
                        relevance=float(hit.distance),
                    )
                )
        sparse = self._sparse_search(query, top_k)
        # 倒数排名融合：同一分块在两类结果中的名次贡献 1/(60+rank) 并累加，再按总分取 top_k
        fused: dict[str, tuple[float, RetrievedChunk]] = {}
        for rank, item in enumerate(dense + sparse, start=1):
            key = item.citation.chunk_id or item.citation.content[:80]
            fused_score = 1.0 / (60 + rank)
            previous = fused.get(key)
            if previous is None:
                fused[key] = (fused_score, item)
                continue
            prev_score, prev_item = previous
            # 同一分块在稠密/稀疏中可能都出现：保留带稠密相关度版本供判定使用
            keep = prev_item if prev_item.relevance is not None or item.relevance is None else item
            fused[key] = (fused_score + prev_score, keep)
        ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)[:top_k]
        return [RetrievedChunk(citation=item.citation, score=score, relevance=item.relevance) for score, item in ranked]


class FakeHybridRetriever(HybridRetriever):
    """测试用检索器：直接返回预设的分块列表。"""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks or []

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.chunks[:top_k]
