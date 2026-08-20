from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError, RuntimeConfigurationError
from app.services.embedding import OpenAIEmbeddingProvider


class MilvusKnowledgeIndexer:
    """将文档分块及其向量写入 Milvus；集合不存在时按需建表并创建索引。"""

    def _collection(self, dimensions: int):
        """获取（或创建）知识库集合，向量维度由嵌入模型决定。"""
        settings = get_settings()
        if not settings.milvus_uri:
            raise RuntimeConfigurationError(["MILVUS_URI"])
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install pymilvus to index documents.", 503) from exc
        connections.connect(alias="default", uri=settings.milvus_uri, token=settings.milvus_token or "", db_name=settings.milvus_database)
        if utility.has_collection(settings.knowledge_collection):
            return Collection(settings.knowledge_collection)
        # 首次使用：按字段定义创建集合，并为向量字段建 AUTOINDEX 余弦索引
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dimensions),
                FieldSchema("document_id", DataType.VARCHAR, max_length=64),
                FieldSchema("chunk_id", DataType.VARCHAR, max_length=64),
                FieldSchema("document_name", DataType.VARCHAR, max_length=512),
                FieldSchema("content", DataType.VARCHAR, max_length=65535),
                FieldSchema("page", DataType.INT64),
            ],
            description="Agentic RAG knowledge chunks",
        )
        collection = Collection(settings.knowledge_collection, schema=schema)
        collection.create_index("vector", {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
        return collection

    def replace_document(self, document_id: str, filename: str, chunks: list[dict[str, object]], vectors: list[list[float]]) -> None:
        """整体替换某文档的向量：先按 document_id 删除旧数据，再批量插入新分块。"""
        collection = self._collection(len(vectors[0]))
        collection.load()
        collection.delete(f'document_id == "{document_id}"')
        collection.insert(
            [
                [str(chunk["chunk_id"]) for chunk in chunks],
                vectors,
                [document_id for _ in chunks],
                [str(chunk["chunk_id"]) for chunk in chunks],
                [filename for _ in chunks],
                [str(chunk["content"])[:65535] for chunk in chunks],
                [int(chunk.get("page") or -1) for chunk in chunks],
            ]
        )
        collection.flush()


class DocumentIngestor:
    """文档摄取：解析文件、切分文本、向量化并写入 Milvus 与 MySQL。"""

    def __init__(self, repository):
        self.repository = repository
        self.embedding = OpenAIEmbeddingProvider()
        self.indexer = MilvusKnowledgeIndexer()

    def _parse(self, path: Path) -> list[str]:
        """用 LlamaIndex 读取文件并按句子切分为文本块（800 字符、120 重叠）。"""
        try:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install LlamaIndex file readers to ingest documents.", 503) from exc
        documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        splitter = SentenceSplitter(chunk_size=800, chunk_overlap=120)
        return [node.text for document in documents for node in splitter.get_nodes_from_documents([document]) if node.text.strip()]

    async def index(self, document_id: str, task_id: str) -> None:
        """执行完整索引流程，并同步更新任务与文档状态。"""
        document = self.repository.get_document(document_id)
        if not document:
            return
        self.repository.update_status(document_id, task_id, "running")
        try:
            # 解析为 CPU 密集操作，放到线程池避免阻塞事件循环
            texts = await asyncio.to_thread(self._parse, Path(str(document["path"])))
            if not texts:
                raise AppError("DOCUMENT_EMPTY", "No readable text was extracted from the document.")
            chunks = [{"chunk_id": str(uuid.uuid4()), "content": text, "page": None} for text in texts]
            vectors = await self.embedding.embed_many([str(chunk["content"]) for chunk in chunks])
            await asyncio.to_thread(
                self.indexer.replace_document,
                document_id,
                str(document["filename"]),
                chunks,
                vectors,
            )
            self.repository.replace_chunks(document_id, chunks)
            self.repository.update_status(document_id, task_id, "completed")
        except Exception as exc:
            # 失败时记录错误摘要，避免后台任务静默丢失
            self.repository.update_status(document_id, task_id, "failed", str(exc)[:1000])
