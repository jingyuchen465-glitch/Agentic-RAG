from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError, RuntimeConfigurationError
from app.services.embedding import OpenAIEmbeddingProvider
from app.services.pdf_pipeline import PdfPipeline


class MilvusKnowledgeIndexer:
    """将文档分块、稠密向量与稀疏向量写入 Milvus；集合不存在时按需建表并创建索引。"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        settings = get_settings()
        if not settings.milvus_uri:
            raise RuntimeConfigurationError(["MILVUS_URI"])
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install pymilvus to index documents.", 503) from exc
        self._client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_database,
        )
        return self._client

    def _ensure_collection(self, dimensions: int) -> str:
        """获取（或创建）知识库集合，向量维度由嵌入模型决定。"""
        client = self._get_client()
        name = get_settings().knowledge_collection
        if client.has_collection(name):
            return name
        from pymilvus import DataType
        from pymilvus.milvus_client.index import IndexParams

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimensions)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="page", datatype=DataType.INT64)

        index_params = IndexParams()
        index_params.add_index(field_name="vector", index_name="vector_idx", index_type="AUTOINDEX", metric_type="COSINE", params={})
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={})
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        return name

    def replace_document(
        self,
        document_id: str,
        filename: str,
        chunks: list[dict[str, object]],
        vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> None:
        """整体替换某文档的向量：先按 document_id 删除旧数据，再批量插入新分块。"""
        client = self._get_client()
        collection = self._ensure_collection(len(vectors[0]))
        client.delete(collection_name=collection, filter=f'document_id == "{document_id}"')
        rows = [
            {
                "id": str(chunk["chunk_id"]),
                "vector": vector,
                "sparse_vector": sparse,
                "document_id": document_id,
                "chunk_id": str(chunk["chunk_id"]),
                "document_name": filename,
                "content": str(chunk["content"])[:65535],
                "page": int(chunk.get("page") or -1),
            }
            for chunk, vector, sparse in zip(chunks, vectors, sparse_vectors)
        ]
        client.insert(collection_name=collection, data=rows)
        client.flush(collection_name=collection)


class DocumentIngestor:
    """文档摄取：解析文件、切分文本、向量化并写入 Milvus 与 MySQL。"""

    def __init__(self, repository):
        self.repository = repository
        self.embedding = OpenAIEmbeddingProvider()
        self.indexer = MilvusKnowledgeIndexer()

    def _split(self, text: str) -> list[str]:
        """递归按章节/段落/换行/句子切分，超长块再用固定长度兜底。"""
        try:
            from llama_index.core import Document
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install LlamaIndex file readers to ingest documents.", 503) from exc
        settings = get_settings()
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=min(settings.chunk_overlap, max(0, settings.chunk_size - 1)),
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[^。！？!?；;.!?]+[。！？!?；;.!?]?|[。！？!?；;.!?]",
        )
        chunks = [node.text.strip() for node in splitter.get_nodes_from_documents([Document(text=text)]) if node.text.strip()]
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= settings.max_chunk_size:
                result.append(chunk)
                continue
            step = max(1, settings.max_chunk_size - settings.chunk_overlap)
            result.extend(chunk[start : start + settings.max_chunk_size] for start in range(0, len(chunk), step) if chunk[start : start + settings.max_chunk_size].strip())
        return result

    def _parse(self, path: Path) -> list[dict[str, object]]:
        """解析并切分文件，返回带页码的 chunk 草稿。"""
        if path.suffix.lower() == ".pdf":
            blocks = asyncio.run(PdfPipeline().parse(path))
            by_page: dict[int, list[str]] = {}
            for block in blocks:
                by_page.setdefault(int(block.get("page") or -1), []).append(str(block["text"]))
            output: list[dict[str, object]] = []
            for page, values in sorted(by_page.items()):
                output.extend({"content": text, "page": None if page < 0 else page} for text in self._split("\n\n".join(values)))
            return output
        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install LlamaIndex file readers to ingest documents.", 503) from exc
        documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        output: list[dict[str, object]] = []
        for document in documents:
            page = document.metadata.get("page_number") or document.metadata.get("page_label")
            output.extend({"content": text, "page": int(page) if str(page).isdigit() else None} for text in self._split(document.text))
        return output

    async def index(self, document_id: str, task_id: str) -> None:
        """执行完整索引流程，并同步更新任务与文档状态。"""
        document = self.repository.get_document(document_id)
        if not document:
            return
        self.repository.update_status(document_id, task_id, "running")
        try:
            chunks = await asyncio.to_thread(self._parse, Path(str(document["path"])))
            if not chunks:
                raise AppError("DOCUMENT_EMPTY", "No readable text was extracted from the document.")
            chunks = [{"chunk_id": str(uuid.uuid4()), **chunk} for chunk in chunks]
            contents = [str(chunk["content"]) for chunk in chunks]
            vectors = await self.embedding.embed_many(contents)
            sparse_vectors = await self.embedding.sparse_embed_many(contents)
            await asyncio.to_thread(
                self.indexer.replace_document,
                document_id,
                str(document["filename"]),
                chunks,
                vectors,
                sparse_vectors,
            )
            self.repository.update_status(document_id, task_id, "completed")
        except Exception as exc:
            self.repository.update_status(document_id, task_id, "failed", str(exc)[:1000])