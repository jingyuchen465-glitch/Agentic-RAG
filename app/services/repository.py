from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError, RuntimeConfigurationError

_DOC_COLLECTION = "knowledge_documents"


class KnowledgeRepository:
    """文档元数据存 Milvus，索引任务状态存 MySQL，对外暴露统一接口。"""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or get_settings().database_url
        self._engine = None
        self._session_factory = None
        self._ready = False
        self._milvus_client = None

    # ── MySQL 任务层 ──────────────────────────────────────────────

    def _session(self):
        if not self.database_url:
            raise RuntimeConfigurationError(["DATABASE_URL"])
        if self._session_factory is None:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            self._engine = create_engine(self.database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        self._ensure_schema()
        return self._session_factory()

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        from app.services.repository_models import Base

        Base.metadata.create_all(self._engine)
        self._ready = True

    # ── Milvus 文档元数据层 ────────────────────────────────────────

    def _milvus(self):
        """懒加载 MilvusClient，复用单一连接。"""
        if self._milvus_client is not None:
            return self._milvus_client
        settings = get_settings()
        if not settings.milvus_uri:
            raise RuntimeConfigurationError(["MILVUS_URI"])
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install pymilvus to manage documents.", 503) from exc
        self._milvus_client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_database,
        )
        return self._milvus_client

    def _doc_collection(self) -> str:
        """获取或创建文档元数据集合（dummy 向量仅满足 Milvus schema 要求）。"""
        client = self._milvus()
        if client.has_collection(_DOC_COLLECTION):
            return _DOC_COLLECTION
        from pymilvus import DataType
        from pymilvus.milvus_client.index import IndexParams

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
        index_params = IndexParams()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP", params={})
        client.create_collection(collection_name=_DOC_COLLECTION, schema=schema, index_params=index_params)
        return _DOC_COLLECTION

    # ── 文档 CRUD ─────────────────────────────────────────────────

    def create_document(self, document_id: str, filename: str, path: str, task_id: str) -> None:
        """写入文档元数据到 Milvus，同时写入任务记录到 MySQL。"""
        from app.services.repository_models import JobRow

        client = self._milvus()
        col = self._doc_collection()
        client.insert(
            collection_name=col,
            data=[
                {
                    "id": document_id,
                    "vector": [0.0, 0.0],
                    "filename": filename,
                    "path": path,
                    "status": "queued",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ],
        )
        client.flush(collection_name=col)
        with self._session() as session:
            session.add(JobRow(id=task_id, type="index", document_id=document_id, status="queued"))
            session.commit()

    def list_documents(self, page: int, size: int) -> dict[str, object]:
        """分页列出文档，按 created_at 降序。"""
        client = self._milvus()
        col = self._doc_collection()
        output_fields = ["id", "filename", "path", "status", "created_at"]
        try:
            results = client.query(collection_name=col, filter="", output_fields=output_fields, offset=page * size, limit=size)
        except Exception:
            # 某些 Milvus 版本不支持空 filter，改用 id != ""
            results = client.query(collection_name=col, filter='id != ""', output_fields=output_fields, offset=page * size, limit=size)
        items = sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)
        content = [
            {"document_id": r["id"], "filename": r.get("filename", ""), "path": r.get("path", ""), "status": r.get("status", ""), "created_at": r.get("created_at", "")}
            for r in items
        ]
        total = client.get_collection_stats(col).get("row_count", 0)
        return {"content": content, "page": page, "size": size, "totalElements": total, "totalPages": max(1, (total + size - 1) // size)}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        client = self._milvus()
        col = self._doc_collection()
        results = client.query(collection_name=col, filter=f'id == "{document_id}"', output_fields=["id", "filename", "path", "status", "created_at"], limit=1)
        if not results:
            return None
        r = results[0]
        return {"document_id": r["id"], "filename": r.get("filename", ""), "path": r.get("path", ""), "status": r.get("status", ""), "created_at": r.get("created_at", "")}

    def delete_document(self, document_id: str) -> None:
        client = self._milvus()
        col = self._doc_collection()
        client.delete(collection_name=col, filter=f'id == "{document_id}"')
        # 同时清理知识库内的分块向量，避免元数据删除后向量残留
        knowledge_collection = get_settings().knowledge_collection
        if client.has_collection(knowledge_collection):
            client.delete(collection_name=knowledge_collection, filter=f'document_id == "{document_id}"')
            client.flush(collection_name=knowledge_collection)
        client.flush(collection_name=col)
        from app.services.repository_models import JobRow
        from sqlalchemy import delete

        with self._session() as session:
            session.execute(delete(JobRow).where(JobRow.document_id == document_id))
            session.commit()

    # ── 任务管理 ──────────────────────────────────────────────────

    def create_reindex_job(self, document_id: str, task_id: str) -> None:
        from app.services.repository_models import JobRow

        with self._session() as session:
            session.add(JobRow(id=task_id, type="reindex", document_id=document_id, status="queued"))
            session.commit()

    def get_task(self, task_id: str) -> dict[str, object] | None:
        from sqlalchemy import select
        from app.services.repository_models import JobRow

        with self._session() as session:
            row = session.scalar(select(JobRow).where(JobRow.id == task_id))
            return row.to_dict() if row else None

    def update_status(self, document_id: str, task_id: str, status: str, error: str | None = None) -> None:
        from sqlalchemy import update
        from app.services.repository_models import JobRow

        with self._session() as session:
            session.execute(update(JobRow).where(JobRow.id == task_id).values(status=status, error=error))
            session.commit()
        client = self._milvus()
        col = self._doc_collection()
        # 先读回已有元数据，再整行 upsert，避免用占位符覆盖 filename/path/created_at
        rows = client.query(collection_name=col, filter=f'id == "{document_id}"', output_fields=["filename", "path", "status", "created_at"], limit=1)
        existing = rows[0] if rows else {}
        client.upsert(
            collection_name=col,
            data=[
                {
                    "id": document_id,
                    "vector": [0.0, 0.0],
                    "filename": existing.get("filename", ""),
                    "path": existing.get("path", ""),
                    "status": status,
                    "created_at": existing.get("created_at", ""),
                }
            ],
        )
        client.flush(collection_name=col)


class InMemoryKnowledgeRepository:
    """内存版仓储（仅用于单元测试），接口与 KnowledgeRepository 保持一致。"""

    def __init__(self):
        self.documents: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, dict[str, Any]] = {}

    def create_document(self, document_id: str, filename: str, path: str, task_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        self.documents[document_id] = {"document_id": document_id, "filename": filename, "path": path, "status": "queued", "created_at": now}
        self.tasks[task_id] = {"task_id": task_id, "type": "index", "status": "queued", "document_id": document_id}

    def list_documents(self, page: int, size: int) -> dict[str, object]:
        values = list(self.documents.values())[page * size : (page + 1) * size]
        total = len(self.documents)
        return {"content": values, "page": page, "size": size, "totalElements": total, "totalPages": (total + size - 1) // size}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self.documents.get(document_id)

    def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self.tasks = {key: value for key, value in self.tasks.items() if value["document_id"] != document_id}

    def create_reindex_job(self, document_id: str, task_id: str) -> None:
        self.tasks[task_id] = {"task_id": task_id, "type": "reindex", "status": "queued", "document_id": document_id}

    def get_task(self, task_id: str) -> dict[str, object] | None:
        return self.tasks.get(task_id)

    def update_status(self, document_id: str, task_id: str, status: str, error: str | None = None) -> None:
        if document_id in self.documents:
            self.documents[document_id]["status"] = status
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["error"] = error