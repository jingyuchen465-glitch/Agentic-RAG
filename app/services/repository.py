from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.errors import RuntimeConfigurationError


class MySQLKnowledgeRepository:
    """SQLAlchemy 仓储：持久化文档与索引任务的元数据。

    连接与表结构按需懒加载（首次访问时创建），避免启动阶段依赖外部数据库。
    """

    def __init__(self, database_url: str | None):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None
        self._ready = False

    def _session(self):
        """获取会话：未配置数据库时抛错，首次使用时创建引擎并建表。"""
        if not self.database_url:
            raise RuntimeConfigurationError(["DATABASE_URL"])
        if self._session_factory is None:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            # pool_pre_ping 在取连接前探活，避免拿到失效连接
            self._engine = create_engine(self.database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        self._ensure_schema()
        return self._session_factory()

    def _ensure_schema(self) -> None:
        """按需创建缺失的数据表，仅执行一次。"""
        if self._ready:
            return
        from app.services.repository_models import Base

        Base.metadata.create_all(self._engine)
        self._ready = True

    def create_document(self, document_id: str, filename: str, path: str, task_id: str) -> None:
        """写入文档记录及其关联的索引任务记录。"""
        from app.services.repository_models import DocumentRow, JobRow

        with self._session() as session:
            session.add(DocumentRow(id=document_id, filename=filename, path=path, status="queued"))
            session.add(JobRow(id=task_id, type="index", document_id=document_id, status="queued"))
            session.commit()

    def list_documents(self, page: int, size: int) -> dict[str, object]:
        """分页查询文档列表，返回分页元信息与内容。"""
        from sqlalchemy import func, select
        from app.services.repository_models import DocumentRow

        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(DocumentRow)) or 0
            rows = session.scalars(select(DocumentRow).order_by(DocumentRow.created_at.desc()).offset(page * size).limit(size)).all()
        content = [row.to_dict() for row in rows]
        return {"content": content, "page": page, "size": size, "totalElements": total, "totalPages": (total + size - 1) // size}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """按 ID 查询单个文档，不存在时返回 None。"""
        from sqlalchemy import select
        from app.services.repository_models import DocumentRow

        with self._session() as session:
            row = session.scalar(select(DocumentRow).where(DocumentRow.id == document_id))
            return row.to_dict() if row else None

    def delete_document(self, document_id: str) -> None:
        """删除文档及其全部关联任务记录（外键级联删除分块）。"""
        from sqlalchemy import delete
        from app.services.repository_models import DocumentRow, JobRow

        with self._session() as session:
            session.execute(delete(JobRow).where(JobRow.document_id == document_id))
            session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
            session.commit()

    def create_reindex_job(self, document_id: str, task_id: str) -> None:
        """为已有文档创建重新索引任务。"""
        from app.services.repository_models import JobRow

        with self._session() as session:
            session.add(JobRow(id=task_id, type="reindex", document_id=document_id, status="queued"))
            session.commit()

    def get_task(self, task_id: str) -> dict[str, object] | None:
        """按 ID 查询索引任务，不存在时返回 None。"""
        from sqlalchemy import select
        from app.services.repository_models import JobRow

        with self._session() as session:
            row = session.scalar(select(JobRow).where(JobRow.id == task_id))
            return row.to_dict() if row else None

    def update_status(self, document_id: str, task_id: str, status: str, error: str | None = None) -> None:
        """同步更新文档与任务的执行状态（queued/running/completed/failed）。"""
        from sqlalchemy import update
        from app.services.repository_models import DocumentRow, JobRow

        with self._session() as session:
            session.execute(update(DocumentRow).where(DocumentRow.id == document_id).values(status=status))
            session.execute(update(JobRow).where(JobRow.id == task_id).values(status=status, error=error))
            session.commit()

    def replace_chunks(self, document_id: str, chunks: list[dict[str, object]]) -> None:
        """先删除旧分块再批量写入新分块，保证与向量库内容一致。"""
        from sqlalchemy import delete
        from app.services.repository_models import ChunkRow

        with self._session() as session:
            session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
            session.add_all(
                ChunkRow(
                    id=str(chunk["chunk_id"]),
                    document_id=document_id,
                    content=str(chunk["content"]),
                    page=chunk.get("page"),
                )
                for chunk in chunks
            )
            session.commit()

    def list_chunks(self) -> list[dict[str, object]]:
        """返回全部分块，供 BM25 稀疏检索使用。"""
        from sqlalchemy import select
        from app.services.repository_models import ChunkRow

        with self._session() as session:
            rows = session.scalars(select(ChunkRow)).all()
            return [{"chunk_id": row.id, "document_id": row.document_id, "content": row.content, "page": row.page} for row in rows]


class InMemoryKnowledgeRepository:
    """内存版仓储（仅用于单元测试），接口与 MySQLKnowledgeRepository 保持一致。"""

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
        self.documents[document_id]["status"] = status
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["error"] = error

    def replace_chunks(self, document_id: str, chunks: list[dict[str, object]]) -> None:
        return None
