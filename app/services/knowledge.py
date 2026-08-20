from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.errors import AppError
from app.services.repository import MySQLKnowledgeRepository
from app.services.ingestion import DocumentIngestor


class KnowledgeService:
    """知识库服务：文件上传、列表、删除、重新索引与任务状态查询。"""

    def __init__(self, repository=None):
        self.repository = repository or MySQLKnowledgeRepository(get_settings().database_url)

    async def upload(self, file: UploadFile) -> dict[str, str]:
        """校验文件类型与大小，落盘并登记文档与索引任务，返回两者的 ID。"""
        settings = get_settings()
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
            raise AppError("UNSUPPORTED_FILE_TYPE", "Only PDF, DOCX, TXT, and Markdown files are supported.")
        content = await file.read()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise AppError("FILE_TOO_LARGE", f"File exceeds {settings.max_upload_size_mb} MB.")
        document_id, task_id = str(uuid.uuid4()), str(uuid.uuid4())
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        target = settings.upload_dir / f"{document_id}{suffix}"
        target.write_bytes(content)
        self.repository.create_document(document_id, file.filename or target.name, str(target), task_id)
        return {"document_id": document_id, "task_id": task_id}

    def list_files(self, page: int, size: int) -> dict[str, object]:
        return self.repository.list_documents(page, size)

    def delete(self, document_id: str) -> None:
        """删除文档：先移除磁盘文件，再删除数据库记录。"""
        document = self.repository.get_document(document_id)
        if not document:
            raise AppError("DOCUMENT_NOT_FOUND", f"Document {document_id} was not found.", 404)
        path = Path(str(document["path"]))
        if path.exists():
            path.unlink()
        self.repository.delete_document(document_id)

    async def reindex(self, document_id: str) -> dict[str, str]:
        """为已有文档创建重新索引任务并返回任务 ID。"""
        if not self.repository.get_document(document_id):
            raise AppError("DOCUMENT_NOT_FOUND", f"Document {document_id} was not found.", 404)
        task_id = str(uuid.uuid4())
        self.repository.create_reindex_job(document_id, task_id)
        return {"document_id": document_id, "task_id": task_id}

    async def index(self, document_id: str, task_id: str) -> None:
        """后台执行文档索引（由 FastAPI BackgroundTasks 调用）。"""
        await DocumentIngestor(self.repository).index(document_id, task_id)

    def task_status(self, task_id: str) -> dict[str, object]:
        task = self.repository.get_task(task_id)
        if not task:
            raise AppError("TASK_NOT_FOUND", f"Task {task_id} was not found.", 404)
        return task
