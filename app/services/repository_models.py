from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类，所有 ORM 模型继承它。"""

    pass


class JobRow(Base):
    """索引任务表：记录文档的 index/reindex 后台任务状态与错误信息。"""

    __tablename__ = "knowledge_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    document_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict[str, object]:
        return {"task_id": self.id, "type": self.type, "document_id": self.document_id, "status": self.status, "error": self.error}