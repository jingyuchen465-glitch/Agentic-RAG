from __future__ import annotations

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiError(BaseModel):
    """统一错误体：机器可读的错误码、人类可读的消息及可选的详情列表。"""

    code: str
    message: str
    details: list[str] = []


class ApiResponse(BaseModel, Generic[T]):
    """统一响应信封：success 标记成败，data 承载成功数据，error 承载失败信息。"""

    success: bool
    data: T | None = None
    error: ApiError | None = None
    timestamp: datetime


def ok(data: T) -> ApiResponse[T]:
    """构造成功响应。"""
    return ApiResponse(success=True, data=data, timestamp=datetime.now(UTC))


def failed(code: str, message: str, details: list[str] | None = None) -> ApiResponse[None]:
    """构造失败响应。"""
    return ApiResponse(
        success=False,
        error=ApiError(code=code, message=message, details=details or []),
        timestamp=datetime.now(UTC),
    )
