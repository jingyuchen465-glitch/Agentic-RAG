from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.core.api import ApiResponse, ok
from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.models import ChatRequest, ChatResult
from app.services.knowledge import KnowledgeService
from app.services.workflow import AgenticWorkflow

router = APIRouter()
knowledge_service = KnowledgeService()
workflow = AgenticWorkflow()


@router.get("/health/live", tags=["health"])
async def live() -> ApiResponse[dict[str, str]]:
    """存活探针：进程可响应即返回 ok。"""
    return ok({"status": "ok"})


@router.get("/health/ready", tags=["health"])
async def ready() -> ApiResponse[dict[str, object]]:
    """就绪探针：检查外部依赖是否全部配置，返回缺失项列表。"""
    settings = get_settings()
    missing = settings.missing_runtime_settings()
    return ok({"ready": not missing, "missing": missing})


try:
    import multipart  # type: ignore  # noqa: F401
except ImportError:
    multipart = None


if multipart is not None:

    @router.post("/knowledge/files", status_code=202, tags=["knowledge"])
    async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> ApiResponse[dict[str, str]]:
        """上传文件：登记后由后台任务异步执行索引。"""
        result = await knowledge_service.upload(file)
        background_tasks.add_task(knowledge_service.index, result["document_id"], result["task_id"])
        return ok(result)

else:

    @router.post("/knowledge/files", status_code=202, tags=["knowledge"])
    async def upload_file(_: Request) -> ApiResponse[dict[str, str]]:
        """缺少 python-multipart 依赖时的降级实现：返回 503。"""
        raise AppError("DEPENDENCY_MISSING", "安装python-multipart来上传文件。", 503)


@router.get("/knowledge/files", tags=["knowledge"])
async def list_files(page: int = 0, size: int = 20) -> ApiResponse[dict[str, object]]:
    """分页列出已上传文件，校验分页参数合法性。"""
    if page < 0 or size < 1 or size > 100:
        raise AppError("INVALID_PAGINATION", "Page必须为>= 0，size必须在1到100之间。")
    return ok(knowledge_service.list_files(page, size))


@router.delete("/knowledge/files/{document_id}", status_code=204, tags=["knowledge"])
async def delete_file(document_id: str) -> None:
    """删除指定文档及其索引。"""
    knowledge_service.delete(document_id)


@router.post("/knowledge/files/{document_id}/reindex", status_code=202, tags=["knowledge"])
async def reindex_file(document_id: str, background_tasks: BackgroundTasks) -> ApiResponse[dict[str, str]]:
    """触发指定文档的重新索引，由后台任务执行。"""
    result = await knowledge_service.reindex(document_id)
    background_tasks.add_task(knowledge_service.index, result["document_id"], result["task_id"])
    return ok(result)


@router.get("/tasks/{task_id}", tags=["tasks"])
async def task_status(task_id: str) -> ApiResponse[dict[str, object]]:
    """查询索引任务状态。"""
    return ok(knowledge_service.task_status(task_id))


@router.post("/chat/query", tags=["chat"])
async def query(request: ChatRequest) -> ApiResponse[ChatResult]:
    """非流式问答：一次性返回完整回答与引用。"""
    return ok(await workflow.run(request))


async def _events(request: ChatRequest) -> AsyncIterator[str]:
    """把工作流事件流格式化为 SSE 文本；AppError 转为 error 事件。"""
    try:
        async for event in workflow.stream(request):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
    except AppError as exc:
        yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': exc.message}, ensure_ascii=False)}\n\n"


@router.post("/chat/query/stream", tags=["chat"])
async def query_stream(request: ChatRequest) -> StreamingResponse:
    """流式问答：以 text/event-stream 输出元数据、引用、评分与答案 token。"""
    return StreamingResponse(_events(request), media_type="text/event-stream")
