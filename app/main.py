from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.api import failed
from app.core.config import get_settings
from app.core.errors import AppError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时确保上传目录存在。"""
    get_settings().upload_dir.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """组装 FastAPI 应用：中间件、路由、静态资源与全局异常处理。"""
    settings = get_settings()
    app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
    # 允许 Vue 开发服务器跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix=settings.api_v1_prefix)
    static_dir = Path("app/static")
    frontend_dist = Path("frontend/dist")
    # 存在时才挂载静态目录，避免开发环境缺少构建产物时报错
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    if frontend_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        """根路径：优先返回前端构建产物，否则回退到静态目录首页。"""
        index = frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return FileResponse(static_dir / "index.html")

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        """业务异常统一转成带错误码的 JSON 响应。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=failed(exc.code, exc.message, exc.details).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
        """兜底异常处理：未预期错误统一返回 500。"""
        return JSONResponse(
            status_code=500,
            content=failed("INTERNAL_ERROR", "An unexpected error occurred.").model_dump(mode="json"),
        )

    return app


app = create_app()
