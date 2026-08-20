class AppError(Exception):
    """业务异常基类：携带错误码、消息、HTTP 状态码与详情，由全局异常处理器统一转成响应。"""

    def __init__(self, code: str, message: str, status_code: int = 400, details: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []


class RuntimeConfigurationError(AppError):
    """外部依赖（MySQL/Redis/Milvus/OpenAI/Tavily）未配置时抛出，返回 503。"""

    def __init__(self, missing: list[str]):
        super().__init__(
            "RUNTIME_NOT_CONFIGURED",
            "Required external services are not configured.",
            503,
            missing,
        )
