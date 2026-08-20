from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """计划任务的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(StrEnum):
    """任务可路由到的智能体类型。"""

    RAG = "rag"
    TOOLS = "tools"
    FALLBACK = "fallback"


class ToolSpec(BaseModel):
    """工具能力描述：向量路由时用它生成嵌入，并与计划任务做语义匹配。"""

    tool_id: str
    name: str
    description: str
    capabilities: list[str]
    required_inputs: list[str]
    constraints: list[str] = Field(default_factory=list)
    enabled: bool = True
    agent_type: AgentType

    def routing_text(self) -> str:
        """将工具描述拼成用于向量化的纯文本。"""
        return "\n".join(
            [
                f"name: {self.name}",
                f"description: {self.description}",
                f"capabilities: {', '.join(self.capabilities)}",
                f"inputs: {', '.join(self.required_inputs)}",
                f"constraints: {', '.join(self.constraints)}",
            ]
        )


class PlannedTask(BaseModel):
    """计划器产出的单个任务节点；dependencies 声明其前置任务，用于调度。"""

    task_id: str
    goal: str
    dependencies: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=lambda: ["query"])
    constraints: list[str] = Field(default_factory=list)
    agent_type: AgentType | None = None
    status: TaskStatus = TaskStatus.PENDING

    def routing_text(self) -> str:
        """将任务目标拼成用于向量化的纯文本，与工具描述做相似度比对。"""
        return "\n".join(
            [
                f"goal: {self.goal}",
                f"inputs: {', '.join(self.required_inputs)}",
                f"constraints: {', '.join(self.constraints)}",
            ]
        )


class RouteCandidate(BaseModel):
    """路由候选项：记录候选工具及其相似度得分。"""

    tool_id: str
    score: float


class RouteDecision(BaseModel):
    """路由决策结果：选中工具、得分、备选列表及决策理由。"""

    selected_tool_id: str | None
    score: float
    alternatives: list[RouteCandidate] = Field(default_factory=list)
    reason: str


class Citation(BaseModel):
    """回答引用来源：来自知识库（文档/分块）或网络（URL）。"""

    source_type: Literal["knowledge_base", "web"]
    document_id: str | None = None
    document_name: str
    chunk_id: str | None = None
    page: int | None = None
    url: str | None = None
    content: str


class ChatRequest(BaseModel):
    """聊天请求体：query 必填，session_id 可选（缺省时服务端新建会话）。"""

    query: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None


class ChatResult(BaseModel):
    """聊天响应体：答案、引用列表、任务执行摘要与警告信息。"""

    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    task_summary: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
