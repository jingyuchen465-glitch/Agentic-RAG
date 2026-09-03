from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.domain.models import PlannedTask, RouteCandidate, RouteDecision, ToolSpec
from app.services.embedding import EmbeddingProvider, OpenAIEmbeddingProvider, cosine_similarity


@dataclass(slots=True)
class ToolVector:
    """工具及其预计算的嵌入向量。"""

    spec: ToolSpec
    embedding: list[float]


class VectorToolRouter:
    """向量工具路由器：将计划任务与已注册工具做语义匹配，决定由哪个工具执行。"""

    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider()
        self._tools: dict[str, ToolVector] = {}

    async def register(self, spec: ToolSpec) -> None:
        """注册单个工具：对其能力描述做向量化并缓存。"""
        vector = await self.embedding_provider.embed(spec.routing_text())
        self._tools[spec.tool_id] = ToolVector(spec, vector)

    async def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            await self.register(spec)

    def _hard_filter(self, task: PlannedTask, spec: ToolSpec) -> bool:
        """硬性过滤：工具必须启用、智能体类型匹配、且任务输入覆盖工具必需输入。"""
        if not spec.enabled:
            return False
        if task.agent_type and spec.agent_type != task.agent_type:
            return False
        return set(spec.required_inputs).issubset(set(task.required_inputs) | {"query"})

    async def route(self, task: PlannedTask) -> RouteDecision:
        """对任务做向量路由：相似度排序后按阈值与歧义度判定最终选择。"""
        if not self._tools:
            return RouteDecision(selected_tool_id=None, score=0.0, reason="No tools registered")
        task_vector = await self.embedding_provider.embed(task.routing_text())
        # 计算任务与每个候选工具的余弦相似度，按得分降序排列
        ranked = sorted(
            (
                (cosine_similarity(task_vector, entry.embedding), entry.spec)
                for entry in self._tools.values()
                if self._hard_filter(task, entry.spec)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            return RouteDecision(selected_tool_id=None, score=0.0, reason="No tool passed hard constraints")
        best_score, best_spec = ranked[0]
        alternatives = [RouteCandidate(tool_id=spec.tool_id, score=round(score, 4)) for score, spec in ranked[1:3]]
        threshold = get_settings().router_threshold
        # 最高分低于阈值：置信度不足，不路由任何工具
        if best_score < threshold:
            return RouteDecision(
                selected_tool_id=None,
                score=round(best_score, 4),
                alternatives=alternatives,
                reason=f"Best score {best_score:.3f} is below threshold {threshold:.3f}",
            )
        # 最高分与次高分差距过小：候选歧义，放弃路由
        if alternatives and best_score - alternatives[0].score < get_settings().router_ambiguity_delta:
            return RouteDecision(
                selected_tool_id=None,
                score=round(best_score, 4),
                alternatives=alternatives,
                reason="Top tool candidates are ambiguous",
            )
        return RouteDecision(
            selected_tool_id=best_spec.tool_id,
            score=round(best_score, 4),
            alternatives=alternatives,
            reason=f"Selected {best_spec.name} by semantic similarity",
        )


def default_tool_specs() -> list[ToolSpec]:
    """返回 MVP 内置的工具清单（RAG 智能体、Text2SQL、网络搜索）。"""
    from app.domain.models import AgentType

    return [
        ToolSpec(
            tool_id="rag-agent",
            name="Knowledge Base RAG Agent",
            description="Search internal documents and answer with grounded citations.",
            capabilities=["document retrieval", "semantic question answering", "citations"],
            required_inputs=["query"],
            constraints=["requires indexed documents"],
            agent_type=AgentType.RAG,
        ),
        ToolSpec(
            tool_id="text2sql-agent",
            name="Text2SQL Agent",
            description="Translate analytical questions into restricted read-only SQL.",
            capabilities=["database query", "aggregations", "structured data"],
            required_inputs=["query"],
            constraints=["read-only", "requires database schema", "disabled in MVP"],
            enabled=False,
            agent_type=AgentType.TOOLS,
        ),
        ToolSpec(
            tool_id="web-search-agent",
            name="Web Search Agent",
            description="Search public web sources when internal documents do not support an answer.",
            capabilities=["web search", "fresh information", "external sources"],
            required_inputs=["query"],
            constraints=["requires Tavily API key"],
            agent_type=AgentType.TOOLS,
        ),
    ]


def mcp_tool_specs(tools: list[dict]) -> tuple[list[ToolSpec], dict[str, dict]]:
    """把 MCP 中台 tools/list 返回的工具转换为路由可用的 ToolSpec。

    每个 MCP 动态工具映射为一个 ``mcp:{name}`` 的工具 ID，agent_type 固定为
    TOOLS，供向量路由与 rag-agent/web-search-agent 一起参与语义匹配。同时返回
    tool_id -> inputSchema 的映射，供执行节点按 schema 绑定调用参数。由于任务
    required_inputs 默认为 ``query``，这里统一将其声明为 ``query`` 以通过硬过滤，
    真正的入参在 tools/call 阶段再按 schema 解析。
    """
    from app.domain.models import AgentType

    specs: list[ToolSpec] = []
    schemas: dict[str, dict] = {}
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        schema = tool.get("inputSchema") or {}
        tool_id = f"mcp:{name}"
        specs.append(
            ToolSpec(
                tool_id=tool_id,
                name=name,
                description=tool.get("description", "") or name,
                capabilities=["external api", name],
                required_inputs=["query"],
                constraints=[],
                enabled=True,
                agent_type=AgentType.TOOLS,
            )
        )
        schemas[tool_id] = schema
    return specs, schemas
