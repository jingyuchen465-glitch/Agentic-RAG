from __future__ import annotations

import json
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.models import PlannedTask


def _get_chat_client() -> Any:
    """模块级单例：复用 ChatOpenAI 客户端，避免每次调用都重新实例化。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise AppError("MODEL_NOT_CONFIGURED", "OPENAI_API_KEY is required.", 503)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise AppError("DEPENDENCY_MISSING", "Install langchain-openai to use the model.", 503) from exc
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )


class OpenAILLMService:
    """基于 OpenAI Chat 模型的大模型服务：改写、意图识别、规划、评分与生成。"""

    def __init__(self):
        self._client = None

    def _model(self) -> Any:
        if self._client is None:
            self._client = _get_chat_client()
        return self._client

    async def _json(self, prompt: str) -> dict[str, Any]:
        """调用模型并强制解析为 JSON 对象，解析失败时抛 502。"""
        response = await self._model().ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        try:
            return json.loads(content.removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError as exc:
            raise AppError("MODEL_OUTPUT_INVALID", "Model did not return the required JSON object.", 502) from exc

    async def rewrite(self, query: str) -> tuple[str, list[str]]:
        """将用户问题改写为独立查询，并生成最多 3 条检索子查询。"""
        result = await self._json(
            "Rewrite the user's Chinese or English question into a standalone query. "
            "Return JSON only: {\"standalone_query\": string, \"retrieval_queries\": [string]}.\n"
            f"User question: {query}"
        )
        return str(result["standalone_query"]), [str(item) for item in result.get("retrieval_queries", [])][:3]

    async def understand(self, query: str) -> dict[str, Any]:
        """单次调用同时完成改写与意图识别，减少一次大模型往返。"""
        result = await self._json(
            "Rewrite the user's Chinese or English question into a standalone query, "
            "generate retrieval sub-queries, and classify the intent. Return JSON only: "
            "{\"standalone_query\":string,\"retrieval_queries\":[string],"
            "\"intent\":\"knowledge_lookup\"|\"multi_step\"|\"tool_request\"|\"chitchat\"}.\n"
            f"User question: {query}"
        )
        return {
            "standalone_query": str(result["standalone_query"]),
            "retrieval_queries": [str(item) for item in result.get("retrieval_queries", [])][:3],
            "intent": str(result.get("intent", "knowledge_lookup")),
        }

    # 单一知识问答强制单任务，防止规划器把同一条问题拆成多个内容重叠、重复检索的子任务。
    async def plan(self, query: str, intent: str) -> list[PlannedTask]:
        """根据意图生成最小任务 DAG；知识问答强制单任务；无任务时抛 502。"""
        result = await self._json(
            "Create a minimal task DAG for an Agentic RAG workflow. Return JSON only: "
            "{\"tasks\":[{\"task_id\":string,\"goal\":string,\"dependencies\":[string],"
            "\"required_inputs\":[string],\"constraints\":[string],\"agent_type\":\"rag\"|\"tools\"|\"fallback\"}]}. "
            "Rules:\n"
            "- If the intent is \"knowledge_lookup\" (a single knowledge question), return EXACTLY ONE `rag` task "
            "whose goal is the full user question; never split a knowledge question into multiple redundant sub-tasks.\n"
            "- Split into multiple tasks ONLY for \"multi_step\" queries whose sub-questions are genuinely "
            "independent and can be answered by different retrievals; avoid overlapping duplicate answers.\n"
            "- Use `rag` for knowledge questions. Do not create a database task unless the query explicitly asks for data analysis.\n"
            f"Intent: {intent}\nQuery: {query}"
        )
        tasks = [PlannedTask.model_validate(item) for item in result.get("tasks", [])]
        if not tasks:
            raise AppError("PLAN_EMPTY", "Task Planner returned no executable tasks.", 502)
        return self._enforce_plan_shape(query, intent, tasks)

    @staticmethod
    def _enforce_plan_shape(
        query: str, intent: str, tasks: list[PlannedTask]
    ) -> list[PlannedTask]:
        """确定性兜底：知识问答无论模型输出什么，都收敛为单个 RAG 任务。"""
        if intent == "knowledge_lookup" and len(tasks) != 1:
            return [PlannedTask(task_id="task_rag", goal=query, required_inputs=["query"], agent_type="rag")]
        return tasks

    async def generate(self, query: str, citations: list[dict[str, Any]]) -> str:
        """基于引用证据生成回答，要求按 [1]、[2] 内联标注来源。"""
        evidence = "\n\n".join(f"[{idx + 1}] {item['content']}" for idx, item in enumerate(citations))
        response = await self._model().ainvoke(
            "Answer in the user's language using only the supplied evidence. "
            "State uncertainty when evidence is incomplete. Cite evidence inline as [1], [2].\n"
            f"Question: {query}\nEvidence:\n{evidence}"
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    async def grade_answer(self, query: str, answer: str, citations: list[dict[str, Any]]) -> tuple[bool, str]:
        """回答评分：校验回答是否被证据支持且切题。"""
        evidence = "\n".join(item["content"] for item in citations)
        result = await self._json(
            "Check whether the answer is supported by evidence and answers the question. "
            "Return JSON only: {\"approved\": boolean, \"reason\": string}.\n"
            f"Question: {query}\nAnswer: {answer}\nEvidence: {evidence}"
        )
        return bool(result.get("approved")), str(result.get("reason", ""))

    async def bind_tool_arguments(
        self, tool_name: str, description: str, input_schema: dict[str, Any], goal: str
    ) -> dict[str, Any]:
        """为 MCP 动态工具生成符合 JSON Schema 的调用入参（仅工具路由路径触发）。"""
        schema = json.dumps(input_schema, ensure_ascii=False) if input_schema else "{}"
        result = await self._json(
            "Fill the arguments for an MCP tool call according to its input JSON Schema. "
            "Use only the schema-provided fields. Return JSON only: the arguments object itself.\n"
            f"Tool: {tool_name}\nDescription: {description}\nGoal: {goal}\n"
            f"Input schema: {schema}"
        )
        return result if isinstance(result, dict) else {}


class FakeLLMService:
    """工作流测试用的显式替身（test double），不发起真实模型调用。"""

    async def understand(self, query: str) -> dict[str, Any]:
        return {"standalone_query": query, "retrieval_queries": [query], "intent": "knowledge_lookup"}

    async def plan(self, query: str, intent: str) -> list[PlannedTask]:
        return [PlannedTask(task_id="task-1", goal=query, agent_type="rag")]

    async def generate(self, query: str, citations: list[dict[str, Any]]) -> str:
        return citations[0]["content"] if citations else "No evidence found."

    async def grade_answer(self, query: str, answer: str, citations: list[dict[str, Any]]) -> tuple[bool, str]:
        return True, "fake answer grade"

    async def bind_tool_arguments(
        self, tool_name: str, description: str, input_schema: dict[str, Any], goal: str
    ) -> dict[str, Any]:
        return {"query": goal}
