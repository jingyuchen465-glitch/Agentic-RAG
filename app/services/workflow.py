from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.models import AgentType, ChatRequest, ChatResult, Citation, PlannedTask, TaskStatus
from app.services.llm import OpenAILLMService
from app.services.retrieval import ConfiguredHybridRetriever, HybridRetriever
from app.services.tool_router import VectorToolRouter, default_tool_specs
from app.services.web_search import TavilySearch
from app.services.session import RedisSessionStore


class GraphState(TypedDict, total=False):
    """LangGraph 节点间共享的状态字典，定义整个工作流的上下文。"""

    session_id: str
    original_query: str
    standalone_query: str
    retrieval_queries: list[str]
    intent: str
    tasks: list[dict[str, Any]]
    active_task_id: str | None
    task_results: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    max_relevance: float
    answer: str
    warnings: list[str]
    retry_counts: dict[str, int]
    document_grade_reason: str
    answer_grade_reason: str


class AgenticWorkflow:
    """Agentic RAG 主工作流：改写→意图→规划→调度→路由→检索→评分→生成→评分。"""

    def __init__(
        self,
        llm: Any | None = None,
        retriever: HybridRetriever | None = None,
        router: VectorToolRouter | None = None,
        web_search: TavilySearch | None = None,
        session_store: RedisSessionStore | None = None,
    ):
        # 依赖可注入，便于测试时替换为 Fake 实现
        self.llm = llm or OpenAILLMService()
        self.retriever = retriever or ConfiguredHybridRetriever()
        self.router = router or VectorToolRouter()
        self.web_search = web_search or TavilySearch()
        self.session_store = session_store or RedisSessionStore()
        self._tools_registered = False
        self._graph = None

    async def _ensure_tools(self) -> None:
        """首次路由前把默认工具注册到路由器（只执行一次）。"""
        if not self._tools_registered:
            await self.router.register_many(default_tool_specs())
            self._tools_registered = True

    def _current_task(self, state: GraphState) -> PlannedTask:
        """从状态中取出当前正在执行的任务，找不到时抛 500。"""
        task_id = state["active_task_id"]
        for task in state["tasks"]:
            if task["task_id"] == task_id:
                return PlannedTask.model_validate(task)
        raise AppError("TASK_NOT_FOUND", "未找到活动工作流任务。", 500)

    async def rewrite(self, state: GraphState) -> GraphState:
        """改写查询为独立查询、生成检索子查询并识别意图（一次调用）。"""
        info = await self.llm.understand(state["original_query"])
        return {
            "standalone_query": info["standalone_query"],
            "retrieval_queries": info["retrieval_queries"] or [info["standalone_query"]],
            "intent": info["intent"],
        }

    async def plan(self, state: GraphState) -> GraphState:
        """生成任务 DAG，并初始化任务结果与重试计数。"""
        tasks = await self.llm.plan(state["standalone_query"], state["intent"])
        return {"tasks": [task.model_dump(mode="json") for task in tasks], "task_results": [], "retry_counts": {}}

    async def schedule(self, state: GraphState) -> GraphState:
        """调度：选出第一个依赖已全部完成且仍处于 PENDING 的任务并置为 RUNNING。"""
        completed = {item["task_id"] for item in state.get("task_results", [])}
        for index, raw_task in enumerate(state["tasks"]):
            task = PlannedTask.model_validate(raw_task)
            if task.status == TaskStatus.PENDING and set(task.dependencies).issubset(completed):
                raw_task["status"] = TaskStatus.RUNNING.value
                return {"tasks": state["tasks"], "active_task_id": task.task_id}
        return {"active_task_id": None}

    def schedule_next(self, state: GraphState) -> str:
        """调度后的分支：有待执行任务则路由，否则收尾。"""
        return "route" if state.get("active_task_id") else "finalize"

    async def route(self, state: GraphState) -> GraphState:
        """对当前任务做向量路由；无可用工具时记录警告并置空答案。"""
        await self._ensure_tools()
        task = self._current_task(state)
        decision = await self.router.route(task)
        if not decision.selected_tool_id:
            return {"warnings": state.get("warnings", []) + [f"{task.task_id}: {decision.reason}"], "answer": ""}
        return {"task_results": state.get("task_results", []) + [{"routing": decision.model_dump()}]}

    def routed_branch(self, state: GraphState) -> str:
        """路由结果分支：未选出工具时兜底走 RAG 检索，选中显式不支持的工具才标记失败。"""
        routing = next((entry.get("routing") for entry in reversed(state.get("task_results", [])) if "routing" in entry), None)
        tool_id = routing.get("selected_tool_id") if routing else None
        if tool_id == "web-search-agent":
            return "web_search"
        if tool_id and tool_id != "rag-agent":
            return "complete_unsupported"
        return "retrieve"

    async def retrieve(self, state: GraphState) -> GraphState:
        """混合检索：取第一条检索子查询，返回知识库引用及最高相关度。"""
        query = state["retrieval_queries"][0]
        chunks = await self.retriever.retrieve(query, get_settings().retrieval_top_k)
        citations = [chunk.citation.model_dump(mode="json") for chunk in chunks]
        max_relevance = max((float(c.relevance) for c in chunks if c.relevance is not None), default=0.0)
        return {"citations": citations, "max_relevance": max_relevance}

    def documents_grounded(self, state: GraphState) -> bool:
        """确定性判定：检索到的最高相关度达到阈值即视为证据充分，避免依赖不稳的模型布尔。"""
        return float(state.get("max_relevance", 0.0)) >= get_settings().retrieval_grounded_threshold

    async def document_grade(self, state: GraphState) -> GraphState:
        """文档评分：按检索相关度确定性判定证据是否充分，并给出可读理由。"""
        count = len(state.get("citations", []))
        relevance = float(state.get("max_relevance", 0.0))
        threshold = get_settings().retrieval_grounded_threshold
        grounded = self.documents_grounded(state)
        reason = (
            f"检索到 {count} 条证据，最高相关度 {relevance:.3f}"
            f"{'（达到' if grounded else '（未达到'}判定阈值 {threshold:.2f}）。"
            f"{'证据充分，可直接回答。' if grounded else '证据不足，转网络搜索补充。'}"
        )
        warning = "Knowledge-base evidence was insufficient."
        warnings = state.get("warnings", []) if grounded else state.get("warnings", []) + [warning]
        return {"document_grade_reason": reason, "document_grounded": grounded, "warnings": warnings}

    def document_branch(self, state: GraphState) -> str:
        """文档评分分支：证据充分则生成回答，否则转网络搜索。"""
        return "generate" if not state.get("warnings", []) or "Knowledge-base evidence was insufficient." not in state["warnings"] else "web_search"

    async def web_search_node(self, state: GraphState) -> GraphState:
        """网络搜索节点：把搜索结果并入引用列表。"""
        results = await self.web_search.search(state["standalone_query"])
        citations = state.get("citations", []) + [result.model_dump(mode="json") for result in results]
        return {"citations": citations}

    async def generate(self, state: GraphState) -> GraphState:
        """基于引用生成最终回答。"""
        return {"answer": await self.llm.generate(state["standalone_query"], state.get("citations", []))}

    async def answer_grade(self, state: GraphState) -> GraphState:
        """回答评分：未通过则累计该任务的重试次数并追加警告。"""
        approved, reason = await self.llm.grade_answer(state["standalone_query"], state["answer"], state.get("citations", []))
        task = self._current_task(state)
        retries = dict(state.get("retry_counts", {}))
        retries[task.task_id] = retries.get(task.task_id, 0) + (0 if approved else 1)
        return {"answer_grade_reason": reason, "answer_approved": approved, "retry_counts": retries, "warnings": state.get("warnings", []) if approved else state.get("warnings", []) + [f"Answer requires revision: {reason}"]}

    def answer_branch(self, state: GraphState) -> str:
        """回答评分分支：重试未超限则重新检索，否则完成当前任务。"""
        task_id = state["active_task_id"]
        retry_count = state.get("retry_counts", {}).get(task_id or "", 0)
        if state.get("warnings", []) and state["warnings"][-1].startswith("Answer requires revision"):
            return "retrieve" if retry_count <= get_settings().max_agent_retries else "complete_task"
        return "complete_task"

    async def complete_task(self, state: GraphState) -> GraphState:
        """将当前任务标记为完成，并把答案写入任务结果。"""
        task = self._current_task(state)
        for raw_task in state["tasks"]:
            if raw_task["task_id"] == task.task_id:
                raw_task["status"] = TaskStatus.COMPLETED.value
        results = [entry for entry in state.get("task_results", []) if "routing" not in entry]
        results.append({"task_id": task.task_id, "status": "completed", "answer": state.get("answer", "")})
        return {"tasks": state["tasks"], "task_results": results, "active_task_id": None}

    async def complete_unsupported(self, state: GraphState) -> GraphState:
        """当前任务选中的工具当前不可用：标记失败并给出可读的说明。"""
        task = self._current_task(state)
        for raw_task in state["tasks"]:
            if raw_task["task_id"] == task.task_id:
                raw_task["status"] = TaskStatus.FAILED.value
        routing = next((entry.get("routing") for entry in reversed(state.get("task_results", [])) if "routing" in entry), None)
        reason = routing.get("reason", "") if routing else ""
        answer = f"当前查询对应的能力暂未开通，无法回答。{reason}" if reason else "当前查询对应的能力暂未开通，无法回答。"
        results = [entry for entry in state.get("task_results", []) if "routing" not in entry]
        results.append({"task_id": task.task_id, "status": "failed", "answer": answer})
        return {"tasks": state["tasks"], "task_results": results, "active_task_id": None}

    async def finalize(self, state: GraphState) -> GraphState:
        """收尾：汇总各任务答案；无答案时给出兜底文案。"""
        answers = [item["answer"] for item in state.get("task_results", []) if item.get("answer")]
        return {"answer": "\n\n".join(answers) or state.get("answer", "Unable to produce an answer.")}

    def _build_graph(self):
        """构建并编译 LangGraph 状态图，定义节点与流转边。"""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "安装langgraph以运行工作流。", 503) from exc
        graph = StateGraph(GraphState)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("plan", self.plan)
        graph.add_node("schedule", self.schedule)
        graph.add_node("route", self.route)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("document_grade", self.document_grade)
        graph.add_node("web_search", self.web_search_node)
        graph.add_node("generate", self.generate)
        graph.add_node("answer_grade", self.answer_grade)
        graph.add_node("complete_task", self.complete_task)
        graph.add_node("complete_unsupported", self.complete_unsupported)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "plan")
        graph.add_edge("plan", "schedule")
        graph.add_conditional_edges("schedule", self.schedule_next, {"route": "route", "finalize": "finalize"})
        graph.add_conditional_edges(
            "route",
            self.routed_branch,
            {"retrieve": "retrieve", "web_search": "web_search", "complete_unsupported": "complete_unsupported"},
        )
        graph.add_edge("retrieve", "document_grade")
        graph.add_conditional_edges("document_grade", self.document_branch, {"generate": "generate", "web_search": "web_search"})
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", "answer_grade")
        graph.add_conditional_edges("answer_grade", self.answer_branch, {"retrieve": "retrieve", "complete_task": "complete_task"})
        graph.add_edge("complete_task", "schedule")
        graph.add_edge("complete_unsupported", "schedule")
        graph.add_edge("finalize", END)
        return graph.compile()

    async def run(self, request: ChatRequest) -> ChatResult:
        """执行一次完整问答：运行状态图、保存会话并组装 ChatResult。"""
        if self._graph is None:
            self._graph = self._build_graph()
        session_id = request.session_id or str(uuid.uuid4())
        state = await self._graph.ainvoke(
            {"session_id": session_id, "original_query": request.query, "warnings": [], "citations": []}
        )
        await self.session_store.save(session_id, {"query": request.query, "answer": state["answer"]})
        return ChatResult(
            session_id=session_id,
            answer=state["answer"],
            citations=[Citation.model_validate(item) for item in state.get("citations", [])],
            task_summary=[item for item in state.get("task_results", []) if "task_id" in item],
            warnings=state.get("warnings", []),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """以 SSE 事件流形式输出结果：逐节点 trace、引用、评分警告、答案 token 与完成事件。"""
        if self._graph is None:
            self._graph = self._build_graph()
        session_id = request.session_id or str(uuid.uuid4())
        yield {"event": "metadata", "data": {"session_id": session_id}}
        seed = {"session_id": session_id, "original_query": request.query, "warnings": [], "citations": []}
        state: dict[str, Any] = dict(seed)
        # stream_mode="updates"：每完成一个节点就吐一次该节点的增量状态，用于逐节点观测
        async for update in self._graph.astream(seed, stream_mode="updates"):
            for node_name, partial in update.items() or []:
                state.update(partial)
                event = self._trace_event(node_name, partial)
                if event:
                    yield event
        await self.session_store.save(session_id, {"query": request.query, "answer": state.get("answer", "")})
        result = ChatResult(
            session_id=session_id,
            answer=state.get("answer", ""),
            citations=[Citation.model_validate(item) for item in state.get("citations", [])],
            task_summary=[item for item in state.get("task_results", []) if "task_id" in item],
            warnings=state.get("warnings", []),
        )

        for citation in result.citations:
            yield {"event": "citation", "data": citation.model_dump(mode="json")}
        yield {"event": "grader", "data": {"warnings": result.warnings}}
        for token in result.answer.split(" "):
            yield {"event": "token", "data": {"content": token + " "}}
        yield {"event": "done", "data": result.model_dump(mode="json")}

    def _trace_event(self, node_name: str, partial: dict[str, Any]) -> dict[str, Any] | None:
        """把单个节点的增量状态转成可观测的 trace 事件；无需展示的节点返回 None。"""
        labels = {
            "rewrite": "Query Rewrite",
            "schedule": "Task Scheduler",
            "plan": "Task Planner",
            "route": "Vector Tool Router",
            "retrieve": "Document Retrieval",
            "document_grade": "Document Grader",
            "web_search": "Web Search",
            "generate": "Answer Generation",
            "answer_grade": "Answer Grader",
        }
        label = labels.get(node_name)
        if not label:
            return None
        data: dict[str, Any] = {"node": node_name, "label": label, "output": self._node_output(node_name, partial)}
        if node_name == "route":
            routing = next((entry.get("routing") for entry in partial.get("task_results", []) if "routing" in entry), None)
            if routing:
                data["tool"] = routing.get("selected_tool_id")
                data["score"] = routing.get("score")
        return {"event": "trace", "data": data}

    @staticmethod
    def _node_output(node_name: str, partial: dict[str, Any]) -> str:
        """根据节点类型把增量状态格式化成人类可读的输出文本。"""
        if node_name == "rewrite":
            queries = partial.get("retrieval_queries") or [partial.get("standalone_query", "")]
            return f"独立问句：{partial.get('standalone_query', '')}\n检索子查询：{', '.join(queries)}\n意图：{partial.get('intent', '')}"
        if node_name == "schedule":
            return f"当前任务：{partial.get('active_task_id') or '（无，进入收尾）'}"
        if node_name == "plan":
            tasks = [
                f"- {task.get('task_id')} | {task.get('goal')} | agent_type={task.get('agent_type')}"
                for task in partial.get("tasks", [])
            ]
            return "\n".join(tasks) or "未生成可执行任务"
        if node_name == "route":
            routing = next((entry.get("routing") for entry in partial.get("task_results", []) if "routing" in entry), None)
            if not routing:
                return "未匹配到可用工具（置信度或硬约束不足）"
            selected = routing.get("selected_tool_id") or "（无，置信度不足）"
            return f"选中工具：{selected}\n相似度：{routing.get('score', '')}\n理由：{routing.get('reason', '')}"
        if node_name == "retrieve":
            citations = partial.get("citations", [])
            lines = [f"- {item.get('content', '')[:60]}" for item in citations[:5]]
            return f"检索到 {len(citations)} 条证据：\n" + "\n".join(lines)
        if node_name == "document_grade":
            reason = partial.get("document_grade_reason", "")
            # 判定结果以理由文本为准，避免 LangGraph 增量流丢失布尔字段导致展示失真
            is_grounded = "证据充分" in reason
            return f"证据是否充分：{is_grounded}\n理由：{reason}"
        if node_name == "web_search":
            citations = partial.get("citations", [])
            web_count = sum(1 for item in citations if item.get("source_type") == "web")
            return f"知识库证据不足，网络搜索补充 {web_count} 条来源"
        if node_name == "generate":
            answer = partial.get("answer", "")
            return answer[:200] + ("…" if len(answer) > 200 else "")
        if node_name == "answer_grade":
            return f"答案是否通过：{partial.get('answer_approved', False)}\n理由：{partial.get('answer_grade_reason', '')}"
        return ""
