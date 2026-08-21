from __future__ import annotations

import uuid
import operator
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.models import AgentType, ChatRequest, ChatResult, Citation, PlannedTask
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
    task_results: Annotated[list[dict[str, Any]], operator.add]
    task_query: str
    citations: list[dict[str, Any]]
    max_relevance: float
    answer: str
    warnings: list[str]
    retry_counts: dict[str, int]
    document_grade_reason: str
    answer_grade_reason: str


class TaskState(TypedDict, total=False):
    """单个任务的子流水线状态。每个任务一个独立实例，彼此不共享，并行执行。"""

    session_id: str
    original_query: str
    standalone_query: str
    retrieval_queries: list[str]
    intent: str
    tasks: list[dict[str, Any]]
    active_task_id: str | None
    task_results: Annotated[list[dict[str, Any]], operator.add]
    task_query: str
    citations: list[dict[str, Any]]
    max_relevance: float
    answer: str
    warnings: list[str]
    retry_counts: dict[str, int]
    document_grade_reason: str
    answer_grade_reason: str
    document_grounded: bool
    answer_approved: bool


class TaskOut(TypedDict, total=False):
    """并行任务子图允许向上层回写的通道。仅聚合字段可安全合并，
    其余标量字段（session_id/answer/citations 等）一旦由多个并行实例各自回写
    就会触发 LangGraph 的 InvalidUpdateError，故在此显式排除。"""

    task_results: Annotated[list[dict[str, Any]], operator.add]


_TRACE_LABELS = {
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

    def _ready_tasks(self, state: GraphState) -> list[dict[str, Any]]:
        """返回依赖已全部完成且尚未执行的待调度任务（并行就绪集）。"""
        done = {item["task_id"] for item in state.get("task_results", []) if item.get("task_id")}
        return [
            raw
            for raw in state["tasks"]
            if raw["task_id"] not in done and set(raw.get("dependencies", [])).issubset(done)
        ]

    def schedule(self, state: GraphState) -> GraphState:
        """调度节点：仅统计待执行任务数供流式观测，真正的并行派发在 dispatch。"""
        return {"pending": len(self._ready_tasks(state))}

    def dispatch(self, state: GraphState) -> list[Any]:
        """调度分发：把每个就绪任务作为独立 Send 并行派发；无任务则送收尾。"""
        from langgraph.types import Send

        ready = self._ready_tasks(state)
        if not ready:
            return [
                Send(
                    "finalize",
                    {
                        "session_id": state.get("session_id"),
                        "standalone_query": state.get("standalone_query", ""),
                        "task_results": state.get("task_results", []),
                    },
                )
            ]
        base_queries = state.get("retrieval_queries") or [state.get("standalone_query", "")]
        # 每个 Send 分支彼此隔离，必须自带完整上下文（含任务身份与就绪状态快照）
        return [
            Send(
                "execute_task",
                {
                    "session_id": state.get("session_id"),
                    "original_query": state.get("original_query"),
                    "standalone_query": state.get("standalone_query", ""),
                    "retrieval_queries": base_queries,
                    "intent": state.get("intent"),
                    "tasks": state["tasks"],
                    "active_task_id": task["task_id"],
                    "task_query": task.get("goal") or base_queries[0],
                    "warnings": [],
                    "retry_counts": {},
                },
            )
            for task in ready
        ]

    async def route(self, state: GraphState) -> GraphState:
        """对当前任务做向量路由；无可用工具时记录警告并置空答案。"""
        await self._ensure_tools()
        task = self._current_task(state)
        decision = await self.router.route(task)
        if not decision.selected_tool_id:
            out: dict[str, Any] = {"warnings": state.get("warnings", []) + [f"{task.task_id}: {decision.reason}"], "answer": ""}
            self._emit_stream("route", out)
            return out
        out = {"task_results": [{"routing": decision.model_dump()}]}
        self._emit_stream("route", out)
        return out

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
        """混合检索：基于任务自身查询返回知识库引用及最高相关度。"""
        query = state.get("task_query") or (state.get("retrieval_queries") or [""])[0]
        chunks = await self.retriever.retrieve(query, get_settings().retrieval_top_k)
        citations = [chunk.citation.model_dump(mode="json") for chunk in chunks]
        max_relevance = max((float(c.relevance) for c in chunks if c.relevance is not None), default=0.0)
        out = {"citations": citations, "max_relevance": max_relevance}
        self._emit_stream("retrieve", out)
        return out

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
        out = {"document_grade_reason": reason, "document_grounded": grounded, "warnings": warnings}
        self._emit_stream("document_grade", out)
        return out

    def document_branch(self, state: GraphState) -> str:
        """文档评分分支：证据充分则生成回答，否则转网络搜索。"""
        return "generate" if not state.get("warnings", []) or "Knowledge-base evidence was insufficient." not in state["warnings"] else "web_search"

    async def web_search_node(self, state: GraphState) -> GraphState:
        """网络搜索节点：基于任务自身查询把搜索结果并入引用列表。"""
        query = state.get("task_query") or state.get("standalone_query", "")
        results = await self.web_search.search(query)
        citations = state.get("citations", []) + [result.model_dump(mode="json") for result in results]
        out = {"citations": citations}
        self._emit_stream("web_search", out)
        return out

    async def generate(self, state: GraphState) -> GraphState:
        """基于引用生成当前任务的回答。"""
        query = state.get("task_query") or state.get("standalone_query", "")
        answer = await self.llm.generate(query, state.get("citations", []))
        out = {"answer": answer}
        self._emit_stream("generate", out)
        return out

    async def answer_grade(self, state: GraphState) -> GraphState:
        """回答评分：未通过则累计该任务的重试次数并追加警告。"""
        query = state.get("task_query") or state.get("standalone_query", "")
        approved, reason = await self.llm.grade_answer(query, state["answer"], state.get("citations", []))
        task = self._current_task(state)
        retries = dict(state.get("retry_counts", {}))
        retries[task.task_id] = retries.get(task.task_id, 0) + (0 if approved else 1)
        warnings = state.get("warnings", []) if approved else state.get("warnings", []) + [f"Answer requires revision: {reason}"]
        out = {"answer_grade_reason": reason, "answer_approved": approved, "retry_counts": retries, "warnings": warnings}
        self._emit_stream("answer_grade", out)
        return out

    def answer_branch(self, state: GraphState) -> str:
        """回答评分分支：重试未超限则重新检索，否则完成当前任务。"""
        task_id = state["active_task_id"]
        retry_count = state.get("retry_counts", {}).get(task_id or "", 0)
        if state.get("warnings", []) and state["warnings"][-1].startswith("Answer requires revision"):
            return "retrieve" if retry_count <= get_settings().max_agent_retries else "complete_task"
        return "complete_task"

    async def complete_task(self, state: GraphState) -> GraphState:
        """将当前任务的答案/引用/警告作为一条增量写入任务结果，供收尾汇总。"""
        task = self._current_task(state)
        return {
            "task_results": [
                {
                    "task_id": task.task_id,
                    "status": "completed",
                    "answer": state.get("answer", ""),
                    "citations": state.get("citations", []),
                    "warnings": state.get("warnings", []),
                }
            ]
        }

    async def complete_unsupported(self, state: GraphState) -> GraphState:
        """当前任务选中的工具不可用：将失败结果作为增量写入任务结果。"""
        task = self._current_task(state)
        routing = next((entry.get("routing") for entry in reversed(state.get("task_results", [])) if "routing" in entry), None)
        reason = routing.get("reason", "") if routing else ""
        answer = f"当前查询对应的能力暂未开通，无法回答。{reason}" if reason else "当前查询对应的能力暂未开通，无法回答。"
        return {"task_results": [{"task_id": task.task_id, "status": "failed", "answer": answer}]}

    async def finalize(self, state: GraphState) -> GraphState:
        """收尾：汇总各任务答案、引用与警告；无答案时给出兜底文案。"""
        results = state.get("task_results", [])
        answers = [item["answer"] for item in results if item.get("answer")]
        citations = [c for item in results for c in item.get("citations", [])]
        warnings = [w for item in results for w in item.get("warnings", [])]
        # 不回写 task_results：该字段是 operator.add reducer，回写会把聚合结果再次追加成双份
        return {
            "answer": "\n\n".join(answers) or state.get("answer", "Unable to produce an answer."),
            "citations": citations or state.get("citations", []),
            "warnings": warnings or state.get("warnings", []),
        }

    def _build_graph(self):
        """构建并编译 LangGraph：外层做改写/规划/并行调度，内层子图为单任务流水线。"""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "安装langgraph以运行工作流。", 503) from exc

        # 内层：单个任务的路由→检索/搜索→评分→生成→完成，每个任务一个独立实例并行运行
        # output_schema 把子图回写限制为 task_results，避免并行实例回写标量字段互相冲突
        task_graph = StateGraph(TaskState, output_schema=TaskOut)
        task_graph.add_node("route", self.route)
        task_graph.add_node("retrieve", self.retrieve)
        task_graph.add_node("document_grade", self.document_grade)
        task_graph.add_node("web_search", self.web_search_node)
        task_graph.add_node("generate", self.generate)
        task_graph.add_node("answer_grade", self.answer_grade)
        task_graph.add_node("complete_task", self.complete_task)
        task_graph.add_node("complete_unsupported", self.complete_unsupported)
        task_graph.add_edge(START, "route")
        task_graph.add_conditional_edges(
            "route",
            self.routed_branch,
            {"retrieve": "retrieve", "web_search": "web_search", "complete_unsupported": "complete_unsupported"},
        )
        task_graph.add_edge("retrieve", "document_grade")
        task_graph.add_conditional_edges("document_grade", self.document_branch, {"generate": "generate", "web_search": "web_search"})
        task_graph.add_edge("web_search", "generate")
        task_graph.add_edge("generate", "answer_grade")
        task_graph.add_conditional_edges("answer_grade", self.answer_branch, {"retrieve": "retrieve", "complete_task": "complete_task"})
        task_graph.add_edge("complete_task", END)
        task_graph.add_edge("complete_unsupported", END)
        task_subgraph = task_graph.compile()

        # 外层：改写→规划→并行调度→收尾；execute_task 即上文编译好的任务子图
        graph = StateGraph(GraphState)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("plan", self.plan)
        graph.add_node("schedule", self.schedule)
        graph.add_node("execute_task", task_subgraph)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "plan")
        graph.add_edge("plan", "schedule")
        graph.add_conditional_edges("schedule", self.dispatch, ["execute_task", "finalize"])
        graph.add_edge("execute_task", "schedule")
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
        # subgraphs=True 把各并行任务子图内的节点事件也上抛；finalize 节点会聚合所有任务结果纠正最终状态
        # 组合 stream_mode + subgraphs 时每个事件为 (namespace, mode, value) 三元组：
        #   - updates 只用于根层状态增量（rewrite/plan/finalize 等），子图内 updates 因 output_schema 裁剪多为 None，跳过
        #   - custom 由子图内任务节点（route/retrieve/grade/generate 等）主动上抛，作为 trace 展示
        async for namespace, mode, value in self._graph.astream(seed, stream_mode=["updates", "custom"], subgraphs=True):
            if mode == "custom":
                yield {"event": "trace", "data": {k: value[k] for k in ("node", "label", "output", "tool", "score") if k in value}}
                continue
            if namespace:  # 子图内的 updates 已改由 custom 上抛，跳过避免重复
                continue
            for node_name, partial in (value or {}).items():
                # 根层控制事件可能携带非字典增量（None），跳过以免崩溃
                if not isinstance(partial, dict):
                    continue
                # task_results 是 operator.add reducer；流式增量需逐个追加，而不是覆盖，否则会丢失并行任务结果
                extra_results = partial.pop("task_results", None) if isinstance(partial.get("task_results"), list) else None
                state.update(partial)
                if extra_results is not None:
                    state.setdefault("task_results", [])
                    state["task_results"].extend(extra_results)
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
        label = _TRACE_LABELS.get(node_name)
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
    def _emit_stream(node_name: str, partial: dict[str, Any]) -> None:
        """任务子图内节点主动上抛可观测 trace（output_schema 会裁剪 updates 通道，
        因此子图内节点改用 custom streaming 事件，确保检索/评分/生成仍能被观测）。"""
        label = _TRACE_LABELS.get(node_name)
        if not label:
            return
        try:
            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
        except Exception:
            return
        data: dict[str, Any] = {"type": "trace", "node": node_name, "label": label, "output": AgenticWorkflow._node_output(node_name, partial)}
        if node_name == "route":
            routing = next((entry.get("routing") for entry in partial.get("task_results", []) if "routing" in entry), None)
            if routing:
                data["tool"] = routing.get("selected_tool_id")
                data["score"] = routing.get("score")
        writer(data)

    @staticmethod
    def _node_output(node_name: str, partial: dict[str, Any]) -> str:
        """根据节点类型把增量状态格式化成人类可读的输出文本。"""
        if node_name == "rewrite":
            queries = partial.get("retrieval_queries") or [partial.get("standalone_query", "")]
            return f"独立问句：{partial.get('standalone_query', '')}\n检索子查询：{', '.join(queries)}\n意图：{partial.get('intent', '')}"
        if node_name == "schedule":
            return f"待执行任务：{partial.get('pending', 0)}"
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
