import asyncio
import time
from types import SimpleNamespace

from app.domain.models import Citation, ChatRequest, PlannedTask, RouteDecision
from app.services.workflow import AgenticWorkflow


QUERIES_HIT = []


class FakeLLM:
    async def understand(self, query):
        return {"standalone_query": query, "retrieval_queries": [query], "intent": "knowledge_lookup"}

    async def plan(self, query, intent):
        return [
            PlannedTask(task_id="t1", goal="Q1", agent_type="rag"),
            PlannedTask(task_id="t2", goal="Q2", agent_type="rag"),
        ]

    async def generate(self, query, citations):
        return "A:" + query

    async def grade_answer(self, query, answer, citations):
        return True, "ok"


class FakeRetriever:
    async def retrieve(self, query, top_k):
        QUERIES_HIT.append(query)
        time.sleep(0.1)  # simulate I/O so overlap is observable
        chunk = SimpleNamespace(
            citation=Citation(source_type="knowledge_base", document_name="doc", document_id="d1", content=f"chunk for {query}"),
            relevance=1.0,
        )
        return [chunk]


class FakeRouter:
    async def register_many(self, specs):
        return None

    async def route(self, task):
        return RouteDecision(selected_tool_id="rag-agent", score=0.9, alternatives=[], reason="rag")


class FakeWeb:
    async def search(self, query):
        return []


class FakeSession:
    async def save(self, session_id, payload):
        return None


async def main():
    wf = AgenticWorkflow(
        llm=FakeLLM(), retriever=FakeRetriever(), router=FakeRouter(), web_search=FakeWeb(), session_store=FakeSession()
    )
    t0 = time.perf_counter()
    result = await wf.run(ChatRequest(query="question"))
    elapsed = time.perf_counter() - t0
    print("elapsed_sec:", round(elapsed, 2))
    print("retrieve_queries_hit:", sorted(QUERIES_HIT))
    print("answer:", repr(result.answer))
    print("citations:", len(result.citations))
    print("task_summary:", [t["task_id"] for t in result.task_summary])
    assert set(QUERIES_HIT) == {"Q1", "Q2"}, "both parallel tasks must retrieve their own query"
    assert "A:Q1" in result.answer and "A:Q2" in result.answer, "both task answers must be aggregated"
    assert len([t for t in result.task_summary if t.get("task_id")]) == 2
    print("SMOKE OK - parallel execution verified")


asyncio.run(main())