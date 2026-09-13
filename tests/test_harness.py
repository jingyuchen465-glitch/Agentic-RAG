from __future__ import annotations

import pytest

from app.core.errors import AppError
from app.domain.models import PlannedTask
from app.harness.contracts import HarnessLimitError, RunContext
from app.harness.plan_validator import validate_plan


def test_plan_validator_rejects_unknown_dependency():
    with pytest.raises(AppError) as error:
        validate_plan([PlannedTask(task_id="a", goal="a", dependencies=["missing"])])
    assert error.value.code == "PLAN_INVALID"


def test_plan_validator_rejects_cycle():
    with pytest.raises(AppError) as error:
        validate_plan(
            [
                PlannedTask(task_id="a", goal="a", dependencies=["b"]),
                PlannedTask(task_id="b", goal="b", dependencies=["a"]),
            ]
        )
    assert error.value.code == "PLAN_CYCLE"


def test_run_context_enforces_step_budget():
    context = RunContext(max_steps=1, timeout_seconds=30)
    context.consume_step("first")
    with pytest.raises(HarnessLimitError):
        context.consume_step("second")


@pytest.mark.asyncio
async def test_evaluation_runner_with_fake_workflow(monkeypatch):
    from app.harness.evaluation import EvalCase, run_evaluation
    from app.services.embedding import HashEmbeddingProvider
    from app.services.llm import FakeLLMService
    from app.services.retrieval import FakeHybridRetriever
    from app.services.tool_router import VectorToolRouter
    from app.services.workflow import AgenticWorkflow

    class NoopSessionStore:
        async def save(self, session_id, state):
            return None

    class NoopWebSearch:
        async def search(self, query):
            return []

    workflow = AgenticWorkflow(
        llm=FakeLLMService(),
        retriever=FakeHybridRetriever(),
        router=VectorToolRouter(HashEmbeddingProvider()),
        session_store=NoopSessionStore(),
        web_search=NoopWebSearch(),
    )
    report = await run_evaluation(workflow, [EvalCase(query="hello", expected_answer_contains=("No evidence",))])
    assert report["pass_rate"] == 1.0
