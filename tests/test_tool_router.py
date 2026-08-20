from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.domain.models import AgentType, PlannedTask
from app.services.embedding import HashEmbeddingProvider
from app.services.tool_router import VectorToolRouter, default_tool_specs


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """get_settings() 带 lru_cache，需在每次测试前后清空，保证 ROUTER_THRESHOLD 等环境变量生效。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_vector_router_selects_rag_for_document_task(monkeypatch):
    """知识库类任务应被路由到 rag-agent（阈值设为 0 以放行低分匹配）。"""
    monkeypatch.setenv("ROUTER_THRESHOLD", "0.0")
    router = VectorToolRouter(HashEmbeddingProvider())
    await router.register_many(default_tool_specs())

    decision = await router.route(
        PlannedTask(
            task_id="task-1",
            goal="搜索内部知识库文档并根据资料回答问题",
            agent_type=AgentType.RAG,
        )
    )

    assert decision.selected_tool_id == "rag-agent"
    assert decision.score >= 0


@pytest.mark.asyncio
async def test_vector_router_rejects_low_confidence(monkeypatch):
    """最高相似度低于阈值时应拒绝路由，并给出 below threshold 理由。"""
    monkeypatch.setenv("ROUTER_THRESHOLD", "1.1")
    router = VectorToolRouter(HashEmbeddingProvider())
    await router.register_many(default_tool_specs())

    decision = await router.route(PlannedTask(task_id="task-1", goal="完全不相关的任务"))

    assert decision.selected_tool_id is None
    assert "below threshold" in decision.reason


@pytest.mark.asyncio
async def test_disabled_text2sql_is_never_selected(monkeypatch):
    """被禁用的 text2sql 工具绝不参与路由，也不得出现在候选中。"""
    monkeypatch.setenv("ROUTER_THRESHOLD", "0.0")
    router = VectorToolRouter(HashEmbeddingProvider())
    await router.register_many(default_tool_specs())

    decision = await router.route(
        PlannedTask(task_id="task-1", goal="查询数据库订单总额", agent_type=AgentType.TOOLS)
    )

    # 阈值 0.0 下其他启用的 TOOLS 工具（如 web-search）可能被选中，
    # 但被禁用的 text2sql 绝不参与路由，也不得出现在候选中。
    assert decision.selected_tool_id != "text2sql-agent"
    assert all(candidate.tool_id != "text2sql-agent" for candidate in decision.alternatives)
