from __future__ import annotations

from collections.abc import Iterable

from app.core.errors import AppError
from app.domain.models import PlannedTask


def validate_plan(tasks: Iterable[PlannedTask], *, max_tasks: int = 16) -> list[PlannedTask]:
    """Validate an LLM-produced task graph before it enters the scheduler."""
    normalized = list(tasks)
    if not normalized:
        raise AppError("PLAN_EMPTY", "Task Planner returned no executable tasks.", 502)
    if len(normalized) > max_tasks:
        raise AppError("PLAN_TOO_LARGE", f"Task plan exceeds the {max_tasks}-task limit.", 502)

    ids = [task.task_id for task in normalized]
    if len(set(ids)) != len(ids) or any(not task_id.strip() for task_id in ids):
        raise AppError("PLAN_INVALID", "Task plan contains duplicate or empty task IDs.", 502)
    known = set(ids)
    for task in normalized:
        if task.task_id in task.dependencies:
            raise AppError("PLAN_CYCLE", f"Task {task.task_id} depends on itself.", 502)
        missing = sorted(set(task.dependencies) - known)
        if missing:
            raise AppError("PLAN_INVALID", f"Task {task.task_id} has unknown dependencies.", 502, missing)

    graph = {task.task_id: set(task.dependencies) for task in normalized}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise AppError("PLAN_CYCLE", "Task plan contains a dependency cycle.", 502)
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)
    return normalized
