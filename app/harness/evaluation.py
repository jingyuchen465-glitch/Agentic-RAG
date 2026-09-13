from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.models import ChatRequest
from app.services.workflow import AgenticWorkflow


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A small, provider-agnostic case for regression evaluation."""

    query: str
    expected_answer_contains: tuple[str, ...] = ()
    expected_citation_count: int | None = None


@dataclass(slots=True)
class EvalResult:
    query: str
    answer: str
    citation_count: int
    passed: bool
    warnings: list[str] = field(default_factory=list)


async def run_evaluation(workflow: AgenticWorkflow, cases: list[EvalCase]) -> dict[str, Any]:
    """Run deterministic or live cases through the same production workflow."""
    results: list[EvalResult] = []
    for case in cases:
        result = await workflow.run(ChatRequest(query=case.query))
        contains_all = all(expected in result.answer for expected in case.expected_answer_contains)
        citation_ok = case.expected_citation_count is None or len(result.citations) >= case.expected_citation_count
        results.append(
            EvalResult(
                query=case.query,
                answer=result.answer,
                citation_count=len(result.citations),
                passed=contains_all and citation_ok,
                warnings=result.warnings,
            )
        )
    passed = sum(item.passed for item in results)
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": [
            {
                "query": item.query,
                "answer": item.answer,
                "citation_count": item.citation_count,
                "passed": item.passed,
                "warnings": item.warnings,
            }
            for item in results
        ],
    }
