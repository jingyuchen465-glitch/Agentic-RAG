from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class RunContext:
    """Per-run budget and identity shared by every workflow node."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_steps: int = 32
    max_retries: int = 2
    timeout_seconds: float = 120.0
    steps: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume_step(self, node: str) -> None:
        """Consume one execution slot and fail closed when the run budget is exhausted."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise HarnessLimitError(f"step budget exceeded at {node}")
        if self.expired:
            raise HarnessLimitError("run deadline exceeded")

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.started_at >= self.timeout_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "steps": self.steps,
        }


class HarnessLimitError(RuntimeError):
    """A run exceeded a Harness budget."""


@dataclass(slots=True)
class ToolResult:
    """Stable result envelope for future executable tools."""

    ok: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent:
    """Structured trace event; the SSE adapter may serialize it directly."""

    run_id: str
    node: str
    status: Literal["started", "completed", "failed"]
    task_id: str | None = None
    duration_ms: float | None = None
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node": self.node,
            "status": self.status,
            "task_id": self.task_id,
            "duration_ms": self.duration_ms,
            "output": self.output,
            **self.metadata,
        }
