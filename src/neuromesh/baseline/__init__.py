"""Track C (Yutong): single-agent baseline. The runner calls only run_baseline()."""

from __future__ import annotations

from neuromesh.contracts import BaselineResult, LLMClient, TaskInstance


def run_baseline(instance: TaskInstance, llm: LLMClient) -> BaselineResult:
    """Give one agent all task information at once, then score its answer."""
    raise NotImplementedError("Track C: run_baseline")


__all__ = ["run_baseline"]
