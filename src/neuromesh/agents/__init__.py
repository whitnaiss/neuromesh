"""Track B (Armman): role prompts, JSON message protocol, round engine."""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import (
    DEFAULT_HISTORY_ROUNDS,
    DEFAULT_ROUNDS,
    LLMClient,
    RunResult,
    TaskInstance,
)


def run_network(
    graph: nx.Graph,
    instance: TaskInstance,
    llm: LLMClient,
    rounds: int = DEFAULT_ROUNDS,
    history_rounds: int = DEFAULT_HISTORY_ROUNDS,
) -> RunResult:
    """Run synchronous message passing and return every call plus final answers."""
    raise NotImplementedError("Track B: run_network")


__all__ = ["run_network"]
