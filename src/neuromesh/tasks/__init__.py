"""Track C (Yutong): Fact Propagation, Consensus, Distributed Aggregation.

Each task implements ``neuromesh.contracts.Task``. Tasks are registered in
``_REGISTRY`` so the runner never needs editing when a task is added.

Agent state written by the round engine (``AgentTurn.state``) per task:

* ``fact_propagation``: ``{"facts": {"F1": "4821", ...}}``
* ``consensus``: ``{"vote": 0 | 1}``
* ``aggregation``: ``{"values": {"visual": 12, ...}, "sum": 345}``

See ``docs/track_c.md`` for the full specification.
"""

from __future__ import annotations

from collections.abc import Callable

from neuromesh.contracts import Task, TaskName
from neuromesh.tasks.aggregation import AggregationTask
from neuromesh.tasks.consensus import ConsensusTask
from neuromesh.tasks.fact_propagation import FactPropagationTask

_REGISTRY: dict[str, Callable[[], Task]] = {
    "fact_propagation": FactPropagationTask,
    "consensus": ConsensusTask,
    "aggregation": AggregationTask,
}

TASK_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def get_task(name: TaskName) -> Task:
    """Return the task object for ``name`` (raises ``KeyError`` for unknown names)."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {', '.join(TASK_NAMES)}") from None


__all__ = [
    "TASK_NAMES",
    "AggregationTask",
    "ConsensusTask",
    "FactPropagationTask",
    "get_task",
]
