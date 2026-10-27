"""Track C (Yutong): Fact Propagation, Consensus, Distributed Aggregation.

Each task implements neuromesh.contracts.Task. Register task classes here so
the runner never needs editing when a task is added.
"""

from __future__ import annotations

from neuromesh.contracts import Task, TaskName


def get_task(name: TaskName) -> Task:
    """Return the task object for ``name``."""
    raise NotImplementedError("Track C: get_task")


__all__ = ["get_task"]
