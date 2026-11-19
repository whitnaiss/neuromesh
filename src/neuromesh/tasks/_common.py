"""Helpers shared by the three tasks (Track C, Yutong).

Everything here is pure and deterministic: no LLM, no I/O. The tasks use it to
read roles off a graph, compute hop distances, and parse the agent ``state``
dicts that the round engine logs in ``CallRecord.state``.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable

import networkx as nx

from neuromesh.contracts import INPUT_ROLE, READOUT_ROLE, YEO7, CallRecord, RunResult

#: Distance reported for agents that cannot be reached from any input agent.
UNREACHABLE = -1


def nodes_with_role(graph: nx.Graph, role: str) -> list[int]:
    """Sorted ids of every node whose ``role`` attribute equals ``role``."""
    return sorted(int(v) for v, r in graph.nodes(data="role") if r == role)


def require_roles(graph: nx.Graph) -> None:
    """Raise ``ValueError`` if any node lacks a valid Yeo role."""
    bad = [v for v, r in graph.nodes(data="role") if r not in YEO7]
    if bad:
        raise ValueError(
            f"{len(bad)} node(s) have no valid Yeo role (e.g. node {bad[0]}); "
            "call neuromesh.graphs.assign_roles before making a task instance"
        )


def input_nodes(graph: nx.Graph) -> list[int]:
    return nodes_with_role(graph, INPUT_ROLE)


def readout_nodes(graph: nx.Graph) -> list[int]:
    return nodes_with_role(graph, READOUT_ROLE)


def hop_distance_from(graph: nx.Graph, sources: Iterable[int]) -> dict[int, int]:
    """Shortest hop count from the nearest source; ``UNREACHABLE`` if none."""
    sources = list(sources)
    dist = {int(v): UNREACHABLE for v in graph.nodes}
    if not sources:
        return dist
    lengths = nx.multi_source_dijkstra_path_length(graph, sources, weight=None)
    for v, d in lengths.items():
        dist[int(v)] = int(d)
    return dist


def rng_for(task: str, seed: int, graph: nx.Graph) -> random.Random:
    """Task RNG seeded by task name, seed and graph identity (stable across runs)."""
    meta = graph.graph
    key = f"{task}|{seed}|{meta.get('topology')}|{meta.get('n')}|{meta.get('seed')}"
    return random.Random(key)


# --------------------------------------------------------------------------- run logs
def states_by_round(result: RunResult) -> dict[int, dict[int, dict]]:
    """``{round: {node_id: state}}`` from the call log (last record wins)."""
    out: dict[int, dict[int, dict]] = {}
    for call in result.calls:
        out.setdefault(int(call.round), {})[int(call.node_id)] = _state_of(call)
    return out


def final_states(result: RunResult) -> dict[int, dict]:
    """Final state per node: ``final_answers`` if given, else the last logged round."""
    if result.final_answers:
        return {int(k): (v or {}) for k, v in result.final_answers.items()}
    final: dict[int, dict] = {}
    for _, states in sorted(states_by_round(result).items()):
        final.update(states)
    return final


def _state_of(call: CallRecord) -> dict:
    return call.state if (call.parse_ok and isinstance(call.state, dict)) else {}


# --------------------------------------------------------------------------- parsing
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict | None:
    """Best-effort parse of the first JSON object in ``text`` (None on failure).

    Tolerates code fences, leading prose and ``<think>`` blocks, which small
    open models often emit even when asked for bare JSON.
    """
    if not text:
        return None
    text = _THINK.sub("", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def as_int(value: object) -> int | None:
    """Coerce ``7``, ``"7"``, ``7.0`` or ``" 7 "`` to ``int``; anything else to None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return None


def norm_text(value: object) -> str:
    """Lower-case, strip and collapse whitespace; used for fact comparison."""
    return " ".join(str(value).strip().lower().split()) if value is not None else ""
